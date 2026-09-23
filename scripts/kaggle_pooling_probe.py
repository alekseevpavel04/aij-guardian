"""Probe for a Kaggle GPU notebook: does vLLM 0.19.1 return the last-token hidden state of a Qwen3.5 hybrid model?

The platform model (Qwen3.6-35B-A3B, `Qwen3_5MoeForConditionalGeneration`) mixes linear and full
attention, and vLLM's own comment says it has no pooling model of that kind, so the path is untested.
Qwen3.5-0.8B has the same code (`qwen3_5.py`) and attention layout without the experts, and fits a T4.
The probe embeds synthetic prompts of growing length with vLLM (runner pooling, convert embed, LAST,
no normalisation) and compares each vector with the final normalised hidden state from transformers.
No competition data is used. Run in a notebook cell after `pip install vllm==0.19.1`:

    !python kaggle_pooling_probe.py
"""

import json
import time

import numpy as np
import torch

MODEL = "Qwen/Qwen3.5-0.8B"
LENGTHS = (500, 4000, 16000, 30000)


def prompts() -> list[str]:
    rng = np.random.default_rng(0)
    words = ["заказ", "рейс", "возврат", "карта", "политика", "клиент", "статус", "order", "flight", "refund"]
    return [" ".join(rng.choice(words, n)) for n in LENGTHS]


def with_vllm(texts: list[str]) -> tuple[np.ndarray, float]:
    from vllm import LLM

    llm = LLM(
        model=MODEL,
        runner="pooling",
        convert="embed",
        pooler_config={"pooling_type": "LAST", "use_activation": False},
        dtype="float16",
        max_model_len=32768,
        language_model_only=True,
        gpu_memory_utilization=0.8,
    )
    start = time.monotonic()
    out = llm.embed(texts)
    seconds = time.monotonic() - start
    vectors = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in out])
    del llm
    torch.cuda.empty_cache()
    return vectors, seconds


def with_transformers(texts: list[str]) -> np.ndarray:
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    vectors = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            hidden = model(input_ids=ids, output_hidden_states=True).hidden_states[-1]
        vectors.append(hidden[0, -1].float().cpu().numpy())
    return np.stack(vectors)


def main() -> None:
    texts = prompts()
    result: dict = {"model": MODEL, "lengths_in_words": list(LENGTHS)}
    try:
        v, seconds = with_vllm(texts)
        result["vllm"] = {"ok": True, "seconds": round(seconds, 2), "dim": int(v.shape[1])}
    except Exception as error:  # the point of the probe is to see how it fails
        result["vllm"] = {"ok": False, "error": f"{type(error).__name__}: {error}"[:2000]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    h = with_transformers(texts)
    cos = (v * h).sum(1) / np.linalg.norm(v, axis=1) / np.linalg.norm(h, axis=1)
    result["vllm_vs_transformers"] = {
        "cosine": [round(float(c), 5) for c in cos],
        "norm_ratio": [round(float(r), 4) for r in np.linalg.norm(v, axis=1) / np.linalg.norm(h, axis=1)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
