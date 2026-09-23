"""Probe for a Kaggle GPU notebook: does vLLM 0.19.1 return the last-token hidden state of a Qwen3.5 hybrid model?

The platform model (Qwen3.6-35B-A3B, `Qwen3_5MoeForConditionalGeneration`) mixes linear and full
attention, and vLLM's own comment says it has no pooling model of that kind, so the path is untested.
Qwen3.5-0.8B has the same code (`qwen3_5.py`) and attention layout without the experts, and fits a T4.

The probe embeds synthetic prompts of growing length (runner pooling, convert embed, LAST, no
normalisation) twice: with the defaults, where a long prompt is prefilled in chunks of 8192 tokens
and prefixes may come from the cache, and as a reference with the whole prompt in one chunk and no
prefix cache. The linear-attention state has to be carried between chunks and cached prefixes; if it
is not, long prompts give vectors unlike the reference while short ones agree. The transformers
build that comes with vLLM 0.19.1 cannot construct this checkpoint, so vLLM is its own reference.
No competition data is used. Run in a notebook cell after `pip install vllm==0.19.1`:

    !python kaggle_pooling_probe.py
"""

import gc
import json
import time

import numpy as np
import torch

MODEL = "Qwen/Qwen3.5-0.8B"
LENGTHS = (500, 4000, 16000, 30000)  # tokens, under max_model_len


def prompts() -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rng = np.random.default_rng(0)
    words = ["заказ", "рейс", "возврат", "карта", "политика", "клиент", "статус", "order", "flight", "refund"]
    ids = tokenizer(" ".join(rng.choice(words, max(LENGTHS))))["input_ids"]
    return [tokenizer.decode(ids[:n]) for n in LENGTHS]


def with_vllm(texts: list[str], **engine) -> tuple[np.ndarray, float]:
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
        # a T4 has no FlashAttention 2, and FlashInfer's JIT build cannot link libcuda on Kaggle;
        # Triton kernels need no C++ build (the platform's A100 would take FlashAttention 2)
        attention_config={"backend": "TRITON_ATTN"},
        gdn_prefill_backend="triton",
        **engine,
    )
    start = time.monotonic()
    out = llm.embed(texts)
    seconds = time.monotonic() - start
    vectors = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in out])
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return vectors, seconds


def main() -> None:
    texts = prompts()
    result: dict = {"model": MODEL, "lengths_in_tokens": list(LENGTHS)}
    try:
        chunked, seconds = with_vllm(texts)
        result["vllm"] = {"ok": True, "seconds": round(seconds, 2), "dim": int(chunked.shape[1])}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        whole, seconds = with_vllm(texts, max_num_batched_tokens=32768, enable_prefix_caching=False)
    except Exception as error:  # the point of the probe is to see how it fails
        result["error"] = f"{type(error).__name__}: {error}"[:2000]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    cos = (chunked * whole).sum(1) / np.linalg.norm(chunked, axis=1) / np.linalg.norm(whole, axis=1)
    result["default_vs_one_chunk_no_cache"] = {
        "cosine": [round(float(c), 5) for c in cos],
        "norm_ratio": [round(float(r), 4) for r in np.linalg.norm(chunked, axis=1) / np.linalg.norm(whole, axis=1)],
        "seconds_one_chunk": round(seconds, 2),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
