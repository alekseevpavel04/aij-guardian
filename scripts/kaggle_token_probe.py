"""Probe for a Kaggle GPU notebook: can vLLM 0.19.1 give every token's hidden state of a Qwen3.5 hybrid model,
and do token probabilities computed from them match vLLM's own?

The uncertainty head (results/turn_ruler.json) needs, for each token of the agent's turn,
the probability the model gave it. At home they come from llama.dll; on the platform they would come
from the same pooling pass that gives the head its last-token vector, with pooling ALL instead of
LAST, times the output matrix. The probe checks, on Qwen3.5-0.8B (same code as the platform model,
fits a T4) and synthetic text:

1. pooling ALL returns one state per token, and its last row equals the LAST vector;
2. log-probabilities from those states times the output matrix (read from the checkpoint) equal the
   prompt_logprobs of vLLM's generate runner, so the states are taken after the final norm;
3. time and memory of ALL at up to 30 000 tokens.

No competition data is used. Run in a notebook cell after `pip install vllm==0.19.1`:

    !python kaggle_token_probe.py
"""

import gc
import json
import time

import numpy as np
import torch

MODEL = "Qwen/Qwen3.5-0.8B"
LENGTHS = (500, 4000, 16000, 30000)
ENGINE = {
    "model": MODEL,
    "dtype": "float16",
    "max_model_len": 32768,
    "language_model_only": True,
    "gpu_memory_utilization": 0.8,
    # a T4 has no FlashAttention 2, and FlashInfer's JIT build cannot link libcuda on Kaggle
    "attention_config": {"backend": "TRITON_ATTN"},
    "gdn_prefill_backend": "triton",
}


def token_ids() -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rng = np.random.default_rng(0)
    words = ["заказ", "рейс", "возврат", "карта", "политика", "клиент", "статус", "order", "flight", "refund"]
    ids = tokenizer(" ".join(rng.choice(words, max(LENGTHS))))["input_ids"]
    return [ids[:n] for n in LENGTHS]


def free(llm) -> None:
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def output_matrix() -> torch.Tensor:
    """lm_head, or the tied input embeddings, straight from the safetensors of the checkpoint."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    folder = snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json"])
    import glob

    names = ("lm_head.weight", "model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
    for path in glob.glob(f"{folder}/*.safetensors"):
        with safe_open(path, "pt") as f:
            for name in names:
                if name in f.keys():
                    return f.get_tensor(name).float().cuda()
    raise KeyError("no output matrix found")


def main() -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ids = token_ids()
    prompts = [TokensPrompt(prompt_token_ids=x) for x in ids]
    result: dict = {"model": MODEL, "lengths_in_tokens": list(LENGTHS)}
    try:
        llm = LLM(
            runner="pooling", convert="embed", pooler_config={"pooling_type": "LAST", "use_activation": False}, **ENGINE
        )
        last = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in llm.embed(prompts)])
        free(llm)

        llm = LLM(
            runner="pooling", convert="embed", pooler_config={"pooling_type": "ALL", "use_activation": False}, **ENGINE
        )
        start = time.monotonic()
        out = llm.encode(prompts, pooling_task="token_embed")
        seconds = time.monotonic() - start
        states = [torch.as_tensor(o.outputs.data).float() for o in out]
        result["all"] = {
            "rows_per_prompt": [int(s.shape[0]) for s in states],
            "dim": int(states[0].shape[1]),
            "seconds": round(seconds, 2),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        cos = [
            float(np.dot(s[-1].numpy(), v) / np.linalg.norm(s[-1].numpy()) / np.linalg.norm(v))
            for s, v in zip(states, last, strict=True)
        ]
        result["all_last_row_vs_last_cosine"] = [round(c, 5) for c in cos]
        print(json.dumps(result, ensure_ascii=False), flush=True)
        free(llm)

        w = output_matrix()
        ours = []
        for s, x in zip(
            states[:2], ids[:2], strict=True
        ):  # the two shorter prompts: logits of every token fit the card
            logp = torch.log_softmax(s.cuda() @ w.T, dim=-1)
            ours.append(logp[torch.arange(len(x) - 1), torch.tensor(x[1:])].cpu().numpy())

        llm = LLM(runner="generate", **ENGINE)
        gen = llm.generate(prompts[:2], SamplingParams(max_tokens=1, prompt_logprobs=0))
        theirs = []
        for o, x in zip(gen, ids[:2], strict=True):
            theirs.append(np.array([d[t].logprob for d, t in zip(o.prompt_logprobs[1:], x[1:], strict=True)]))
        free(llm)
        result["token_logprob_ours_vs_vllm"] = [
            {"max_abs_diff": round(float(np.abs(a - b).max()), 4), "corr": round(float(np.corrcoef(a, b)[0, 1]), 5)}
            for a, b in zip(ours, theirs, strict=True)
        ]
    except Exception as error:  # the point of the probe is to see how it fails
        result["error"] = f"{type(error).__name__}: {error}"[:2000]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
