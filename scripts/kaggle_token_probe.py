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
                if name in f.keys():  # noqa: SIM118 (safe_open has no __contains__)
                    return f.get_tensor(name).float().cuda()
    raise KeyError("no output matrix found")


def stage(name: str) -> None:
    """One engine per process: in vLLM 0.19 deleting an LLM does not stop its engine process, so a
    second engine in the same process finds the card taken. Results go to stage_<name>.npz."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ids = token_ids()
    prompts = [TokensPrompt(prompt_token_ids=x) for x in ids]
    out: dict = {}
    if name == "last":
        llm = LLM(
            runner="pooling", convert="embed", pooler_config={"pooling_type": "LAST", "use_activation": False}, **ENGINE
        )
        out["last"] = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in llm.embed(prompts)])
    elif name == "all":
        llm = LLM(
            runner="pooling", convert="embed", pooler_config={"pooling_type": "ALL", "use_activation": False}, **ENGINE
        )
        start = time.monotonic()
        res = llm.encode(prompts, pooling_task="token_embed")
        out["seconds"] = np.array(time.monotonic() - start)
        for k, o in enumerate(res):
            out[f"states_{k}"] = torch.as_tensor(o.outputs.data).float().numpy()
    else:
        llm = LLM(runner="generate", **ENGINE)
        gen = llm.generate(prompts[:2], SamplingParams(max_tokens=1, prompt_logprobs=0))
        for k, (o, x) in enumerate(zip(gen, ids[:2], strict=True)):
            out[f"logprob_{k}"] = np.array([d[t].logprob for d, t in zip(o.prompt_logprobs[1:], x[1:], strict=True)])
    np.savez(f"stage_{name}.npz", **out)


def main() -> None:
    import subprocess
    import sys

    if len(sys.argv) > 1:
        stage(sys.argv[1])
        return
    result: dict = {"model": MODEL, "lengths_in_tokens": list(LENGTHS)}
    for name in ("last", "all", "generate"):
        code = subprocess.call([sys.executable, __file__, name])
        result[f"stage_{name}_exit"] = code
        if code:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
    last = np.load("stage_last.npz")["last"]
    all_ = np.load("stage_all.npz")
    states = [all_[f"states_{k}"] for k in range(len(LENGTHS))]
    result["all"] = {
        "rows_per_prompt": [int(s.shape[0]) for s in states],
        "dim": int(states[0].shape[1]),
        "seconds": round(float(all_["seconds"]), 2),
    }
    result["all_last_row_vs_last_cosine"] = [
        round(float(np.dot(s[-1], v) / np.linalg.norm(s[-1]) / np.linalg.norm(v)), 5)
        for s, v in zip(states, last, strict=True)
    ]
    gen = np.load("stage_generate.npz")
    w = output_matrix()
    ids = token_ids()
    comparison = []
    for k in range(2):  # the two shorter prompts
        s, x = torch.as_tensor(states[k]).cuda(), ids[k]
        ours = []
        for a in range(0, len(x) - 1, 512):  # logits over the whole vocabulary, in slices
            logp = torch.log_softmax(s[a : a + 512] @ w.T, dim=-1)
            ours.append(logp[torch.arange(logp.shape[0]), torch.tensor(x[a + 1 : a + 1 + logp.shape[0]])].cpu())
        ours = torch.cat(ours).numpy()[: len(x) - 1]
        theirs = gen[f"logprob_{k}"]
        comparison.append(
            {
                "max_abs_diff": round(float(np.abs(ours - theirs).max()), 4),
                "corr": round(float(np.corrcoef(ours, theirs)[0, 1]), 5),
            }
        )
    result["token_logprob_ours_vs_vllm"] = comparison
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
