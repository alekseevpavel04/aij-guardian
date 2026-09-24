"""Frozen vectors of a small encoder for every organiser context, to join the judge model's vector in one head.

Two settings. deepvk/USER2-base sees a 4096-token window (the start of the prompt with the tool list,
the end of the history, the turn), mean-pooled as in the AI-text detector project. Qwen3-Embedding-0.6B
takes 32 768 tokens, so it reads almost every context whole; a longer one keeps its first 1024 tokens
(the system prompt and the tools) and its end, and the vector is its last token, as that model is
pooled. The row is given as the baseline lays it out (the prompt, then the turn). Vectors go to
data/interim/embeddings/<name>_org.jsonl in the format of embed_rows.py.

    python scripts/encoder_embed.py --model deepvk/USER2-base --max-len 4096 --pooling mean --name user2_window
    python scripts/encoder_embed.py --model Qwen/Qwen3-Embedding-0.6B --max-len 32768 --pooling last \
        --name qwen3emb_full
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from guardian import journal
from guardian.env import setup
from guardian.judge import user_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from embed_rows import encode  # noqa: E402

HEAD_TOKENS = 1024
INSTRUCTION = "Instruct: Does the agent's last turn contain a context hallucination?\nQuery: "


def ids_for(tok, text: str, max_len: int, prefix: list[int], suffix: list[int]) -> list[int]:
    body = tok(text, add_special_tokens=False)["input_ids"]
    room = max_len - len(prefix) - len(suffix)
    if len(body) > room:
        body = body[:HEAD_TOKENS] + body[-(room - HEAD_TOKENS) :]
    return prefix + body + suffix


def main() -> None:
    setup()
    # after setup: huggingface_hub reads HF_HOME (on D) when it is imported
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--pooling", choices=("mean", "last"), required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    frame = pd.read_parquet(args.contexts)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if args.pooling == "last":
        prefix = tok(INSTRUCTION, add_special_tokens=False)["input_ids"]
        suffix = [tok.eos_token_id] if tok.eos_token_id is not None else []
    else:
        prefix, suffix = [tok.cls_token_id], [tok.sep_token_id]
    log = ROOT / "data/interim/embeddings" / f"{args.name}_org.jsonl"
    journal.repair(log)
    done = {r["id"] for r in journal.read(log)}
    todo = frame[~frame.id.isin(done)]
    print(f"{len(frame)} rows, {len(done)} done before, {len(todo)} to go", flush=True)
    started = time.monotonic()
    with torch.no_grad():
        for i, row in enumerate(todo.itertuples(), 1):
            ids = ids_for(tok, user_text(row.prompt, row.response), args.max_len, prefix, suffix)
            hidden = model(input_ids=torch.tensor([ids], device="cuda")).last_hidden_state[0].float()
            vector = hidden.mean(0) if args.pooling == "mean" else hidden[-1]
            vector = vector.cpu().numpy().astype(np.float32)
            journal.append(log, {"id": row.id, "vector": encode(vector), "dim": int(vector.size), "tokens": len(ids)})
            if i % 25 == 0:
                rate = (time.monotonic() - started) / i
                print(f"{len(done) + i}/{len(frame)}, eta {rate * (len(todo) - i) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
