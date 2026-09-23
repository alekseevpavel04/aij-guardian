"""Hidden state of the verdict position for every row, from llama-server in embedding mode (T4.H1).

Each row is rendered exactly as the judge sees it (the baseline's system prompt, the row, the empty
thinking block), so the last token is the one whose next-token distribution is the judge's verdict;
with `--pooling last` the server returns that token's final hidden state. Vectors go to
data/interim/embeddings/<name>.jsonl as base64 float32, one durable line per row, so a run that
stops continues with the same command.

The server has to be started for this, generation is not available in embedding mode. Long rows are
processed in chunks of ubatch tokens, so there is no length limit, but in this mode ubatch 4096 takes
7.8 of the card's 8 GB and pushes the machine into paging (113 tokens/s); ubatch 2048 holds 6.4 GB and
runs at about 850 tokens/s, the longest row of the 46 (65 481 tokens) in 79 s:

    python scripts/serve_judge.py --set embeddings= pooling=last embd-normalize=-1 ubatch-size=2048 batch-size=2048
    python scripts/embed_rows.py --data valid --name q4_last_valid
    python scripts/embed_rows.py --data ext_val --limit 400 --name q4_last_extval400
    python scripts/embed_rows.py --rows data/synthetic/organizer_contexts.parquet --name q4_last_org
    python scripts/embed_rows.py --rows data/synthetic/pairs/*_02*/pairs.parquet --name q4_last_pairs_02
"""

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from guardian import journal
from guardian.env import setup
from guardian.judge import render

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from judge_eval import load  # noqa: E402

BASELINE_PROMPT = ROOT / "data/raw/baseline/baseline/src/hallucination_detector/prompt.md"


def post(url: str, payload: dict, timeout: float = 3600) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def embed(server: str, text: str) -> tuple[np.ndarray, int]:
    out = post(f"{server}/embeddings", {"input": text, "model": "judge"})
    return np.asarray(out["data"][0]["embedding"], dtype=np.float32), out["usage"]["prompt_tokens"]


def encode(vector: np.ndarray) -> str:
    return base64.b64encode(vector.astype(np.float32).tobytes()).decode()


def decode(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32)


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", choices=("valid", "ext_dev", "ext_val"))
    source.add_argument("--rows", type=Path, nargs="+", help="parquet files with id, prompt and response columns")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--system-prompt", type=Path, default=BASELINE_PROMPT)
    ap.add_argument("--server", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--longest-first", action="store_true", help="probe: start with the longest rows")
    args = ap.parse_args()

    frame = pd.concat(map(pd.read_parquet, args.rows)) if args.rows else load(args.data, args.limit, args.seed)
    system_prompt = args.system_prompt.read_text(encoding="utf-8")
    log = ROOT / "data/interim/embeddings" / f"{args.name}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    journal.repair(log)
    done = {r["id"] for r in journal.read(log)}
    todo = frame[~frame.id.isin(done)]
    if args.longest_first:
        todo = todo.iloc[np.argsort(-(todo.prompt.str.len() + todo.response.str.len()).to_numpy())]
    if args.max_rows:
        todo = todo.head(args.max_rows)
    print(f"{len(frame)} rows, {len(done)} embedded before, {len(todo)} to go", flush=True)
    started = time.monotonic()
    for i, row in enumerate(todo.itertuples(), 1):
        t0 = time.monotonic()
        vector, tokens = embed(args.server, render(system_prompt, row.prompt, row.response, thinking=False))
        journal.append(
            log,
            {
                "id": row.id,
                "vector": encode(vector),
                "dim": int(vector.size),
                "prompt_tokens": tokens,
                "seconds": round(time.monotonic() - t0, 2),
            },
        )
        elapsed = time.monotonic() - started
        print(
            f"{len(done) + i}/{len(frame)} {row.id[:60]:<60} {tokens} tokens {time.monotonic() - t0:.1f} s, "
            f"eta {elapsed / i * (len(todo) - i) / 60:.0f} min",
            flush=True,
        )


if __name__ == "__main__":
    main()
