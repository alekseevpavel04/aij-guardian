"""Several vectors per row from one pass: the mean over the turn's tokens and states at chosen positions (I-24).

llama-server started with `--pooling none` returns the final hidden state of every token. Each row is
rendered as for embed_rows.py (the baseline's prompt, the row, the empty thinking block), the turn's
token span is found by tokenizing the text up to it, and only reduced vectors are kept, one journal per
kind, in the format of embed_rows.py so that train_head_ru.vectors reads them:

- turn_mean: mean over the tokens of the turn;
- turn_last: the turn's last token;
- history_end: the token just before the turn, which has read the history but not the turn;
- tail_mean: mean over the tokens after the turn (closing tags and the empty thinking block);
- final: the last token, the same position as pooling last (a check against q4_last_org).

The full answer of a 65 000-token row is over a gigabyte of JSON, so the numbers are parsed with numpy
rather than json.

    python scripts/serve_judge.py --set embeddings= pooling=none embd-normalize=-1 ubatch-size=2048 batch-size=2048
    python scripts/embed_tokens.py --rows data/synthetic/organizer_contexts.parquet --prefix q4_tok --max-rows 3
"""

import argparse
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
from embed_rows import BASELINE_PROMPT, encode  # noqa: E402

KINDS = ("turn_mean", "turn_last", "history_end", "tail_mean", "final")


def post(url: str, payload: dict) -> bytes:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return response.read()


def count_tokens(server: str, text: str) -> int:
    return len(json.loads(post(f"{server}/tokenize", {"content": text, "add_special": False}))["tokens"])


def all_states(server: str, text: str, dim: int = 2048) -> np.ndarray:
    raw = post(f"{server}/embedding", {"content": text})
    start = raw.index(b'"embedding"')
    body = raw[raw.index(b"[", start) : raw.rindex(b"]]") + 2]
    flat = np.array(body.replace(b"[", b" ").replace(b"]", b" ").split(b","), dtype=np.float32)
    return flat.reshape(-1, dim)


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs="+", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--system-prompt", type=Path, default=BASELINE_PROMPT)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    frame = pd.concat(map(pd.read_parquet, args.rows))
    system_prompt = args.system_prompt.read_text(encoding="utf-8")
    logs = {k: ROOT / "data/interim/embeddings" / f"{args.prefix}_{k}_org.jsonl" for k in KINDS}
    for log in logs.values():
        journal.repair(log)
    done = {r["id"] for r in journal.read(logs["final"])}
    todo = frame[~frame.id.isin(done)]
    if args.max_rows:
        todo = todo.head(args.max_rows)
    print(f"{len(frame)} rows, {len(done)} done before, {len(todo)} to go", flush=True)
    started = time.monotonic()
    for i, row in enumerate(todo.itertuples(), 1):
        t0 = time.monotonic()
        text = render(system_prompt, row.prompt, row.response, thinking=False)
        cut = text.rindex("<response>\n") + len("<response>\n")
        before = count_tokens(args.server, text[:cut])
        through = count_tokens(args.server, text[: cut + len(row.response)])
        states = all_states(args.server, text)
        vectors = {
            "turn_mean": states[before:through].mean(0),
            "turn_last": states[through - 1],
            "history_end": states[before - 1],
            "tail_mean": states[through:].mean(0),
            "final": states[-1],
        }
        # the final kind is written last: a row counts as done only when all its kinds are written
        for kind in KINDS:
            journal.append(
                logs[kind],
                {
                    "id": row.id,
                    "vector": encode(vectors[kind]),
                    "dim": int(vectors[kind].size),
                    "prompt_tokens": len(states),
                    "turn_tokens": through - before,
                    "seconds": round(time.monotonic() - t0, 2),
                },
            )
        elapsed = time.monotonic() - started
        print(
            f"{len(done) + i}/{len(frame)} {row.id[:50]:<50} {len(states)} tokens, turn {through - before}, "
            f"{time.monotonic() - t0:.1f} s, eta {elapsed / i * (len(todo) - i) / 60:.0f} min",
            flush=True,
        )


if __name__ == "__main__":
    main()
