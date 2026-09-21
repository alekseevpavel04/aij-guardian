"""Label unlabelled rows with the privileged labeller (task reference plus reasoning), resumably.

Which rows are labelled is decided by a salted hash of the row id against `--share`, so the choice
of a row does not depend on which other rows exist: simulations added later extend the sample and
never reshuffle it. Every labelled row is appended to data/interim/labels/<name>.jsonl with its
score and the labeller's reasoning; a rerun of the same command continues where it stopped.

    python scripts/label_rows.py --name labeler_v1_pseudo --share 0.3
"""

import argparse
import hashlib
import time
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup
from guardian.judge import JudgeClient

ROOT = Path(__file__).resolve().parents[1]
SALT = "aij-guardian pseudo-test labels"


def chosen(row_id: str, share: float) -> bool:
    digest = hashlib.sha256(f"{SALT}:{row_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < share


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=ROOT / "data/synthetic/pseudo_rows.parquet")
    ap.add_argument("--name", required=True)
    ap.add_argument("--share", type=float, default=1.0, help="share of rows to label, chosen by id hash")
    ap.add_argument("--prompt", type=Path, default=ROOT / "configs/prompts/labeler_v1.md")
    ap.add_argument("--thinking-tokens", type=int, default=2048)
    ap.add_argument("--answer-prefix", default="Метка: ")
    ap.add_argument("--server", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--split", default=None, help="keep rows of one split (column `split`), e.g. train")
    args = ap.parse_args()

    frame = pd.read_parquet(args.rows)
    frame = frame[[chosen(i, args.share) for i in frame.id]]
    if args.split:
        frame = frame[frame.split == args.split]
    # consecutive turns of one dialogue share their prompt, so the server's prefix cache is reused
    order = [c for c in ("domain", "task", "task_id", "trial", "turn") if c in frame.columns]
    frame = frame.sort_values(order)
    log = ROOT / "data/interim/labels" / f"{args.name}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    journal.repair(log)
    done = {r["id"] for r in journal.read(log)}
    todo = frame[~frame.id.isin(done)]
    if args.max_rows:
        todo = todo.head(args.max_rows)
    client = JudgeClient(
        args.server, thinking=True, thinking_tokens=args.thinking_tokens, answer_prefix=args.answer_prefix
    )
    prompt = args.prompt.read_text(encoding="utf-8")
    print(f"{len(frame)} rows chosen, {len(done)} labelled before, {len(todo)} to go", flush=True)
    started = time.monotonic()
    for i, row in enumerate(todo.itertuples(), 1):
        # an empty reference (no unique τ² task) sends no <reference> block at all
        verdict = client.score(prompt, row.prompt, row.response, row.reference or None)
        journal.append(
            log,
            {
                "id": row.id,
                "score": verdict.score,
                "reasoning": verdict.reasoning,
                "reasoning_tokens": verdict.completion_tokens,
                "prompt_tokens": verdict.prompt_tokens,
                "seconds": round(verdict.seconds, 2),
                "labeller": args.prompt.name,
                "thinking_tokens": args.thinking_tokens,
            },
        )
        elapsed = time.monotonic() - started
        print(
            f"{len(done) + i}/{len(frame)} {row.id[:60]:<60} score {verdict.score} {verdict.seconds:.0f} s, "
            f"eta {elapsed / i * (len(todo) - i) / 60:.0f} min",
            flush=True,
        )


if __name__ == "__main__":
    main()
