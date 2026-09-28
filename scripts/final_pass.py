"""The judge model's pass for the final heads over a file of rows (guardian.extract), in batches, resumable.

On a rented A100 it computes the training rows (organiser contexts, pairs, τ², simulations) with the platform's
engine; inside the submission, scripts/predict.py runs it on the test rows in a process of its own, so a crash or
a hang cannot take the run down and the card is free again when it is stopped. Writes, per row, in the formats of
the home pass (scripts/embed_llama.py, scripts/turn_tokens.py):

- <out-dir>/<prefix>_final_org.jsonl: the last-token vector;
- <out-dir>/<prefix>_uncertainty_org.jsonl: token counts and the turn's log-probabilities, entropies, gaps;
- <out-dir>/<prefix>_parts_org.jsonl: the part of the turn each token belongs to;
- <out-dir>/<prefix>_skipped.jsonl: rows longer than the model's window.

The uncertainty record is written last, so a row counts as done only when all its records are there.

    python scripts/final_pass.py --rows rows.parquet --prefix fp8_org --model-dir /models/qwen36 \
        --system-prompt prompt.md --out-dir features
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardian import journal  # noqa: E402
from guardian.extract import Pass, encode  # noqa: E402


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if path.suffix == ".jsonl":
            rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            import pandas as pd

            rows += pd.read_parquet(path)[["id", "prompt", "response"]].to_dict("records")
    return rows


def batches(rows: list[dict], chars: int, most: int):
    """Consecutive rows up to about `chars` characters (a token is 3-4 characters) or `most` rows."""
    batch, size = [], 0
    for row in rows:
        n = len(row["prompt"]) + len(row["response"])
        if batch and (size + n > chars or len(batch) >= most):
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += n
    if batch:
        yield batch


def pass_job(engine: Pass, rows: list[dict], prefix: str, out_dir: Path, args: argparse.Namespace) -> bool:
    """One file of rows under one prefix; False when the time ran out."""
    logs = {k: out_dir / f"{prefix}_{k}_org.jsonl" for k in ("final", "parts", "uncertainty")}
    skipped_log = out_dir / f"{prefix}_skipped.jsonl"
    for log in [*logs.values(), skipped_log]:
        journal.repair(log)
    done = {r["id"] for r in journal.read(logs["uncertainty"])} | {r["id"] for r in journal.read(skipped_log)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{prefix}: {len(rows)} rows, {len(done)} done before, {len(todo)} to go", flush=True)
    passed, tokens, t0 = 0, 0, time.monotonic()
    for batch in batches(todo, args.batch_chars, args.batch_rows):
        if args.stop_at is not None and time.time() > args.stop_at:
            print("time is up", flush=True)
            return False
        for row, record in zip(batch, engine.run(batch), strict=True):
            if record is None:
                journal.append(skipped_log, {"id": row["id"]})
                continue
            journal.append(
                logs["final"],
                {"id": record["id"], "vector": encode(record["vector"]), "dim": int(record["vector"].size)},
            )
            journal.append(logs["parts"], {"id": record["id"], "parts": record["parts"]})
            journal.append(
                logs["uncertainty"],
                {k: record[k] for k in ("id", "prompt_tokens", "turn_tokens", "logp", "entropy", "margin")},
            )
            tokens += record["prompt_tokens"]
        passed += len(batch)
        elapsed = time.monotonic() - t0
        print(
            f"{prefix}: {len(done) + passed}/{len(rows)} rows, {tokens / elapsed:.0f} tokens/s, {elapsed:.0f} s",
            flush=True,
        )
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs="+", default=None, help="parquet or jsonl with id, prompt, response")
    ap.add_argument("--prefix", default=None)
    ap.add_argument(
        "--job", nargs="+", default=[], help="rows_file:prefix pairs, passed in order with one engine (the rental)"
    )
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--system-prompt", type=Path, required=True, help="the judge prompt (the baseline's)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="the rest holds the output matrix")
    ap.add_argument("--engine-json", default=None, help="more vLLM engine arguments, e.g. dtype for a T4")
    ap.add_argument("--batch-chars", type=int, default=600_000)
    ap.add_argument("--batch-rows", type=int, default=32)
    ap.add_argument("--stop-at", type=float, default=None, help="unix time after which no new batch starts")
    args = ap.parse_args()

    jobs = [(read_rows(args.rows), args.prefix)] if args.rows else []
    for job in args.job:
        path, prefix = job.rsplit(":", 1)
        jobs.append((read_rows([Path(path)]), prefix))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    engine = Pass(
        args.model_dir,
        args.system_prompt.read_text(encoding="utf-8"),
        args.max_model_len,
        args.gpu_memory_utilization,
        json.loads(args.engine_json) if args.engine_json else None,
    )
    print(f"engine up in {time.monotonic() - started:.0f} s", flush=True)
    for rows, prefix in jobs:
        if not pass_job(engine, rows, prefix, args.out_dir, args):
            break


if __name__ == "__main__":
    main()
