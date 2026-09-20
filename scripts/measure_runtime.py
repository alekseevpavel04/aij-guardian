"""Measure how long the predictor takes per example and project it onto the 30 minute limit.

Timings are not reproducible bit for bit, so results/runtime.json is the only result file that
scripts/verify_numbers.py does not re-check; it records which machine produced it instead.

    python scripts/measure_runtime.py --repeats 3
"""

import argparse
import json
import platform
import time
from pathlib import Path

from guardian.env import setup
from guardian.io import read_rows
from guardian.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]
LIMIT_SECONDS = 30 * 60


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data/raw/valid.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/runtime.json")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--strategy", choices=("constant", "rules"), default="rules")
    args = ap.parse_args()

    rows = read_rows(args.input)
    predictor = Predictor(strategy=args.strategy)

    timings = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        for row in rows:
            predictor.predict(row["prompt"], row["response"])
        timings.append(time.perf_counter() - started)

    best = min(timings)
    per_row = best / len(rows)
    chars = sum(len(row["prompt"]) + len(row["response"]) for row in rows) / len(rows)
    report = {
        "strategy": args.strategy,
        "rows": len(rows),
        "mean_chars_per_row": round(chars),
        "repeats": args.repeats,
        "seconds_best_pass": round(best, 3),
        "ms_per_row": round(per_row * 1000, 2),
        "rows_per_30_min_limit": int(LIMIT_SECONDS / per_row),
        "machine": {
            "cpu": platform.processor(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "note": "CPU only, single process; the competition runs on one A100 with a 30 minute limit",
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.strategy}: {report['ms_per_row']} ms per row on {report['mean_chars_per_row']} chars, "
        f"{report['rows_per_30_min_limit']} rows would fit in 30 minutes"
    )


if __name__ == "__main__":
    main()
