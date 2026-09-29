"""The submission's run time on the rented A100 (29.09.2026), read from the logs the rental brought home.

`scripts/rental.sh timing` ran `predict.py` of the submission folder on 548 test-like rows with a GPU memory log
beside it. The platform gives 30 minutes, weight loading included; the task asks for a 2x margin.

    python scripts/rental_timing.py
"""

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_SECONDS = 30 * 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=ROOT / "submission_work/rental/logs")
    ap.add_argument("--out", type=Path, default=ROOT / "results/rental_timing.json")
    args = ap.parse_args()

    summary = (args.logs / "rental_timing.log").read_text(encoding="utf-8")
    rows, predicted, seconds = map(float, re.search(r"(\d+) rows, (\d+) predicted 1, .* ([\d.]+) s \(", summary).groups())
    worker = (args.logs / "rental_timing_pass.log").read_text(encoding="utf-8", errors="replace")
    engine = float(re.search(r"engine up in (\d+) s", worker).group(1))
    last = re.findall(r"(\d+)/(\d+) rows, (\d+) tokens/s, (\d+) s", worker)[-1]
    with (args.logs / "rental_gpu.csv").open(encoding="utf-8") as f:
        peak = max(int(r[" memory.used [MiB]"].split()[0]) for r in csv.DictReader(f))
    result = {
        "card": "A100 PCIe 80 GB (vast.ai)",
        "engine": "vLLM 0.19.1, Qwen3.6-35B-A3B FP8, pooling of every position",
        "rows": int(rows),
        "rows_labelled_1": int(predicted),
        "total_seconds": seconds,
        "seconds_per_row": round(seconds / rows, 3),
        "engine_start_seconds": engine,
        "pass_tokens_per_second": int(last[2]),
        "peak_gpu_memory_mib": peak,
        "limit_seconds": LIMIT_SECONDS,
        "margin": round(LIMIT_SECONDS / seconds, 2),
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
