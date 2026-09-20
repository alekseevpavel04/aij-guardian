"""Submission entrypoint: python scripts/predict.py --input <test.csv> --output <predictions.csv>

One output row per input row, in input order, label 0 or 1. A failure on one example is written to
stderr and that example gets the default label; the run continues. This is a one-shot batch on a
platform that gives no second chance, not a habit to carry into the rest of the code.
"""

import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTHONHASHSEED", "0")

from guardian.io import main_argv, read_rows, write_predictions
from guardian.predictor import Predictor

CONFIG = Path(__file__).resolve().parents[1] / "model" / "config.json"


def main() -> None:
    started = time.monotonic()
    input_path, output_path = main_argv()
    predictor = Predictor.from_config(CONFIG) if CONFIG.exists() else Predictor()

    rows = read_rows(input_path)
    ids, labels = [], []
    failures = 0
    for i, row in enumerate(rows):
        ids.append(row.get("id", str(i)))
        try:
            labels.append(predictor.predict(row.get("prompt", ""), row.get("response", "")))
        except Exception as error:  # one bad example must not cost the whole run
            failures += 1
            print(f"example {row.get('id', i)}: {type(error).__name__}: {error}", file=sys.stderr)
            labels.append(predictor.default_label)

    write_predictions(ids, labels, output_path)
    elapsed = time.monotonic() - started
    print(
        f"{len(rows)} rows, {sum(labels)} predicted 1, {failures} failed, {elapsed:.1f} s "
        f"({elapsed / max(len(rows), 1) * 1000:.1f} ms per row)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
