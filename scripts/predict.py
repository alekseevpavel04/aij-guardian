"""Submission entrypoint: python scripts/predict.py --input <test.csv> --output <predictions.csv>

One output row per input row, in input order, label 0 or 1, written as parquet whatever the output
extension (see guardian/io.py). A failure on one example is written to stderr and that example gets
the default label; the run continues. This is a one-shot batch on a platform that gives no second
chance, not a habit to carry into the rest of the code.
"""

import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
# `pip install .` runs without network on the platform; if it fails or installs nothing, the
# package is still importable straight from the archive.
sys.path.insert(0, str(ROOT / "src"))

from guardian.io import main_argv, read_rows, write_predictions  # noqa: E402
from guardian.parsing import format_recognised  # noqa: E402
from guardian.predictor import Predictor  # noqa: E402

CONFIG = ROOT / "model" / "config.json"


def main() -> None:
    started = time.monotonic()
    input_path, output_path = main_argv()
    predictor = Predictor.from_config(CONFIG) if CONFIG.exists() else Predictor()

    rows = read_rows(input_path)
    ids, labels = [], []
    failures = unrecognised = 0
    for i, row in enumerate(rows):
        ids.append(row.get("id", str(i)))
        try:
            # A row with no system turn or no tool list is one no check can fire on: a share near 1
            # means the test uses a transcript format the parser does not know.
            unrecognised += not format_recognised(row.get("prompt", ""))
            labels.append(predictor.predict(row.get("prompt", ""), row.get("response", "")))
        except Exception as error:  # one bad example must not cost the whole run
            failures += 1
            print(f"example {row.get('id', i)}: {type(error).__name__}: {error}", file=sys.stderr)
            labels.append(predictor.default_label)

    written = write_predictions(ids, labels, output_path)
    elapsed = time.monotonic() - started
    print(
        f"{len(rows)} rows, {sum(labels)} predicted 1, {failures} failed, {unrecognised} with unrecognised format, "
        f"{written} written, {elapsed:.1f} s "
        f"({elapsed / max(len(rows), 1) * 1000:.1f} ms per row)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
