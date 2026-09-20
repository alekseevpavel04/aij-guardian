"""Reading the evaluation input and writing predictions.

The platform passes `--input <path_to_test.csv>`, but organizers said in the comments that both
parquet and CSV are accepted, and at least one participant only got past the pipeline by reading
parquet first and falling back to CSV. So the format is decided by content,
not by extension.
"""

from pathlib import Path

import pandas as pd

PARQUET_MAGIC = b"PAR1"
REQUIRED_OUTPUT_COLUMNS = ["id", "label"]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    head = path.open("rb").read(4)
    if head == PARQUET_MAGIC:
        return pd.read_parquet(path)
    try:
        return pd.read_parquet(path)
    except Exception:  # the file is not parquet; CSV is the documented fallback
        return pd.read_csv(path, dtype={"id": str}, keep_default_na=False)


def write_predictions(ids: list, labels: list, path: str | Path) -> None:
    """Write id,label with one row per input row, in input order, labels as int 0/1."""
    if len(ids) != len(labels):
        raise ValueError(f"{len(ids)} ids but {len(labels)} labels")
    frame = pd.DataFrame({"id": ids, "label": [int(v) for v in labels]})
    bad = set(frame["label"].unique()) - {0, 1}
    if bad:
        raise ValueError(f"labels must be 0 or 1, got {sorted(bad)}")
    frame.to_csv(path, index=False)
