"""Reading the evaluation input and writing predictions.

The platform passes `--input <path_to_test.csv>`, but organizers said in the comments that both
parquet and CSV are accepted, and at least one participant only got past the pipeline by reading
parquet first and falling back to CSV. So the format is decided by content,
not by extension.

Nothing here needs pandas: this module runs inside the submission, where every dependency is one
more thing that can fail at `pip install .` on a machine without network.
"""

import csv
import sys
from pathlib import Path

PARQUET_MAGIC = b"PAR1"
# Transcripts run to hundreds of thousands of characters in one field, far past the csv default of
# 128 KiB. sys.maxsize overflows the C long behind this limit on Windows, so use a 32-bit ceiling.
csv.field_size_limit(2**31 - 1)


def looks_like_parquet(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == PARQUET_MAGIC


def read_rows(path: str | Path) -> list[dict[str, str]]:
    """Rows of the input table as dicts of strings, in file order."""
    path = Path(path)
    if looks_like_parquet(path):
        return _read_parquet(path)
    try:
        return _read_csv(path)
    except (UnicodeDecodeError, csv.Error):
        return _read_parquet(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [{k: "" if v is None else v for k, v in row.items()} for row in csv.DictReader(handle)]


def _read_parquet(path: Path) -> list[dict[str, str]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    return [
        {name: "" if values[i] is None else str(values[i]) for name, values in columns.items()}
        for i in range(table.num_rows)
    ]


def write_predictions(ids: list, labels: list, path: str | Path) -> None:
    """Write id,label with one row per input row, in input order, labels as int 0/1."""
    if len(ids) != len(labels):
        raise ValueError(f"{len(ids)} ids but {len(labels)} labels")
    values = [int(v) for v in labels]
    bad = sorted(set(values) - {0, 1})
    if bad:
        raise ValueError(f"labels must be 0 or 1, got {bad}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        writer.writerows(zip(ids, values, strict=True))


def main_argv(argv: list[str] | None = None) -> tuple[Path, Path]:
    """Parse --input / --output the way the platform passes them."""
    argv = list(sys.argv[1:] if argv is None else argv)
    values: dict[str, str] = {}
    for flag in ("--input", "--output"):
        if flag not in argv:
            raise SystemExit(f"missing {flag}")
        values[flag] = argv[argv.index(flag) + 1]
    return Path(values["--input"]), Path(values["--output"])
