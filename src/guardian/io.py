"""Reading the evaluation input and writing predictions.

The platform passes `--input <path_to_test.csv>` and `--output <path_to_predictions.csv>`, but the
extensions do not describe the formats. Input is accepted in both formats (organizer, 07.09.2026),
so it is read by content. Output is read back by the metric stage with pyarrow as parquet: a CSV
there fails stage 22 with "Parquet magic bytes not found in footer" (organizer's log quoted in the
comments, 01.09.2026). So predictions are always written as parquet, whatever the extension says.

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


def write_predictions(ids: list, labels: list, path: str | Path) -> str:
    """Write id,label with one row per input row, in input order, labels as int 0/1.

    Returns the format written. Parquet is the one the platform reads; CSV is written only when
    pyarrow is missing, so that a broken environment still leaves an output to look at.
    """
    if len(ids) != len(labels):
        raise ValueError(f"{len(ids)} ids but {len(labels)} labels")
    values = [int(v) for v in labels]
    bad = sorted(set(values) - {0, 1})
    if bad:
        raise ValueError(f"labels must be 0 or 1, got {bad}")
    ids = [str(i) for i in ids]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow is not installed, writing CSV instead of parquet", file=sys.stderr)
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "label"])
            writer.writerows(zip(ids, values, strict=True))
        return "csv"
    table = pa.table({"id": pa.array(ids, pa.string()), "label": pa.array(values, pa.int64())})
    pq.write_table(table, path)
    return "parquet"


def main_argv(argv: list[str] | None = None) -> tuple[Path, Path]:
    """Parse --input / --output the way the platform passes them."""
    argv = list(sys.argv[1:] if argv is None else argv)
    values: dict[str, str] = {}
    for flag in ("--input", "--output"):
        if flag not in argv:
            raise SystemExit(f"missing {flag}")
        values[flag] = argv[argv.index(flag) + 1]
    return Path(values["--input"]), Path(values["--output"])
