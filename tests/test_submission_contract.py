"""The contract the platform checks: one row out per row in, same order, label 0 or 1.

The last test runs scripts/predict.py as a subprocess, which is the only way to check the thing
that actually gets submitted rather than an import of its parts.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from guardian.io import read_rows, write_predictions
from guardian.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]
ROWS = [
    {"id": "a1", "prompt": "⟦SYSTEM⟧\nrules", "response": "⟦ASSISTANT · ход 1⟧\nok"},
    {"id": "a2", "prompt": 'line with "quotes"\nand a newline', "response": "text, with comma"},
    {"id": "a3", "prompt": "⟦USER⟧\nпривет", "response": "⟦ASSISTANT · ход 1⟧\nда"},
]


def write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_csv_with_newlines_and_quotes_survives_a_roundtrip(tmp_path):
    rows = read_rows(write_csv(tmp_path / "test.csv", ROWS))
    assert [r["id"] for r in rows] == ["a1", "a2", "a3"]
    assert rows[1]["prompt"] == 'line with "quotes"\nand a newline'
    assert rows[2]["response"] == "⟦ASSISTANT · ход 1⟧\nда"


def test_a_field_longer_than_the_csv_default_limit_is_read(tmp_path):
    long_row = [{"id": "big", "prompt": "x" * 300_000, "response": "ok"}]
    rows = read_rows(write_csv(tmp_path / "big.csv", long_row))
    assert len(rows[0]["prompt"]) == 300_000


def test_parquet_is_read_even_when_the_extension_says_csv(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "test.csv"
    pq.write_table(pa.Table.from_pylist(ROWS), path)
    assert [r["id"] for r in read_rows(path)] == ["a1", "a2", "a3"]


def test_predictions_have_one_row_per_input_in_order(tmp_path):
    out = tmp_path / "predictions.csv"
    write_predictions([r["id"] for r in ROWS], [1, 0, 1], out)
    written = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["id"] for r in written] == ["a1", "a2", "a3"]
    assert [r["label"] for r in written] == ["1", "0", "1"]


def test_length_mismatch_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="2 ids but 1 labels"):
        write_predictions(["a", "b"], [1], tmp_path / "p.csv")


def test_non_binary_label_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="must be 0 or 1"):
        write_predictions(["a"], [2], tmp_path / "p.csv")


def test_boolean_labels_are_written_as_int(tmp_path):
    out = tmp_path / "p.csv"
    write_predictions(["a", "b"], [True, False], out)
    assert [r["label"] for r in csv.DictReader(out.open(encoding="utf-8"))] == ["1", "0"]


def test_ids_that_look_numeric_keep_their_leading_zeros(tmp_path):
    rows = [{"id": "007", "prompt": "a", "response": "b"}, {"id": "010", "prompt": "c", "response": "d"}]
    assert [r["id"] for r in read_rows(write_csv(tmp_path / "t.csv", rows))] == ["007", "010"]


def test_constant_predictor_labels_everything(prompt):
    predictor = Predictor(strategy="constant", label=1)
    assert predictor.predict(prompt, "anything") == 1
    assert predictor.default_label == 1


def test_rule_predictor_defaults_to_zero_when_an_example_fails():
    assert Predictor(strategy="rules").default_label == 0


def test_entrypoint_runs_end_to_end(tmp_path):
    source = write_csv(tmp_path / "test.csv", ROWS)
    output = tmp_path / "predictions.csv"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/predict.py"), "--input", str(source), "--output", str(output)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    written = list(csv.DictReader(output.open(encoding="utf-8")))
    assert [r["id"] for r in written] == [r["id"] for r in ROWS]
    assert {r["label"] for r in written} <= {"0", "1"}


def test_entrypoint_reports_a_missing_flag(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/predict.py"), "--input", str(tmp_path / "x.csv")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
    )
    assert result.returncode != 0
    assert "--output" in result.stderr
