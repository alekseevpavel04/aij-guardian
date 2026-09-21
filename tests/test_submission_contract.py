"""The contract the platform checks: one row out per row in, same order, label 0 or 1, as parquet.

The last tests build the real archive, unpack it and run scripts/predict.py from it as a subprocess,
which is the only way to check the thing that actually gets submitted rather than an import of its
parts.
"""

import csv
import subprocess
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from guardian.io import PARQUET_MAGIC, read_rows, write_predictions
from guardian.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_submission  # noqa: E402


def read_predictions(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


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
    path = tmp_path / "test.csv"
    pq.write_table(pa.Table.from_pylist(ROWS), path)
    assert [r["id"] for r in read_rows(path)] == ["a1", "a2", "a3"]


def test_predictions_are_parquet_even_when_the_path_says_csv(tmp_path):
    out = tmp_path / "predictions.csv"
    assert write_predictions([r["id"] for r in ROWS], [1, 0, 1], out) == "parquet"
    assert out.read_bytes()[:4] == PARQUET_MAGIC


def test_predictions_have_one_row_per_input_in_order(tmp_path):
    out = tmp_path / "predictions.csv"
    write_predictions([r["id"] for r in ROWS], [1, 0, 1], out)
    assert read_predictions(out) == [{"id": "a1", "label": 1}, {"id": "a2", "label": 0}, {"id": "a3", "label": 1}]


def test_length_mismatch_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="2 ids but 1 labels"):
        write_predictions(["a", "b"], [1], tmp_path / "p.csv")


def test_non_binary_label_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="must be 0 or 1"):
        write_predictions(["a"], [2], tmp_path / "p.csv")


def test_boolean_labels_are_written_as_int(tmp_path):
    out = tmp_path / "p.csv"
    write_predictions(["a", "b"], [True, False], out)
    assert [r["label"] for r in read_predictions(out)] == [1, 0]


def test_ids_that_look_numeric_keep_their_leading_zeros(tmp_path):
    rows = [{"id": "007", "prompt": "a", "response": "b"}, {"id": "010", "prompt": "c", "response": "d"}]
    assert [r["id"] for r in read_rows(write_csv(tmp_path / "t.csv", rows))] == ["007", "010"]


def test_constant_predictor_labels_everything(prompt):
    predictor = Predictor(strategy="constant", label=1)
    assert predictor.predict(prompt, "anything") == 1
    assert predictor.default_label == 1


def test_rule_predictor_defaults_to_zero_when_an_example_fails():
    assert Predictor(strategy="rules").default_label == 0


def run_entrypoint(root: Path, source: Path, output: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(root / "scripts/predict.py"), "--input", str(source), "--output", str(output)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=root,
    )


def test_entrypoint_runs_end_to_end(tmp_path):
    source = write_csv(tmp_path / "test.csv", ROWS)
    output = tmp_path / "predictions.csv"
    result = run_entrypoint(ROOT, source, output)
    assert result.returncode == 0, result.stderr
    written = read_predictions(output)
    assert [r["id"] for r in written] == [r["id"] for r in ROWS]
    assert {r["label"] for r in written} <= {0, 1}


@pytest.fixture
def archive(tmp_path):
    build_submission.build(tmp_path / "submission", "rules", 1, "0.0.0")
    path = tmp_path / "guardian.zip"
    build_submission.zip_dir(tmp_path / "submission", path)
    return path


def test_archive_has_the_files_at_its_root_with_unix_permissions(archive):
    assert build_submission.check_archive(archive) == []
    names = zipfile.ZipFile(archive).namelist()
    assert "scripts/predict.py" in names
    assert not any(n.startswith("submission/") for n in names)


def test_archive_check_rejects_a_top_level_folder_and_windows_permissions(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("submission/scripts/predict.py", "")
    problems = build_submission.check_archive(bad)
    assert any("scripts/predict.py is not at the root" in p for p in problems)
    assert any("permissions" in p for p in problems)


def test_unpacked_archive_runs_on_parquet_input(archive, tmp_path):
    unpacked = tmp_path / "unpacked"
    zipfile.ZipFile(archive).extractall(unpacked)
    source = tmp_path / "test.csv"
    pq.write_table(pa.Table.from_pylist(ROWS), source)
    output = tmp_path / "out" / "predictions.csv"
    output.parent.mkdir()
    result = run_entrypoint(unpacked, source, output)
    assert result.returncode == 0, result.stderr
    assert "parquet written" in result.stderr
    assert [r["id"] for r in read_predictions(output)] == [r["id"] for r in ROWS]


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
