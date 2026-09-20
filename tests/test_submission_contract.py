"""The contract the platform checks: one row out per row in, same order, label is 0 or 1."""

import pandas as pd
import pytest

from guardian.io import read_table, write_predictions

ROWS = [
    {"id": "a1", "prompt": "⟦SYSTEM⟧\nrules", "response": "⟦ASSISTANT⟧\nok"},
    {"id": "a2", "prompt": 'line with "quotes"\nand a newline', "response": "text, with comma"},
    {"id": "a3", "prompt": "⟦USER⟧\nпривет", "response": "⟦ASSISTANT⟧\nда"},
]


@pytest.fixture
def frame():
    return pd.DataFrame(ROWS)


def test_roundtrip_csv_with_newlines_and_quotes(tmp_path, frame):
    path = tmp_path / "test.csv"
    frame.to_csv(path, index=False)
    loaded = read_table(path)
    assert list(loaded["id"]) == ["a1", "a2", "a3"]
    assert loaded.loc[1, "prompt"] == 'line with "quotes"\nand a newline'
    assert loaded.loc[2, "response"] == "⟦ASSISTANT⟧\nда"


def test_parquet_is_read_even_with_csv_extension(tmp_path, frame):
    path = tmp_path / "test.csv"
    frame.to_parquet(path)
    loaded = read_table(path)
    assert list(loaded["id"]) == ["a1", "a2", "a3"]


def test_predictions_have_one_row_per_input_in_order(tmp_path, frame):
    out = tmp_path / "predictions.csv"
    write_predictions(list(frame["id"]), [1, 0, 1], out)
    written = pd.read_csv(out, dtype={"id": str})
    assert list(written.columns) == ["id", "label"]
    assert list(written["id"]) == list(frame["id"])
    assert written["label"].tolist() == [1, 0, 1]
    assert written["label"].dtype.kind == "i"


def test_length_mismatch_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="2 ids but 1 labels"):
        write_predictions(["a", "b"], [1], tmp_path / "p.csv")


def test_non_binary_label_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="must be 0 or 1"):
        write_predictions(["a"], [2], tmp_path / "p.csv")


def test_boolean_and_float_labels_are_written_as_int(tmp_path):
    out = tmp_path / "p.csv"
    write_predictions(["a", "b"], [True, 0.0], out)
    assert pd.read_csv(out)["label"].tolist() == [1, 0]


def test_ids_that_look_numeric_keep_their_string_form(tmp_path):
    path = tmp_path / "test.csv"
    pd.DataFrame({"id": ["007", "010"], "prompt": ["a", "b"], "response": ["c", "d"]}).to_csv(path, index=False)
    loaded = read_table(path)
    assert list(loaded["id"]) == ["007", "010"]
