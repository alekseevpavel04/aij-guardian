import pytest

from guardian.journal import CorruptJournal, append, read, repair, write_atomic


def test_records_round_trip(tmp_path):
    path = tmp_path / "log.jsonl"
    assert read(path) == []
    append(path, {"id": "a", "score": 0.5})
    append(path, {"id": "б", "score": None})
    assert read(path) == [{"id": "a", "score": 0.5}, {"id": "б", "score": None}]


def test_a_half_written_last_line_is_ignored_and_cut_before_appending(tmp_path):
    path = tmp_path / "log.jsonl"
    append(path, {"id": "a"})
    with path.open("ab") as f:
        f.write(b'{"id": "b", "sco')  # power cut in the middle of an append
    assert read(path) == [{"id": "a"}]
    assert repair(path) == len(b'{"id": "b", "sco')
    append(path, {"id": "c"})
    assert read(path) == [{"id": "a"}, {"id": "c"}]


def test_repair_leaves_a_clean_or_missing_file_alone(tmp_path):
    path = tmp_path / "log.jsonl"
    assert repair(path) == 0
    append(path, {"id": "a"})
    assert repair(path) == 0
    path.write_bytes(b"")
    assert repair(path) == 0


def test_a_broken_line_before_the_end_is_corruption(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"id": "a"}\nnot json\n{"id": "c"}\n')
    with pytest.raises(CorruptJournal):
        read(path)


def test_atomic_write_replaces_the_file_whole(tmp_path):
    path = tmp_path / "summary.json"
    write_atomic(path, "one")
    write_atomic(path, "two")
    assert path.read_text(encoding="utf-8") == "two"
    assert not (tmp_path / "summary.json.tmp").exists()
