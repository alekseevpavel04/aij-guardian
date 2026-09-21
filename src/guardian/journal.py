"""Durable records for long runs that must survive the computer being switched off.

A run appends one JSON line per finished unit of work and forces it to disk. After a power cut the
last line may be half-written; `read` drops such a tail and `repair` cuts it off the file before the
run appends again, so a resumed run never glues a new record onto a broken one. A broken line
anywhere else is not a power cut but corruption, and is raised rather than skipped.
Whole files (summaries, finished simulations) are written to a temporary name and renamed.
"""

import json
import os
from pathlib import Path


class CorruptJournal(ValueError):
    pass


def repair(path: Path) -> int:
    """Cut an unfinished last line off the file; returns the number of bytes removed."""
    if not path.exists():
        return 0
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    keep = data.rfind(b"\n") + 1
    with path.open("r+b") as f:
        f.truncate(keep)
        f.flush()
        os.fsync(f.fileno())
    return len(data) - keep


def read(path: Path) -> list[dict]:
    """All complete records; a half-written last line is ignored, a broken earlier one raises."""
    if not path.exists():
        return []
    lines = path.read_bytes().split(b"\n")
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            if number == len(lines):  # the last line had no newline yet: an interrupted append
                break
            raise CorruptJournal(f"{path}: line {number} is not JSON") from error
    return records


def append(path: Path, record: dict) -> None:
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("ab") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
