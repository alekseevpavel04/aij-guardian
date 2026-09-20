"""Recompute every published number and fail if a committed file disagrees.

Each result file in `results/` is regenerated into a temporary directory by the script that owns it,
then compared with what is committed. A number that appears in the README or in a report but cannot
be reproduced this way does not exist.

    python scripts/verify_numbers.py
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# result file -> (script, extra arguments that redirect its output into the temporary directory)
GENERATED = {
    "leaderboard_forensics.json": ("leaderboard_forensics.py", lambda tmp: ["--out", str(tmp / "x.json")]),
    "eda.json": ("eda.py", lambda tmp: ["--out-dir", str(tmp)]),
    "trivial_baselines.json": ("eda.py", lambda tmp: ["--out-dir", str(tmp)]),
    "rules_baseline.json": ("rules_baseline.py", lambda tmp: ["--out", str(tmp / "x.json")]),
}
NEEDS_DATA = {"eda.json", "trivial_baselines.json", "rules_baseline.json"}
DATA = ROOT / "data/raw/valid.parquet"


def run(script: str, extra: list[str], tmp: Path) -> None:
    result = subprocess.run(
        [PYTHON, str(ROOT / "scripts" / script), *extra], capture_output=True, text=True, encoding="utf-8", cwd=ROOT
    )
    if result.returncode != 0:
        raise SystemExit(f"{script} failed:\n{result.stderr}")


def first_difference(a: object, b: object, path: str = "") -> str | None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key}: present in only one of the two"
            if (diff := first_difference(a[key], b[key], f"{path}.{key}")) is not None:
                return diff
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} items vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if (diff := first_difference(x, y, f"{path}[{i}]")) is not None:
                return diff
        return None
    return None if a == b else f"{path}: {a!r} committed, {b!r} recomputed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    if not DATA.exists():
        print(f"{DATA} is missing: the checks that need the labelled file cannot run", file=sys.stderr)

    failures = []
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        done = set()
        for name, (script, extra) in GENERATED.items():
            if name in NEEDS_DATA and not DATA.exists():
                print(f"skip {name}: needs {DATA.name}")
                continue
            if script not in done:
                run(script, extra(tmp), tmp)
                done.add(script)
            committed_path = args.results / name
            # scripts that take --out write to the name given there, scripts that take --out-dir keep theirs
            recomputed_path = tmp / ("x.json" if "--out" in extra(tmp) else name)
            if not committed_path.exists():
                failures.append(f"{name}: not committed")
                continue
            committed = json.loads(committed_path.read_text(encoding="utf-8"))
            recomputed = json.loads(recomputed_path.read_text(encoding="utf-8"))
            recomputed.pop("input", None)
            committed.pop("input", None)
            diff = first_difference(committed, recomputed)
            if diff:
                failures.append(f"{name}{diff}")
            else:
                print(f"ok  {name}")

    if failures:
        print("\nmismatches:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("\nall recomputed numbers match the committed files")


if __name__ == "__main__":
    main()
