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
    "taxonomy_summary.json": ("taxonomy_summary.py", lambda tmp: ["--out", str(tmp / "x.json")]),
    "submission_counts.json": ("submission_counts.py", lambda tmp: ["--out", str(tmp / "x.json")]),
    "external_apb.json": (
        "build_apb.py",
        lambda tmp: ["--out", str(tmp / "apb.parquet"), "--stats", str(tmp / "x.json")],
    ),
    "external_trajerr.json": (
        "build_trajerr.py",
        lambda tmp: ["--out", str(tmp / "trajerr.parquet"), "--stats", str(tmp / "x.json")],
    ),
    "external_splits.json": (
        "split_external.py",
        lambda tmp: [
            "--sources",
            str(tmp / "apb.parquet"),
            str(tmp / "trajerr.parquet"),
            "--splits",
            str(tmp / "splits.json"),
            "--out",
            str(tmp / "x.json"),
        ],
    ),
    "rules_external.json": (
        "rules_external.py",
        lambda tmp: [
            "--sources",
            str(tmp / "apb.parquet"),
            str(tmp / "trajerr.parquet"),
            "--splits",
            str(tmp / "splits.json"),
            "--out",
            str(tmp / "x.json"),
        ],
    ),
    "source_match.json": ("source_match.py", lambda tmp: ["--out", str(tmp / "x.json")]),
    "organizer_contexts.json": (
        "extract_contexts.py",
        lambda tmp: ["--out", str(tmp / "contexts.parquet"), "--summary", str(tmp / "x.json")],
    ),
    "budget_model.json": ("budget_model.py", lambda tmp: ["--out", str(tmp / "x.json")]),
}
# runtime.json is a timing, tau2_match.json needs the network, baseline_inventory.json the unpacked
# organiser archive and tau2_reference.json a local clone of tau2-bench, so none of them is re-checked here
DATA = ROOT / "data/raw/valid.parquet"
# downloaded sources, not in git
EXTERNAL = [ROOT / "data/external/apb/tau2/test.jsonl", Path("D:/VScode_projects/TrajDebug/data/trajerrbench")]
LEADERBOARD = [ROOT / "data/external/tau2_leaderboard/submissions"]
REQUIRES = {
    "eda.json": [DATA],
    "trivial_baselines.json": [DATA],
    "rules_baseline.json": [DATA],
    "external_apb.json": [DATA, *EXTERNAL],
    "external_trajerr.json": [DATA, *EXTERNAL],
    "external_splits.json": [DATA, *EXTERNAL],
    "rules_external.json": [DATA, *EXTERNAL],
    "source_match.json": [DATA, *LEADERBOARD],
    "organizer_contexts.json": [DATA],
}
# judge runs: the summary is recomputed from the jsonl of scores, no server call is made
JUDGE_RUNS = {
    "baseline_prompt_q4_valid": [
        "--data",
        "valid",
        "--system-prompt",
        "data/raw/baseline/baseline/src/hallucination_detector/prompt.md",
    ],
    "v1_q4_valid": ["--data", "valid", "--system-prompt", "configs/prompts/judge_v1.md"],
    "baseline_prompt_think1024_q4_valid": [
        "--data",
        "valid",
        "--system-prompt",
        "data/raw/baseline/baseline/src/hallucination_detector/prompt.md",
        "--thinking",
        "--thinking-tokens",
        "1024",
    ],
    "v1_q4_extval400": [
        "--data",
        "ext_val",
        "--limit",
        "400",
        "--seed",
        "0",
        "--system-prompt",
        "configs/prompts/judge_v1.md",
    ],
    "baseline_prompt_q4_extval400": [
        "--data",
        "ext_val",
        "--limit",
        "400",
        "--seed",
        "0",
        "--system-prompt",
        "data/raw/baseline/baseline/src/hallucination_detector/prompt.md",
    ],
    "baseline_prompt_think1024_q4_extval400_trajerr": [
        "--data",
        "ext_val",
        "--limit",
        "400",
        "--seed",
        "0",
        "--source",
        "trajerrbench",
        "--system-prompt",
        "data/raw/baseline/baseline/src/hallucination_detector/prompt.md",
        "--thinking",
        "--thinking-tokens",
        "1024",
    ],
    "newerr_q4_valid": [
        "--data",
        "valid",
        "--system-prompt",
        "data/raw/baseline/baseline/src/hallucination_detector/prompt.md",
        "--append-prompt",
        "configs/prompts/only_new_errors.md",
    ],
    "newerr_q4_extval400": [
        "--data",
        "ext_val",
        "--limit",
        "400",
        "--seed",
        "0",
        "--system-prompt",
        "data/raw/baseline/baseline/src/hallucination_detector/prompt.md",
        "--append-prompt",
        "configs/prompts/only_new_errors.md",
    ],
    "labeler_v1_think2048_valid": [
        "--data",
        "valid",
        "--system-prompt",
        "configs/prompts/labeler_v1.md",
        "--reference",
        "tau2",
        "--thinking",
        "--thinking-tokens",
        "2048",
        "--answer-prefix",
        "Метка: ",
    ],
    "labeler_v2_think2048_valid": [
        "--data",
        "valid",
        "--system-prompt",
        "configs/prompts/labeler_v2.md",
        "--reference",
        "tau2",
        "--thinking",
        "--thinking-tokens",
        "2048",
        "--answer-prefix",
        "Метка: ",
    ],
}
JUDGE_COMPARES = {
    "v1_q4_valid__baseline_prompt_q4_valid": ["--a", "v1_q4_valid", "--b", "baseline_prompt_q4_valid"],
    "v1_q4_valid__baseline_prompt_q4_valid__rules_silent": [
        "--a",
        "v1_q4_valid",
        "--b",
        "baseline_prompt_q4_valid",
        "--where-rules-silent",
    ],
    "v1_q4_extval400__baseline_prompt_q4_extval400": ["--a", "v1_q4_extval400", "--b", "baseline_prompt_q4_extval400"],
    "v1_q4_extval400__baseline_prompt_q4_extval400__rules_silent": [
        "--a",
        "v1_q4_extval400",
        "--b",
        "baseline_prompt_q4_extval400",
        "--where-rules-silent",
    ],
    "v1_q4_extval400__baseline_prompt_q4_extval400__trajerrbench": [
        "--a",
        "v1_q4_extval400",
        "--b",
        "baseline_prompt_q4_extval400",
        "--source",
        "trajerrbench",
    ],
    "v1_q4_extval400__baseline_prompt_q4_extval400__agentprocessbench__rules_silent": [
        "--a",
        "v1_q4_extval400",
        "--b",
        "baseline_prompt_q4_extval400",
        "--source",
        "agentprocessbench",
        "--where-rules-silent",
    ],
    "baseline_prompt_think1024_q4_valid__baseline_prompt_q4_valid": [
        "--a",
        "baseline_prompt_think1024_q4_valid",
        "--b",
        "baseline_prompt_q4_valid",
    ],
    "baseline_prompt_think1024_q4_extval400_trajerr__baseline_prompt_q4_extval400__trajerrbench": [
        "--a",
        "baseline_prompt_think1024_q4_extval400_trajerr",
        "--b",
        "baseline_prompt_q4_extval400",
        "--source",
        "trajerrbench",
    ],
    "newerr_q4_valid__baseline_prompt_q4_valid": ["--a", "newerr_q4_valid", "--b", "baseline_prompt_q4_valid"],
    "newerr_q4_extval400__baseline_prompt_q4_extval400": [
        "--a",
        "newerr_q4_extval400",
        "--b",
        "baseline_prompt_q4_extval400",
    ],
    "newerr_q4_extval400__baseline_prompt_q4_extval400__trajerrbench": [
        "--a",
        "newerr_q4_extval400",
        "--b",
        "baseline_prompt_q4_extval400",
        "--source",
        "trajerrbench",
    ],
}
for _name, _args in JUDGE_RUNS.items():
    GENERATED[f"judge_{_name}.json"] = (
        "judge_eval.py",
        lambda tmp, a=_args, n=_name: [*a, "--name", n, "--out", str(tmp / "x.json")],
    )
    REQUIRES[f"judge_{_name}.json"] = [DATA, *EXTERNAL, ROOT / "data/interim/judge" / f"{_name}.jsonl"]
for _name, _args in JUDGE_COMPARES.items():
    GENERATED[f"judge_compare_{_name}.json"] = (
        "judge_compare.py",
        lambda tmp, a=_args: [*a, "--out", str(tmp / "x.json")],
    )
    REQUIRES[f"judge_compare_{_name}.json"] = [DATA, *EXTERNAL]


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
            missing = [path for path in REQUIRES.get(name, []) if not path.exists()]
            if missing:
                print(f"skip {name}: needs {', '.join(str(path) for path in missing)}")
                continue
            # one run per script and arguments: eda.py writes two files, judge_eval.py serves many runs
            key = (script, *extra(tmp))
            if key not in done:
                run(script, extra(tmp), tmp)
                done.add(key)
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
