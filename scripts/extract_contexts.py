"""Cut every earlier agent turn out of the organiser's prompts as a context of its own.

Each prompt of valid.parquet holds the whole history the organiser's agent produced: every past
"⟦ASSISTANT · ход N⟧" is a turn that could itself have been a row. Cutting the prompt right before
that marker gives the context, and the turn up to its first tool result gives the response, exactly
the layout of the data (a response never contains TOOL_RESPONSE lines).

Dialogues are split in half by τ² task, alternately within each domain in the order of a salted
hash: the train half feeds labelling and minimal pairs, the check half is never trained on.
A cut that reproduces one of the 46 labelled rows is marked and kept out of both halves.

    python scripts/extract_contexts.py
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from guardian.env import setup
from guardian.tau2_tasks import reference_text, unique_task

ROOT = Path(__file__).resolve().parents[1]
TURN_RE = re.compile(r"^⟦ASSISTANT · ход (\d+)⟧[ \t]*$", re.MULTILINE)
NEXT_MARKER_RE = re.compile(r"^⟦[A-Z_]+[^⟧\n]*⟧[ \t]*$", re.MULTILINE)
RESULT_RE = re.compile(r"^[ \t]*← TOOL_RESPONSE ", re.MULTILINE)
SALT = "aij-guardian organiser contexts"


def cuts(prompt: str) -> list[tuple[int, str, str]]:
    """(turn, context, response) for every numbered agent turn inside a prompt."""
    out = []
    for m in TURN_RE.finditer(prompt):
        following = NEXT_MARKER_RE.search(prompt, m.end())
        block = prompt[m.start() : following.start() if following else len(prompt)]
        result = RESULT_RE.search(block)
        response = (block[: result.start()] if result else block).rstrip()
        out.append((int(m.group(1)), prompt[: m.start()].rstrip("\n"), response))
    return out


def halves(tasks: list[str]) -> dict[str, str]:
    """train / check, alternating within each domain in salted-hash order."""
    split = {}
    by_domain: dict[str, list[str]] = {}
    for task in tasks:
        by_domain.setdefault(task.split("__")[0], []).append(task)
    for domain_tasks in by_domain.values():
        ordered = sorted(domain_tasks, key=lambda t: hashlib.sha256(f"{SALT}:{t}".encode()).hexdigest())
        for i, task in enumerate(ordered):
            split[task] = "train" if i % 2 == 0 else "check"
    return split


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", type=Path, default=ROOT / "data/raw/valid.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--summary", type=Path, default=ROOT / "results/organizer_contexts.json")
    args = ap.parse_args()

    valid = pd.read_parquet(args.valid)
    labelled = {(r.id.split("::")[0], int(r.id.split("::t")[1])): (r.prompt, r.response) for r in valid.itertuples()}
    rows: dict[tuple[str, int], dict] = {}
    mismatches = []
    for r in valid.itertuples():
        task = r.id.split("::")[0]
        for turn, context, response in cuts(r.prompt):
            key = (task, turn)
            if key in rows:
                if (rows[key]["prompt"], rows[key]["response"]) != (context, response):
                    mismatches.append(f"{task}::t{turn}")
                continue
            if key in labelled and labelled[key] != (context, response):
                mismatches.append(f"{task}::t{turn} differs from the labelled row")
            task_ref = unique_task(r.id)
            rows[key] = {
                "id": f"org__{task}::t{turn}",
                "task": task,
                "domain": task.split("__")[0],
                "turn": turn,
                "prompt": context,
                "response": response,
                "has_calls": "→ TOOL_CALL" in response,
                "reference": reference_text(task_ref) if task_ref else "",
                "is_labelled_row": key in labelled,
            }
    if mismatches:
        raise SystemExit(f"cuts disagree with each other or with the labelled rows: {mismatches}")

    frame = pd.DataFrame(list(rows.values())).sort_values(["task", "turn"])
    split = halves(sorted(frame.task.unique()))
    frame["split"] = [
        ("labelled" if lab else split[t]) for t, lab in zip(frame.task, frame.is_labelled_row, strict=True)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    summary = {
        "contexts": len(frame),
        "tasks": int(frame.task.nunique()),
        "by_split": {s: int((frame.split == s).sum()) for s in ("train", "check", "labelled")},
        "by_domain_and_split": {f"{d}/{s}": int(n) for (d, s), n in frame.groupby(["domain", "split"]).size().items()},
        "with_calls": int(frame.has_calls.sum()),
        "with_reference": int((frame.reference != "").sum()),
        "check_tasks": sorted(t for t, s in split.items() if s == "check"),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "check_tasks"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
