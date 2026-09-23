"""Where each labeller disagrees with the organiser: misses (organiser 1, labeller 0) and false alarms.

Every labeller is counted on the rows it has labelled, which differ: the local runs cover all 46, the
Claude batches only the rows they were given, so the rows count is printed next to every number.
The baseline judge is included at its fixed threshold 0.5 for reference.

    python scripts/labeller_errors.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
LOCAL = {
    "judge_baseline": "baseline_prompt_q4_valid",
    "local_v1": "labeler_v1_think2048_valid",
    "local_v1s": "labeler_v1s_think2048_valid",
    "local_v4": "labeler_v4_think2048_valid",
}
CLAUDE = {"sonnet": "sonnet.jsonl", "opus": "opus.jsonl", "opus_v4": "opus_v4.jsonl"}


def claude_labels(file_name: str) -> dict[str, int]:
    out = {}
    for path in sorted((ROOT / "data/synthetic/sonnet_labels").glob(f"*/{file_name}")):
        out |= {r["id"]: int(r["label"]) for r in journal.read(path)}
    return out


def counts(frame: pd.DataFrame) -> dict:
    y, p = frame.label, frame.pred
    return {
        "rows": len(frame),
        "true_positive": int(((y == 1) & (p == 1)).sum()),
        "miss": int(((y == 1) & (p == 0)).sum()),
        "false_alarm": int(((y == 0) & (p == 1)).sum()),
        "true_negative": int(((y == 0) & (p == 0)).sum()),
    }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/labeller_errors.json")
    args = ap.parse_args()

    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet")
    valid["domain"] = valid.id.str.split("__").str[0]
    halves = json.loads((ROOT / "results/valid_halves.json").read_text(encoding="utf-8"))
    half = {i: name for name, s in halves.items() for i in s["ids"]}
    labellers = {
        n: {r["id"]: int(r["score"] >= 0.5) for r in journal.read(ROOT / "data/interim/judge" / f"{run}.jsonl")}
        for n, run in LOCAL.items()
    }
    labellers |= {n: claude_labels(f) for n, f in CLAUDE.items()}

    result = {}
    for name, labels in labellers.items():
        frame = valid[valid.id.isin(labels)].copy()
        frame["pred"] = frame.id.map(labels)
        frame["half"] = frame.id.map(half)
        if frame.empty:
            continue
        result[name] = {
            "all": counts(frame),
            "by_half": {h: counts(g) for h, g in frame.groupby("half")},
            "by_domain": {d: counts(g) for d, g in frame.groupby("domain")},
            "misses": sorted(frame[(frame.label == 1) & (frame.pred == 0)].id),
            "false_alarms": sorted(frame[(frame.label == 0) & (frame.pred == 1)].id),
        }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, r in result.items():
        a = r["all"]
        print(
            f"{name:<15} rows {a['rows']:>2}  TP {a['true_positive']:>2}  miss {a['miss']:>2}  "
            f"false alarm {a['false_alarm']:>2}  TN {a['true_negative']:>2}"
        )


if __name__ == "__main__":
    main()
