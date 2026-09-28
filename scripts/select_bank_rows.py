"""The banking sample that goes to labelling and the token pass: whole dialogues, the organiser's tasks first.

A simulated banking dialogue runs past the organiser's longest context (231947 characters of prompt and
response, about 65k tokens), and its later turns would not fit the token pass (76k). Turns longer than that
are dropped, so the sample stays within the organiser's length range; a dialogue keeps its turns up to there.

    python scripts/select_bank_rows.py --tag kbank --turns 350
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORGANISER_MAX_CHARS = 231947


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="kbank")
    ap.add_argument("--turns", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frame = pd.read_parquet(ROOT / f"data/synthetic/tau2_ru_{args.tag}_rows.parquet")
    frame = frame[frame.prompt.str.len() + frame.response.str.len() <= ORGANISER_MAX_CHARS]
    organiser = {i.split("__")[1].split("::")[0] for i in pd.read_parquet(ROOT / "data/raw/valid.parquet").id}
    organiser = {t for t in organiser if t.startswith("task_")}
    ids = sorted(frame.task_id.unique())
    random.Random(args.seed).shuffle(ids)
    ids.sort(key=lambda t: t not in organiser)

    picked, turns = [], 0
    for task in ids:
        if turns >= args.turns:
            break
        picked.append(task)
        turns += int((frame.task_id == task).sum())
    keep = frame[frame.task_id.isin(picked)]
    out = ROOT / f"data/synthetic/tau2_ru_{args.tag}_s{args.turns}_rows.parquet"
    keep.to_parquet(out, index=False)
    summary = {
        "dialogues": len(picked),
        "organiser_tasks": sorted(set(picked) & organiser),
        "turns": len(keep),
        "max_chars": int((keep.prompt.str.len() + keep.response.str.len()).max()),
        "file": out.name,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
