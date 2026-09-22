"""Agreement of Claude labelling batches (Sonnet, Opus) with the organiser, next to the local labellers.

The rows of a batch were chosen by hand (the ones the local labeller got wrong plus a random few), so
the counts describe these rows only and are not an estimate over the 46.

    python scripts/sonnet_label_agreement.py --batch valid_10 --labellers sonnet opus
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
LOCAL = {"local_v1": "labeler_v1_think2048_valid", "local_v1s": "labeler_v1s_think2048_valid"}


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--labellers", nargs="+", default=["sonnet"], help="<name>.jsonl files in the batch folder")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    folder = ROOT / "data/synthetic/sonnet_labels" / args.batch
    ids = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet").set_index("id").loc[ids]
    frame = pd.DataFrame({"id": ids, "organiser": valid.label.to_numpy().astype(int)})
    for name in args.labellers:
        labels = {r["id"]: r["label"] for r in journal.read(folder / f"{name}.jsonl")}
        frame[name] = frame.id.map(labels)
    for name, run in LOCAL.items():
        scores = {r["id"]: r["score"] for r in journal.read(ROOT / "data/interim/judge" / f"{run}.jsonl")}
        frame[name] = frame.id.map(lambda i, s=scores: int(s[i] >= 0.5))
    names = [*args.labellers, *LOCAL]
    result = {
        "batch": args.batch,
        "rows": len(frame),
        "organiser_positives": int(frame.organiser.sum()),
        "missing": {name: int(frame[name].isna().sum()) for name in args.labellers},
        "agree_with_organiser": {name: int((frame[name] == frame.organiser).sum()) for name in names},
        "rows_detail": [
            {"id": r["id"], "organiser": int(r["organiser"])}
            | {name: None if pd.isna(r[name]) else int(r[name]) for name in names}
            for r in frame.to_dict("records")
        ],
    }
    out = args.out or ROOT / "results" / f"sonnet_labels_{args.batch}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
