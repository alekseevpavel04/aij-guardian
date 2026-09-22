"""Agreement of a Sonnet labelling batch with the organiser, next to the local labellers on the same rows.

The rows of a batch were chosen by hand (the ones the local labeller got wrong plus a random few), so
the counts describe these rows only and are not an estimate over the 46.

    python scripts/sonnet_label_agreement.py --batch valid_10
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
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    folder = ROOT / "data/synthetic/sonnet_labels" / args.batch
    ids = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    sonnet = {r["id"]: r for r in journal.read(folder / "sonnet.jsonl")}
    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet").set_index("id").loc[ids]
    frame = pd.DataFrame({"id": ids, "organiser": valid.label.to_numpy().astype(int)})
    frame["sonnet"] = frame.id.map(lambda i: sonnet[i]["label"] if i in sonnet else None)
    for name, run in LOCAL.items():
        scores = {r["id"]: r["score"] for r in journal.read(ROOT / "data/interim/judge" / f"{run}.jsonl")}
        frame[name] = frame.id.map(lambda i, s=scores: int(s[i] >= 0.5))
    missing = int(frame.sonnet.isna().sum())
    result = {
        "batch": args.batch,
        "rows": len(frame),
        "organiser_positives": int(frame.organiser.sum()),
        "sonnet_missing": missing,
        "agree_with_organiser": {name: int((frame[name] == frame.organiser).sum()) for name in ("sonnet", *LOCAL)},
        "rows_detail": [
            {
                "id": r.id,
                "organiser": r.organiser,
                "sonnet": None if pd.isna(r.sonnet) else int(r.sonnet),
                "local_v1": r.local_v1,
                "local_v1s": r.local_v1s,
            }
            for r in frame.itertuples()
        ],
    }
    out = args.out or ROOT / "results" / f"sonnet_labels_{args.batch}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
