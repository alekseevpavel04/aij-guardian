"""Split the 46 labelled rows into a half that labelling rules are derived from and a half they are measured on.

Within each (domain, label) cell the rows are ordered by a salted hash and dealt alternately, so both
halves keep the mix of domains and labels. Rules for a labeller may cite only rows of `derive`;
agreement with the organiser is reported on `holdout`.

    python scripts/split_valid.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
SALT = "aij-guardian labelling rules"


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/valid_halves.json")
    args = ap.parse_args()

    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet")
    valid["domain"] = valid.id.str.split("__").str[0]
    halves: dict[str, list[str]] = {"derive": [], "holdout": []}
    for _, cell in valid.groupby(["domain", "label"]):
        ordered = sorted(cell.id, key=lambda i: hashlib.sha256(f"{SALT}:{i}".encode()).hexdigest())
        for k, row_id in enumerate(ordered):
            halves["derive" if k % 2 == 0 else "holdout"].append(row_id)
    label = valid.set_index("id").label
    summary = {
        name: {"rows": len(ids), "positives": int(label[ids].sum()), "ids": sorted(ids)} for name, ids in halves.items()
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({name: (s["rows"], s["positives"]) for name, s in summary.items()})


if __name__ == "__main__":
    main()
