"""Freeze the external validation: split the external sets into ext_dev and ext_val by τ² user.

Rows of one user share database records across tasks and trajectories, and the same user appears in
AgentProcessBench and TrajErrBench, so the split is by the `group` column over the union of both
sets, never by row. A group goes to ext_val when a salted hash of its name falls below the share:
the assignment of a group does not depend on which other groups exist, so adding a source later does
not move the groups already frozen.

ext_val is for measurements only. Errors on it are not read; if they are, docs/DECISIONS.md records
it and the part is considered spent.

    python scripts/split_external.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (ROOT / "data/external/apb_tau2.parquet", ROOT / "data/external/trajerr_tau2.parquet")
SALT = "aij-guardian ext_val 2026-09-21"


def in_val(group: str, share: float) -> bool:
    digest = hashlib.sha256(f"{SALT}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < share


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, nargs="+", default=list(SOURCES))
    ap.add_argument("--val-share", type=float, default=0.4)
    ap.add_argument("--splits", type=Path, default=ROOT / "data/external/splits.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/external_splits.json")
    args = ap.parse_args()

    frame = pd.concat([pd.read_parquet(s) for s in args.sources], ignore_index=True)
    groups = sorted(frame.group.unique())
    val_groups = {g for g in groups if in_val(g, args.val_share)}
    frame["part"] = frame.group.map(lambda g: "ext_val" if g in val_groups else "ext_dev")

    args.splits.write_text(
        json.dumps({part: sorted(frame.id[frame.part == part]) for part in ("ext_dev", "ext_val")}, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "salt": SALT,
        "val_share_of_groups": args.val_share,
        "groups": len(groups),
        "val_groups": sorted(val_groups),
        "parts": {},
    }
    for part, g in frame.groupby("part"):
        summary["parts"][part] = {
            "rows": len(g),
            "positives": int(g.weak_label.sum()),
            "groups": int(g.group.nunique()),
            "by_source": {s: {"rows": len(x), "positives": int(x.weak_label.sum())} for s, x in g.groupby("source")},
            "by_domain": {d: {"rows": len(x), "positives": int(x.weak_label.sum())} for d, x in g.groupby("domain")},
        }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for part, s in summary["parts"].items():
        print(part, s["rows"], "rows,", s["positives"], "positives,", s["groups"], "groups", s["by_domain"])


if __name__ == "__main__":
    main()
