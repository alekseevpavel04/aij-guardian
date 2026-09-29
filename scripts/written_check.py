"""The local labeller on the written turns that the blind Opus check kept as correct.

v1s is Qwen3.6, another model family than the writer and the checker (both Opus). Its false alarms on
the organiser's rows are rare (1 of 23 zeros), so a 1 from it on a turn written as correct is a flag
worth reading; a 0 says little, it misses subtle errors. Reports how many of the checked turns v1s
labels 1, by writer version, pair and domain, with the ids and v1s's reasoning excerpt.

    python scripts/written_check.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="labeler_v1s_written_correct")
    ap.add_argument("--rows", type=Path, default=ROOT / "data/interim/written_correct.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/written_check.json")
    args = ap.parse_args()

    rows = pd.read_parquet(args.rows)
    labelled = {r["id"]: r for r in journal.read(ROOT / "data/interim/labels" / f"{args.name}.jsonl")}
    rows = rows[rows.id.isin(labelled)].copy()
    rows["v1s"] = rows.id.map(lambda i: int(labelled[i]["score"] >= 0.5))
    rows["writer"] = rows.id.str.extract(r"pair__\w+?_(0\d)")[0].map({"02": "v3", "03": "v4"})
    written = pd.concat([pd.read_parquet(f) for f in (ROOT / "data/synthetic/pairs").glob("*_0[23]*/pairs.parquet")])
    rows["pair"] = rows.id.map(written.set_index("id").pair)

    def count(part: pd.DataFrame) -> dict:
        return {"turns": len(part), "v1s_1": int(part.v1s.sum())}

    flagged = rows[rows.v1s == 1]
    result = {
        "labelled": len(rows),
        "of_checked": len(pd.read_parquet(args.rows)),
        **count(rows),
        "by_writer_and_pair": {f"{w}_{p}": count(g) for (w, p), g in rows.groupby(["writer", "pair"])},
        "by_domain": {d: count(g) for d, g in rows.groupby("domain")},
        # ids only: the labeller's reasoning quotes the organiser's dialogues, which stay out of the repository
        "flagged": [r.id for r in flagged.itertuples()],
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "flagged"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
