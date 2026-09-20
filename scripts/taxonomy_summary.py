"""Join the error taxonomy with what the deterministic checks caught.

`results/error_taxonomy.csv` is a judgement, not a measurement: the 23 positive examples were sorted
into categories by the agent from the organizer's `explanation` field. It has not been reviewed by
the author, and `results/taxonomy_summary.json` says so in `provenance`.

    python scripts/taxonomy_summary.py
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
NOISY = "unsupported_number"


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", type=Path, default=ROOT / "results/error_taxonomy.csv")
    ap.add_argument("--evidence", type=Path, default=ROOT / "results/rules_evidence.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/taxonomy_summary.json")
    args = ap.parse_args()

    with args.taxonomy.open(encoding="utf-8") as handle:
        taxonomy = list(csv.DictReader(handle))
    evidence = {e["id"]: e for e in json.loads(args.evidence.read_text(encoding="utf-8"))}

    missing = [row["id"] for row in taxonomy if row["id"] not in evidence]
    if missing:
        raise SystemExit(f"taxonomy has ids that are not in the evidence file: {missing}")

    caught = defaultdict(list)
    for row in taxonomy:
        fired = [k for k, v in evidence[row["id"]]["evidence"].items() if v and k != NOISY]
        caught[row["category"]].append((row["id"], fired))

    by_category = {}
    for category, rows in sorted(caught.items(), key=lambda kv: -len(kv[1])):
        hit = [example for example, fired in rows if fired]
        by_category[category] = {
            "examples": len(rows),
            "caught_by_checks": len(hit),
            "checks_that_fired": sorted({check for _, fired in rows for check in fired}),
        }

    total = len(taxonomy)
    report = {
        "provenance": (
            "categories assigned by the agent from the organizer's explanation field for the 23 positive "
            "examples of valid.parquet; not reviewed by the author"
        ),
        "positives": total,
        "mechanisms": dict(Counter(row["mechanism"] for row in taxonomy)),
        "caught_by_checks": sum(v["caught_by_checks"] for v in by_category.values()),
        "by_category": by_category,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{total} positives, {report['caught_by_checks']} caught by the deterministic checks")
    print(f"{'category':>28} {'n':>3} {'caught':>7}  checks")
    for category, m in by_category.items():
        print(f"{category:>28} {m['examples']:>3} {m['caught_by_checks']:>7}  {', '.join(m['checks_that_fired'])}")


if __name__ == "__main__":
    main()
