"""Integer counts behind each of our leaderboard submissions and what changed between them.

results/submissions.csv holds what the platform showed for each submission (F1, precision, recall,
5 decimals). The number of positives in the public test comes from results/leaderboard_forensics.json.
For every row the script recovers TP, predicted positives k and FP = k - TP, and the change in TP and
FP from the previous submission, which is what says whether an added check was precise on the test.

    python scripts/submission_counts.py
"""

import argparse
import csv
import json
from pathlib import Path

from guardian.env import setup
from guardian.metrics import counts_from_published

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["date", "archive_sha256", "strategy", "f1", "precision", "recall", "note"]


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", type=Path, default=ROOT / "results/submissions.csv")
    ap.add_argument("--forensics", type=Path, default=ROOT / "results/leaderboard_forensics.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/submission_counts.json")
    args = ap.parse_args()

    forensics = json.loads(args.forensics.read_text(encoding="utf-8"))
    positives, rows = forensics["positives_minimal"], forensics["test_size_lower_bound"]
    with args.submissions.open(encoding="utf-8") as f:
        submissions = list(csv.DictReader(f))

    out, previous = [], None
    for s in submissions:
        entry = {k: s[k] for k in COLUMNS}
        try:
            c = counts_from_published(float(s["f1"]), float(s["precision"]), float(s["recall"]), positives)
        except ValueError as error:
            out.append(entry | {"counts": None, "why_no_counts": str(error)})
            previous = None
            continue
        counts = {"tp": c.true_positives, "predicted_positive": c.predicted_positive}
        counts["fp"] = c.predicted_positive - c.true_positives
        counts["predicted_share"] = round(c.predicted_positive / rows, 4)
        if previous is not None:
            counts["delta_tp"] = counts["tp"] - previous["tp"]
            counts["delta_fp"] = counts["fp"] - previous["fp"]
            added = counts["delta_tp"] + counts["delta_fp"]
            # precision of what was added, meaningful when a version only adds predictions
            counts["precision_of_added"] = round(counts["delta_tp"] / added, 4) if added > 0 else None
        out.append(entry | {"counts": counts})
        previous = counts

    result = {"positives": positives, "rows": rows, "source": str(args.forensics.name), "submissions": out}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for e in out:
        print(e["date"], e["strategy"], e["f1"], e["counts"] or e["why_no_counts"])
    print(f"{len(out)} submissions, test with {positives} positives in {rows} rows")


if __name__ == "__main__":
    main()
