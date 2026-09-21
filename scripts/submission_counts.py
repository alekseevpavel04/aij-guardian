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


def candidates_from_f1(f1_text: str, positives: int, rows: int) -> list[dict]:
    """Every (TP, k) pair that reproduces an F1 published without precision and recall.

    F1 = 2 TP / (k + P) alone does not fix the counts; the list is what can honestly be said until the
    other two rates are read off the platform. The tolerance follows the number of decimals shown.
    """
    f1 = float(f1_text)
    decimals = len(f1_text.split(".")[1]) if "." in f1_text else 0
    tol = 0.5 * 10**-decimals
    return [
        {"tp": tp, "predicted_positive": k, "precision": round(tp / k, 4), "recall": round(tp / positives, 4)}
        for tp in range(1, positives + 1)
        for k in range(tp, rows - positives + tp + 1)
        if abs(2 * tp / (k + positives) - f1) <= tol
    ]


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
        if not s["precision"] or not s["recall"]:
            candidates = candidates_from_f1(s["f1"], positives, rows)
            out.append(entry | {"counts": None, "why_no_counts": "only F1 known", "candidates": candidates})
            previous = None
            continue
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
        for c in e.get("candidates", []):
            print("   candidate", c)
    print(f"{len(out)} submissions, test with {positives} positives in {rows} rows")


if __name__ == "__main__":
    main()
