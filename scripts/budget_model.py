"""How F1 on the public test depends on the number of rows labelled 1, under a binormal ROC.

One point of the baseline's ROC on the test is known from the leaderboard: 94 of 130 positives and
160 of 418 negatives flagged. An equal-variance binormal ROC through that point has separation
d' = z(TPR) - z(FPR). For that d', and for better rankers (d' + 0.3, d' + 0.6), the script finds
the threshold that flags k rows and reports TP and F1. It is a model, not a measurement: it says how
much a budget can buy for a given ranking quality, and that without better ranking it buys little.

    python scripts/budget_model.py
"""

import argparse
import json
from pathlib import Path

from scipy.optimize import brentq
from scipy.stats import norm

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--forensics", type=Path, default=ROOT / "results/leaderboard_forensics.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/budget_model.json")
    args = ap.parse_args()

    forensics = json.loads(args.forensics.read_text(encoding="utf-8"))
    positives, rows = forensics["positives_minimal"], forensics["test_size_lower_bound"]
    negatives = rows - positives
    baseline = next(t for t in forensics["teams"] if t["team"] == "Baseline" and t["f1_leaderboard"] > 0.45)
    tpr = baseline["true_positives"] / positives
    fpr = baseline["false_positives"] / negatives
    d = float(norm.ppf(tpr) - norm.ppf(fpr))

    curves = {}
    for label, shift in (("baseline", 0.0), ("plus_0_3", 0.3), ("plus_0_6", 0.6)):
        dd = d + shift
        points = []
        for k in range(60, 301, 20):
            cut = brentq(lambda c, dd=dd, k=k: positives * norm.sf(c - dd) + negatives * norm.sf(c) - k, -10, 10)
            tp = positives * norm.sf(cut - dd)
            points.append({"k": k, "tp": round(tp, 1), "f1": round(2 * tp / (k + positives), 4)})
        best = max(points, key=lambda p: p["f1"])
        curves[label] = {
            "d_prime": round(dd, 3),
            "auroc": round(float(norm.cdf(dd / 2**0.5)), 3),
            "best": best,
            "points": points,
        }

    result = {
        "baseline_point": {"tp": baseline["true_positives"], "fp": baseline["false_positives"]},
        "positives": positives,
        "negatives": negatives,
        "curves": curves,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for label, c in curves.items():
        print(label, "AUROC", c["auroc"], "best", c["best"], [(p["k"], p["f1"]) for p in c["points"][::2]])


if __name__ == "__main__":
    main()
