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

    def tp_at(k: float, dd: float) -> float:
        cut = brentq(lambda c: positives * norm.sf(c - dd) + negatives * norm.sf(c) - k, -10, 10)
        return positives * norm.sf(cut - dd)

    # which test AUROC a submission's (TP, k) implies, and the F1 that ranking would give at other budgets;
    # submission 7 shows only F1 = 0.491935, and 2 TP / (k + 130) = 61 / 124 leaves TP 61 of k 118
    # (the next solution, 122 of 366, would be a recall of 0.94); submission 8 (F1 0.495575) leaves 56 of 96, 84 of
    # 209 or 112 of 322, and only 84 of 209 fits a lower threshold than submission 7's with about 170 flags expected
    implied = {}
    for name, tp, k in (
        ("cascade_think0", 71, 173),
        ("baseline", baseline["true_positives"], baseline["true_positives"] + baseline["false_positives"]),
        ("head_b", 77, 170),
        ("head_b_without_pairs", 75, 170),
        ("final_threshold", 61, 118),
        ("final_threshold_1.117", 84, 209),
    ):
        dd = brentq(lambda x, tp=tp, k=k: tp_at(k, x) - tp, 0.01, 5)
        implied[name] = {
            "tp": tp,
            "k": k,
            "auroc": round(float(norm.cdf(dd / 2**0.5)), 3),
            "f1_at_k": {kk: round(2 * tp_at(kk, dd) / (kk + positives), 3) for kk in (118, 140, 170, 200)},
        }
    by_auroc = []
    for auc in (0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.87, 0.90):
        dd = float(2**0.5 * norm.ppf(auc))
        f1s = {k: 2 * tp_at(k, dd) / (k + positives) for k in range(60, 331, 2)}
        best = max(f1s, key=f1s.get)
        by_auroc.append({"auroc": auc, "f1_at_172": round(f1s[172], 4), "best_k": best, "best_f1": round(f1s[best], 4)})

    result = {
        "baseline_point": {"tp": baseline["true_positives"], "fp": baseline["false_positives"]},
        "positives": positives,
        "negatives": negatives,
        "curves": curves,
        "implied_test_auroc": implied,
        "f1_by_auroc": by_auroc,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for label, c in curves.items():
        print(label, "AUROC", c["auroc"], "best", c["best"], [(p["k"], p["f1"]) for p in c["points"][::2]])


if __name__ == "__main__":
    main()
