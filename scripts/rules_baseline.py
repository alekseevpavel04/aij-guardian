"""Measure each deterministic check on the labelled file, one at a time and as an OR.

Nothing here is fitted: every check is derived from the tool schema in the context, not from the
labels. The checks were written after reading the 23 explanations in valid.parquet, so their
numbers on that same file are optimistic and are reported as such.

    python scripts/rules_baseline.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from guardian.env import setup
from guardian.io import read_rows
from guardian.metrics import binary_scores, bootstrap_ci, clopper_pearson_upper, projected_scores
from guardian.rules import CHECKS, Context, evidence

ROOT = Path(__file__).resolve().parents[1]
FORENSICS = ROOT / "results/leaderboard_forensics.json"
SNAPSHOT = ROOT / "results/leaderboard/public_2026-09-20.csv"


def leaderboard() -> tuple[float | None, list[float]]:
    """Measured share of positives in the public test, and the published F1 values to rank against."""
    if not FORENSICS.exists():
        return None, []
    share = json.loads(FORENSICS.read_text(encoding="utf-8")).get("positive_share")
    scores = sorted((float(r["f1"]) for r in csv.DictReader(SNAPSHOT.open(encoding="utf-8"))), reverse=True)
    return share, scores


def rank_of(f1: float, scores: list[float]) -> int:
    return sum(1 for s in scores if s > f1) + 1


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data/raw/valid.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/rules_baseline.json")
    args = ap.parse_args()

    frame = pd.DataFrame(read_rows(args.input))
    y = frame["label"].to_numpy().astype(int)

    names = [*CHECKS, "unsupported_number"]
    hits = {name: np.zeros(len(frame), dtype=int) for name in names}
    per_example = []
    for i, row in enumerate(frame.itertuples()):
        found = evidence(row.prompt, row.response, Context.from_prompt(row.prompt))
        for name in names:
            hits[name][i] = len(found[name])
        per_example.append({"id": row.id, "label": int(row.label), "evidence": {k: v for k, v in found.items() if v}})

    report = {"input": str(args.input), "rows": len(frame), "positives": int(y.sum()), "checks": {}}
    for name in names:
        fired = (hits[name] > 0).astype(int)
        s = binary_scores(y, fired)
        f1, lo, hi = bootstrap_ci(y, fired)
        report["checks"][name] = {
            "fired": int(fired.sum()),
            "fired_on_positive": int(((fired == 1) & (y == 1)).sum()),
            "fired_on_negative": int(((fired == 1) & (y == 0)).sum()),
            "precision": round(s.precision, 4),
            "recall": round(s.recall, 4),
            "f1": round(f1, 4),
            "f1_ci95": [round(lo, 4), round(hi, 4)],
        }

    schema_checks = ["unknown_tool", "unknown_argument", "missing_required_argument", "enum_violation"]
    combos = {
        "any_schema_check": schema_checks,
        "any_schema_or_repeat": [*schema_checks, "repeated_failed_call"],
        "any_check_except_numbers": names[:-1],
        "any_check": names,
    }
    report["combinations"] = {}
    for label, members in combos.items():
        fired = (np.sum([hits[m] for m in members], axis=0) > 0).astype(int)
        s = binary_scores(y, fired)
        f1, lo, hi = bootstrap_ci(y, fired)
        report["combinations"][label] = {
            "members": members,
            "fired": int(fired.sum()),
            "precision": round(s.precision, 4),
            "recall": round(s.recall, 4),
            "f1": round(f1, 4),
            "f1_ci95": [round(lo, 4), round(hi, 4)],
        }

    # The labelled file is balanced 23/23 while the competition test is not, and precision depends on
    # that share. Recall and the false positive rate do not, so they are re-expressed for other shares.
    best = "any_check_except_numbers"
    fired = (np.sum([hits[m] for m in combos[best]], axis=0) > 0).astype(int)
    negatives = int((y == 0).sum())
    false_positives = int(((fired == 1) & (y == 0)).sum())
    recall = binary_scores(y, fired).recall
    fpr_observed = false_positives / negatives
    fpr_upper = clopper_pearson_upper(false_positives, negatives)
    report["projection"] = {
        "detector": best,
        "recall": round(recall, 4),
        "false_positive_rate_observed": round(fpr_observed, 4),
        "false_positive_rate_upper95": round(fpr_upper, 4),
        "note": (
            f"{false_positives} false positives out of {negatives} negatives; the upper bound is what "
            "carries to a larger test set, the point estimate of 0 does not"
        ),
        "by_base_rate": {},
    }
    share, scores = leaderboard()
    report["projection"]["measured_share_of_positives"] = share
    for base_rate in sorted({0.5, 1 / 3, share or 0.25, 0.25}, reverse=True):
        precision_obs, f1_obs = projected_scores(recall, fpr_observed, base_rate)
        precision_up, f1_up = projected_scores(recall, fpr_upper, base_rate)
        _, f1_ones = projected_scores(1.0, 1.0, base_rate)
        report["projection"]["by_base_rate"][f"{base_rate:.4f}"] = {
            "rules_precision_at_observed_fpr": round(precision_obs, 4),
            "rules_f1_at_observed_fpr": round(f1_obs, 4),
            "rules_precision_at_upper_fpr": round(precision_up, 4),
            "rules_f1_at_upper_fpr": round(f1_up, 4),
            "always_1_f1": round(f1_ones, 4),
            "rank_at_observed_fpr": rank_of(f1_obs, scores) if scores else None,
            "rank_at_upper_fpr": rank_of(f1_up, scores) if scores else None,
        }

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "results/rules_evidence.json").write_text(
        json.dumps(per_example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"{len(frame)} rows, {int(y.sum())} positive\n")
    header = f"{'check':>28}  {'fired':>5} {'on 1':>5} {'on 0':>5}  {'prec':>5} {'rec':>5}  F1 [95%]"
    print(header)
    for name, m in report["checks"].items():
        print(
            f"{name:>28}  {m['fired']:>5} {m['fired_on_positive']:>5} {m['fired_on_negative']:>5}  "
            f"{m['precision']:>5.3f} {m['recall']:>5.3f}  {m['f1']:.3f} [{m['f1_ci95'][0]:.3f}; {m['f1_ci95'][1]:.3f}]"
        )
    print()
    for label, m in report["combinations"].items():
        print(
            f"{label:>28}  {m['fired']:>5} {'':>5} {'':>5}  {m['precision']:>5.3f} {m['recall']:>5.3f}  "
            f"{m['f1']:.3f} [{m['f1_ci95'][0]:.3f}; {m['f1_ci95'][1]:.3f}]"
        )

    pr = report["projection"]
    print()
    print(
        f"projection of {pr['detector']} (recall {pr['recall']}, FPR {pr['false_positive_rate_observed']}, "
        f"upper {pr['false_positive_rate_upper95']}) onto other shares of positives:"
    )
    print(f"measured share of positives in the public test: {pr['measured_share_of_positives']}")
    print(f"{'share of 1':>12} {'F1 (FPR=obs)':>14} {'rank':>5} {'F1 (FPR=upper)':>16} {'rank':>5} {'always-1':>9}")
    for value, m in pr["by_base_rate"].items():
        print(
            f"{float(value):>12.4f} {m['rules_f1_at_observed_fpr']:>14.3f} {m['rank_at_observed_fpr']!s:>5} "
            f"{m['rules_f1_at_upper_fpr']:>16.3f} {m['rank_at_upper_fpr']!s:>5} {m['always_1_f1']:>9.3f}"
        )


if __name__ == "__main__":
    main()
