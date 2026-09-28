"""A fixed threshold against a fixed share of the file when the test's share of errors is not the one we expect.

The submissions so far flag the rows a rule fires on plus the top 30 % of the file by score (1.31 flags per error
at the test's 130 of 548); the final model can instead flag each row whose score reaches a threshold chosen on the
contexts. From the out-of-fold scores of the organiser's contexts (tau2ru_ruler.py --export writes them), files of
548 rows are drawn with a set share of errors (errors and correct contexts drawn with replacement, whole dialogues
are not kept together) and both rules label them; the threshold of each context is the one chosen on the other
folds, so it has not seen the row it decides.

    python scripts/threshold_shift.py --oof submission_work/final_q4.oof.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

from guardian.metrics import binary_scores

ROOT = Path(__file__).resolve().parents[1]
SHARES = (0.12, 0.17, 0.2372, 0.30, 0.35)


def top_share(score: np.ndarray, fired: np.ndarray, share: float) -> np.ndarray:
    """The submissions' rule: the fired rows, then the top `share` of the file by score (guardian.cascade)."""
    flagged = fired.copy()
    k = round(share * len(score))
    order = np.argsort(-score, kind="stable")
    flagged[order[:k]] = True
    return flagged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--teacher", choices=("v1s", "opus"), default="v1s")
    ap.add_argument("--rows", type=int, default=548)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--share", type=float, default=0.30, help="share of the file the fixed rule flags")
    ap.add_argument("--out", type=Path, default=ROOT / "results/threshold_shift.json")
    args = ap.parse_args()

    d = np.load(args.oof)
    y, score, fired, held = d[args.teacher].astype(int), d["score"], d["fired"].astype(bool), d["held_flags"]
    errors, correct = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(0)
    result = {"oof": str(args.oof), "teacher": args.teacher, "rows": args.rows, "draws": args.draws, "shares": {}}
    for share in SHARES:
        f1_share, f1_thr, flags_share, flags_thr, diff = [], [], [], [], []
        k = round(share * args.rows)
        for _ in range(args.draws):
            idx = np.concatenate([rng.choice(errors, k), rng.choice(correct, args.rows - k)])
            a = top_share(score[idx], fired[idx], args.share)
            b = held[idx]
            fa, fb = binary_scores(y[idx], a.astype(int)).f1, binary_scores(y[idx], b.astype(int)).f1
            f1_share.append(fa)
            f1_thr.append(fb)
            diff.append(fb - fa)
            flags_share.append(a.sum() / k)
            flags_thr.append(b.sum() / k)
        result["shares"][str(share)] = {
            "f1_fixed_share": round(float(np.mean(f1_share)), 4),
            "f1_threshold": round(float(np.mean(f1_thr)), 4),
            "threshold_minus_share": round(float(np.mean(diff)), 4),
            "ci95": [round(float(v), 4) for v in np.percentile(diff, [2.5, 97.5])],
            "flags_per_error_fixed_share": round(float(np.mean(flags_share)), 2),
            "flags_per_error_threshold": round(float(np.mean(flags_thr)), 2),
        }
        print(share, json.dumps(result["shares"][str(share)]), flush=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
