"""Which absolute threshold to send after submission 7.

Submission 7 took the threshold that maximises F1 on our contexts, where the heads rank far better than on the
test (AUROC about 0.93 against about 0.76), and flagged too few rows (61 of 118). The threshold should instead flag
the number of rows that is best for a ranking of the test's quality. Three checks of that:

1. backtest: the binormal model (scripts/budget_model.py) fitted on one submission predicts another at a different
   number of flags: head B (170) -> submission 7 (118), and our judge (173) -> the organiser's baseline (254);
2. the test F1 curve for submission 7's implied AUROC and its band: where the best number of flags lies, and how
   flat F1 is around it;
3. without the binormal assumption: our own out-of-fold scores (submission_work/final_fp8.oof.npz) spoiled until
   their AUROC is the test's, by noise on the score or by flipped labels; the best number of flags per error there;
4. the flipped-labels shape backtested like the binormal one: the share of flipped labels fitted on head B at 170
   flags, then the prediction for submission 7 at 118. The two shapes disagree on the direction (the binormal best
   is near 195 flags, the flipped one near 118), and this pair of submissions is the only test point that tells
   them apart.

The number of flags is turned into a threshold on the 548 rows of the rental's timing run (the submission's pass
on test-like rows), scaled by what the test itself showed at the sent threshold (118 flags against 102).

    python scripts/threshold_choice.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from guardian import journal
from guardian.cascade import rule_labels
from guardian.extract import decode
from guardian.final import Final
from guardian.uncertainty import features

ROOT = Path(__file__).resolve().parents[1]
POSITIVES, NEGATIVES = 130, 418


def tp_at(k: float, d: float) -> float:
    cut = brentq(lambda c: POSITIVES * norm.sf(c - d) + NEGATIVES * norm.sf(c) - k, -10, 10)
    return POSITIVES * norm.sf(cut - d)


def separation(tp: int, k: int) -> float:
    return brentq(lambda d: tp_at(k, d) - tp, 0.01, 5)


def auroc_of(d: float) -> float:
    return float(norm.cdf(d / 2**0.5))


def f1_curve(d: float) -> dict[int, float]:
    return {k: 2 * tp_at(k, d) / (k + POSITIVES) for k in range(80, 331)}


def plateau(curve: dict[int, float], within: float = 0.005) -> dict:
    best = max(curve, key=curve.get)
    near = [k for k, f in curve.items() if f >= curve[best] - within]
    return {"best_k": best, "best_f1": round(curve[best], 4), f"k_within_{within}": [min(near), max(near)]}


def backtest() -> list[dict]:
    out = []
    for source, target in (
        (("head_b", 77, 170), ("final_threshold", 61, 118)),
        (("cascade_think0", 71, 173), ("baseline", 94, 254)),
    ):
        d = separation(source[1], source[2])
        predicted = tp_at(target[2], d)
        out.append(
            {
                "from": source[0],
                "to": target[0],
                "from_auroc": round(auroc_of(d), 3),
                "to_k": target[2],
                "predicted_tp": round(predicted, 1),
                "actual_tp": target[1],
                "predicted_f1": round(2 * predicted / (target[2] + POSITIVES), 4),
                "actual_f1": round(2 * target[1] / (target[2] + POSITIVES), 4),
            }
        )
    return out


def best_flags_per_error(combined: np.ndarray, y: np.ndarray) -> np.ndarray:
    """F1 at each number of flags k = 1..n, rows ranked by `combined` (random order among ties)."""
    order = np.argsort(-combined, kind="stable")
    tp = np.cumsum(y[order])
    k = np.arange(1, len(y) + 1)
    return 2 * tp / (k + y.sum())


def spoiled(score: np.ndarray, fired: np.ndarray, y: np.ndarray, target: float, repeats: int, seed: int) -> dict:
    """Noise on the score and flipped labels, each tuned so that the mean AUROC is `target`."""
    rng = np.random.default_rng(seed)
    base = np.where(fired, 10.0, score)

    def noisy(sigma: float, r: np.random.Generator) -> np.ndarray:
        return np.where(fired, 10.0, score + r.normal(0, sigma, len(score)))

    def flipped(p: float, r: np.random.Generator) -> np.ndarray:
        # positives lose and negatives gain labels so that the share of errors becomes the test's (130 of 548)
        q = (POSITIVES / (POSITIVES + NEGATIVES) * len(y) - y.sum() * (1 - p)) / (len(y) - y.sum())
        flip = r.random(len(y)) < np.where(y == 1, p, max(q, 0.0))
        return np.where(flip, 1 - y, y)

    out = {}
    for name, grid in (("noise_on_score", np.linspace(0.0, 1.5, 151)), ("flipped_labels", np.linspace(0, 0.6, 121))):
        chosen = None
        for value in grid:
            r = np.random.default_rng(seed)
            aucs = []
            for _ in range(50):
                if name == "noise_on_score":
                    aucs.append(roc_auc_score(y, noisy(value, r)))
                else:
                    aucs.append(roc_auc_score(flipped(value, r), base))
            if np.mean(aucs) <= target:
                chosen = float(value)
                break
        curves, errors, aucs = [], [], []
        for _ in range(repeats):
            if name == "noise_on_score":
                combined, labels = noisy(chosen, rng), y
            else:
                combined, labels = base + rng.random(len(y)) * 1e-9, flipped(chosen, rng)
            curves.append(best_flags_per_error(combined, labels) if labels.sum() else np.zeros(len(y)))
            errors.append(labels.sum())
            aucs.append(roc_auc_score(labels, combined))
        mean = np.mean(curves, axis=0)
        best = int(np.argmax(mean)) + 1
        near = np.flatnonzero(mean >= mean.max() - 0.005) + 1
        per_error = float(np.mean(errors))
        out[name] = {
            "parameter": round(chosen, 4),
            "auroc": round(float(np.mean(aucs)), 3),
            "errors": round(per_error, 1),
            "best_flags": best,
            "best_flags_per_error": round(best / per_error, 2),
            "flags_per_error_within_0.005": [round(near.min() / per_error, 2), round(near.max() / per_error, 2)],
            "best_f1": round(float(mean.max()), 4),
        }
    return out


def flipped_shape(score: np.ndarray, fired: np.ndarray, y: np.ndarray, repeats: int, seed: int) -> dict:
    """Test TP at k flags when the test's labels are ours with a share p of them flipped (errors made 130 of 548)."""
    rng = np.random.default_rng(seed)
    base = np.where(fired, 10.0, score)
    n = len(y)
    ks = (118, 140, 170, 200)

    def tp(p: float) -> dict[int, float]:
        q = max((POSITIVES / (POSITIVES + NEGATIVES) * n - y.sum() * (1 - p)) / (n - y.sum()), 0.0)
        recall = dict.fromkeys(ks, 0.0)
        for _ in range(repeats):
            labels = np.where(rng.random(n) < np.where(y == 1, p, q), 1 - y, y)
            hits = np.cumsum(labels[np.argsort(-(base + rng.random(n) * 1e-9))])
            for k in ks:
                recall[k] += hits[round(k / (POSITIVES + NEGATIVES) * n) - 1] / labels.sum() / repeats
        return {k: POSITIVES * r for k, r in recall.items()}

    grid = {round(p, 3): tp(p) for p in np.arange(0.0, 0.5, 0.01)}
    p_b = min(grid, key=lambda p: abs(grid[p][170] - 77))
    p_7 = min(grid, key=lambda p: abs(grid[p][118] - 61))

    def summary(p: float) -> dict:
        return {"p": p} | {f"f1_at_{k}": round(2 * grid[p][k] / (k + POSITIVES), 4) for k in ks} | {
            "tp_at_118": round(grid[p][118], 1)}

    return {"fitted_on_head_b_at_170": summary(p_b), "fitted_on_submission_7_at_118": summary(p_7), "actual_tp_at_118": 61}


def timing_scores(rental: Path, final: Path) -> tuple[np.ndarray, np.ndarray]:
    folder = rental / "guardian_final_7s7hlgq7"
    vec = {r["id"]: decode(r["vector"]) for r in journal.read(folder / "test_final_org.jsonl")}
    unc = {r["id"]: r for r in journal.read(folder / "test_uncertainty_org.jsonl")}
    par = {r["id"]: r["parts"] for r in journal.read(folder / "test_parts_org.jsonl")}
    model = Final.load(final)
    checks = tuple(json.loads(final.read_text(encoding="utf-8"))["checks"])
    rows = pd.read_parquet(ROOT / "data/rental/timing.parquet")
    ids = [str(i) for i in range(len(rows))]  # predict.py names the rows by their position
    s = model.score(np.stack([vec[i] for i in ids]), np.stack([features(unc[i], par[i]) for i in ids]))
    return s, np.array(rule_labels(rows.to_dict("records"), checks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rental", type=Path, default=ROOT / "submission_work/rental")
    ap.add_argument("--final", type=Path, default=ROOT / "submission_work/final_fp8.json")
    ap.add_argument("--repeats", type=int, default=400)
    ap.add_argument("--out", type=Path, default=ROOT / "results/threshold_choice.json")
    args = ap.parse_args()

    d7 = separation(61, 118)
    band = {}
    for name, d in (("submission_7", d7), ("minus_0.02", 2**0.5 * norm.ppf(auroc_of(d7) - 0.02)),
                    ("plus_0.02", 2**0.5 * norm.ppf(auroc_of(d7) + 0.02))):
        band[name] = {"auroc": round(auroc_of(d), 3)} | plateau(f1_curve(d))
    curve7 = f1_curve(d7)
    band["submission_7"]["f1_at_k"] = {k: round(curve7[k], 4) for k in (118, 140, 160, 170, 180, 190, 200, 220, 250)}

    oof = np.load(args.final.with_suffix(".oof.npz"), allow_pickle=True)
    score, fired, y = oof["score"], oof["fired"].astype(bool), oof["v1s"].astype(int)
    home = {
        "rows": len(y),
        "errors_v1s": int(y.sum()),
        "auroc_v1s": round(roc_auc_score(y, np.where(fired, 10.0, score)), 3),
        "spoiled_to_test_auroc": spoiled(score, fired, y, auroc_of(d7), args.repeats, 0),
        "flipped_shape_backtest": flipped_shape(score, fired, y, args.repeats, 0),
    }

    s, rules = timing_scores(args.rental, args.final)
    sent = json.loads(args.final.read_text(encoding="utf-8"))["threshold"]
    scale = 118 / int((rules | (s >= sent)).sum())
    candidates = np.unique(np.concatenate([s, [sent]]))
    to_threshold = {}
    for k in (140, 160, 170, 180, 190, 200, 220):
        flags = np.array([(rules | (s >= t)).sum() * scale for t in candidates])
        t = float(candidates[np.flatnonzero(flags >= k).max()])
        to_threshold[k] = {"threshold": round(t, 4), "timing_rows_flagged": int((rules | (s >= t)).sum())}

    result = {
        "backtest": backtest(),
        "test_curve": band,
        "home": home,
        "timing_rows": {"rows": len(s), "rules": int(rules.sum()), "sent_threshold": sent,
                        "flagged_at_sent": int((rules | (s >= sent)).sum()), "test_flagged_at_sent": 118,
                        "scale": round(scale, 4), "threshold_for_test_flags": to_threshold},
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
