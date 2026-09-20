"""Response-level metrics, threshold selection and bootstrap intervals.

The public leaderboard is measured on a few hundred examples, so every number carries an interval:
a difference of 0.01 F1 on that sample size is noise.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class Scores:
    f1: float
    precision: float
    recall: float
    true_positives: int
    predicted_positive: int
    positives: int


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Scores:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    pred_pos = int((y_pred == 1).sum())
    pos = int((y_true == 1).sum())
    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / pos if pos else 0.0
    f1 = 2 * tp / (pos + pred_pos) if pos + pred_pos else 0.0
    return Scores(f1, precision, recall, tp, pred_pos, pos)


def f1_at_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> float:
    return binary_scores(y_true, (np.asarray(score) >= threshold).astype(int)).f1


def best_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1, evaluated at midpoints between observed scores.

    Midpoints rather than observed values, so the threshold does not sit exactly on a training
    score and flip on a tie when the same model is applied to new data.
    """
    score = np.asarray(score, dtype=float)
    uniq = np.unique(score)
    cuts = np.concatenate([[uniq[0] - 1e-9], (uniq[:-1] + uniq[1:]) / 2, [uniq[-1] + 1e-9]])
    f1s = np.array([f1_at_threshold(y_true, score, c) for c in cuts])
    best = int(f1s.argmax())
    return float(cuts[best]), float(f1s[best])


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Point estimate of F1 and a percentile bootstrap interval over examples."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        stats[i] = binary_scores(y_true[idx], y_pred[idx]).f1
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return binary_scores(y_true, y_pred).f1, float(lo), float(hi)


def bootstrap_diff_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap of F1(a) - F1(b): both predictions are resampled on the same indices."""
    y_true = np.asarray(y_true).astype(int)
    y_pred_a = np.asarray(y_pred_a).astype(int)
    y_pred_b = np.asarray(y_pred_b).astype(int)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        stats[i] = binary_scores(y_true[idx], y_pred_a[idx]).f1 - binary_scores(y_true[idx], y_pred_b[idx]).f1
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    point = binary_scores(y_true, y_pred_a).f1 - binary_scores(y_true, y_pred_b).f1
    return point, float(lo), float(hi)


def clopper_pearson_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    """Upper bound of the one-sided 95% interval for a rate, used when nothing was observed.

    With 0 false positives out of 23 negatives the point estimate is 0, which is not a claim that
    can be carried over to a test set of a different size: this is the number to carry instead.
    """
    if trials == 0:
        return 1.0
    if successes == trials:
        return 1.0
    return float(beta.ppf(1 - alpha, successes + 1, trials - successes))


def projected_scores(recall: float, false_positive_rate: float, base_rate: float) -> tuple[float, float]:
    """Precision and F1 a detector would show at a given share of positives.

    Recall and the false positive rate are properties of the detector and do not depend on how
    frequent hallucinations are; precision does. The share of positives in the competition test is
    unknown, so the numbers measured on a balanced file are re-expressed for other shares instead of
    being quoted as if they transferred unchanged.
    """
    tp = base_rate * recall
    fp = (1 - base_rate) * false_positive_rate
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, f1
