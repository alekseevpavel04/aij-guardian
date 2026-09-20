import numpy as np

from guardian.metrics import best_threshold, binary_scores, bootstrap_ci, bootstrap_diff_ci


def test_all_ones_baseline_matches_the_closed_form():
    y = np.array([1] * 130 + [0] * 260)
    s = binary_scores(y, np.ones_like(y))
    assert s.recall == 1.0
    assert s.precision == 130 / 390
    assert abs(s.f1 - 2 * 130 / (130 + 390)) < 1e-12


def test_all_zeros_gives_zero_not_a_crash():
    y = np.array([1, 0, 1])
    s = binary_scores(y, np.zeros_like(y))
    assert (s.f1, s.precision, s.recall) == (0.0, 0.0, 0.0)


def test_reconstructed_leaderboard_row():
    # yost49158127 on 2026-09-20: 130 positives, 195 predicted positive, 96 true positives
    y = np.array([1] * 130 + [0] * 260)
    pred = np.zeros_like(y)
    pred[:96] = 1
    pred[130 : 130 + 99] = 1
    s = binary_scores(y, pred)
    assert round(s.f1, 5) == 0.59077
    assert round(s.precision, 5) == 0.49231
    assert round(s.recall, 5) == 0.73846


def test_best_threshold_finds_the_separating_cut():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    thr, f1 = best_threshold(y, score)
    assert f1 == 1.0
    assert 0.2 < thr < 0.8


def test_best_threshold_never_returns_less_than_the_all_ones_baseline():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    score = rng.random(200)
    _, f1 = best_threshold(y, score)
    assert f1 >= binary_scores(y, np.ones_like(y)).f1


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    pred = np.where(rng.random(300) < 0.8, y, 1 - y)
    point, lo, hi = bootstrap_ci(y, pred, n_resamples=400, seed=7)
    assert lo <= point <= hi
    assert hi - lo > 0


def test_bootstrap_is_reproducible_with_a_seed():
    y = np.array([1, 0] * 50)
    pred = np.array([1] * 100)
    assert bootstrap_ci(y, pred, n_resamples=200, seed=3) == bootstrap_ci(y, pred, n_resamples=200, seed=3)


def test_paired_difference_of_identical_predictions_is_exactly_zero():
    y = np.array([1, 0, 1, 1, 0, 0])
    pred = np.array([1, 1, 0, 1, 0, 1])
    point, lo, hi = bootstrap_diff_ci(y, pred, pred, n_resamples=200, seed=5)
    assert (point, lo, hi) == (0.0, 0.0, 0.0)
