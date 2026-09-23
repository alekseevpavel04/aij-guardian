import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from guardian.head import Head, scoring_order


def test_head_scores_as_the_sklearn_model_it_was_exported_from(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 8)) * rng.uniform(0.5, 3, 8) + rng.normal(size=8)
    y = (x[:, 0] + rng.normal(size=60) > 0).astype(int)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=0.3).fit(scaler.transform(x), y)
    path = tmp_path / "head.json"
    exported = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": model.coef_[0].tolist()}
    path.write_text(json.dumps(exported | {"intercept": float(model.intercept_[0])}))
    head = Head.load(path)
    expected = model.predict_proba(scaler.transform(x))[:, 1]
    assert np.allclose([head.score(row.tolist()) for row in x], expected, atol=1e-9)


def test_head_rejects_a_vector_of_another_size():
    head = Head([0.0] * 3, [1.0] * 3, [1.0] * 3, 0.0)
    try:
        head.score([0.0] * 4)
    except ValueError:
        return
    raise AssertionError("a vector of the wrong size was scored")


def test_turns_with_a_tool_call_are_scored_first_and_shorter_first():
    rows = [
        {"prompt": "x" * 30, "response": "text only"},
        {"prompt": "x" * 20, "response": "\t→ TOOL_CALL get_order: {}"},
        {"prompt": "x" * 10, "response": "text only"},
        {"prompt": "x" * 5, "response": "ok\n\t→ TOOL_CALL cancel: {}"},
    ]
    assert scoring_order(rows) == [3, 1, 2, 0]


def test_scores_shifted_on_the_platform_map_back_to_the_home_scale():
    from guardian.head import to_home_scale

    home = [0.05, 0.1, 0.2, 0.4, 0.8]
    platform = [2 * h + 0.3 for h in home]  # the same rows, another engine: scaled and shifted
    assert all(abs(to_home_scale(p, platform, home) - h) < 1e-12 for p, h in zip(platform, home, strict=True))
    assert abs(to_home_scale(2 * 0.3 + 0.3, platform, home) - 0.3) < 1e-12
    assert to_home_scale(-5, platform, home) == 0.05 and to_home_scale(5, platform, home) == 0.8
