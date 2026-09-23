import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from guardian.head import Head


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
