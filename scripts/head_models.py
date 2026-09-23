"""Other models than the logistic head on the context ruler; their combinations are in head_combos.py.

Every model gets the same training rows as head B (v1s labels, a quarter of the Opus pairs, three draws
of the pairs) in the same folds over dialogues, so a difference is the model's, not the data's. The
settings are fixed before the run and every model is reported: with about 60 positives in a fold and
2048 features the best of many is an optimistic pick, not a result. The
out-of-fold scores go to data/interim/head_models_scores.npz for scripts/head_combos.py; averaging
linear heads on one vector gives a linear head again, so only models of another kind or on other data
can add anything.

    python scripts/head_models.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import V3_V4, Ruler, judge_direction  # noqa: E402
from judge_eval import load  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import vectors  # noqa: E402


class Reduced:
    """PCA to `n` components, then the inner model; trees split poorly on 2048 raw features."""

    def __init__(self, n: int, inner) -> None:
        self.pca, self.inner = PCA(n, random_state=0), inner

    def fit(self, x, y, sample_weight=None):
        self.inner.fit(self.pca.fit_transform(x), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, x):
        return self.inner.predict_proba(self.pca.transform(x))


class Unweighted:
    """For models whose fit takes no weights; every row here weighs 1 anyway unless external data is added."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def fit(self, x, y, sample_weight=None):
        self.inner.fit(x, y)
        return self

    def predict_proba(self, x):
        return self.inner.predict_proba(x)


class RandomProducts:
    """The vector plus products x_i * x_j of random pairs i < j, `factor` times as many features in all,
    then a logistic head: a quadratic head on a fixed random subset of the pairs (seeded, so the platform
    draws the same pairs)."""

    def __init__(self, factor: int, c: float, seed: int = 0) -> None:
        self.factor, self.c, self.seed = factor, c, seed

    def expand(self, x: np.ndarray) -> np.ndarray:
        return np.hstack([x, x[:, self.left] * x[:, self.right]])

    def fit(self, x, y, sample_weight=None):
        d = x.shape[1]
        rng = np.random.default_rng(self.seed)
        n = (self.factor - 1) * d
        i, j = rng.integers(0, d, 2 * n), rng.integers(0, d, 2 * n)
        keep = np.unique(np.stack([np.minimum(i, j), np.maximum(i, j)], 1)[i != j], axis=0)
        keep = keep[rng.permutation(len(keep))[:n]]
        self.left, self.right = keep[:, 0], keep[:, 1]
        z = self.expand(x)
        self.scaler = StandardScaler().fit(z)
        self.model = LogisticRegression(C=self.c, max_iter=5000)
        self.model.fit(self.scaler.transform(z), y, sample_weight=sample_weight)
        return self

    def decision_function(self, x):
        return self.model.decision_function(self.scaler.transform(self.expand(x)))


def lgbm(**kw) -> LGBMClassifier:
    base = dict(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=7,
        min_child_samples=10,
        colsample_bytree=0.2,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        random_state=0,
        verbose=-1,
        n_jobs=8,
    )
    return LGBMClassifier(**(base | kw))


def mlp(layers: tuple[int, ...]) -> MLPClassifier:
    return MLPClassifier(layers, alpha=1.0, max_iter=300, early_stopping=False, random_state=0)


MODELS = {
    "tree_depth4": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=0),
    "random_forest": lambda: RandomForestClassifier(
        500, max_features="sqrt", min_samples_leaf=3, n_jobs=8, random_state=0
    ),
    "extra_trees": lambda: ExtraTreesClassifier(500, max_features="sqrt", min_samples_leaf=3, n_jobs=8, random_state=0),
    "lightgbm": lgbm,
    "pca64_lightgbm": lambda: Reduced(64, lgbm(colsample_bytree=0.5)),
    "pca64_random_forest": lambda: Reduced(
        64, RandomForestClassifier(500, max_features="sqrt", min_samples_leaf=3, n_jobs=8, random_state=0)
    ),
    "mlp_64": lambda: mlp((64,)),
    "mlp_64_32": lambda: mlp((64, 32)),
    "mlp_128_64_32": lambda: mlp((128, 64, 32)),
    "svm_rbf": lambda: SVC(C=1.0, kernel="rbf", gamma="scale"),
    "knn_25": lambda: Unweighted(KNeighborsClassifier(25, weights="distance")),
    "lda_shrinkage": lambda: Unweighted(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "logistic_l1": lambda: LogisticRegression(penalty="l1", C=0.01, solver="liblinear", random_state=0),
    "random_products_x8_c0_003": lambda: RandomProducts(8, 0.003),
    "random_products_x8_c0_001": lambda: RandomProducts(8, 0.001),
}


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--models", nargs="*", default=list(MODELS), help="a subset of MODELS")
    ap.add_argument("--out", type=Path, default=ROOT / "results/head_models.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org") | vectors("q4_last_extdev1000")
    for name in ("q4_last_pairs_02", "q4_last_pairs_03"):
        vecs |= vectors(name)
    dev = load("ext_dev", 1000, 0)
    external = (np.stack([vecs[i] for i in dev.id]), dev.label.to_numpy().astype(int))
    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), external)
    v1s = ruler.label["v1s"]

    base = ruler.scores(v1s, 0.25)
    scores = {"head_b_plus_ext_dev_0_5": ruler.scores(v1s, 0.25, external_weight=0.5)}
    scores["judge_no_thinking"] = ruler.x @ judge_direction()
    for name in args.models:
        make = MODELS[name]
        scores[name] = ruler.scores(v1s, 0.25, make=make)
        print(name, ruler.auroc_ci(scores[name])["auroc"], flush=True)
    e1 = vectors("q4e1_last_org")
    scores["head_on_prompt_e1_no_pairs"] = Ruler(contexts, e1, ruler.pairs.iloc[:0], None).scores(v1s, 0.0)

    out = ROOT / "data/interim/head_models_scores.npz"
    np.savez(out, head_b=base, truth=ruler.truth, agree=ruler.agree, tasks=ruler.tasks.astype(str), **scores)
    result = {
        "rows": len(ruler.ctx),
        "agreed_rows": int(ruler.agree.sum()),
        "agreed_positives": int(ruler.truth[ruler.agree].sum()),
        "head_b": ruler.auroc_ci(base),
        "models": {k: {**ruler.auroc_ci(s), "minus_head_b": ruler.paired(s, base)} for k, s in scores.items()},
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
