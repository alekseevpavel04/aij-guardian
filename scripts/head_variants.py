"""Other heads and extra inputs on the vectors already computed, for the training set of decision 26.

Training rows: the organiser's training contexts with the Opus labels plus all written turns kept by
scripts/pairs_effect.py. Every variant picks its own setting by cross-validation on the training rows
grouped by dialogue, never by the measured sets, and is then measured on the validation contexts
(Opus labels, and the part both labellers agree on), the organiser's 46 rows and ext_val.

Inputs: the last-token vector; optionally the counts of the deterministic checks (guardian.rules)
and two shape features of the turn (log length, number of tool calls). Heads: logistic regression
(the current one, also with L1 and on an L2-normalised vector), PCA before it, a small MLP, and
gradient boosting on PCA components. A single MLP depends a lot on its seed, so it is also measured
as the mean of ten seeds, with the penalty chosen by cross-validation of that mean.

Every variant after the first was tried after looking at the 46 rows, so the best of them is an
optimistic estimate; the bootstrap against the judge is there to show how wide it is.

    python scripts/head_variants.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler

from guardian import journal
from guardian.env import setup
from guardian.rules import features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from judge_eval import load  # noqa: E402
from pairs_effect import V3, V4, written  # noqa: E402
from train_head_ru import JUDGE, PAIR_VECTORS, TEACHERS, auroc, labels, vectors  # noqa: E402

C_GRID = (0.001, 0.01, 0.1, 1.0)
SEEDS = 10


def extras(rows: pd.DataFrame) -> np.ndarray:
    rules = pd.DataFrame([features(p, r) for p, r in zip(rows.prompt, rows.response, strict=True)])
    shape = np.column_stack([np.log1p(rows.response.str.len()), rows.response.str.count("TOOL_CALL")])
    return np.hstack([rules.to_numpy(dtype=np.float32), shape.astype(np.float32)])


class SeedMean:
    """The mean probability of MLPs that differ only in their seed."""

    def __init__(self, alpha: float, seeds: int = SEEDS, **mlp) -> None:
        self.alpha = alpha
        self.seeds = seeds
        self.mlp = mlp or {"hidden_layer_sizes": (256,), "early_stopping": True, "max_iter": 500}

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SeedMean":
        self.models = [
            make_pipeline(StandardScaler(), MLPClassifier(alpha=self.alpha, random_state=seed, **self.mlp)).fit(x, y)
            for seed in range(self.seeds)
        ]
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.mean([m.predict_proba(x) for m in self.models], axis=0)


def variants() -> dict[str, list]:
    lr = [make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)) for c in C_GRID]
    return {
        "lr": lr,
        "lr_l1": [
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, penalty="l1", solver="liblinear", max_iter=5000, random_state=0),
            )
            for c in (0.01, 0.1, 1.0)
        ],
        "lr_unit_norm": [
            make_pipeline(Normalizer(), StandardScaler(), LogisticRegression(C=c, max_iter=5000)) for c in C_GRID
        ],
        "pca64_lr": [
            make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=c, max_iter=5000))
            for c in C_GRID
        ],
        "pca256_lr": [
            make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(C=c, max_iter=5000))
            for c in C_GRID
        ],
        "mlp256": [
            make_pipeline(
                StandardScaler(),
                MLPClassifier((256,), alpha=a, early_stopping=True, max_iter=500, random_state=0),
            )
            for a in (0.01, 0.1, 1.0, 10.0)
        ],
        "mlp256_seed_mean": [SeedMean(a) for a in (0.1, 1.0, 10.0)],
        # a fixed number of epochs instead of early stopping on a random tenth of the rows, which is
        # small and not grouped by dialogue; one seed per setting while choosing, ten for the chosen one
        "mlp_tuned_seed_mean": [
            SeedMean(
                alpha,
                hidden_layer_sizes=(hidden,),
                learning_rate_init=rate,
                batch_size=batch,
                max_iter=epochs,
                early_stopping=False,
                seeds=1,
            )
            for hidden in (32, 128)
            for rate in (1e-4, 1e-3)
            for alpha in (1.0, 10.0)
            for batch in (32, 128)
            for epochs in (30, 100)
        ],
        "pca64_boosting": [
            make_pipeline(
                StandardScaler(),
                PCA(64, random_state=0),
                FunctionTransformer(),
                HistGradientBoostingClassifier(max_depth=d, learning_rate=0.05, max_iter=300, random_state=0),
            )
            for d in (2, 3)
        ],
    }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/head_variants.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org")
    for name in PAIR_VECTORS:
        vecs |= vectors(name)
    base = contexts[contexts.split == "train"].copy()
    base["label"] = base.id.map(labels(TEACHERS["opus"].format(split="train")))
    pairs = written(V3 + V4, contexts)
    train = pd.concat([base.assign(group=base.task), pairs], ignore_index=True)[
        ["id", "prompt", "response", "label", "group"]
    ]
    check = contexts[contexts.split == "check"].copy()
    opus = labels(TEACHERS["opus"].format(split="check"))
    v1s = labels(TEACHERS["v1s"].format(split="check"))
    check["label"] = check.id.map(opus)
    agreed = np.array([opus[i] == v1s[i] for i in check.id])
    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet")
    clean = ~valid.id.str.split("::").str[0].isin(set(contexts.task[contexts.split == "train"]))
    ext = load("ext_val", 400, 0)
    valid_vecs, ext_vecs = vectors("q4_last_valid"), vectors("q4_last_extval400")
    sets = {
        "train": (train, vecs),
        "check": (check, vecs),
        "valid": (valid, valid_vecs),
        "ext": (ext, ext_vecs),
    }
    plain = {k: np.stack([v[i] for i in rows.id]) for k, (rows, v) in sets.items()}
    extra = {k: extras(rows) for k, (rows, _) in sets.items()}
    judged = {r["id"]: r["score"] for r in journal.read(ROOT / "data/interim/judge" / f"{JUDGE}.jsonl")}
    y = train.label.to_numpy().astype(int)
    groups = train.group.to_numpy()

    result: dict = {
        "train_rows": len(train),
        "judge_on_valid_all": auroc(valid.label.to_numpy(), valid.id.map(judged).to_numpy(float)),
        "heads": {},
    }
    for use_extra in (False, True):
        inputs = {k: np.hstack([plain[k], extra[k]]) if use_extra else plain[k] for k in sets}
        for name, candidates in variants().items():
            cv = []
            for model in candidates:
                oof = np.zeros(len(y))
                for fit, held in GroupKFold(n_splits=5).split(inputs["train"], y, groups):
                    oof[held] = model.fit(inputs["train"][fit], y[fit]).predict_proba(inputs["train"][held])[:, 1]
                cv.append(auroc(y, oof))
            best = candidates[int(np.argmax(cv))]
            if isinstance(best, SeedMean):
                best.seeds = SEEDS
                result.setdefault("chosen_mlp", {})[name + ("_extra" if use_extra else "")] = {
                    "alpha": best.alpha,
                    **{k: v for k, v in best.mlp.items() if k != "hidden_layer_sizes"},
                    "hidden": best.mlp["hidden_layer_sizes"][0],
                }
            best = best.fit(inputs["train"], y)
            score = {k: best.predict_proba(inputs[k])[:, 1] for k in ("check", "valid", "ext")}
            yc, yv = check.label.to_numpy().astype(int), valid.label.to_numpy().astype(int)
            judge = valid.id.map(judged).to_numpy(float)
            boot = np.random.default_rng(0)
            diffs = []
            for _ in range(2000):
                i = boot.integers(0, len(yv), len(yv))
                if 0 < yv[i].sum() < len(i):
                    diffs.append(auroc(yv[i], score["valid"][i]) - auroc(yv[i], judge[i]))
            result["heads"][name + ("_extra" if use_extra else "")] = {
                "cv_auroc": max(cv),
                "check_opus": auroc(yc, score["check"]),
                "check_agreed": auroc(yc[agreed], score["check"][agreed]),
                "valid_all": auroc(yv, score["valid"]),
                "valid_clean": auroc(yv[clean.to_numpy()], score["valid"][clean.to_numpy()]),
                "valid_all_minus_judge_ci95": [round(float(v), 3) for v in np.percentile(diffs, [2.5, 97.5])],
                "ext_val": auroc(ext.label.to_numpy().astype(int), score["ext"]),
            }
            print(name, use_extra, result["heads"][name + ("_extra" if use_extra else "")], flush=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
