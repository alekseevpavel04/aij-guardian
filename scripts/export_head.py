"""Train the submission head and export it for guardian.head (decision 28).

Training rows: the organiser's training contexts with a labeller's labels (optionally positives only
on turns that call a tool) plus the written pairs of a share of the contexts (scripts/label_variants.py).
C is chosen by cross-validation grouped by dialogue, as in train_head_ru.measure, which also measures
a single exported head, so the numbers next to the weights are those of exactly this head.

With `--bag N` the head is the mean of N heads trained on the pairs of N different random shares of
the contexts (seeds 0..N-1). The mean of their logits is itself linear, so it is exported in the same
format with mean 0 and scale 1, and it no longer depends on which share one seed happened to draw
(on the 46 rows single seeds range from 0.81 to 0.90).

    python scripts/export_head.py --teacher opus --calls-only --pair-share 0.25 --name head_a
    python scripts/export_head.py --teacher opus --calls-only --pair-share 0.25 --bag 10 --name head_a_bag10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from judge_eval import load  # noqa: E402
from label_variants import COLUMNS, natural, pair_subset  # noqa: E402
from pairs_effect import V3, V4, written  # noqa: E402
from train_head_ru import C_GRID, PAIR_VECTORS, TEACHERS, auroc, design, labels, measure, vectors  # noqa: E402


def fit(train: pd.DataFrame, vecs: dict) -> tuple[StandardScaler, LogisticRegression, float]:
    x, y, groups = design(train, vecs, False), train.label.to_numpy().astype(int), train.group.to_numpy()
    cv = {}
    for c in C_GRID:
        oof = np.zeros(len(y))
        for part, held in GroupKFold(n_splits=5).split(x, y, groups):
            scaler = StandardScaler().fit(x[part])
            model = LogisticRegression(C=c, max_iter=5000).fit(scaler.transform(x[part]), y[part])
            oof[held] = model.predict_proba(scaler.transform(x[held]))[:, 1]
        cv[c] = auroc(y, oof)
    best = max(cv, key=lambda c: cv[c] or 0)
    scaler = StandardScaler().fit(x)
    return scaler, LogisticRegression(C=best, max_iter=5000).fit(scaler.transform(x), y), best


def evaluate(weights: np.ndarray, bias: float, contexts: pd.DataFrame) -> dict:
    """A head folded into raw weights, on the same sets as train_head_ru.measure."""
    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet")
    clean = ~valid.id.str.split("::").str[0].isin(set(contexts.task[contexts.split == "train"])).to_numpy()
    vv, ev, org = vectors("q4_last_valid"), vectors("q4_last_extval400"), vectors("q4_last_org")
    sv, yv = np.stack([vv[i] for i in valid.id]) @ weights + bias, valid.label.to_numpy().astype(int)
    ext = load("ext_val", 400, 0)
    se = np.stack([ev[i] for i in ext.id]) @ weights + bias
    check = contexts[contexts.split == "check"]
    sc = np.stack([org[i] for i in check.id]) @ weights + bias
    opus = check.id.map(labels(TEACHERS["opus"].format(split="check"))).to_numpy().astype(int)
    agreed = opus == check.id.map(labels(TEACHERS["v1s"].format(split="check"))).to_numpy().astype(int)
    return {
        "valid": {
            "all": {"rows": len(yv), "head_auroc": auroc(yv, sv)},
            "clean": {"rows": int(clean.sum()), "head_auroc": auroc(yv[clean], sv[clean])},
            "leaked": {"rows": int((~clean).sum()), "head_auroc": auroc(yv[~clean], sv[~clean])},
        },
        "ext_val": {"rows": len(ext), "head_auroc": auroc(ext.label.to_numpy().astype(int), se)},
        "check_contexts_agreed": {"rows": int(agreed.sum()), "head_auroc": auroc(opus[agreed], sc[agreed])},
        "check_contexts_opus_labels": {"rows": len(opus), "head_auroc": auroc(opus, sc)},
    }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--teacher", choices=sorted(TEACHERS), required=True)
    ap.add_argument("--calls-only", action="store_true", help="positives only on turns that call a tool")
    ap.add_argument("--pair-share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bag", type=int, default=0, help="mean of this many heads over pair seeds 0..N-1")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org")
    for name in PAIR_VECTORS:
        vecs |= vectors(name)
    base = natural(contexts, args.teacher, args.calls_only)
    all_pairs = written(V3 + V4, contexts)
    out = ROOT / "configs/heads" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "head": out.relative_to(ROOT).as_posix(),
        "teacher": args.teacher,
        "calls_only": args.calls_only,
        "pair_share": args.pair_share,
    }

    def rows_for(seed: int) -> pd.DataFrame:
        if not args.pair_share:
            return base[COLUMNS]
        return pd.concat([base[COLUMNS], pair_subset(all_pairs, args.pair_share, seed)[COLUMNS]], ignore_index=True)

    if args.bag:
        weights, bias, chosen = np.zeros(0), 0.0, []
        for seed in range(args.bag):
            scaler, model, c = fit(rows_for(seed), vecs)
            w = model.coef_[0] / scaler.scale_
            weights = w if not len(weights) else weights + w
            bias += float(model.intercept_[0] - w @ scaler.mean_)
            chosen.append(c)
        weights, bias = weights / args.bag, bias / args.bag
        head = {
            "mean": [0.0] * len(weights),
            "scale": [1.0] * len(weights),
            "coef": weights.tolist(),
            "intercept": bias,
        }
        summary |= {"bag_seeds": list(range(args.bag)), "chosen_c": chosen, **evaluate(weights, bias, contexts)}
    else:
        train = rows_for(args.seed)
        scaler, model, best = fit(train, vecs)
        head = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
        }
        measured = measure(train, vecs, contexts, False)
        assert measured["chosen_c"] == best
        summary |= {
            "seed": args.seed,
            "train_rows": len(train),
            "train_positives": int(train.label.sum()),
            "written_rows": len(train) - len(base),
            "chosen_c": best,
            "valid": measured["valid"],
            "ext_val": measured["ext_val"],
            "check_contexts_agreed": measured["check_contexts_agreed"],
            "check_contexts_opus_labels": measured["check_contexts_opus_labels"],
        }
    out.write_text(json.dumps(head) + "\n", encoding="utf-8")
    (ROOT / "results" / f"{args.name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
