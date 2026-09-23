"""Train the submission head and export it for guardian.head (decision 28).

Training rows: the organiser's training contexts with a labeller's labels (optionally positives only
on turns that call a tool) plus the written pairs of a share of the contexts (scripts/label_variants.py).
C is chosen by cross-validation grouped by dialogue, as in train_head_ru.measure, which also measures
the exported head, so the numbers next to the weights are those of exactly this head.

    python scripts/export_head.py --teacher opus --calls-only --pair-share 0.25 --name head_a
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
from label_variants import COLUMNS, natural, pair_subset  # noqa: E402
from pairs_effect import V3, V4, written  # noqa: E402
from train_head_ru import C_GRID, PAIR_VECTORS, TEACHERS, auroc, design, measure, vectors  # noqa: E402


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--teacher", choices=sorted(TEACHERS), required=True)
    ap.add_argument("--calls-only", action="store_true", help="positives only on turns that call a tool")
    ap.add_argument("--pair-share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org")
    for name in PAIR_VECTORS:
        vecs |= vectors(name)
    base = natural(contexts, args.teacher, args.calls_only)
    pairs = pair_subset(written(V3 + V4, contexts), args.pair_share, args.seed) if args.pair_share else None
    train = base[COLUMNS] if pairs is None else pd.concat([base[COLUMNS], pairs[COLUMNS]], ignore_index=True)

    x, y, groups = design(train, vecs, False), train.label.to_numpy().astype(int), train.group.to_numpy()
    cv = {}
    for c in C_GRID:
        oof = np.zeros(len(y))
        for fit, held in GroupKFold(n_splits=5).split(x, y, groups):
            scaler = StandardScaler().fit(x[fit])
            model = LogisticRegression(C=c, max_iter=5000).fit(scaler.transform(x[fit]), y[fit])
            oof[held] = model.predict_proba(scaler.transform(x[held]))[:, 1]
        cv[c] = auroc(y, oof)
    best = max(cv, key=lambda c: cv[c] or 0)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=best, max_iter=5000).fit(scaler.transform(x), y)

    head = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }
    out = ROOT / "configs/heads" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(head) + "\n", encoding="utf-8")
    measured = measure(train, vecs, contexts, False)
    assert measured["chosen_c"] == best
    summary = {
        "head": str(out.relative_to(ROOT)).replace("\\", "/"),
        "teacher": args.teacher,
        "calls_only": args.calls_only,
        "pair_share": args.pair_share,
        "seed": args.seed,
        "train_rows": len(train),
        "train_positives": int(y.sum()),
        "written_rows": 0 if pairs is None else len(pairs),
        "chosen_c": best,
        "valid": measured["valid"],
        "ext_val": measured["ext_val"],
        "check_contexts_agreed": measured["check_contexts_agreed"],
        "check_contexts_opus_labels": measured["check_contexts_opus_labels"],
    }
    (ROOT / "results" / f"{args.name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
