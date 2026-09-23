"""Can the verdict-position vector tell a turn Opus wrote from a turn the agent wrote?

If it can, a head trained on written pairs may learn the author instead of the error. The probe
separates written turns labelled correct from natural training turns that Opus labelled 0, and
written error turns from natural turns labelled 1, with a logistic regression cross-validated by
dialogue. AUROC near 0.5 means the vector carries little of the author.

    python scripts/authorship_probe.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pairs_effect import V3, V4, written  # noqa: E402
from train_head_ru import PAIR_VECTORS, TEACHERS, auroc, labels, vectors  # noqa: E402


def probe(natural: pd.DataFrame, synthetic: pd.DataFrame, vecs: dict) -> dict:
    rows = pd.concat([natural.assign(y=0), synthetic.assign(y=1)], ignore_index=True)
    x, y, groups = np.stack([vecs[i] for i in rows.id]), rows.y.to_numpy(), rows.task.to_numpy()
    oof = np.zeros(len(y))
    for fit, held in GroupKFold(n_splits=5).split(x, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000))
        oof[held] = model.fit(x[fit], y[fit]).predict_proba(x[held])[:, 1]
    return {"natural": len(natural), "written": len(synthetic), "auroc": auroc(y, oof)}


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/authorship_probe.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org")
    for name in PAIR_VECTORS:
        if (ROOT / "data/interim/embeddings" / f"{name}.jsonl").exists():
            vecs |= vectors(name)
    natural = contexts[contexts.split == "train"].copy()
    natural["label"] = natural.id.map(labels(TEACHERS["opus"].format(split="train")))
    pairs = written(V3 + V4, contexts)
    pairs = pairs[pairs.id.isin(vecs)]
    result = {
        "correct": probe(natural[natural.label == 0], pairs[pairs.kind == "correct"], vecs),
        "error": probe(natural[natural.label == 1], pairs[pairs.kind == "error"], vecs),
        "written_batches_with_vectors": sorted(pairs.batch.unique()),
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
