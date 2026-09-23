"""Heads trained on the v1s and on the Opus labels of the training contexts, compared on the validation contexts.

Both heads use the C chosen by their own run of train_head_ru.py (results/head_ru.json and
results/head_ru_opus.json). The AUROC difference is bootstrapped over validation dialogues, since
contexts of one dialogue share their history, and measured against each labeller's labels in turn.

    python scripts/teacher_compare.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_head_ru import TEACHERS, auroc, design, labels, vectors  # noqa: E402

RESULTS = {"v1s": "head_ru.json", "opus": "head_ru_opus.json"}


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/head_ru_teacher_compare.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org")
    train, check = contexts[contexts.split == "train"], contexts[contexts.split == "check"]
    x, x_check = design(train, vecs, False), design(check, vecs, False)
    score = {}
    for teacher, result in RESULTS.items():
        y = train.id.map(labels(TEACHERS[teacher].format(split="train"))).to_numpy().astype(int)
        c = json.loads((ROOT / "results" / result).read_text(encoding="utf-8"))["chosen_c"]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(x, y)
        score[teacher] = model.predict_proba(x_check)[:, 1]

    tasks = check.task.to_numpy()
    groups = [np.where(tasks == t)[0] for t in np.unique(tasks)]
    out = {"dialogues": len(groups), "contexts": len(check), "repeats": args.repeats, "by_labels": {}}
    for yardstick in TEACHERS:
        y = check.id.map(labels(TEACHERS[yardstick].format(split="check"))).to_numpy().astype(int)
        rng = np.random.default_rng(args.seed)
        diffs = []
        for _ in range(args.repeats):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            if 0 < y[idx].sum() < len(idx):
                diffs.append(auroc(y[idx], score["opus"][idx]) - auroc(y[idx], score["v1s"][idx]))
        out["by_labels"][yardstick] = {
            "auroc_v1s_teacher": auroc(y, score["v1s"]),
            "auroc_opus_teacher": auroc(y, score["opus"]),
            "difference": round(auroc(y, score["opus"]) - auroc(y, score["v1s"]), 4),
            "ci95": [round(float(v), 3) for v in np.percentile(diffs, [2.5, 97.5])],
        }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
