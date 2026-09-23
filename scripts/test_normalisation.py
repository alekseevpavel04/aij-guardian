"""Standardising the vectors by the statistics of the scored set itself instead of the training set.

On the platform the vectors come from vLLM with FP8 weights, not from llama.cpp with Q4, so every
dimension may be shifted and scaled a little. A linear head standardises its input with the training
means and deviations; if the scored set supplies its own, a systematic shift largely cancels. This
checks two things on the head trained on the Opus labels of the training contexts:

- the cost of the own statistics when there is no shift (validation contexts, the 46 rows, ext_val);
- a simulated shift: every dimension moved by a random offset of `k` training deviations and scaled
  by a random factor around 1, with the training statistics and with the set's own.

    python scripts/test_normalisation.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from judge_eval import load  # noqa: E402
from train_head_ru import TEACHERS, auroc, labels, vectors  # noqa: E402

SHIFTS = (0.05, 0.1, 0.2, 0.5)


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/test_normalisation.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    org = vectors("q4_last_org")
    train = contexts[contexts.split == "train"]
    y = train.id.map(labels(TEACHERS["opus"].format(split="train"))).to_numpy().astype(int)
    x = np.stack([org[i] for i in train.id])
    c = json.loads((ROOT / "results/head_ru_opus.json").read_text(encoding="utf-8"))["chosen_c"]
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=c, max_iter=5000).fit(scaler.transform(x), y)

    check = contexts[contexts.split == "check"]
    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet")
    ext = load("ext_val", 400, 0)
    valid_vecs, ext_vecs = vectors("q4_last_valid"), vectors("q4_last_extval400")
    sets = {
        "check_opus": (
            np.stack([org[i] for i in check.id]),
            check.id.map(labels(TEACHERS["opus"].format(split="check"))),
        ),
        "valid_46": (np.stack([valid_vecs[i] for i in valid.id]), valid.label),
        "ext_val": (np.stack([ext_vecs[i] for i in ext.id]), ext.label),
    }

    def score(z: np.ndarray, own: bool) -> np.ndarray:
        standard = StandardScaler().fit(z) if own else scaler
        return model.decision_function(standard.transform(z))

    rng = np.random.default_rng(args.seed)
    result: dict = {"chosen_c": c, "sets": {}}
    for name, (z, labels_) in sets.items():
        target = labels_.to_numpy().astype(int)
        entry = {
            "rows": len(z),
            "training_statistics": auroc(target, score(z, False)),
            "own_statistics": auroc(target, score(z, True)),
            "shifted": {},
        }
        for k in SHIFTS:
            offset = rng.normal(0, k, z.shape[1]) * scaler.scale_
            factor = 1 + rng.normal(0, k, z.shape[1])
            moved = z * factor + offset
            entry["shifted"][str(k)] = {
                "training_statistics": auroc(target, score(moved, False)),
                "own_statistics": auroc(target, score(moved, True)),
            }
        result["sets"][name] = entry
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
