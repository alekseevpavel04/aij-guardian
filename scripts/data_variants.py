"""Does head B gain from more data: the rest of the organiser's contexts, the 46 rows, external rows?

Every variant is head B's recipe (v1s labels, all error types, a bag of 10 heads over draws of 25% of
the pairs) on a different base, measured on ext_val. ext_val is independent of the Russian rows but
not of ext_dev, which comes from the same two sources; so external rows are also measured across
sources, trained on one source's ext_dev rows and measured on the other source's ext_val rows. The
gain of the largest Russian base over head B is bootstrapped over ext_val trajectories.

    python scripts/data_variants.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_head import fit  # noqa: E402
from judge_eval import load  # noqa: E402
from label_variants import COLUMNS, natural, pair_subset  # noqa: E402
from pairs_effect import V3, V4, written  # noqa: E402
from train_head_ru import PAIR_VECTORS, TEACHERS, auroc, labels, vectors  # noqa: E402

BAG = 10


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "results/data_variants.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org") | vectors("q4_last_valid") | vectors("q4_last_extdev1000")
    for name in PAIR_VECTORS:
        vecs |= vectors(name)
    pairs = written(V3 + V4, contexts)
    train = natural(contexts, "v1s", False)[COLUMNS]
    check = contexts[contexts.split == "check"]
    check = check.assign(
        label=check.id.map(labels(TEACHERS["v1s"].format(split="check"))).astype(int), group=check.task
    )[COLUMNS]
    organiser = pd.read_parquet(ROOT / "data/raw/valid.parquet")
    organiser = organiser.assign(group=organiser.id.str.split("::").str[0])[COLUMNS]
    dev = load("ext_dev", 1000, 0)
    dev = dev.assign(group=dev.id.str.split("::").str[0])
    ext = load("ext_val", 400, 0)
    ev = vectors("q4_last_extval400")
    x_ext, y_ext, source = np.stack([ev[i] for i in ext.id]), ext.label.to_numpy().astype(int), ext.source.to_numpy()

    def bagged(base: pd.DataFrame) -> np.ndarray:
        weights, bias = 0.0, 0.0
        for seed in range(BAG):
            scaler, model, _ = fit(pd.concat([base, pair_subset(pairs, 0.25, seed)[COLUMNS]], ignore_index=True), vecs)
            w = model.coef_[0] / scaler.scale_
            weights, bias = weights + w, bias + model.intercept_[0] - float(w @ scaler.mean_)
        return x_ext @ (weights / BAG) + bias / BAG

    def measured(score: np.ndarray) -> dict:
        out = {"ext_val": auroc(y_ext, score)}
        for name in ("agentprocessbench", "trajerrbench"):
            m = source == name
            out[name] = auroc(y_ext[m], score[m])
        return out

    russian = pd.concat([train, check, organiser], ignore_index=True)
    bases = {
        "head_b_train_half": train,
        "plus_check_contexts": pd.concat([train, check], ignore_index=True),
        "plus_check_and_organiser_46": russian,
        "plus_all_ext_dev_same_sources": pd.concat([russian, dev[COLUMNS]], ignore_index=True),
        "plus_ext_dev_trajerrbench_only": pd.concat([russian, dev[dev.source == "trajerrbench"][COLUMNS]]),
        "plus_ext_dev_agentprocessbench_only": pd.concat([russian, dev[dev.source == "agentprocessbench"][COLUMNS]]),
    }
    scores = {name: bagged(base) for name, base in bases.items()}
    result = {name: {"rows": len(bases[name]), **measured(s)} for name, s in scores.items()}

    trajectory = ext.id.str.split("::").str[0].to_numpy()
    groups = [np.where(trajectory == t)[0] for t in np.unique(trajectory)]
    rng = np.random.default_rng(0)
    a, b = scores["head_b_train_half"], scores["plus_check_and_organiser_46"]
    diffs = []
    for _ in range(args.repeats):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        diffs.append(auroc(y_ext[idx], b[idx]) - auroc(y_ext[idx], a[idx]))
    result["russian_all_minus_head_b"] = {
        "difference": round(auroc(y_ext, b) - auroc(y_ext, a), 4),
        "ci95": [round(float(v), 3) for v in np.percentile(diffs, [2.5, 97.5])],
        "trajectories": len(groups),
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
