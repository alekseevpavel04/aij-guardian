"""LightGBM heads of growing capacity with 800 and with all external rows, on the context ruler.

The logistic head stops gaining from external rows at about 800 (results/external_variants.json); a
model with more capacity might still use more. Every setting trains with a quarter of the pairs and
external rows weighted to half the Russian rows, as head B, and is measured alone and in the system:
the rules first, then the mean of ranks of the head and the uncertainty head at weight 0.5. The
settings are fixed before the run and all are reported.

    python scripts/lgbm_grid.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from guardian import journal
from guardian.env import setup
from guardian.rules import Context, evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import V3_V4, Ruler  # noqa: E402
from head_combos import ranks  # noqa: E402
from head_models import lgbm  # noqa: E402
from judge_eval import load  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import vectors  # noqa: E402
from turn_ruler import RULES, part_summary, summary  # noqa: E402

SETTINGS = {
    "leaves7_trees300_lr0.03_col0.2": {},
    "leaves7_trees600_lr0.03_col0.2": {"n_estimators": 600},
    "leaves15_trees300_lr0.03_col0.2": {"num_leaves": 15},
    "leaves31_trees300_lr0.03_col0.2_minchild20": {"num_leaves": 31, "min_child_samples": 20},
    "leaves15_trees1000_lr0.01_col0.2": {"num_leaves": 15, "n_estimators": 1000, "learning_rate": 0.01},
    "leaves7_trees300_lr0.03_col0.5": {"colsample_bytree": 0.5},
    "leaves3_trees600_lr0.05_col0.2": {"num_leaves": 3, "n_estimators": 600, "learning_rate": 0.05},
    "leaves15_trees600_lr0.03_col0.1_l2_10": {
        "num_leaves": 15,
        "n_estimators": 600,
        "colsample_bytree": 0.1,
        "reg_lambda": 10.0,
    },
}


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/lgbm_grid.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    ext_vecs = vectors("q4_last_extdev_all")
    vecs = vectors("q4_last_org") | ext_vecs
    for name in ("q4_last_pairs_02", "q4_last_pairs_03"):
        vecs |= vectors(name)
    d = load("ext_dev", None, 0)
    d = d[d.id.isin(ext_vecs)].reset_index(drop=True)
    small = load("ext_dev", 800, 0)
    externals = {
        "ext_800": (np.stack([vecs[i] for i in small.id]), small.label.to_numpy().astype(int)),
        "ext_all": (np.stack([vecs[i] for i in d.id]), d.label.to_numpy().astype(int)),
    }
    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), None)
    y = ruler.label["v1s"]
    no_pairs = pd.DataFrame(columns=["id", "task", "context_id", "label"])
    folder = ROOT / "data/interim/embeddings"
    stats = {r["id"]: r for r in journal.read(folder / "q4_tok_uncertainty_org.jsonl")}
    feats = {
        r["id"]: np.concatenate([summary(stats[r["id"]]), part_summary(stats[r["id"]], r["parts"])])
        for r in journal.read(folder / "q4_tok_parts_org.jsonl")
    }
    unc = Ruler(contexts, feats, no_pairs, None).scores(y, 0.0, make=lambda: LogisticRegression(C=1.0, max_iter=5000))
    fired = np.array(
        [
            any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
            for r in ruler.ctx.itertuples()
        ]
    )

    def system(head: np.ndarray) -> np.ndarray:
        return ranks(head) + 0.5 * ranks(unc) + 10 * fired

    result: dict = {"settings": {}}
    ruler.external = externals["ext_all"]
    logistic = ruler.scores(y, 0.25, external_weight=0.5)
    reference = system(logistic)
    result["logistic_ext_all"] = {"alone": ruler.auroc_ci(logistic), "system": ruler.auroc_ci(reference)}
    for name, extra in SETTINGS.items():
        row = {}
        for ext_name, ext in externals.items():
            ruler.external = ext
            s = ruler.scores(y, 0.25, external_weight=0.5, make=lambda extra=extra: lgbm(**extra))
            row[ext_name] = {
                "alone": ruler.auroc_ci(s)["auroc"],
                "system_minus_logistic_system": ruler.paired(system(s), reference),
            }
        result["settings"][name] = row
        print(name, json.dumps(row), flush=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
