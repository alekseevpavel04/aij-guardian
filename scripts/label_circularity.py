"""Is the v1s view favoured because head B learns from v1s labels? Heads trained on each labeller, measured by each.

Head B (logistic, a quarter of the pairs, as submitted) and the candidate's LightGBM head with τ² are trained
out of fold on the contexts' v1s labels and, separately, on their Opus labels, and each is measured by both
labellers on all 384 contexts: AUROC, and F1 at the test's 1.31 flags per error with the rules first. If a head
is best on its own labeller, the measure is partly circular and the fair reading is across labellers.

    python scripts/label_circularity.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from guardian.env import setup
from guardian.rules import Context, evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import V3_V4, Ruler  # noqa: E402
from head_models import lgbm  # noqa: E402
from judge_eval import load  # noqa: E402
from pairs_effect import written  # noqa: E402
from ruler_vs_test import flags  # noqa: E402
from train_head_ru import auroc, binary_scores, vectors  # noqa: E402
from turn_ruler import RULES  # noqa: E402


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/label_circularity.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org") | vectors("q4_last_extdev_all")
    for name in ("q4_last_pairs_02", "q4_last_pairs_03"):
        vecs |= vectors(name)
    tau = load("ext_dev", None, 0)
    tau = tau[tau.id.isin(vecs)]
    external = (np.stack([vecs[i] for i in tau.id]), tau.label.to_numpy())
    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), external)
    fired = np.array(
        [
            any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
            for r in ruler.ctx.itertuples()
        ]
    )
    result: dict = {}
    # "both": the mean of the two labels, a disagreement enters as a soft 0.5 (Ruler.scores duplicates it)
    both = (ruler.label["v1s"] + ruler.label["opus"]) / 2
    for trained, y in (("v1s", ruler.label["v1s"]), ("opus", ruler.label["opus"]), ("both", both)):
        heads = {
            "head_b_logistic": ruler.scores(y, 0.25),
            "head_b_lightgbm_tau2": ruler.scores(y, 0.25, external_weight=0.5, make=lgbm),
        }
        for head, s in heads.items():
            for measured in ("v1s", "opus"):
                truth = ruler.label[measured]
                total = round(170 / 130 * truth.sum())
                key = f"{head}:trained_{trained}:measured_{measured}"
                result[key] = {
                    "auroc": round(float(auroc(truth, s + 10 * fired)), 4),
                    "f1_at_test_point": round(binary_scores(truth, flags(s, fired, 0.3, total)).f1, 4),
                }
                print(key, json.dumps(result[key]), flush=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
