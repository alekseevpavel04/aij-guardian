"""Heads on a middle layer, alone and joined with the last layer, on the context ruler.

Vectors of layer N come from a model cut after block N (scripts/truncate_gguf.py). They exist only for
the organiser's contexts, so every head here is trained without pairs and without external data, and
the reference is head B0 on the last layer (v1s labels, no pairs), which the ruler puts level with head
B. A join is the two vectors side by side under one logistic head; a mean of ranks is two heads; the
product join adds the element-wise product of the two vectors, each dimension first standardised over
all contexts (no labels, so the platform can do the same); the difference is layer 40 minus layer N;
LightGBM on the join mixes the layers non-linearly inside the head.

    python scripts/layer_ruler.py --layers 20
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
from context_ruler import Ruler  # noqa: E402
from head_combos import ranks  # noqa: E402
from head_models import lgbm  # noqa: E402
from train_head_ru import vectors  # noqa: E402


def standardised(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / x.std(0)


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results/layer_ruler.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    no_pairs = pd.DataFrame(columns=["id", "task", "context_id", "label"])
    last = vectors("q4_last_org")
    ruler = Ruler(contexts, last, no_pairs, None)
    v1s = ruler.label["v1s"]
    base = ruler.scores(v1s, 0.0)
    scores = {}
    for n in args.layers:
        mid = vectors(f"q4_L{n}_org")
        scores[f"layer_{n}"] = Ruler(contexts, mid, no_pairs, None).scores(v1s, 0.0)
        joined = {i: np.concatenate([last[i], mid[i]]) for i in mid}
        scores[f"layer_40_joined_with_{n}"] = Ruler(contexts, joined, no_pairs, None).scores(v1s, 0.0)
        scores[f"mean_rank_layer_40_and_{n}"] = ranks(base) + ranks(scores[f"layer_{n}"])
        ids = list(mid)
        a, b = (standardised(np.stack([v[i] for i in ids])) for v in (last, mid))
        product = {i: np.concatenate([a[k], b[k], a[k] * b[k]]) for k, i in enumerate(ids)}
        scores[f"layer_40_joined_with_{n}_and_product"] = Ruler(contexts, product, no_pairs, None).scores(v1s, 0.0)
        difference = {i: last[i] - mid[i] for i in ids}
        scores[f"layer_40_minus_{n}"] = Ruler(contexts, difference, no_pairs, None).scores(v1s, 0.0)
        scores[f"lightgbm_on_layer_40_joined_with_{n}"] = Ruler(contexts, joined, no_pairs, None).scores(
            v1s, 0.0, make=lgbm
        )
    if len(args.layers) > 1:
        joined = {i: np.concatenate([last[i]] + [vectors(f"q4_L{n}_org")[i] for n in args.layers]) for i in last}
        scores["layer_40_joined_with_all"] = Ruler(contexts, joined, no_pairs, None).scores(v1s, 0.0)
    result = {
        "rows": len(ruler.ctx),
        "agreed_rows": int(ruler.agree.sum()),
        "agreed_positives": int(ruler.truth[ruler.agree].sum()),
        "head_b0_layer_40": ruler.auroc_ci(base),
        "variants": {k: {**ruler.auroc_ci(s), "minus_head_b0": ruler.paired(s, base)} for k, s in scores.items()},
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
