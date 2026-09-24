"""Other vectors of the organiser's contexts, alone and with the judge model's vector, on the context ruler.

For vectors that exist only for the contexts (a small encoder's, scripts/encoder_embed.py), so every
head trains without pairs and the reference is head B0 (the last-token vector, v1s labels, no pairs),
level with head B on the ruler. Each vector is measured alone, joined with the B0 vector under one
logistic head, and as a head of its own in a mean of ranks with B0 at weights fixed beforehand.

    python scripts/vector_ruler.py --vectors user2_window qwen3emb_full --out results/encoder_vectors.json
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
from train_head_ru import vectors  # noqa: E402

WEIGHTS = (0.25, 0.5, 1.0)


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--vectors", nargs="+", required=True, help="names of <name>_org.jsonl in data/interim/embeddings")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    no_pairs = pd.DataFrame(columns=["id", "task", "context_id", "label"])
    last = vectors("q4_last_org")
    ruler = Ruler(contexts, last, no_pairs, None)
    y = ruler.label["v1s"]
    base = ruler.scores(y, 0.0)
    scores = {}
    for name in args.vectors:
        v = vectors(f"{name}_org")
        scores[name] = Ruler(contexts, v, no_pairs, None).scores(y, 0.0)
        joined = {i: np.concatenate([last[i], v[i]]) for i in v}
        scores[f"last_joined_with_{name}"] = Ruler(contexts, joined, no_pairs, None).scores(y, 0.0)
        for w in WEIGHTS:
            scores[f"mean_rank_last_and_{name}_w{w:g}"] = ranks(base) + w * ranks(scores[name])
    result = {
        "rows": len(ruler.ctx),
        "agreed_rows": int(ruler.agree.sum()),
        "agreed_positives": int(ruler.truth[ruler.agree].sum()),
        "head_b0": ruler.auroc_ci(base),
        "variants": {k: {**ruler.auroc_ci(s), "minus_head_b0": ruler.paired(s, base)} for k, s in scores.items()},
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
