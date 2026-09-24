"""What to do with the contexts where the two labellers disagree, on the context ruler.

Head B trains on the v1s labels of every context. Here the contexts where v1s and Opus disagree are
dropped, labelled by either labeller, by their union or intersection, given a soft label (0 and 1 at
half weight each), or weighed up and down, always with the quarter of the pairs head B has. The ruler
measures on the contexts both labellers agree on, which favours training on them alone a little: the
held-out rows are of the same kind. So every head is also measured against each labeller's labels on
all contexts.

    python scripts/label_disagreement.py
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
from context_ruler import V3_V4, Ruler  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import PAIR_VECTORS, auroc, vectors  # noqa: E402


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/label_disagreement.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org")
    for name in PAIR_VECTORS:
        if (ROOT / "data/interim/embeddings" / f"{name}.jsonl").exists():
            vecs |= vectors(name)
    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), None)
    v1s, opus = ruler.label["v1s"], ruler.label["opus"]
    split = ~ruler.agree

    def weighted(w: float) -> np.ndarray:
        return np.where(split, w, 1.0)

    base = ruler.scores(v1s, 0.25)
    variants = {
        "opus": ruler.scores(opus, 0.25),
        "agreed_only": ruler.scores(v1s, 0.25, row_weight=weighted(0.0)),
        "union": ruler.scores(v1s | opus, 0.25),
        "intersection": ruler.scores(v1s & opus, 0.25),
        "soft_on_disagreement": ruler.scores(np.where(split, 0.5, v1s), 0.25),
    }
    for w in (0.5, 2.0, 3.0):
        for teacher, y in (("v1s", v1s), ("opus", opus)):
            variants[f"{teacher}_disagreement_weight_{w:g}"] = ruler.scores(y, 0.25, row_weight=weighted(w))

    def against(s: np.ndarray) -> dict:
        return {"v1s_all_rows": auroc(v1s, s), "opus_all_rows": auroc(opus, s)}

    result = {
        "rows": len(v1s),
        "disagreements": int(split.sum()),
        "v1s_1_opus_0": int((split & (v1s == 1)).sum()),
        "v1s_0_opus_1": int((split & (opus == 1)).sum()),
        "disagreements_with_a_tool_call": int((split & ruler.calls).sum()),
        "agreed_rows": int(ruler.agree.sum()),
        "agreed_positives": int(ruler.truth[ruler.agree].sum()),
        "head_b": {**ruler.auroc_ci(base), **against(base)},
        "variants": {
            k: {**ruler.auroc_ci(s), "minus_head_b": ruler.paired(s, base), **against(s)} for k, s in variants.items()
        },
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
