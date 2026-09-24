"""A head that tells the kinds of error apart, scored as 1 - P(no error), against B0 on the context ruler.

"Error" mixes very different things, and one linear head must find a single direction for all of
them. Here the positives carry one of five kinds, assigned by the agent from the Opus explanation (or
the v1s reasoning where only v1s flags the turn), in data/interim/labels/error_types_org.jsonl: T a
call that breaks the tool list or schema, E escalation or refusal with an obvious step untried, F an
invented fact, number or rule, C a claimed action or capability that did not happen, P an action the
policy forbids. A multinomial logistic head gets its own direction per kind. Every head here trains
without pairs (the written pairs carry no kind), so the reference is B0.

    python scripts/multiclass_ruler.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import C, Ruler  # noqa: E402
from train_head_ru import vectors  # noqa: E402

KINDS = "TEFCP"


class AnyError:
    """Multinomial head over no-error and the kinds; the score is the probability of any error."""

    def __init__(self, c: float) -> None:
        self.c = c

    def fit(self, x, y, sample_weight=None):
        self.model = LogisticRegression(C=self.c, max_iter=5000).fit(x, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, x):
        p = self.model.predict_proba(x)
        none = p[:, list(self.model.classes_).index(0)]
        return np.column_stack([none, 1 - none])


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/multiclass_ruler.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    no_pairs = pd.DataFrame(columns=["id", "task", "context_id", "label"])
    ruler = Ruler(contexts, vectors("q4_last_org"), no_pairs, None)
    kinds = {r["id"]: r["type"] for r in journal.read(ROOT / "data/interim/labels/error_types_org.jsonl")}
    kind = ruler.ctx.id.map(kinds).fillna("").to_numpy()
    base = ruler.scores(ruler.label["v1s"], 0.0)
    scores = {}
    for teacher in ("v1s", "union"):
        y = ruler.label["v1s"] if teacher == "v1s" else ruler.label["v1s"] | ruler.label["opus"]
        multi = np.array([1 + KINDS.index(k) if yy and k else 0 for yy, k in zip(y, kind, strict=True)])
        missing = int(((y == 1) & (multi == 0)).sum())
        if missing:
            raise ValueError(f"{missing} positives have no kind")
        for c in (C, 3 * C, C / 3):
            scores[f"{teacher}_multiclass_c{c:g}"] = ruler.scores(multi, 0.0, make=lambda c=c: AnyError(c))
        if teacher == "union":
            scores["union_binary"] = ruler.scores(y, 0.0)
    counts = {k: int(((kind == k) & (ruler.label["v1s"] == 1)).sum()) for k in KINDS}
    result = {
        "rows": len(ruler.ctx),
        "agreed_rows": int(ruler.agree.sum()),
        "agreed_positives": int(ruler.truth[ruler.agree].sum()),
        "v1s_positives_by_kind": counts,
        "head_b0": ruler.auroc_ci(base),
        "variants": {k: {**ruler.auroc_ci(s), "minus_head_b0": ruler.paired(s, base)} for k, s in scores.items()},
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
