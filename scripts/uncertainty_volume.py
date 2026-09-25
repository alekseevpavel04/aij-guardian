"""Does the uncertainty head gain from less training data? A curve over the share of contexts it learns from.

Every fold trains the head of turn_ruler.py on a random share of its training contexts (five seeds,
scores averaged), plus two cleaner variants: only contexts where v1s and Opus agree, and only
contexts without a tool call in the turn. Measured alone and in the system (rules first, then the
mean of ranks of head B0 and the uncertainty head at weight 0.5) against the head on all contexts.

    python scripts/uncertainty_volume.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from guardian.env import setup
from guardian.rules import Context, evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import Ruler  # noqa: E402
from head_combos import ranks  # noqa: E402
from train_head_ru import vectors  # noqa: E402
from turn_ruler import RULES  # noqa: E402
from uncertainty_source_grid import features  # noqa: E402

SHARES = (0.25, 0.5, 0.75)
SEEDS = 5


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/uncertainty_volume.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    no_pairs = pd.DataFrame(columns=["id", "task", "context_id", "label"])
    b = Ruler(contexts, vectors("q4_last_org"), no_pairs, None)
    y = b.label["v1s"]
    fired = np.array(
        [
            any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
            for r in b.ctx.itertuples()
        ]
    )
    base = ranks(b.scores(y, 0.0)) + 10 * fired
    u = Ruler(contexts, features("q4_tok"), no_pairs, None)

    def head(weight: np.ndarray | None) -> np.ndarray:
        return u.scores(y, 0.0, make=lambda: LogisticRegression(C=1.0, max_iter=5000), row_weight=weight)

    ref = head(None)
    result = {"all_contexts": {"alone": u.auroc_ci(ref)["auroc"], "system": u.auroc_ci(base + 0.5 * ranks(ref))["auroc"]}}

    def report(name: str, s: np.ndarray) -> None:
        result[name] = {
            "alone": u.auroc_ci(s)["auroc"],
            "system_minus_all_contexts": u.paired(base + 0.5 * ranks(s), base + 0.5 * ranks(ref)),
        }
        print(name, json.dumps(result[name]), flush=True)

    for share in SHARES:
        total = np.zeros(len(y))
        for seed in range(SEEDS):
            keep = np.random.default_rng(seed).random(len(y)) < share
            total += ranks(head(keep.astype(float)))
        report(f"share_{share}", total)
    report("agreed_labels_only", head(u.agree.astype(float)))
    report("text_turns_only", head((~u.calls).astype(float)))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
