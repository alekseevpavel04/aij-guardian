"""The uncertainty head trained on more than the organiser's contexts, on the context ruler.

The head of turn_ruler.py learns from 384 contexts only, because the token pass (embed_llama.py) was
run on them alone. Here it also learns from external rows that went through the same pass (weighted
to half of the Russian rows, as head B takes its external data), fold by fold as the ruler does, and
is compared alone and in the submitted system: the rules first, then the mean of the ranks of head B0
and the uncertainty head at weight 0.5 (decision of 25.09).

    python scripts/uncertainty_data_ruler.py --external q4_tok_extdev
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
from context_ruler import Ruler  # noqa: E402
from head_combos import ranks  # noqa: E402
from judge_eval import load  # noqa: E402
from train_head_ru import vectors  # noqa: E402
from turn_ruler import RULES, part_summary, summary  # noqa: E402

WEIGHT = 0.5


def features(prefix: str) -> dict[str, np.ndarray]:
    folder = ROOT / "data/interim/embeddings"
    stats = {r["id"]: r for r in journal.read(folder / f"{prefix}_uncertainty_org.jsonl")}
    return {
        r["id"]: np.concatenate([summary(stats[r["id"]]), part_summary(stats[r["id"]], r["parts"])])
        for r in journal.read(folder / f"{prefix}_parts_org.jsonl")
    }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--external", nargs="+", default=["q4_tok_extdev"], help="token-pass prefixes of external rows")
    ap.add_argument("--out", type=Path, default=ROOT / "results/uncertainty_data_ruler.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    no_pairs = pd.DataFrame(columns=["id", "task", "context_id", "label"])
    ruler = Ruler(contexts, vectors("q4_last_org"), no_pairs, None)
    y = ruler.label["v1s"]
    base = ruler.scores(y, 0.0)
    fired = np.array(
        [
            any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
            for r in ruler.ctx.itertuples()
        ]
    )
    submitted = ranks(base) + 10 * fired

    own = features("q4_tok")
    unc = Ruler(contexts, own, no_pairs, None)

    def head():
        return LogisticRegression(C=1.0, max_iter=5000)

    scores = {"contexts_only": unc.scores(y, 0.0, make=head)}
    labels = load("ext_dev", None, 0).set_index("id").label
    for prefix in args.external:
        ext = features(prefix)
        ids = [i for i in ext if i in labels.index]
        unc.external = (np.stack([ext[i] for i in ids]), labels.loc[ids].to_numpy().astype(int))
        scores[f"contexts_plus_{prefix}"] = unc.scores(y, 0.0, external_weight=WEIGHT, make=head)
        scores[f"contexts_plus_{prefix}_weight_1"] = unc.scores(y, 0.0, external_weight=1.0, make=head)
    result = {
        "agreed_rows": int(ruler.agree.sum()),
        "agreed_positives": int(ruler.truth[ruler.agree].sum()),
        "system_rules_then_b0": ruler.auroc_ci(submitted),
        "variants": {
            k: {
                "alone": ruler.auroc_ci(s),
                "system_with_weight_0_5_minus_system": ruler.paired(submitted + WEIGHT * ranks(s), submitted),
            }
            for k, s in scores.items()
        },
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
