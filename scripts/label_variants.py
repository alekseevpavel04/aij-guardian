"""Which labels of the natural training contexts bring the head closest to the organiser (decision 28).

On the organiser's 46 rows 21 of 23 positives are turns with a tool call, while the privileged
labellers also flag text-only turns (an unsupported statement, an invented reason, an offer). The
head is trained on each labeller's labels as they are and with positives kept only on turns that
call a tool, with no pairs, with the pairs of a quarter of the contexts and with all pairs; then the
share of pairs is varied for the v1s labels. Measured on the 46 rows (both halves), ext_val and the
validation contexts where both labellers agree.

    python scripts/label_variants.py
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
from pairs_effect import V3, V4, written  # noqa: E402
from train_head_ru import PAIR_VECTORS, TEACHERS, labels, measure, vectors  # noqa: E402

COLUMNS = ["id", "prompt", "response", "label", "group"]
SHARES = (0.1, 0.25, 0.5, 0.75)
DRAWS = 3


def has_call(responses: pd.Series) -> np.ndarray:
    return responses.str.contains("TOOL_CALL").to_numpy()


def natural(contexts: pd.DataFrame, teacher: str, calls_only: bool) -> pd.DataFrame:
    train = contexts[contexts.split == "train"].copy()
    label = train.id.map(labels(TEACHERS[teacher].format(split="train"))).to_numpy().astype(bool)
    if calls_only:
        label &= has_call(train.response)
    return train.assign(label=label.astype(int), group=train.task)


def pair_subset(pairs: pd.DataFrame, share: float, seed: int) -> pd.DataFrame:
    """The written turns of a random share of the contexts that have pairs."""
    contexts = np.array(sorted(pairs.context_id.unique()))
    keep = np.random.default_rng(seed).choice(contexts, round(share * len(contexts)), replace=False)
    return pairs[pairs.context_id.isin(keep)]


def summary(result: dict) -> dict:
    valid = result["valid"]
    return {
        "train_rows": result["train_rows"],
        "train_positives": result["train_positives"],
        "valid_all": valid["all"]["head_auroc"],
        "valid_clean": valid["clean"]["head_auroc"],
        "valid_other": valid["leaked"]["head_auroc"],
        "ext_val": result["ext_val"]["head_auroc"],
        "check_agreed": result["check_contexts_agreed"]["head_auroc"],
    }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/label_variants.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org")
    for name in PAIR_VECTORS:
        vecs |= vectors(name)
    pairs = written(V3 + V4, contexts)
    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet")
    calls = has_call(valid.response)
    out: dict = {
        "organiser_46": {
            "positives": int(valid.label.sum()),
            "positives_with_call": int((calls & (valid.label == 1)).sum()),
            "negatives_with_call": int((calls & (valid.label == 0)).sum()),
        },
        "labellers": {},
        "heads": {},
        "v1s_calls_only_pair_curve": {},
    }
    for teacher in TEACHERS:
        for split in ("train", "check"):
            part = contexts[contexts.split == split]
            label = part.id.map(labels(TEACHERS[teacher].format(split=split))).to_numpy().astype(bool)
            out["labellers"][f"{teacher}_{split}"] = {
                "positives": int(label.sum()),
                "with_call": int((label & has_call(part.response)).sum()),
            }

    def head(train: pd.DataFrame, extra: pd.DataFrame | None) -> dict:
        rows = train if extra is None else pd.concat([train, extra[COLUMNS]], ignore_index=True)
        return summary(measure(rows[COLUMNS], vecs, contexts, False))

    for teacher in TEACHERS:
        for calls_only in (False, True):
            base = natural(contexts, teacher, calls_only)
            name = teacher + ("_calls_only" if calls_only else "")
            out["heads"][name] = {
                "no_pairs": head(base, None),
                "pairs_25": head(base, pair_subset(pairs, 0.25, 0)),
                "all_pairs": head(base, pairs),
            }
    base = natural(contexts, "v1s", True)
    for share in SHARES:
        draws = [head(base, pair_subset(pairs, share, seed)) for seed in range(DRAWS)]
        out["v1s_calls_only_pair_curve"][str(share)] = {
            key: round(float(np.mean([d[key] for d in draws])), 4) for key in draws[0]
        }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
