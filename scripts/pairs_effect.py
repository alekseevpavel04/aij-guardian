"""How much the written minimal pairs add to the head on Russian contexts (decision 24).

Every head is trained on the Opus labels of the training contexts plus a set of written turns and
measured by train_head_ru.measure. Written turns that copy the agent's own turn are left out (they
duplicate a context), and so are turns written as correct that the blind checker labelled 1
(scripts/pair_verification.py). Sets compared:

- no pairs, the v3 batches of 23.09 only, the v4 batches only, all Opus pairs, all plus the Sonnet pilot;
- a curve over the share of contexts whose pairs are kept (a share of contexts, not of turns, so a
  kept context brings all its turns), three random draws per share;
- pairs A only and pairs B only;
- full pairs only: an error turn is kept only with its written correct twin. Dropping copies leaves
  their error twins behind, so written turns lean to label 1, and the vector can tell a written turn
  from a natural one (scripts/authorship_probe.py): a head may then learn "written, so an error";
- all pairs with the written turns weighted to 25% or 50% of the natural rows' total weight;
- all pairs with a "written" indicator feature, which is 0 for every row the head is measured on.

The head with all Opus pairs is compared with the head without pairs on the validation contexts by a
bootstrap over dialogues.

    python scripts/pairs_effect.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from guardian import journal
from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_head_ru import PAIR_VECTORS, PILOT, TEACHERS, auroc, labels, measure, vectors  # noqa: E402

V3 = ["airline_02", "retail_02a", "retail_02b", "telecom_02a", "telecom_02b", "banking_02a", "banking_02b"]
V4 = [
    *(f"airline_03{s}" for s in "abcdef"),
    "retail_03",
    "telecom_03a",
    "telecom_03b",
    *(f"banking_03{s}" for s in "abf"),
]
SHARES = (0.25, 0.5, 0.75)
DRAWS = 3


def written(batches: list[str], contexts: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat([pd.read_parquet(ROOT / "data/synthetic/pairs" / b / "pairs.parquet") for b in batches])
    frame = frame[frame.context_id.map(contexts.set_index("id").split) == "train"]
    rejected = {
        r["id"]
        for r in journal.read(ROOT / "data/interim/labels/opus_verify_pairs.jsonl")
        if r["role"] == "check" and r["score"] >= 0.5
    }
    copy = frame["flags"].map(lambda f: "same_as_original" in list(f))
    return frame[~copy & ~frame.id.isin(rejected)].assign(group=lambda f: f.task)


def full_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    kinds = frame.groupby(["context_id", "pair"]).kind.transform("nunique")
    return frame[kinds == 2]


def summary(result: dict) -> dict:
    return {
        "train_rows": result["train_rows"],
        "train_positives": result["train_positives"],
        "check_opus_auroc": result["check_contexts_opus_labels"]["head_auroc"],
        "check_opus_true_positives_at_test_share": result["check_contexts_opus_labels"]["head_at_test_share"][
            "true_positives"
        ],
        "check_agreed_auroc": result["check_contexts_agreed"]["head_auroc"],
        "check_v1s_auroc": result["check_contexts"]["head_auroc"],
        "valid_clean_auroc": result["valid"]["clean"]["head_auroc"],
        "valid_leaked_auroc": result["valid"]["leaked"]["head_auroc"],
        "ext_val_auroc": result["ext_val"]["head_auroc"],
    }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/pairs_effect.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org")
    for name in PAIR_VECTORS:
        vecs |= vectors(name)
    base = contexts[contexts.split == "train"].copy()
    base["label"] = base.id.map(labels(TEACHERS["opus"].format(split="train")))
    base["group"] = base.task
    columns = ["id", "prompt", "response", "label", "group"]

    def head(
        extra: pd.DataFrame | None, scores: dict | None = None, share: float | None = None, indicator: bool = False
    ) -> dict:
        if extra is None:
            return summary(measure(base, vecs, contexts, False, scores))
        train = pd.concat([base.assign(written=0.0), extra[columns].assign(written=1.0)], ignore_index=True)
        # written turns together weigh `share` of the natural rows, natural rows weigh 1 each
        train["weight"] = np.where(train.written == 1, share * len(base) / len(extra), 1.0) if share else 1.0
        if not indicator:
            train = train.drop(columns="written")
        return summary(measure(train, vecs, contexts, False, scores))

    opus_pairs = written(V3 + V4, contexts)
    out: dict = {"written_turns_used": len(opus_pairs), "sets": {}, "curve": {}}
    base_scores, all_scores = {}, {}
    out["sets"]["none"] = head(None, base_scores)
    out["sets"]["v3"] = head(written(V3, contexts))
    out["sets"]["v4"] = head(written(V4, contexts))
    out["sets"]["all_opus"] = head(opus_pairs, all_scores)
    out["sets"]["all_opus_and_pilot"] = head(pd.concat([opus_pairs, written(PILOT, contexts)]))
    out["sets"]["pairs_A"] = head(opus_pairs[opus_pairs.pair == "A"])
    out["sets"]["pairs_B"] = head(opus_pairs[opus_pairs.pair == "B"])
    out["sets"]["full_pairs"] = head(full_pairs(opus_pairs))
    out["sets"]["full_pairs_B"] = head(full_pairs(opus_pairs[opus_pairs.pair == "B"]))
    out["sets"]["weight_25"] = head(opus_pairs, share=0.25)
    out["sets"]["weight_50"] = head(opus_pairs, share=0.5)
    out["sets"]["indicator"] = head(opus_pairs, indicator=True)
    out["sets"]["indicator_weight_50"] = head(opus_pairs, share=0.5, indicator=True)

    rng = np.random.default_rng(args.seed)
    kept_contexts = np.array(sorted(opus_pairs.context_id.unique()))
    for share in SHARES:
        draws = []
        for _ in range(DRAWS):
            keep = rng.choice(kept_contexts, round(share * len(kept_contexts)), replace=False)
            draws.append(head(opus_pairs[opus_pairs.context_id.isin(keep)]))
        out["curve"][str(share)] = {
            key: round(float(np.mean([d[key] for d in draws])), 4) for key in draws[0] if draws[0][key] is not None
        }
    out["curve"]["1.0"] = out["sets"]["all_opus"]

    check = contexts[contexts.split == "check"].set_index("id")
    opus = labels(TEACHERS["opus"].format(split="check"))
    v1s = labels(TEACHERS["v1s"].format(split="check"))
    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet").set_index("id")
    clean = valid[~valid.index.str.split("::").str[0].isin(set(contexts.task[contexts.split == "train"]))]
    ids = np.array(sorted(check.index) + sorted(clean.index))
    tasks = np.concatenate([check.task.reindex(sorted(check.index)), clean.index.str.split("::").str[0].sort_values()])
    groups = [np.where(tasks == t)[0] for t in np.unique(tasks)]
    s0, s1 = np.array([base_scores[i] for i in ids]), np.array([all_scores[i] for i in ids])
    in_check = np.isin(ids, check.index)
    # validation contexts carry Opus labels, the clean rows of valid.parquet the organiser's
    y = np.array([opus[i] if c else int(valid.at[i, "label"]) for i, c in zip(ids, in_check, strict=True)])
    agreed = np.array([c and opus[i] == v1s[i] for i, c in zip(ids, in_check, strict=True)])
    out["all_opus_minus_none"] = {}
    for name, mask in (("check_opus", in_check), ("check_agreed", agreed), ("valid_clean", ~in_check)):
        boot = np.random.default_rng(args.seed)
        own = [g[mask[g]] for g in groups if mask[g].any()]
        diffs = []
        for _ in range(args.repeats):
            idx = np.concatenate([own[i] for i in boot.integers(0, len(own), len(own))])
            if 0 < y[idx].sum() < len(idx):
                diffs.append(auroc(y[idx], s1[idx]) - auroc(y[idx], s0[idx]))
        out["all_opus_minus_none"][name] = {
            "difference": round(auroc(y[mask], s1[mask]) - auroc(y[mask], s0[mask]), 4),
            "ci95": [round(float(v), 3) for v in np.percentile(diffs, [2.5, 97.5])],
            "rows": int(mask.sum()),
            "dialogues": len(np.unique(tasks[mask])),
        }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
