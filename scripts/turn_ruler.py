"""Turn states and token uncertainty (scripts/embed_llama.py) against head B0 on the context ruler (I-24, I-25).

The new vectors exist only for the organiser's contexts, so every head here trains without pairs, and
the reference is B0 (the last-token vector, v1s labels, no pairs), level with head B on the ruler.
Uncertainty is summarised from the per-token records: log-probability of the turn's tokens (mean, min,
spread, low quantiles, share below 0.1 and 0.01), entropy (mean, max, mean of the ten highest), margin
and the turn's length; a logistic head on these alone, and in a mean of ranks with B0 at weights fixed
beforehand (all reported). When scripts/turn_tokens.py has marked each token as part of a call's
argument value, an argument name, a tool name, layout or free text, the same summaries per part (layout
left out) are added,
and the system as it is submitted (the rules first, then the head) is compared with and without the
uncertainty head in the mean of ranks.

    python scripts/turn_ruler.py --prefix q4_tok
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
from train_head_ru import vectors  # noqa: E402

KINDS = ("turn_mean", "turn_last", "history_end")
# the checks in the submitted predictor: every call-level check (guardian.predictor.DEFAULT_CHECKS)
# except repeated_call, left out since submission 1
RULES = (
    "unknown_tool",
    "locked_tool",
    "unknown_argument",
    "missing_required_argument",
    "enum_violation",
    "repeated_failed_call",
    "unsupported_argument_value",
    "no_op_replacement",
)
WEIGHTS = (0.25, 0.5, 1.0)


def summary(r: dict) -> np.ndarray:
    lp, h, m = (np.asarray(r[k], dtype=float) for k in ("logp", "entropy", "margin"))
    top = np.sort(h)[-10:]
    return np.array(
        [
            lp.mean(),
            lp.min(),
            lp.std(),
            np.quantile(lp, 0.05),
            np.quantile(lp, 0.25),
            (lp < np.log(0.1)).mean(),
            (lp < np.log(0.01)).mean(),
            h.mean(),
            h.max(),
            top.mean(),
            m.mean(),
            np.log(len(lp)),
        ]
    )


def part_summary(r: dict, parts: list[str]) -> np.ndarray:
    """For argument values, argument names, tool names and text (layout left out): count, lowest and
    mean log-probability, share below 0.01, mean and highest entropy."""
    lp, h, kinds = np.asarray(r["logp"], dtype=float), np.asarray(r["entropy"], dtype=float), np.asarray(parts)
    out = []
    for part in ("value", "key", "name", "text"):
        x, e = lp[kinds == part], h[kinds == part]
        out += (
            [np.log1p(len(x)), x.min(), x.mean(), (x < np.log(0.01)).mean(), e.mean(), e.max()] if len(x) else [0.0] * 6
        )
    return np.array(out)


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--prefix", default="q4_tok")
    ap.add_argument("--out", type=Path, default=ROOT / "results/turn_ruler.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    no_pairs = pd.DataFrame(columns=["id", "task", "context_id", "label"])
    last = vectors("q4_last_org")
    ruler = Ruler(contexts, last, no_pairs, None)
    y = ruler.label["v1s"]
    base = ruler.scores(y, 0.0)
    scores = {}
    for kind in KINDS:
        v = vectors(f"{args.prefix}_{kind}_org")
        scores[kind] = Ruler(contexts, v, no_pairs, None).scores(y, 0.0)
        joined = {i: np.concatenate([last[i], v[i]]) for i in v}
        scores[f"last_joined_with_{kind}"] = Ruler(contexts, joined, no_pairs, None).scores(y, 0.0)
        for w in WEIGHTS:
            scores[f"mean_rank_last_and_{kind}_w{w:g}"] = ranks(base) + w * ranks(scores[kind])

    records = {
        r["id"]: summary(r)
        for r in journal.read(ROOT / "data/interim/embeddings" / f"{args.prefix}_uncertainty_org.jsonl")
    }
    unc = Ruler(contexts, records, no_pairs, None)
    scores["uncertainty"] = unc.scores(y, 0.0, make=lambda: LogisticRegression(C=1.0, max_iter=5000))
    for w in WEIGHTS:
        scores[f"mean_rank_last_and_uncertainty_w{w:g}"] = ranks(base) + w * ranks(scores["uncertainty"])
    parts_log = ROOT / "data/interim/embeddings" / f"{args.prefix}_parts_org.jsonl"
    if parts_log.exists():
        stats = {
            r["id"]: r for r in journal.read(ROOT / "data/interim/embeddings" / f"{args.prefix}_uncertainty_org.jsonl")
        }
        both = {
            r["id"]: np.concatenate([records[r["id"]], part_summary(stats[r["id"]], r["parts"])])
            for r in journal.read(parts_log)
        }
        parted = Ruler(contexts, both, no_pairs, None)
        scores["uncertainty_with_parts"] = parted.scores(y, 0.0, make=lambda: LogisticRegression(C=1.0, max_iter=5000))
        for w in WEIGHTS:
            scores[f"mean_rank_last_and_uncertainty_with_parts_w{w:g}"] = ranks(base) + w * ranks(
                scores["uncertainty_with_parts"]
            )
    if parts_log.exists():
        fired = np.array(
            [
                any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
                for r in ruler.ctx.itertuples()
            ]
        )
        submitted = ranks(base) + 10 * fired
        for w in WEIGHTS[:2]:
            scores[f"system_rules_b0_plus_uncertainty_with_parts_w{w:g}"] = submitted + w * ranks(
                scores["uncertainty_with_parts"]
            )
    single = {
        "turn_logp_mean": -unc.x[:, 0],
        "turn_logp_min": -unc.x[:, 1],
        "share_below_0_1": unc.x[:, 5],
        "entropy_top10": unc.x[:, 9],
    }
    result = {
        "rows": len(ruler.ctx),
        "agreed_rows": int(ruler.agree.sum()),
        "agreed_positives": int(ruler.truth[ruler.agree].sum()),
        "head_b0": ruler.auroc_ci(base),
        "single_uncertainty_features": {k: ruler.auroc_ci(s)["auroc"] for k, s in single.items()},
        "variants": {
            k: {**ruler.auroc_ci(s), "minus_head_b0": ruler.paired(s, base)}
            for k, s in scores.items()
            if not k.startswith("system_")
        },
        "system_rules_then_b0": ruler.auroc_ci(submitted) if parts_log.exists() else None,
        "system_variants": {
            k: {**ruler.auroc_ci(s), "minus_system": ruler.paired(s, submitted)}
            for k, s in scores.items()
            if k.startswith("system_")
        },
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
