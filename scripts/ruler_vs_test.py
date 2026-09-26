"""The ruler against the leaderboard: every submission with an analogue on the contexts, and a bridge to the test.

A submission labels a row 1 when a rule fires or when its score is in the top 30 % (guardian.cascade,
results/submissions.csv). The same decision is taken on the 384 organiser contexts with out-of-fold
scores: rules only; the no-reasoning judge (the hidden state times the judge direction, Pearson 0.99
with its logit); head A (Opus labels on tool-call turns, a quarter of the pairs); head B (v1s labels,
a quarter of the pairs); head B without pairs. The judge with reasoning (think1024) has no analogue.
Each is scored by F1 against the v1s labels, the Opus labels, and on the agreed contexts, and set next to
its F1 on the test. A line through the five points (test F1 against ruler F1) maps the systems that were
not submitted (the candidate and the candidate with simulation rows, from tau2ru_ruler.py's saved scores);
its leave-one-out error says how far to trust it.

    python scripts/ruler_vs_test.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from guardian.env import setup
from guardian.rules import Context, evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import V3_V4, Ruler, judge_direction  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import binary_scores, vectors  # noqa: E402
from turn_ruler import RULES  # noqa: E402

SHARE = 0.30  # the submissions flagged the top 30 % of scored rows besides the rules
TEST = {  # strategy prefix in results/submissions.csv -> name here
    "rules without repeated_call": "rules_only",
    "cascade think0": "judge_no_thinking",
    "head A": "head_a",
    "head B: vLLM pooling + head_b_bag10": "head_b",
    "head B without pairs": "head_b_no_pairs",
}


def flags(score: np.ndarray | None, fired: np.ndarray) -> np.ndarray:
    if score is None:
        return fired.astype(int)
    count = round(SHARE * len(score))
    top = np.zeros(len(score), dtype=bool)
    top[np.argsort(-score)[:count]] = True
    return (top | fired).astype(int)


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument(
        "--simulation-scores", type=Path, nargs="*", default=[ROOT / "results/tau2q36_ruler_opus_quick.scores.npz"]
    )
    ap.add_argument("--out", type=Path, default=ROOT / "results/ruler_vs_test.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org")
    for name in ("q4_last_pairs_02", "q4_last_pairs_03"):
        vecs |= vectors(name)
    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), None)
    v1s, opus = ruler.label["v1s"], ruler.label["opus"]
    fired = np.array(
        [
            any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
            for r in ruler.ctx.itertuples()
        ]
    )
    systems: dict[str, np.ndarray | None] = {
        "rules_only": None,
        "judge_no_thinking": ruler.x @ judge_direction(),
        "head_a": ruler.scores(opus & ruler.calls, 0.25),
        "head_b": ruler.scores(v1s, 0.25),
        "head_b_no_pairs": ruler.scores(v1s, 0.0),
    }
    for path in args.simulation_scores:
        if not path.exists():
            continue
        saved = np.load(path)
        order = pd.Series(range(len(saved["ids"])), index=saved["ids"]).loc[ruler.ctx.id].to_numpy()
        # the saved system puts the rules first by +10; the submission's decision takes them apart
        systems["candidate"] = saved["reference"][order] - 10 * fired
        for key in saved.files:
            if key not in ("ids", "reference"):
                systems[f"candidate_plus_{path.stem.split('_ruler')[0]}_{key}"] = saved[key][order] - 10 * fired

    test = {}
    with open(ROOT / "results/submissions.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for prefix, name in TEST.items():
                if row["strategy"].startswith(prefix):
                    test[name] = float(row["f1"])

    views = {
        "v1s_all": (np.ones(len(v1s), bool), v1s),
        "opus_all": (np.ones(len(v1s), bool), opus),
        "agreed_v1s": (ruler.agree, v1s),
    }
    table = {}
    for name, score in systems.items():
        f = flags(score, fired)
        table[name] = {"test_f1": test.get(name), "flagged_share": round(float(f.mean()), 3)}
        for view, (mask, truth) in views.items():
            table[name][view] = round(binary_scores(truth[mask], f[mask]).f1, 4)
        if score is not None:
            table[name]["auroc_v1s_all"] = round(float(_auroc(v1s, score + 10 * fired)), 4)
    bridges = {}
    submitted = [n for n in table if table[n]["test_f1"] is not None]
    for view in views:
        xs = np.array([table[n][view] for n in submitted])
        ys = np.array([table[n]["test_f1"] for n in submitted])
        slope, intercept = np.polyfit(xs, ys, 1)
        loo = []
        for k in range(len(xs)):
            keep = np.arange(len(xs)) != k
            s, i = np.polyfit(xs[keep], ys[keep], 1)
            loo.append(abs(s * xs[k] + i - ys[k]))
        bridges[view] = {
            "pearson": round(float(np.corrcoef(xs, ys)[0, 1]), 3),
            "slope": round(float(slope), 3),
            "intercept": round(float(intercept), 3),
            "leave_one_out_abs_error_mean": round(float(np.mean(loo)), 3),
            "predicted_test_f1": {
                n: round(float(slope * table[n][view] + intercept), 3) for n in table if table[n]["test_f1"] is None
            },
        }
    # the rules-only point sits far from the rest and carries any line through it; the four scored systems
    # alone are the honest test of whether a measure ranks close variants as the leaderboard does
    heads = [n for n in submitted if n != "rules_only"]
    ys = np.array([table[n]["test_f1"] for n in heads])
    for measure in ("v1s_all", "opus_all", "agreed_v1s", "auroc_v1s_all"):
        xs = np.array([table[n][measure] for n in heads])
        slope, intercept = np.polyfit(xs, ys, 1)
        bridges[f"scored_submissions_only:{measure}"] = {
            "pearson": round(float(np.corrcoef(xs, ys)[0, 1]), 3),
            "slope": round(float(slope), 3),
            "predicted_test_f1": {
                n: round(float(slope * table[n][measure] + intercept), 3) for n in table if table[n]["test_f1"] is None
            },
        }
    result = {
        "share": SHARE,
        "systems": table,
        "bridges": bridges,
        "not_reproducible": ["cascade think1024 (0.450331)"],
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    r = pd.Series(s).rank().to_numpy()
    pos = y.astype(bool)
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


if __name__ == "__main__":
    main()
