"""Can the submission decide row by row, with one fixed threshold instead of "rules or the top 30 % of the file"?

The candidate now adds the ranks of head B and the uncertainty head within the file, which ties every
decision to the other rows. Here the two heads are joined by a small logistic regression on their
logits instead (a probability per row), and a fixed threshold on it is chosen at home. Everything is
out of fold over the 40 dialogues: the heads (as in component_ablation.py: LightGBM head B with a quarter
of the pairs and the τ² steps, the uncertainty head), the stacker (fitted on the other folds' out-of-fold
head scores), and the threshold (chosen on four folds, applied to the fifth). Measures, by the v1s labels
on all 384 contexts (the measure that ranks the submissions as the leaderboard does): AUROC, F1 at 1.31
flags per error (the test's point), F1 of the fixed threshold, and per held-out fold how many flags per
error the threshold gives; the rules label 1 on their own in every variant.

    python scripts/threshold_study.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from guardian import journal
from guardian.env import setup
from guardian.rules import Context, evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import FOLDS, V3_V4, Ruler  # noqa: E402
from head_combos import ranks  # noqa: E402
from head_models import lgbm  # noqa: E402
from judge_eval import load  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import auroc, binary_scores, vectors  # noqa: E402
from turn_ruler import RULES, part_summary, summary  # noqa: E402

EMB = ROOT / "data/interim/embeddings"
TEST_FLAGS_PER_ERROR = 170 / 130


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def f1_flags(truth: np.ndarray, flagged: np.ndarray) -> float:
    return round(binary_scores(truth, flagged.astype(int)).f1, 4)


def top_matched(score: np.ndarray, fired: np.ndarray, total: int) -> np.ndarray:
    order = np.argsort(-score)
    flagged = fired.copy()
    for i in order:
        if flagged.sum() >= total:
            break
        flagged[i] = True
    return flagged


def best_threshold(p: np.ndarray, truth: np.ndarray, fired: np.ndarray) -> float:
    candidates = np.unique(np.round(p, 4))
    scores = [binary_scores(truth, (fired | (p >= t)).astype(int)).f1 for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/threshold_study.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org") | vectors("q4_last_extdev_all")
    for name in ("q4_last_pairs_02", "q4_last_pairs_03"):
        vecs |= vectors(name)
    tau = load("ext_dev", None, 0)
    tau = tau[tau.id.isin(vecs)]
    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), None)
    y, x, tasks, pairs = ruler.label["v1s"], ruler.x, ruler.tasks, ruler.pairs
    stats = {r["id"]: r for r in journal.read(EMB / "q4_tok_uncertainty_org.jsonl")}
    feats = {
        r["id"]: np.concatenate([summary(stats[r["id"]]), part_summary(stats[r["id"]], r["parts"])])
        for r in journal.read(EMB / "q4_tok_parts_org.jsonl")
    }
    xu = np.stack([feats[i] for i in ruler.ctx.id])
    fired = np.array(
        [
            any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
            for r in ruler.ctx.itertuples()
        ]
    )
    folds = list(GroupKFold(FOLDS).split(x, y, tasks))
    fold_of = np.zeros(len(y), dtype=int)
    for k, (_, held) in enumerate(folds):
        fold_of[held] = k

    pb, pu = np.zeros(len(y)), np.zeros(len(y))
    xt, yt = np.stack([vecs[i] for i in tau.id]), tau.label.to_numpy()
    for fit, held in folds:
        held_tasks = set(tasks[held])
        for seed in range(3):
            p = pairs[~pairs.task.isin(held_tasks)]
            keep = np.random.default_rng(seed).choice(
                sorted(p.context_id.unique()), round(0.25 * p.context_id.nunique()), replace=False
            )
            p = p[p.context_id.isin(keep)]
            n = len(fit) + len(p)
            xf = np.vstack([x[fit], np.stack([vecs[i] for i in p.id]), xt])
            yf = np.concatenate([y[fit], p.label.to_numpy(), yt]).astype(int)
            wf = np.concatenate([np.ones(n), np.full(len(yt), 0.5 * n / len(yt))])
            sc = StandardScaler().fit(xf)
            m = lgbm().fit(sc.transform(xf), yf, sample_weight=wf)
            pb[held] += m.predict_proba(sc.transform(x[held]))[:, 1] / 3
        sc = StandardScaler().fit(xu[fit])
        mu = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(xu[fit]), y[fit])
        pu[held] = mu.predict_proba(sc.transform(xu[held]))[:, 1]

    # the stacker for a held-out fold learns only from the other folds' out-of-fold head scores
    z = np.column_stack([logit(pb), logit(pu)])
    stacked = np.zeros(len(y))
    for k in range(FOLDS):
        train, held = fold_of != k, fold_of == k
        st = LogisticRegression(C=1.0, max_iter=5000).fit(z[train], y[train])
        stacked[held] = st.predict_proba(z[held])[:, 1]
    rank_sum = ranks(pb) + 0.5 * ranks(pu)

    matched = round(TEST_FLAGS_PER_ERROR * y.sum())
    result: dict = {
        "rows": len(y),
        "errors_v1s": int(y.sum()),
        "rules_fire": int(fired.sum()),
        "ranking": {},
    }
    for name, s in (("rank_sum_in_file", rank_sum), ("stacked_probability", stacked)):
        result["ranking"][name] = {
            "auroc_v1s": round(float(auroc(y, s + 10 * fired)), 4),
            "f1_at_test_flags_per_error": f1_flags(y, top_matched(s, fired, matched)),
        }

    # fixed threshold: chosen on four folds' out-of-fold probabilities, applied to the fifth
    flagged = np.zeros(len(y), dtype=bool)
    per_fold = []
    for k in range(FOLDS):
        train, held = fold_of != k, fold_of == k
        t = best_threshold(stacked[train], y[train], fired[train])
        f = fired[held] | (stacked[held] >= t)
        flagged[held] = f
        per_fold.append(
            {
                "fold": k,
                "threshold": round(t, 4),
                "rows": int(held.sum()),
                "errors": int(y[held].sum()),
                "flagged": int(f.sum()),
                "flags_per_error": round(float(f.sum() / max(1, y[held].sum())), 2),
                "f1": f1_flags(y[held], f),
            }
        )
    # the F1 maximum on 77 rows is noisy; for a calibrated probability the F1-optimal threshold is half the
    # best reachable F1 (Lipton, Elkan, Narayanaswamy 2014), a rule that barely depends on the sample
    rules = {"half_of_best_f1": None, "fixed_0.30": 0.30, "fixed_0.35": 0.35, "fixed_0.40": 0.40}
    result["threshold_rules"] = {}
    for name, fixed in rules.items():
        flagged_r, rows_r = np.zeros(len(y), dtype=bool), []
        for k in range(FOLDS):
            train, held = fold_of != k, fold_of == k
            if fixed is None:
                t = best_threshold(stacked[train], y[train], fired[train])
                t = binary_scores(y[train], (fired[train] | (stacked[train] >= t)).astype(int)).f1 / 2
            else:
                t = fixed
            f = fired[held] | (stacked[held] >= t)
            flagged_r[held] = f
            rows_r.append(
                {"threshold": round(float(t), 3), "flags_per_error": round(float(f.sum() / max(1, y[held].sum())), 2)}
            )
        result["threshold_rules"][name] = {
            "pooled_f1_held_out": f1_flags(y, flagged_r),
            "pooled_flags_per_error": round(float(flagged_r.sum() / y.sum()), 2),
            "per_fold": rows_r,
        }
    np.savez(
        ROOT / "data/interim/threshold_study_scores.npz", pb=pb, pu=pu, stacked=stacked, y=y, fired=fired, fold=fold_of
    )
    t_all = best_threshold(stacked, y, fired)
    result["fixed_threshold"] = {
        "pooled_f1_held_out": f1_flags(y, flagged),
        "pooled_flags_per_error": round(float(flagged.sum() / y.sum()), 2),
        "per_fold": per_fold,
        "threshold_on_all_contexts": round(t_all, 4),
        "calibration_mean_probability_vs_error_share": [round(float(stacked.mean()), 3), round(float(y.mean()), 3)],
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
