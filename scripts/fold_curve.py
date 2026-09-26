"""Does the candidate still gain from more of the organiser's own data? The ruler with 5, 10, 20 and 40 folds.

With k folds over the 40 dialogues each model learns from (k − 1)/k of them: 80 %, 90 %, 95 %, 97.5 %.
If the system's AUROC keeps rising along that curve, more data like the organiser's should help, which
is what the Russian τ² simulations are for; if it is flat, the limit is elsewhere. Downward the same
question was answered by uncertainty_volume.py (25–75 % of the contexts). Parts as in
component_ablation.py: head B as logistic regression and LightGBM with a quarter of the v3/v4 pairs and
the τ² steps (three seeds), the uncertainty head, rules first. Scores of different folds come from
different models, so more folds also mix more models into one ranking; that noise works against the
larger k, not for it.

    python scripts/fold_curve.py
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
from context_ruler import V3_V4, C, Ruler  # noqa: E402
from head_combos import ranks  # noqa: E402
from head_models import lgbm  # noqa: E402
from judge_eval import load  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import vectors  # noqa: E402
from turn_ruler import RULES, part_summary, summary  # noqa: E402

EMB = ROOT / "data/interim/embeddings"


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--folds", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--out", type=Path, default=ROOT / "results/fold_curve.json")
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
    xt, yt = np.stack([vecs[i] for i in tau.id]), tau.label.to_numpy()

    def head_b(folds: int, model: str) -> np.ndarray:
        out = np.zeros(len(y))
        for fit, held in GroupKFold(folds).split(x, y, tasks):
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
                m = LogisticRegression(C=C, max_iter=5000) if model == "logistic" else lgbm()
                m.fit(sc.transform(xf), yf, sample_weight=wf)
                xh = sc.transform(x[held])
                out[held] += m.decision_function(xh) if model == "logistic" else m.predict_proba(xh)[:, 1]
        return out

    def uncertainty(folds: int) -> np.ndarray:
        out = np.zeros(len(y))
        for fit, held in GroupKFold(folds).split(xu, y, tasks):
            sc = StandardScaler().fit(xu[fit])
            m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(xu[fit]), y[fit])
            out[held] = m.predict_proba(sc.transform(xu[held]))[:, 1]
        return out

    result: dict = {"folds": {}}
    base: dict = {}
    for k in args.folds:
        u = uncertainty(k)
        row = {"train_share": round((k - 1) / k, 3), "uncertainty_alone": ruler.auroc_ci(u)["auroc"]}
        for model in ("logistic", "lightgbm"):
            b = head_b(k, model)
            system = ranks(b) + 0.5 * ranks(u) + 10 * fired
            row[model] = {"head_b_alone": ruler.auroc_ci(b)["auroc"], "system": ruler.auroc_ci(system)["auroc"]}
            if k == args.folds[0]:
                base[model] = (b, system)
            else:
                row[model]["system_minus_first"] = ruler.paired(system, base[model][1])
                row[model]["head_b_minus_first"] = ruler.paired(b, base[model][0])
        if k == args.folds[0]:
            base["u"] = u
        else:
            row["uncertainty_minus_first"] = ruler.paired(u, base["u"])
        result["folds"][str(k)] = row
        print(k, json.dumps(row), flush=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
