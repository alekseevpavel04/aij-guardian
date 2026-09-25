"""Head B against the amount of τ² steps it learns from, up to ext_dev plus all of ext_val, on the context ruler.

Points: 800 and 2000 rows of ext_dev (whole trajectories, fixed seed), all 3626 rows of ext_dev,
and ext_dev with all 2691 rows of ext_val (decision 37: ext_val is no longer a check and goes into
training). Head B is trained as in component_ablation.py (the contexts' v1s labels, a quarter of the
v3/v4 pairs, three seeds; the external rows together weigh half of the Russian rows), as logistic
regression and as LightGBM, and measured alone and in the system (rules first, then the mean of ranks
of the head and the uncertainty head at weight 0.5), each point against the ext_dev point.

    python scripts/ext_volume_curve.py
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
from context_ruler import FOLDS, V3_V4, C, Ruler  # noqa: E402
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
    ap.add_argument("--out", type=Path, default=ROOT / "results/ext_volume_curve.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org") | vectors("q4_last_extdev_all") | vectors("q4_last_extval_all")
    for name in ("q4_last_pairs_02", "q4_last_pairs_03"):
        vecs |= vectors(name)
    points = {
        "ext_dev_800": load("ext_dev", 800, 0),
        "ext_dev_2000": load("ext_dev", 2000, 0),
        "ext_dev_all": load("ext_dev", None, 0),
    }
    points["ext_dev_plus_ext_val"] = pd.concat([points["ext_dev_all"], load("ext_val", None, 0)])
    points = {k: v[v.id.isin(vecs)] for k, v in points.items()}

    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), None)
    y, x, tasks, pairs = ruler.label["v1s"], ruler.x, ruler.tasks, ruler.pairs
    stats = {r["id"]: r for r in journal.read(EMB / "q4_tok_uncertainty_org.jsonl")}
    feats = {
        r["id"]: np.concatenate([summary(stats[r["id"]]), part_summary(stats[r["id"]], r["parts"])])
        for r in journal.read(EMB / "q4_tok_parts_org.jsonl")
    }
    no_pairs = pd.DataFrame(columns=["id", "task", "context_id", "label"])
    unc = Ruler(contexts, feats, no_pairs, None).scores(y, 0.0, make=lambda: LogisticRegression(C=1.0, max_iter=5000))
    fired = np.array(
        [
            any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
            for r in ruler.ctx.itertuples()
        ]
    )

    def head(tau: pd.DataFrame, model: str) -> np.ndarray:
        out = np.zeros(len(y))
        xt, yt = np.stack([vecs[i] for i in tau.id]), tau.label.to_numpy()
        for fit, held in GroupKFold(FOLDS).split(x, y, tasks):
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

    def system(h: np.ndarray) -> np.ndarray:
        return ranks(h) + 0.5 * ranks(unc) + 10 * fired

    result: dict = {"rows": {k: [len(v), int(v.label.sum())] for k, v in points.items()}, "models": {}}
    for model in ("logistic", "lightgbm"):
        scores = {k: head(v, model) for k, v in points.items()}
        base = scores["ext_dev_all"]
        result["models"][model] = {
            k: {
                "alone": ruler.auroc_ci(s)["auroc"],
                "system": ruler.auroc_ci(system(s))["auroc"],
                "alone_minus_ext_dev_all": ruler.paired(s, base),
                "system_minus_ext_dev_all": ruler.paired(system(s), system(base)),
            }
            for k, s in scores.items()
        }
        print(model, json.dumps(result["models"][model]), flush=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
