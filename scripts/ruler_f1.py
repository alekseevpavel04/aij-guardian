"""Why the ruler reads 0.98 while the test gives F1 0.51: the same systems under other measures.

The context ruler reports AUROC on the 343 contexts where the v1s and Opus labels agree. Here the
submitted system (rules, then the logistic head B without τ²) and the candidate (rules, LightGBM head B
with τ², the uncertainty head at 0.5) are scored, out of fold as in component_ablation.py, on:
the agreed contexts, all 384 contexts with the v1s labels, and all 384 with the Opus labels; each by
AUROC and by F1 when the top 23.72 % are flagged (the share of positives on the competition's test).

    python scripts/ruler_f1.py
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
from train_head_ru import TEST_SHARE, at_share, auroc, vectors  # noqa: E402
from turn_ruler import RULES, part_summary, summary  # noqa: E402

EMB = ROOT / "data/interim/embeddings"


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/ruler_f1.json")
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
    no_pairs = pd.DataFrame(columns=["id", "task", "context_id", "label"])
    unc = Ruler(contexts, feats, no_pairs, None).scores(y, 0.0, make=lambda: LogisticRegression(C=1.0, max_iter=5000))
    fired = np.array(
        [
            any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
            for r in ruler.ctx.itertuples()
        ]
    )

    def head(with_tau: bool, model: str) -> np.ndarray:
        out = np.zeros(len(y))
        for fit, held in GroupKFold(FOLDS).split(x, y, tasks):
            held_tasks = set(tasks[held])
            for seed in range(3):
                p = pairs[~pairs.task.isin(held_tasks)]
                keep = np.random.default_rng(seed).choice(
                    sorted(p.context_id.unique()), round(0.25 * p.context_id.nunique()), replace=False
                )
                p = p[p.context_id.isin(keep)]
                xs, ys = [x[fit], np.stack([vecs[i] for i in p.id])], [y[fit], p.label.to_numpy()]
                n = len(fit) + len(p)
                ws = [np.ones(n)]
                if with_tau:
                    xs.append(np.stack([vecs[i] for i in tau.id]))
                    ys.append(tau.label.to_numpy())
                    ws.append(np.full(len(tau), 0.5 * n / len(tau)))
                xf, yf, wf = np.vstack(xs), np.concatenate(ys).astype(int), np.concatenate(ws)
                sc = StandardScaler().fit(xf)
                m = LogisticRegression(C=C, max_iter=5000) if model == "logistic" else lgbm()
                m.fit(sc.transform(xf), yf, sample_weight=wf)
                xh = sc.transform(x[held])
                out[held] += m.decision_function(xh) if model == "logistic" else m.predict_proba(xh)[:, 1]
        return out

    systems = {
        "submitted": ranks(head(False, "logistic")) + 10 * fired,
        "candidate": ranks(head(True, "lightgbm")) + 0.5 * ranks(unc) + 10 * fired,
    }
    views = {
        "agreed_contexts_v1s": (ruler.agree, ruler.label["v1s"]),
        "all_contexts_v1s": (np.ones(len(y), bool), ruler.label["v1s"]),
        "all_contexts_opus": (np.ones(len(y), bool), ruler.label["opus"]),
    }
    result: dict = {"test_share": TEST_SHARE, "views": {}}
    for view, (mask, truth) in views.items():
        result["views"][view] = {"rows": int(mask.sum()), "positives": int(truth[mask].sum())}
        for name, s in systems.items():
            result["views"][view][name] = {
                "auroc": round(float(auroc(truth[mask], s[mask])), 4),
                "at_test_share": at_share(truth[mask], s[mask]),
            }
        print(view, json.dumps(result["views"][view]), flush=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
