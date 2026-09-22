"""A linear head on the verdict-position hidden state, trained on ext_dev and measured elsewhere (T4.H1).

The head is a logistic regression on the standardised vector, optionally with the rule features
(`guardian.rules.features`). Its strength is chosen by cross-validation on the training rows grouped
by trajectory, so steps of one trajectory never sit on both sides of a fold. The evaluation sets are
never used for any choice; the judge's scores on the same rows are reported next to the head's.

    python scripts/train_head.py --train q4_last_extdev1000 --eval q4_last_extval400 q4_last_valid
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from guardian import journal
from guardian.env import setup
from guardian.rules import features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from embed_rows import decode  # noqa: E402
from judge_eval import load  # noqa: E402

# embedding run -> the rows it was computed on (data, limit), as in scripts/embed_rows.py
RUNS = {
    "q4_last_extdev1000": ("ext_dev", 1000),
    "q4_last_extval400": ("ext_val", 400),
    "q4_last_valid": ("valid", None),
}
JUDGE = {"q4_last_extval400": "baseline_prompt_q4_extval400", "q4_last_valid": "baseline_prompt_q4_valid"}
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def matrix(name: str, with_rules: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    data, limit = RUNS[name]
    frame = load(data, limit, 0).set_index("id")
    records = journal.read(ROOT / "data/interim/embeddings" / f"{name}.jsonl")
    ids = [r["id"] for r in records]
    missing = set(frame.index) - set(ids)
    if missing:
        raise SystemExit(f"{name}: {len(missing)} rows of {len(frame)} have no vector yet")
    x = np.stack([decode(r["vector"]) for r in records])
    rows = frame.loc[ids]
    if with_rules:
        rules = pd.DataFrame([features(p, r) for p, r in zip(rows.prompt, rows.response, strict=True)])
        x = np.hstack([x, rules.to_numpy(dtype=np.float32)])
    groups = np.array([i.split("::")[0] for i in ids])
    return x, rows.label.to_numpy().astype(int), groups, rows.reset_index()


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="q4_last_extdev1000")
    ap.add_argument("--eval", nargs="+", default=["q4_last_extval400", "q4_last_valid"])
    ap.add_argument("--rules", action="store_true", help="append the rule features to the vector")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    x, y, groups, _ = matrix(args.train, args.rules)
    folds = GroupKFold(n_splits=5)
    cv = {}
    for c in C_GRID:
        oof = np.zeros(len(y))
        for fit, held in folds.split(x, y, groups):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
            oof[held] = model.fit(x[fit], y[fit]).predict_proba(x[held])[:, 1]
        cv[c] = round(float(roc_auc_score(y, oof)), 4)
    best = max(cv, key=cv.get)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=5000)).fit(x, y)

    result = {
        "train": args.train,
        "train_rows": len(y),
        "train_positives": int(y.sum()),
        "rules": args.rules,
        "cv_auroc_by_c": {str(c): v for c, v in cv.items()},
        "chosen_c": best,
        "eval": {},
    }
    for name in args.eval:
        xe, ye, _, rows = matrix(name, args.rules)
        score = model.predict_proba(xe)[:, 1]
        entry = {"rows": len(ye), "positives": int(ye.sum()), "head_auroc": round(float(roc_auc_score(ye, score)), 4)}
        if name in JUDGE:
            judged = {r["id"]: r["score"] for r in journal.read(ROOT / "data/interim/judge" / f"{JUDGE[name]}.jsonl")}
            js = rows.id.map(judged).to_numpy(dtype=float)
            entry["judge"] = JUDGE[name]
            entry["judge_auroc"] = round(float(roc_auc_score(ye, js)), 4)
            # the two scores on one scale: rank average, no fitting on the evaluation rows
            ranks = (pd.Series(score).rank().to_numpy() + pd.Series(js).rank().to_numpy()) / 2
            entry["rank_mean_auroc"] = round(float(roc_auc_score(ye, ranks)), 4)
        result["eval"][name] = entry
    out = args.out or ROOT / "results" / f"head_{args.train}{'_rules' if args.rules else ''}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
