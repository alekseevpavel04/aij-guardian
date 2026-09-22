"""A linear head trained on Russian rows of the organiser's own dialogues, measured on the held-out half.

Training rows: the organiser's contexts of the train half of τ² tasks (scripts/extract_contexts.py),
labelled by the privileged labeller, optionally with the minimal pairs written for those contexts.
Evaluation, never used for any choice:

- the contexts of the check half, with the same labeller's labels;
- the rows of valid.parquet that share no dialogue with a training context (a task of the check half,
  or a first turn with no earlier context), with the organiser's labels; the rows whose dialogue has
  training contexts are reported apart as leaked.

The judge's scores on the valid rows are reported next to the head's. C is chosen by
cross-validation on the training rows grouped by task.

    python scripts/train_head_ru.py
    python scripts/train_head_ru.py --pairs --rules
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

C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
JUDGE = "baseline_prompt_q4_valid"


def vectors(name: str) -> dict[str, np.ndarray]:
    return {r["id"]: decode(r["vector"]) for r in journal.read(ROOT / "data/interim/embeddings" / f"{name}.jsonl")}


def labels(name: str) -> dict[str, int]:
    return {r["id"]: int(r["score"] >= 0.5) for r in journal.read(ROOT / "data/interim/labels" / f"{name}.jsonl")}


def design(rows: pd.DataFrame, vecs: dict[str, np.ndarray], with_rules: bool) -> np.ndarray:
    missing = [i for i in rows.id if i not in vecs]
    if missing:
        raise SystemExit(f"{len(missing)} rows have no vector yet, e.g. {missing[0]}")
    x = np.stack([vecs[i] for i in rows.id])
    if with_rules:
        rules = pd.DataFrame([features(p, r) for p, r in zip(rows.prompt, rows.response, strict=True)])
        x = np.hstack([x, rules.to_numpy(dtype=np.float32)])
    return x


def auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    return round(float(roc_auc_score(y, s)), 4) if 0 < y.sum() < len(y) else None


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--pairs", action="store_true", help="add the written minimal pairs to training")
    ap.add_argument("--rules", action="store_true", help="append the rule features to the vector")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    context_vecs = vectors("q4_last_org")
    train = contexts[contexts.split == "train"].copy()
    train["label"] = train.id.map(labels("labeler_v1s_org_train"))
    train["group"] = train.task
    if args.pairs:
        pairs = pd.read_parquet(ROOT / "data/interim/pilot_written.parquet")
        pairs = pairs[pairs.context_id.map(contexts.set_index("id").split) == "train"].copy()
        pairs["group"] = pairs.task
        context_vecs |= vectors("q4_last_pilot_written")
        train = pd.concat([train, pairs[["id", "prompt", "response", "label", "group"]]], ignore_index=True)
    train = train.dropna(subset=["label"])
    x, y, groups = design(train, context_vecs, args.rules), train.label.to_numpy().astype(int), train.group.to_numpy()

    cv = {}
    for c in C_GRID:
        oof = np.zeros(len(y))
        for fit, held in GroupKFold(n_splits=5).split(x, y, groups):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
            oof[held] = model.fit(x[fit], y[fit]).predict_proba(x[held])[:, 1]
        cv[c] = auroc(y, oof)
    best = max(cv, key=lambda c: cv[c] or 0)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=5000)).fit(x, y)

    check = contexts[contexts.split == "check"].copy()
    check["label"] = check.id.map(labels("labeler_v1s_org_check"))
    check = check.dropna(subset=["label"])
    check_score = model.predict_proba(design(check, context_vecs, args.rules))[:, 1]

    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet")
    valid["task"] = valid.id.str.split("::").str[0]
    train_tasks = set(contexts[contexts.split == "train"].task)
    valid["half"] = np.where(valid.task.isin(train_tasks), "leaked", "clean")
    valid_score = model.predict_proba(design(valid, vectors("q4_last_valid"), args.rules))[:, 1]
    judged = {r["id"]: r["score"] for r in journal.read(ROOT / "data/interim/judge" / f"{JUDGE}.jsonl")}
    judge_score = valid.id.map(judged).to_numpy(dtype=float)

    result = {
        "train_rows": len(y),
        "train_positives": int(y.sum()),
        "pairs": args.pairs,
        "rules": args.rules,
        "cv_auroc_by_c": {str(c): v for c, v in cv.items()},
        "chosen_c": best,
        "check_contexts": {
            "rows": len(check),
            "positives": int(check.label.sum()),
            "head_auroc": auroc(check.label.to_numpy().astype(int), check_score),
        },
        "valid": {},
    }
    for half in ("clean", "leaked"):
        mask = (valid.half == half).to_numpy()
        yv = valid.label.to_numpy().astype(int)[mask]
        result["valid"][half] = {
            "rows": int(mask.sum()),
            "positives": int(yv.sum()),
            "head_auroc": auroc(yv, valid_score[mask]),
            "judge_auroc": auroc(yv, judge_score[mask]),
        }
    name = "head_ru" + ("_pairs" if args.pairs else "") + ("_rules" if args.rules else "")
    out = args.out or ROOT / "results" / f"{name}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
