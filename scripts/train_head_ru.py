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
    python scripts/train_head_ru.py --pairs airline_01 retail_01 airline_02 --out results/x.json
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
from guardian.metrics import binary_scores
from guardian.rules import features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from embed_rows import decode  # noqa: E402
from judge_eval import load  # noqa: E402

C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
PILOT = ["airline_01", "retail_01"]
# every embedding journal of written turns; a batch's rows are looked up by id
PAIR_VECTORS = ("q4_last_pilot_written", "q4_last_pairs_02", "q4_last_pairs_03")
JUDGE = "baseline_prompt_q4_valid"
JUDGE_EXT = "baseline_prompt_q4_extval400"
TEACHERS = {"v1s": "labeler_v1s_org_{split}", "opus": "opus_v4_org_{split}"}
TEST_SHARE = 0.2372  # share of positives on the competition's test, the rate the submission flags at


def vectors(name: str) -> dict[str, np.ndarray]:
    return {r["id"]: decode(r["vector"]) for r in journal.read(ROOT / "data/interim/embeddings" / f"{name}.jsonl")}


def labels(name: str) -> dict[str, int]:
    return {r["id"]: int(r["score"] >= 0.5) for r in journal.read(ROOT / "data/interim/labels" / f"{name}.jsonl")}


def at_share(y: np.ndarray, s: np.ndarray, share: float = TEST_SHARE) -> dict:
    flagged = np.zeros(len(s), dtype=int)
    flagged[np.argsort(-s)[: round(share * len(s))]] = 1
    scores = binary_scores(y, flagged)
    return {"flagged": int(flagged.sum()), "true_positives": scores.true_positives, "f1": round(scores.f1, 4)}


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
    ap.add_argument(
        "--pairs", nargs="*", metavar="BATCH", help="add the minimal pairs of these batches (no names: the pilot)"
    )
    ap.add_argument("--rules", action="store_true", help="append the rule features to the vector")
    ap.add_argument("--teacher", choices=sorted(TEACHERS), default="v1s", help="labels of the training contexts")
    ap.add_argument("--drop-copies", action="store_true", help="leave out written turns equal to the original")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    context_vecs = vectors("q4_last_org")
    train = contexts[contexts.split == "train"].copy()
    train["label"] = train.id.map(labels(TEACHERS[args.teacher].format(split="train")))
    train["group"] = train.task
    batches = None if args.pairs is None else args.pairs or PILOT
    if batches:
        pairs = pd.concat([pd.read_parquet(ROOT / "data/synthetic/pairs" / b / "pairs.parquet") for b in batches])
        pairs = pairs[pairs.context_id.map(contexts.set_index("id").split) == "train"].copy()
        pairs["group"] = pairs.task
        if args.drop_copies:
            pairs = pairs[~pairs["flags"].map(lambda f: "same_as_original" in list(f))]
        for name in PAIR_VECTORS:
            if (ROOT / "data/interim/embeddings" / f"{name}.jsonl").exists():
                context_vecs |= vectors(name)
        train = pd.concat([train, pairs[["id", "prompt", "response", "label", "group"]]], ignore_index=True)
    result = {
        "pairs": batches is not None,
        "pair_batches": batches,
        "rules": args.rules,
        "teacher": args.teacher,
        "drop_copies": args.drop_copies,
        **measure(train, context_vecs, contexts, args.rules),
    }
    name = "head_ru" + ("_opus" if args.teacher == "opus" else "") + ("_pairs" if batches else "")
    name += "_rules" if args.rules else ""
    out = args.out or ROOT / "results" / f"{name}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def measure(
    train: pd.DataFrame, context_vecs: dict, contexts: pd.DataFrame, rules: bool, scores: dict | None = None
) -> dict:
    train = train.dropna(subset=["label"])
    x, y, groups = design(train, context_vecs, rules), train.label.to_numpy().astype(int), train.group.to_numpy()
    # optional: per-row weights, and a "written" indicator that takes up what tells a written turn from a
    # natural one during training and is 0 for every row the head is measured on
    weight = train["weight"].to_numpy(dtype=float) if "weight" in train else np.ones(len(y))
    indicator = "written" in train
    if indicator:
        x = np.hstack([x, train["written"].to_numpy(dtype=np.float32)[:, None]])

    def features_of(rows: pd.DataFrame, vecs: dict) -> np.ndarray:
        z = design(rows, vecs, rules)
        return np.hstack([z, np.zeros((len(z), 1), dtype=np.float32)]) if indicator else z

    def fitted(c: float, rows: np.ndarray):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        return model.fit(x[rows], y[rows], logisticregression__sample_weight=weight[rows])

    cv = {}
    for c in C_GRID:
        oof = np.zeros(len(y))
        for fit, held in GroupKFold(n_splits=5).split(x, y, groups):
            oof[held] = fitted(c, fit).predict_proba(x[held])[:, 1]
        cv[c] = auroc(y, oof)
    best = max(cv, key=lambda c: cv[c] or 0)
    model = fitted(best, np.arange(len(y)))

    check = contexts[contexts.split == "check"].copy()
    check["label"] = check.id.map(labels("labeler_v1s_org_check"))
    check = check.dropna(subset=["label"])
    check_score = model.predict_proba(features_of(check, context_vecs))[:, 1]
    check_opus = check.id.map(labels(TEACHERS["opus"].format(split="check"))).to_numpy().astype(int)
    if scores is not None:  # for a paired comparison of heads on the same contexts
        scores.update(zip(check.id, check_score, strict=True))

    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet")
    valid["task"] = valid.id.str.split("::").str[0]
    train_tasks = set(contexts[contexts.split == "train"].task)
    valid["half"] = np.where(valid.task.isin(train_tasks), "leaked", "clean")
    valid_score = model.predict_proba(features_of(valid, vectors("q4_last_valid")))[:, 1]
    if scores is not None:
        scores.update(zip(valid.id, valid_score, strict=True))
    judged = {r["id"]: r["score"] for r in journal.read(ROOT / "data/interim/judge" / f"{JUDGE}.jsonl")}
    judge_score = valid.id.map(judged).to_numpy(dtype=float)

    result = {
        "train_rows": len(y),
        "train_positives": int(y.sum()),
        "cv_auroc_by_c": {str(c): v for c, v in cv.items()},
        "chosen_c": best,
        "check_contexts": {
            "rows": len(check),
            "positives": int(check.label.sum()),
            "head_auroc": auroc(check.label.to_numpy().astype(int), check_score),
            "head_at_test_share": at_share(check.label.to_numpy().astype(int), check_score),
        },
        "check_contexts_opus_labels": {
            "rows": len(check_opus),
            "positives": int(check_opus.sum()),
            "head_auroc": auroc(check_opus, check_score),
            "head_at_test_share": at_share(check_opus, check_score),
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
    yv = valid.label.to_numpy().astype(int)
    result["valid"]["all"] = {
        "rows": len(yv),
        "positives": int(yv.sum()),
        "head_auroc": auroc(yv, valid_score),
        "judge_auroc": auroc(yv, judge_score),
    }
    # both labellers agree: fewer label errors, a check that does not side with either teacher
    agreed = check.label.to_numpy().astype(int) == check_opus
    result["check_contexts_agreed"] = {
        "rows": int(agreed.sum()),
        "positives": int(check_opus[agreed].sum()),
        "head_auroc": auroc(check_opus[agreed], check_score[agreed]),
    }
    # the English external rows with their own weak labels: a check that nothing broke, not of gain
    ext = load("ext_val", 400, 0)
    ext_score = model.predict_proba(features_of(ext, vectors("q4_last_extval400")))[:, 1]
    judged = {r["id"]: r["score"] for r in journal.read(ROOT / "data/interim/judge" / f"{JUDGE_EXT}.jsonl")}
    result["ext_val"] = {
        "rows": len(ext),
        "positives": int(ext.label.sum()),
        "head_auroc": auroc(ext.label.to_numpy().astype(int), ext_score),
        "judge_auroc": auroc(ext.label.to_numpy().astype(int), ext.id.map(judged).to_numpy(dtype=float)),
    }
    return result


if __name__ == "__main__":
    main()
