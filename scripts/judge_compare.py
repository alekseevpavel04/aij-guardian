"""Compare two judge runs on the same rows: AUROC of each and a paired bootstrap of the difference.

Steps of one trajectory are correlated, so the bootstrap resamples trajectories (the id before
"::"), not rows. On valid.parquet every row is its own trajectory.

`--where-rules-silent` keeps only rows on which no check of the predictor fires: in the cascade the
judge decides only there, so that is where a better judge has to be better.

    python scripts/judge_compare.py --a v1_q4_extval400 --b baseline_prompt_q4_extval400
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from guardian.env import setup
from guardian.predictor import DEFAULT_CHECKS
from guardian.rules import Context, evidence

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = (ROOT / "data/external/apb_tau2.parquet", ROOT / "data/external/trajerr_tau2.parquet")


def rows() -> pd.DataFrame:
    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet", columns=["id", "prompt", "response", "label"])
    valid["source"] = "valid"
    columns = ["id", "prompt", "response", "weak_label", "source"]
    external = pd.concat([pd.read_parquet(p, columns=columns) for p in EXTERNAL])
    return pd.concat([valid, external.rename(columns={"weak_label": "label"})], ignore_index=True)


def rules_fire(row) -> bool:
    found = evidence(row.prompt, row.response, Context.from_prompt(row.prompt))
    return any(found[name] for name in DEFAULT_CHECKS if name != "repeated_call")


def scores(name: str) -> pd.Series:
    path = ROOT / "data/interim/judge" / f"{name}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame(records).drop_duplicates("id", keep="last").set_index("id").score


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default=None, help="keep one source: valid, agentprocessbench, trajerrbench")
    ap.add_argument("--where-rules-silent", action="store_true")
    args = ap.parse_args()

    frame = pd.DataFrame({"a": scores(args.a), "b": scores(args.b)}).dropna()
    frame = frame.join(rows().set_index("id"), how="inner")
    if args.source:
        frame = frame[frame.source == args.source]
    if args.where_rules_silent:
        frame = frame[[not rules_fire(r) for r in frame.itertuples()]]
    frame["trajectory"] = frame.index.str.split("::").str[0]
    groups = {t: idx for t, idx in frame.groupby("trajectory").indices.items()}
    keys = list(groups)
    y, a, b = frame.label.to_numpy(), frame.a.to_numpy(), frame.b.to_numpy()

    rng = np.random.default_rng(args.seed)
    diffs = []
    for _ in range(args.resamples):
        idx = np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        if 0 < y[idx].sum() < len(idx):
            diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    result = {
        "a": args.a,
        "b": args.b,
        "source": args.source,
        "where_rules_silent": args.where_rules_silent,
        "rows": len(frame),
        "positives": int(y.sum()),
        "trajectories": len(keys),
        "auroc_a": round(float(roc_auc_score(y, a)), 4),
        "auroc_b": round(float(roc_auc_score(y, b)), 4),
        "difference": round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 4),
        "difference_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "resamples": len(diffs),
    }
    suffix = (f"__{args.source}" if args.source else "") + ("__rules_silent" if args.where_rules_silent else "")
    out = ROOT / "results" / f"judge_compare_{args.a}__{args.b}{suffix}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
