"""Paired difference in AUROC (v1s or Opus labels of the organiser contexts) between two tau2ru_ruler.py runs, from the
out-of-fold system scores they save, with a bootstrap over dialogues. Both runs score the same contexts, so the
dialogue-to-dialogue noise they share cancels in the difference.

    python scripts/paired_variants.py --a strong_kbank_ruler_opus_norepeats_simtotal1 \
        --b strong955_ruler_opus_norepeats_simtotal1
"""

import argparse
import json
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = ROOT / "data/synthetic/organizer_contexts.parquet"
sys.path.insert(0, str(ROOT / "scripts"))
from train_head_ru import TEACHERS, auroc, labels  # noqa: E402

VIEWS = (*TEACHERS, "agreed")
TEST_FLAGS_PER_ERROR = 1.31  # the submissions flagged 1.31 rows per error of the test


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def f1_at_flags(y: np.ndarray, s: np.ndarray) -> float:
    """F1 when the top 1.31 x (number of errors) rows are flagged, the test's operating point."""
    k = round(TEST_FLAGS_PER_ERROR * y.sum())
    top = np.zeros(len(s), bool)
    top[np.argsort(-s, kind="stable")[:k]] = True
    hits = (top & (y == 1)).sum()
    return float(2 * hits / (k + y.sum()))


METRICS = {"auroc": auroc, "ap": average_precision, "f1_at_flags": f1_at_flags}


@cache
def views() -> dict[str, dict[str, int]]:
    return {t: labels(TEACHERS[t].format(split="train")) | labels(TEACHERS[t].format(split="check")) for t in TEACHERS}


@cache
def tasks() -> pd.Series:
    return pd.read_parquet(CONTEXTS).set_index("id").task


def compare(
    a_name: str,
    b_name: str,
    teacher: str = "v1s",
    domain: str | None = None,
    key: str = "both_w1.0",
    repeats: int = 2000,
    b_key: str | None = None,
    metric: str = "auroc",
) -> dict:
    """`b_name` "reference" is the system without simulations that the run `a_name` saved with it; teacher
    "agreed" keeps only the contexts where v1s and Opus agree."""
    a = np.load(ROOT / "results" / f"{a_name}.scores.npz")
    b_key = b_key or key
    b = (
        {"ids": a["ids"], b_key: a["reference"]}
        if b_name == "reference"
        else np.load(ROOT / "results" / f"{b_name}.scores.npz")
    )
    if not (a["ids"] == b["ids"]).all():
        raise SystemExit("the two runs score different contexts")
    view = views()
    found = view["v1s"] if teacher == "agreed" else view[teacher]
    keep = np.array([domain is None or i.split("__")[1] == domain for i in a["ids"]])
    if teacher == "agreed":
        keep &= np.array([view["v1s"][i] == view["opus"][i] for i in a["ids"]])
    ids = pd.Series(a["ids"][keep])
    truth = ids.map(found).to_numpy().astype(int)
    task = ids.map(tasks()).to_numpy()
    groups = [np.where(task == t)[0] for t in np.unique(task)]
    sa, sb = a[key][keep], b[b_key][keep]

    score = METRICS[metric]
    rng = np.random.default_rng(0)
    diffs, one_class = [], 0
    for _ in range(repeats):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if truth[idx].min() == truth[idx].max():  # a resample of a few dialogues can hold no error at all
            one_class += 1
            continue
        diffs.append(score(truth[idx], sa[idx]) - score(truth[idx], sb[idx]))
    diffs = np.array(diffs)
    return {
        "a": a_name if key == "both_w1.0" else f"{a_name}:{key}",
        "b": b_name if b_key == "both_w1.0" else f"{b_name}:{b_key}",
        "domain": domain,
        "contexts": len(ids),
        "teacher": teacher,
        "errors": int(truth.sum()),
        "dialogues": len(groups),
        "metric": metric,
        f"{metric}_a": round(float(score(truth, sa)), 4),
        f"{metric}_b": round(float(score(truth, sb)), 4),
        "a_minus_b": round(float(score(truth, sa) - score(truth, sb)), 4),
        "ci95": [round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])],
        "share_above_zero": round(float((diffs > 0).mean()), 3),
        "repeats": repeats,
        "resamples_without_both_classes": one_class,
    }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="results/<name>.scores.npz, the variant")
    ap.add_argument("--b", required=True, help="results/<name>.scores.npz to compare with, or `reference`")
    ap.add_argument("--key", default="both_w1.0")
    ap.add_argument("--b-key", default=None, help="the score of run b to compare with (default: --key)")
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--domain", default=None, help="only the contexts of one domain")
    ap.add_argument("--teacher", choices=VIEWS, default="v1s", help="whose labels of the contexts")
    ap.add_argument("--metric", choices=tuple(METRICS), default="auroc")
    args = ap.parse_args()

    result = compare(args.a, args.b, args.teacher, args.domain, args.key, args.repeats, args.b_key, args.metric)
    suffix = (f"__{args.domain}" if args.domain else "") + (f"__{args.teacher}" if args.teacher != "v1s" else "")
    suffix += (f"__{args.key}" if args.key != "both_w1.0" else "") + (f"__{args.b_key}" if args.b_key else "")
    suffix += f"__{args.metric}" if args.metric != "auroc" else ""
    out = ROOT / "results" / f"paired_{args.a}__vs__{args.b}{suffix}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
