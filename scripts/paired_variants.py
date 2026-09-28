"""Paired difference in AUROC (v1s labels, all organiser contexts) between two tau2ru_ruler.py runs, from the
out-of-fold system scores they save, with a bootstrap over dialogues. Both runs score the same contexts, so the
dialogue-to-dialogue noise they share cancels in the difference.

    python scripts/paired_variants.py --a strong_kbank_ruler_opus_norepeats_simtotal1 \
        --b strong955_ruler_opus_norepeats_simtotal1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_head_ru import TEACHERS, auroc, labels  # noqa: E402


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="results/<name>.scores.npz, the variant")
    ap.add_argument("--b", required=True, help="results/<name>.scores.npz, what it is compared with")
    ap.add_argument("--key", default="both_w1.0")
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    args = ap.parse_args()

    a, b = (np.load(ROOT / "results" / f"{n}.scores.npz") for n in (args.a, args.b))
    if not (a["ids"] == b["ids"]).all():
        raise SystemExit("the two runs score different contexts")
    tasks = pd.read_parquet(args.contexts).set_index("id").task
    found = labels(TEACHERS["v1s"].format(split="train")) | labels(TEACHERS["v1s"].format(split="check"))
    ids = pd.Series(a["ids"])
    truth = ids.map(found).to_numpy().astype(int)
    groups = [np.where(ids.map(tasks).to_numpy() == t)[0] for t in np.unique(ids.map(tasks))]
    sa, sb = a[args.key], b[args.key]

    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(args.repeats):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        diffs.append(auroc(truth[idx], sa[idx]) - auroc(truth[idx], sb[idx]))
    diffs = np.array(diffs)
    result = {
        "a": args.a,
        "b": args.b,
        "contexts": len(ids),
        "errors_v1s": int(truth.sum()),
        "dialogues": len(groups),
        "auroc_a": round(float(auroc(truth, sa)), 4),
        "auroc_b": round(float(auroc(truth, sb)), 4),
        "a_minus_b": round(float(auroc(truth, sa) - auroc(truth, sb)), 4),
        "ci95": [round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])],
        "share_above_zero": round(float((diffs > 0).mean()), 3),
        "repeats": args.repeats,
    }
    out = ROOT / "results" / f"paired_{args.a}__vs__{args.b}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
