"""The judge's verdict read off the final system's own pass, on the organiser contexts.

The last state of the pass sits where the organiser's judge prompt expects its one-token answer, so
state . (output row of "1" - row of "0") is the judge's logit without reasoning. Measured alone and added to the
final system's out-of-fold score as a rank at weights 0.25, 0.5 and 1, each paired against the system with a
bootstrap over dialogues on the agreed contexts, v1s and Opus. Submission 10 sent weight 0.25.

    python scripts/judge_signal.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from add_judge import judge_direction  # noqa: E402
from head_combos import ranks  # noqa: E402
from paired_variants import compare  # noqa: E402
from train_head_ru import vectors  # noqa: E402

WEIGHTS = (0.25, 0.5, 1.0)


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", type=Path, default=ROOT / "submission_work/final_fp8.oof.npz")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "data/raw/baseline/baseline/model")
    ap.add_argument("--context-vectors", default="fp8_contexts_final_org")
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "results/judge_signal.json")
    args = ap.parse_args()

    oof = np.load(args.oof, allow_pickle=True)
    ids, fired = oof["ids"].astype(str), oof["fired"].astype(bool)
    v = vectors(args.context_vectors)
    d = judge_direction(args.model_dir)
    judge = np.array([v[i] @ d for i in ids])
    scores = {"ids": ids, "system": oof["score"] + 10 * fired, "judge": judge}
    for w in WEIGHTS:
        scores[f"system_judge{w}"] = ranks(oof["score"]) + w * ranks(judge) + 10 * fired
    name = "judge_signal"
    np.savez(ROOT / "results" / f"{name}.scores.npz", **scores)
    result = {"contexts": len(ids), "paired_against_the_system": {}}
    for key in ("judge", *(f"system_judge{w}" for w in WEIGHTS)):
        for view in ("agreed", "v1s", "opus"):
            r = compare(name, name, view, key=key, b_key="system", repeats=args.repeats)
            result["paired_against_the_system"][f"{key}_{view}"] = {
                k: r[k] for k in ("contexts", "errors", "auroc_a", "auroc_b", "a_minus_b", "ci95", "share_above_zero")
            }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for k, r in result["paired_against_the_system"].items():
        print(k, r["auroc_a"], r["auroc_b"], r["a_minus_b"], r["ci95"])


if __name__ == "__main__":
    main()
