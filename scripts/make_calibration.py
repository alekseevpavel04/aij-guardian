"""A calibration set and a per-row threshold for a head archive (DECISIONS.md, 33 and 35).

The head decides each test row on its own by comparing its score with a threshold fixed at home. The
platform's engine (FP8, vLLM) shifts the scores against the home one (Q4, llama.cpp), so the archive
carries calibration rows: organiser contexts with their home scores. On the platform the same rows go
through the platform's engine first, and guardian.head.to_home_scale maps every test score onto the
home scale by matching quantiles. The rows come from the validation half, which the heads of the B
kind never trained on, so their home scores are what the head gives to unseen rows; they are spread
over the dialogues round-robin. The threshold is the home score above which `--flag-share` of the
validation contexts fall.

The texts are the organiser's data and stay in data/ (not in git); the ids, the threshold and the
share of each kind of context above it go to results/.

    python scripts/make_calibration.py --head configs/heads/head_b_bag10.json --name cal_head_b
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from guardian.env import setup
from guardian.head import Head

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_head_ru import TEACHERS, labels, vectors  # noqa: E402


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--flag-share", type=float, default=0.31)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    head = Head.load(args.head)
    vecs = vectors("q4_last_org")
    check = pd.read_parquet(args.contexts).query("split == 'check'").copy()
    check["home_score"] = [head.score(vecs[i].tolist()) for i in check.id]
    threshold = float(np.quantile(check.home_score, 1 - args.flag_share))

    rng = random.Random(args.seed)
    by_task = {t: list(g.index) for t, g in check.groupby("task")}
    for rows in by_task.values():
        rng.shuffle(rows)
    order = sorted(by_task)
    rng.shuffle(order)
    chosen: list[int] = []
    while len(chosen) < args.n and any(by_task.values()):
        for task in order:
            if by_task[task] and len(chosen) < args.n:
                chosen.append(by_task[task].pop())
    rows = check.loc[chosen]
    out = ROOT / "data/interim/calibration" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            [
                {"id": r.id, "prompt": r.prompt, "response": r.response, "home_score": r.home_score}
                for r in rows.itertuples()
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    opus = check.id.map(labels(TEACHERS["opus"].format(split="check"))).to_numpy().astype(int)
    v1s = check.id.map(labels(TEACHERS["v1s"].format(split="check"))).to_numpy().astype(int)
    flagged = (check.home_score >= threshold).to_numpy()
    summary = {
        "head": args.head.as_posix(),
        "calibration_rows": len(rows),
        "calibration_ids": list(rows.id),
        "dialogues": int(rows.task.nunique()),
        "threshold_home_scale": threshold,
        "flag_share_target": args.flag_share,
        "validation_contexts": len(check),
        "flagged_validation_contexts": int(flagged.sum()),
        "flagged_among_positives_v1s": f"{int((flagged & (v1s == 1)).sum())} of {int(v1s.sum())}",
        "flagged_among_positives_opus": f"{int((flagged & (opus == 1)).sum())} of {int(opus.sum())}",
        "calibration_rows_above_threshold": int((rows.home_score >= threshold).sum()),
    }
    (ROOT / "results" / f"{args.name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "calibration_ids"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
