"""Measure each deterministic check on the external sets, per part, source and domain.

Only counts are written for ext_val: which rows a check fired on is not stored, so the errors on it
cannot be read by accident. The external labels are broader than the competition
label: a -1 step in AgentProcessBench or a decisive error in TrajErrBench is any wrong step, including
wrong reasoning, so recall here is a lower bound of nothing in particular and precision is the number
that matters (T2.6).

    python scripts/rules_external.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from guardian.env import setup
from guardian.metrics import binary_scores
from guardian.predictor import DEFAULT_CHECKS
from guardian.rules import CHECKS, Context, evidence

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (ROOT / "data/external/apb_tau2.parquet", ROOT / "data/external/trajerr_tau2.parquet")


def scores(y: np.ndarray, fired: np.ndarray) -> dict:
    s = binary_scores(y, fired)
    return {
        "fired": int(fired.sum()),
        "fired_on_positive": s.true_positives,
        "fired_on_negative": int(fired.sum()) - s.true_positives,
        "precision": round(s.precision, 4) if fired.sum() else None,
        "recall": round(s.recall, 4),
    }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, nargs="+", default=list(SOURCES))
    ap.add_argument("--splits", type=Path, default=ROOT / "data/external/splits.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/rules_external.json")
    args = ap.parse_args()

    frame = pd.concat([pd.read_parquet(s) for s in args.sources], ignore_index=True)
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    part = {i: name for name, ids in splits.items() for i in ids}
    frame["part"] = frame.id.map(part)

    started = time.monotonic()
    names = [*CHECKS, "unsupported_number"]
    hits = np.zeros((len(frame), len(names)), dtype=int)
    for i, row in enumerate(frame.itertuples()):
        found = evidence(row.prompt, row.response, Context.from_prompt(row.prompt))
        hits[i] = [len(found[n]) > 0 for n in names]
    elapsed = time.monotonic() - started
    predictor = hits[:, [names.index(n) for n in DEFAULT_CHECKS]].any(axis=1).astype(int)

    report = {"rows": len(frame), "seconds": round(elapsed, 1), "parts": {}}
    for part_name in ("ext_dev", "ext_val"):
        mask = (frame.part == part_name).to_numpy()
        y = frame.weak_label.to_numpy()[mask]
        entry = {
            "rows": int(mask.sum()),
            "positives": int(y.sum()),
            "predictor": scores(y, predictor[mask]),
            "checks": {n: scores(y, hits[mask, j]) for j, n in enumerate(names)},
            "predictor_by_source": {},
            "predictor_by_domain": {},
        }
        for column, key in (("source", "predictor_by_source"), ("domain", "predictor_by_domain")):
            for value in sorted(frame[column].unique()):
                sub = mask & (frame[column] == value).to_numpy()
                entry[key][value] = scores(frame.weak_label.to_numpy()[sub], predictor[sub])
        report["parts"][part_name] = entry

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(frame)} rows in {elapsed:.0f} s")
    for part_name, entry in report["parts"].items():
        print(f"\n{part_name}: {entry['rows']} rows, {entry['positives']} positive; predictor {entry['predictor']}")
        rows = [*entry["checks"].items(), *entry["predictor_by_source"].items(), *entry["predictor_by_domain"].items()]
        for name, m in rows:
            print(f"{name:>28}  fired {m['fired']:>4}  on 1 {m['fired_on_positive']:>4}  precision {m['precision']}")


if __name__ == "__main__":
    main()
