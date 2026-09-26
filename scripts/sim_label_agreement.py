"""Agreement of the local v1s labeller and Opus on the simulation turns both labelled.

python scripts/sim_label_agreement.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from guardian import journal

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1s", default="labeler_v1s_strong")
    ap.add_argument(
        "--opus-dirs", nargs="+", default=["opus_labels_tau2q36", "opus_labels_tau2glm", "opus_labels_tau2q36n"]
    )
    ap.add_argument("--out", type=Path, default=ROOT / "results/sim_label_agreement.json")
    args = ap.parse_args()

    v1s = {r["id"]: int(r["score"] >= 0.5) for r in journal.read(ROOT / "data/interim/labels" / f"{args.v1s}.jsonl")}
    opus = {
        r["id"]: int(r["label"])
        for d in args.opus_dirs
        for f in (ROOT / "data/synthetic" / d).glob("*/opus.jsonl")
        for r in journal.read(f)
    }
    common = sorted(i for i in v1s if i in opus)
    per = defaultdict(lambda: {"turns": 0, "agree": 0, "v1s_errors": 0, "opus_errors": 0})
    for i in common:
        p = per[i.split("__")[0]]
        p["turns"] += 1
        p["agree"] += v1s[i] == opus[i]
        p["v1s_errors"] += v1s[i]
        p["opus_errors"] += opus[i]
    result = {
        "turns": len(common),
        "agree": sum(v1s[i] == opus[i] for i in common),
        "v1s_1_opus_0": sum(v1s[i] == 1 and opus[i] == 0 for i in common),
        "v1s_0_opus_1": sum(v1s[i] == 0 and opus[i] == 1 for i in common),
        "per_agent": dict(per),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
