"""Gain against the amount of simulation rows: the candidate with a random share of the simulated dialogues.

Runs tau2ru_ruler.py --quick (both heads, weight 1) for each share and seed and collects, per point, the
rows and errors used, the old ruler's difference to the candidate (AUROC on the agreed contexts, paired
bootstrap), both heads alone, and the new ruler by the independent v1s labels (AUROC and F1 at the test's
share on all contexts). The same --source arguments as tau2ru_ruler.py.

    python scripts/sim_scale_curve.py --tag q36 --source "rows|prefix|opus_dir" --extra --unc-skip glm
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed to tau2ru_ruler.py (keep it last)")
    args = ap.parse_args()

    out = ROOT / f"results/sim_scale_curve_{args.tag}.json"
    tmp = ROOT / "data/interim/scale_curve"
    tmp.mkdir(parents=True, exist_ok=True)
    points = []
    for fraction in args.fractions:
        for seed in range(1 if fraction == 1.0 else args.seeds):
            path = tmp / f"{args.tag}_{fraction}_{seed}.json"
            if not path.exists():
                command = [sys.executable, "scripts/tau2ru_ruler.py", "--labels", "opus", "--quick", "--out", str(path)]
                command += ["--fraction", str(fraction), "--seed", str(seed)]
                for source in args.source:
                    command += ["--source", source]
                command += args.extra
                subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
            r = json.loads(path.read_text(encoding="utf-8"))
            v = r["variants"]["both_w1.0"]
            points.append(
                {
                    "fraction": fraction,
                    "seed": seed,
                    "rows": r["rows"],
                    "errors": r["positives"],
                    "old_ruler_system_minus_candidate": v["system_minus_reference"]["difference"],
                    "old_ruler_ci95": v["system_minus_reference"]["ci95"],
                    "head_b_alone": v["head_b_alone"],
                    "uncertainty_alone": v["uncertainty_alone"],
                    "v1s_auroc_all_contexts": v["new_ruler_all_contexts"]["v1s"]["auroc"],
                    "v1s_f1_minus_candidate": v["new_ruler_all_contexts"]["v1s"]["f1_minus_reference"],
                }
            )
            print(json.dumps(points[-1]), flush=True)
            out.write_text(json.dumps({"sources": args.source, "points": points}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
