"""Every agent turn of the stored Russian simulations as a competition row, with its task reference.

Reads data/synthetic/tau2_ru/<domain>/*.json (written by simulate_tau2.py) and writes
data/synthetic/pseudo_rows.parquet. Deterministic and cheap: rerun it whenever simulations are added.
The reference is the simulation's own task, known exactly, unlike the cut ids of the labelled file.

    python scripts/build_pseudo_rows.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from guardian.env import setup
from guardian.tau2_sim import organiser_systems, steps
from guardian.tau2_tasks import reference_text, tasks

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulations", type=Path, default=ROOT / "data/synthetic/tau2_ru")
    ap.add_argument("--valid", type=Path, default=ROOT / "data/raw/valid.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic/pseudo_rows.parquet")
    ap.add_argument("--tag", default="sim", help="id prefix, one per agent model, so two agents never share an id")
    args = ap.parse_args()

    systems = organiser_systems(args.valid)
    rows = []
    for path in sorted(args.simulations.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        domain, task_id = record["domain"], record["task_id"]
        task = next(t for t in tasks(domain) if t["id"] == task_id)
        reward = (record["simulation"].get("reward_info") or {}).get("reward")
        for step in steps(record, systems):
            rows.append(
                {
                    "id": f"{args.tag}__{domain}__{path.stem}::t{step.turn}",
                    "agent_model": record["model"],
                    "prompt": step.prompt,
                    "response": step.response,
                    "domain": domain,
                    "task_id": task_id,
                    "trial": record["trial"],
                    "turn": step.turn,
                    "simulation_reward": reward,
                    "reference": reference_text(task),
                }
            )
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    if len(frame):
        print(frame.groupby("domain").agg(rows=("id", "size"), simulations=("task_id", "nunique")))
    print(f"{len(frame)} rows from {len(list(args.simulations.glob('*/*.json')))} simulations -> {args.out}")


if __name__ == "__main__":
    main()
