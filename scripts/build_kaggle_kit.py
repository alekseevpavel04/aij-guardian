"""The kit for Kaggle simulation sessions (scripts/kaggle_sim.py): code, the agent instruction, task lists.

The task lists hold τ² tasks never simulated here before (in any data/synthetic/tau2_ru*/ folder) and not among
the organiser's tasks (their telecom ids are cut, so every task whose cut id matches is left out too). Each
run gets its own disjoint list, domains interleaved in the organiser's proportions without banking (airline
33 %, retail 20 %, telecom 47 %), so a session stopped early is still balanced. The organiser's system block
is not in the kit; the kit holds the agent instruction found in his data, so the Kaggle dataset must stay private.

    python scripts/build_kaggle_kit.py --runs 3 --per-run 150
"""

import argparse
import json
import random
import zipfile
from pathlib import Path

import pandas as pd

from guardian.env import setup
from guardian.tau2_tasks import CUT, data_id, tasks

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "scripts/kaggle_sim.py",
    "scripts/simulate_tau2.py",
    "src/guardian/__init__.py",
    "src/guardian/journal.py",
    "src/guardian/parsing.py",
    "src/guardian/render.py",
    "src/guardian/tau2_sim.py",
    "configs/agent_instruction_ru.txt",
    "configs/tau2_requirements.txt",
    "configs/organiser_banking_policy.txt",
]
SHARES = {"airline": 0.33, "retail": 0.20, "telecom": 0.47}


def used_tasks() -> set[tuple[str, str]]:
    used = set()
    for folder in (ROOT / "data/synthetic").glob("tau2_ru*"):
        for path in folder.glob("*/*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            used.add((record["domain"], record["task_id"]))
    return used


def listed_in_kit(kit: Path) -> set[tuple[str, str]]:
    """Tasks of the current kit's lists: a session may still be running one of them."""
    if not kit.exists():
        return set()
    with zipfile.ZipFile(kit) as z:
        names = [n for n in z.namelist() if n.startswith("tasks/")]
        return {(d, t) for n in names for d, t in json.loads(z.read(n))}


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--per-run", type=int, default=150)
    ap.add_argument("--out", type=Path, default=ROOT / "data/kaggle/aij-sim-kit.zip")
    args = ap.parse_args()

    organiser = set(pd.read_parquet(ROOT / "data/synthetic/organizer_contexts.parquet").task)
    used = used_tasks() | listed_in_kit(args.out)
    pools = {}
    for domain in SHARES:
        pool = [
            t["id"]
            for t in tasks(domain)
            if (domain, t["id"]) not in used
            and f"{domain}__{data_id(t['id'])}"[: len(domain) + 2 + CUT] not in organiser
        ]
        random.Random(0).shuffle(pool)
        pools[domain] = pool
    lists = []
    for _ in range(args.runs):
        run = []
        want = {d: round(s * args.per_run) for d, s in SHARES.items()}
        # a scarce domain (airline has 50 tasks) is shared evenly between the runs instead of going to the first
        take = {d: min(want[d], len(pools[d]) // (args.runs - len(lists))) for d in SHARES}
        picked = {d: [pools[d].pop() for _ in range(take[d])] for d in SHARES}
        while any(picked.values()):  # interleave in proportion, so an early stop keeps the mix
            for domain in sorted(SHARES, key=lambda d: -len(picked[d]) / max(1, want[d])):
                if picked[domain]:
                    run.append([domain, picked[domain].pop(0)])
                    break
        lists.append(run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(ROOT / name, name)
        for k, run in enumerate(lists, 1):
            z.writestr(f"tasks/tasks_run{k}.json", json.dumps(run, ensure_ascii=False, indent=0))
        # banking_knowledge: its own list and the Russian knowledge base (scripts/kb_translation.py)
        bank = [
            ["banking_knowledge", t["id"]]
            for t in tasks("banking_knowledge")
            if ("banking_knowledge", t["id"]) not in used and f"banking_knowledge__{t['id']}" not in organiser
        ]
        z.writestr("tasks/tasks_bank.json", json.dumps(bank, ensure_ascii=False, indent=0))
        for path in sorted((ROOT / "data/synthetic/banking_kb_ru").glob("*.json")):
            z.write(path, f"data/synthetic/banking_kb_ru/{path.name}")
        # τ²'s cache of the documents' qwen3-embedding-8b embeddings, made at home (kaggle_sim.py explains why)
        for path in sorted((ROOT / "data/interim/kb_cache_build/data/.embeddings_cache").glob("*")):
            z.write(path, f"data/kb_embeddings_cache/{path.name}")
    counts = [{d: sum(1 for x in run if x[0] == d) for d in SHARES} for run in lists]
    print(args.out, "runs:", counts, "left in the pools:", {d: len(p) for d, p in pools.items()}, "banking:", len(bank))


if __name__ == "__main__":
    main()
