"""Which simulated dialogues go to labelling and the token pass at each data milestone.

The Kaggle sessions gave 4261 turns in one night, more than Opus labels within the budget at once (about 2.8k
tokens a turn), and two thirds of them telecom: airline has only 50 τ² tasks, and a telecom dialogue runs to
about 21 turns against 10 in the organiser's. So the turns go in nested stages of whole dialogues: one seeded
order per agent and domain, stage k takes a longer prefix of it, so every stage holds the one before (the scale
curve compares like with like). All airline dialogues come first; retail and telecom follow in the organiser's
proportion without banking (20 : 47 of turns); the agents alternate, so each stage has them in equal parts.

    python scripts/select_stage_rows.py --tags kq36n kq36 --turns 1300 --stage 1
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHARES = {"retail": 20, "telecom": 47}


def dialogue(frame: pd.DataFrame) -> pd.Series:
    return frame.domain + "|" + frame.task_id.astype(str) + "|" + frame.trial.astype(str)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--turns", type=int, required=True, help="new turns in this stage, all tags together")
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = {t: pd.read_parquet(ROOT / f"data/synthetic/tau2_ru_{t}_rows.parquet") for t in args.tags}
    order: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for tag, frame in frames.items():
        sizes = frame.groupby(dialogue(frame)).size()
        for domain in ("airline", *SHARES):
            ids = sorted(i for i in sizes.index if i.startswith(domain + "|"))
            random.Random(f"{args.seed}|{tag}|{domain}").shuffle(ids)
            order[(tag, domain)] = [(i, int(sizes[i])) for i in ids]

    picked = {t: [] for t in args.tags}
    counts = {d: 0 for d in ("airline", *SHARES)}
    for tag in args.tags:
        for i, n in order[(tag, "airline")]:
            picked[tag].append(i)
            counts["airline"] += n
    cursor = {k: 0 for k in order}
    turn = 0
    while sum(counts.values()) < args.turns:
        domain = min(SHARES, key=lambda d: counts[d] / SHARES[d])
        tags = args.tags[turn % len(args.tags) :] + args.tags[: turn % len(args.tags)]
        tag = next((t for t in tags if cursor[(t, domain)] < len(order[(t, domain)])), None)
        turn += 1
        if tag is None:
            SHARES.pop(domain)
            if not SHARES:
                break
            continue
        i, n = order[(tag, domain)][cursor[(tag, domain)]]
        cursor[(tag, domain)] += 1
        picked[tag].append(i)
        counts[domain] += n

    summary = {"stage": args.stage, "turns_by_domain": counts, "dialogues": {}}
    for tag, frame in frames.items():
        keep = frame[dialogue(frame).isin(set(picked[tag]))]
        out = ROOT / f"data/synthetic/tau2_ru_{tag}_s{args.stage}_rows.parquet"
        keep.to_parquet(out, index=False)
        summary["dialogues"][tag] = {"dialogues": len(picked[tag]), "turns": len(keep), "file": str(out.name)}
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
