"""Batches of written turns for a blind check by a second Claude subagent.

The check is aimed at turns written as correct: a wrong one would enter training as a hallucination
labelled 0, which no code check can catch. The checker labels every turn with the labeller's rules
(labeler_v4.md), without the intended label, the writer's explanation or the error twin of the turn,
since a turn read next to its broken copy looks right by contrast. As a control of the checker's
attention the batch also carries the error turns whose correct twin is a copy of the agent's own turn
(such copies are not checked), so the rate at which the checker finds a planted error is measured.

Turns are grouped by dialogue over all the given pair batches, so a history is read once; the order
of turns inside a dialogue is shuffled so that kinds do not come in a pattern.

    python scripts/build_verify_batch.py --pairs airline_02 retail_02a ... --prefix v1
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from guardian.env import setup
from guardian.labeller import without_one_call_rule

ROOT = Path(__file__).resolve().parents[1]
MARKER = "⟦ASSISTANT⟧"


def history(prompt: str) -> str:
    return prompt[prompt.find(MARKER) :]


def latest(rows: pd.DataFrame, contexts: pd.DataFrame) -> pd.Series:
    return contexts.loc[rows.context_id].sort_values("turn").iloc[-1]


def candidates(frame: pd.DataFrame) -> pd.DataFrame:
    copy = frame["flags"].map(lambda f: "same_as_original" in list(f))
    copied_pairs = set(zip(frame.context_id[copy], frame.pair[copy], strict=True))
    control = (frame.kind == "error") & [
        (c, p) in copied_pairs for c, p in zip(frame.context_id, frame.pair, strict=True)
    ]
    frame = frame.assign(role=None)
    frame.loc[(frame.kind == "correct") & ~copy, "role"] = "check"
    frame.loc[control, "role"] = "control"
    return frame.dropna(subset=["role"])


def write_batch(folder: Path, system: str, dialogues: list[pd.DataFrame], contexts: pd.DataFrame, args) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "labeller.md").write_text(args.labeller.read_text(encoding="utf-8"), encoding="utf-8")
    (folder / "system.md").write_text(system + "\n", encoding="utf-8")
    parts, manifest = [], []
    for rows in dialogues:
        last = latest(rows, contexts)
        parts.append(f"# Диалог {last.task}\n\n## Эталон задачи (агент его не видел)\n\n{last.reference or 'нет'}\n")
        parts.append(f"## История (до хода {last.turn})\n\n```\n{history(last.prompt)}\n```\n")
        for row in rows.itertuples():
            turn = int(contexts.at[row.context_id, "turn"])
            manifest.append({"id": row.id, "role": row.role})
            parts.append(
                f"## Ход `{row.id}`\n\nРазмечаемый ход: ход {turn}. Видна только история до маркера "
                f"`⟦ASSISTANT · ход {turn}⟧`; исходный ход {turn} и всё после него здесь не существуют.\n\n"
                f"```\n{row.response}\n```\n"
            )
    (folder / "batch.md").write_text("\n".join(parts), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    instruction = args.instruction.read_text(encoding="utf-8")
    (folder / "prompt.md").write_text(instruction.format(folder=folder.resolve(), n=len(manifest)), encoding="utf-8")
    return len(manifest)


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--pairs", nargs="+", required=True, help="pair batch names")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--max-kib", type=int, default=200)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labeller", type=Path, default=ROOT / "configs/prompts/labeler_v4.md")
    ap.add_argument("--instruction", type=Path, default=ROOT / "configs/prompts/turn_verifier_v1.md")
    ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic/verify")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts).set_index("id")
    written = pd.concat([pd.read_parquet(ROOT / "data/synthetic/pairs" / b / "pairs.parquet") for b in args.pairs])
    frame = candidates(written.reset_index(drop=True))
    rng = random.Random(args.seed)
    total = 0
    for domain, rows in frame.groupby("domain"):
        first = contexts.loc[rows.context_id.iloc[0]]
        system = without_one_call_rule(first.prompt[: first.prompt.find(MARKER)].rstrip())
        dialogues = []
        for _, d in rows.groupby("task"):
            d = d.sample(frac=1, random_state=rng.randrange(2**32))
            for i in range(0, len(d), args.max_turns):
                dialogues.append(d.iloc[i : i + args.max_turns])
        size_of = {id(d): len(history(latest(d, contexts).prompt).encode()) / 1024 for d in dialogues}
        dialogues.sort(key=lambda d: -size_of[id(d)])
        batches: list[list[pd.DataFrame]] = []
        size = 0.0
        for d in dialogues:
            full = batches and sum(map(len, batches[-1])) + len(d) > args.max_turns
            if not batches or size + size_of[id(d)] > args.max_kib or full:
                batches.append([])
                size = 0.0
            batches[-1].append(d)
            size += size_of[id(d)]
        for n, group in enumerate(batches, 1):
            name = f"{args.prefix}_{domain}_{n}"
            count = write_batch(args.out / name, system, group, contexts, args)
            kib = sum((args.out / name / f).stat().st_size for f in ("batch.md", "system.md", "labeller.md")) / 1024
            print(f"{name}: {count} turns from {len(group)} dialogues, {kib:.0f} KiB")
            total += count
    print(f"{total} turns, {int((frame.role == 'control').sum())} of them controls")


if __name__ == "__main__":
    main()
