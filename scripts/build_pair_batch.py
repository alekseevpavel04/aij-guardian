"""Batches of organiser contexts for a Claude subagent that writes minimal pairs.

To keep the subagent cheap, the domain's ⟦SYSTEM⟧ block (policy and tool schemas) goes to one file
that is read once, and each dialogue is shown once, up to the latest chosen turn: an earlier cut of
the same dialogue is marked by its turn number, and the subagent is told that only the history
before that turn is visible there. A manifest lists the contexts the answers must refer to.

`--count` contexts (default: all) are taken from the domain's split round-robin over its dialogues,
skipping every context that some earlier batch already has. A batch holds whole dialogues up to
`--max-batch` contexts, since a split dialogue's history is read twice; only a dialogue with more
contexts than that is cut into runs of consecutive turns. Several batches get the suffixes a, b, ...
With `--labels` the writer sees each original turn's label and reason from that label journal. With
`--correct-by` only contexts that every given label journal scores 0 are taken, and the writer is told
that the original turn was judged correct (pair_writer_v5 plants one error in the original turn).

    python scripts/build_pair_batch.py --domain airline --batch airline_03 --labels opus_v4_org_train
    # the batches of 23.09 (decision 22) were built with --count and pair_writer_v3.md
"""

import argparse
import json
import math
import random
import string
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]


def used_contexts(root: Path) -> set[str]:
    return {m["context_id"] for f in root.glob("*/manifest.json") for m in json.loads(f.read_text(encoding="utf-8"))}


def pick(frame: pd.DataFrame, count: int | None, skip: set[str], rng: random.Random) -> dict[str, list[int]]:
    free = {task: sorted(rows.turn[~rows.id.isin(skip)]) for task, rows in frame.groupby("task")}
    order = sorted(t for t in free if free[t])
    rng.shuffle(order)
    for turns in free.values():
        rng.shuffle(turns)
    count = count or sum(map(len, free.values()))
    picks: dict[str, list[int]] = {}
    taken = 0
    while taken < count and any(free[t] for t in order):
        for task in order:
            if free[task] and taken < count:
                picks.setdefault(task, []).append(int(free[task].pop()))
                taken += 1
    return {task: sorted(turns) for task, turns in picks.items()}


def write_batch(
    out: Path,
    frame: pd.DataFrame,
    picks: list[tuple[str, list[int]]],
    writer: str,
    known: dict[str, dict],
    args_correct: bool = False,
) -> int:
    out.mkdir(parents=True, exist_ok=True)
    system = frame.prompt.iloc[0][: frame.prompt.iloc[0].find("⟦ASSISTANT⟧")].rstrip()
    (out / "system.md").write_text(system + "\n", encoding="utf-8")
    parts, manifest = [], []
    for task, turns in picks:
        rows = frame[frame.task == task]
        last = rows[rows.turn == turns[-1]].iloc[0]
        dialog = last.prompt[last.prompt.find("⟦ASSISTANT⟧") :]
        parts.append(f"# Диалог {task}\n\n## Эталон задачи (агент его не видел)\n\n{last.reference or 'нет'}\n")
        parts.append(f"## История (до хода {turns[-1]})\n\n```\n{dialog}\n```\n")
        for turn in turns:
            row = rows[rows.turn == turn].iloc[0]
            manifest.append({"context_id": row.id, "task": task, "turn": turn})
            if args_correct:
                heading, reason = "Исходный ход агента (два независимых разметчика считают его правильным)", ""
            elif row.id in known:
                label = known[row.id]
                heading = f"Исходный ход агента, метка независимого разметчика {int(label['score'])}"
                reason = f"\nПричина метки: {label['explanation']}\n"
            else:
                heading, reason = "Исходный ход агента (его метка неизвестна)", ""
            parts.append(
                f"## Контекст `{row.id}`\n\nПроверяемый ход: ход {turn}. Агенту видна только история до маркера "
                f"`⟦ASSISTANT · ход {turn}⟧`, всё после него для этого контекста не существует.\n\n"
                f"{heading}:\n\n```\n{row.response}\n```\n{reason}"
            )
    (out / "batch.md").write_text("\n".join(parts), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # the exact instruction the subagent gets is kept with the batch it wrote
    (out / "prompt.md").write_text(writer.replace("{batch_dir}", str(out.resolve())), encoding="utf-8")
    return len(manifest)


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch", required=True)
    ap.add_argument("--count", type=int, default=None, help="default: every free context of the domain")
    ap.add_argument("--max-batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic/pairs")
    ap.add_argument("--writer-prompt", type=Path, default=ROOT / "configs/prompts/pair_writer_v4.md")
    ap.add_argument("--answers", default="opus.jsonl", help="file the subagent writes into the batch folder")
    ap.add_argument("--labels", default=None, help="label journal whose label and reason the writer is shown")
    ap.add_argument("--correct-by", nargs="+", default=None, help="keep contexts all these journals score 0")
    args = ap.parse_args()

    frame = pd.read_parquet(args.contexts)
    frame = frame[(frame.domain == args.domain) & (frame.split == args.split) & (frame.turn >= 2)]
    if args.correct_by:
        journals = [ROOT / "data/interim/labels" / f"{name}.jsonl" for name in args.correct_by]
        scored = [{r["id"]: r["score"] for r in journal.read(path)} for path in journals]
        frame = frame[[all(s.get(i, 1) < 0.5 for s in scored) for i in frame.id]]
    picks = pick(frame, args.count, used_contexts(args.out), random.Random(args.seed))
    known = (
        {r["id"]: r for r in journal.read(ROOT / "data/interim/labels" / f"{args.labels}.jsonl")} if args.labels else {}
    )
    # a dialogue with more contexts than a batch holds is cut into runs of consecutive turns
    runs = []
    for task, turns in picks.items():
        pieces = math.ceil(len(turns) / args.max_batch)
        step = math.ceil(len(turns) / pieces)
        runs += [(task, turns[i : i + step]) for i in range(0, len(turns), step)]
    groups: list[list[tuple[str, list[int]]]] = [[]]
    for task, turns in sorted(runs, key=lambda run: -len(run[1])):
        if groups[-1] and sum(len(t) for _, t in groups[-1]) + len(turns) > args.max_batch:
            groups.append([])
        groups[-1].append((task, turns))
    writer = args.writer_prompt.read_text(encoding="utf-8")
    writer = writer.replace("{domain}", args.domain).replace("{answers}", args.answers)
    for n, group in enumerate(groups):
        name = args.batch if len(groups) == 1 else f"{args.batch}{string.ascii_lowercase[n]}"
        count = write_batch(args.out / name, frame, group, writer, known, bool(args.correct_by))
        size = sum((args.out / name / f).stat().st_size for f in ("batch.md", "system.md"))
        print(f"{name}: {count} contexts from {len(group)} dialogues, {size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
