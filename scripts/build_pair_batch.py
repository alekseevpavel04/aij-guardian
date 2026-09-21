"""A batch of organiser contexts for a Sonnet subagent that writes minimal pairs.

To keep the subagent cheap, the domain's ⟦SYSTEM⟧ block (policy and tool schemas) goes to one file
that is read once, and each dialogue is shown once, up to the latest chosen turn: an earlier cut of
the same dialogue is marked by its turn number, and the subagent is told that only the history
before that turn is visible there. A manifest lists the contexts the answers must refer to.

    python scripts/build_pair_batch.py --domain airline --batch airline_01 --tasks 5 --per-task 2
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch", required=True)
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic/pairs")
    args = ap.parse_args()

    frame = pd.read_parquet(args.contexts)
    frame = frame[(frame.domain == args.domain) & (frame.split == args.split) & (frame.turn >= 2)]
    rng = random.Random(args.seed)
    tasks = sorted(frame.task.unique())
    rng.shuffle(tasks)
    out = args.out / args.batch
    out.mkdir(parents=True, exist_ok=True)

    system = frame.prompt.iloc[0][: frame.prompt.iloc[0].find("⟦ASSISTANT⟧")].rstrip()
    (out / "system.md").write_text(system + "\n", encoding="utf-8")

    parts, manifest = [], []
    for task in tasks[: args.tasks]:
        rows = frame[frame.task == task].sort_values("turn")
        # an early and a late turn, so both short and long histories are covered
        picks = sorted(rng.sample(list(rows.turn), min(args.per_task, len(rows))))
        last = rows[rows.turn == picks[-1]].iloc[0]
        dialog = last.prompt[last.prompt.find("⟦ASSISTANT⟧") :]
        parts.append(f"# Диалог {task}\n\n## Эталон задачи (агент его не видел)\n\n{last.reference or 'нет'}\n")
        parts.append(f"## История (до хода {picks[-1]})\n\n```\n{dialog}\n```\n")
        for turn in picks:
            row = rows[rows.turn == turn].iloc[0]
            manifest.append({"context_id": row.id, "task": task, "turn": int(turn)})
            parts.append(
                f"## Контекст `{row.id}`\n\nПроверяемый ход: ход {turn}. Агенту видна только история до маркера "
                f"`⟦ASSISTANT · ход {turn}⟧`, всё после него для этого контекста не существует.\n\n"
                f"Исходный ход агента (его метка неизвестна):\n\n```\n{row.response}\n```\n"
            )
    (out / "batch.md").write_text("\n".join(parts), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    size = (out / "batch.md").stat().st_size + (out / "system.md").stat().st_size
    print(f"{len(manifest)} contexts from {min(args.tasks, len(tasks))} dialogues, {size / 1024:.0f} KiB -> {out}")


if __name__ == "__main__":
    main()
