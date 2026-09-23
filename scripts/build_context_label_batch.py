"""Batches of organiser contexts for a Claude subagent acting as the privileged labeller.

A context is a past agent turn of one of the organiser's dialogues (scripts/extract_contexts.py), so
all contexts of a dialogue share its history. Each dialogue is shown once, up to its latest context,
and every context of it is labelled against the history before its own turn marker; this is what
makes labelling a whole split affordable. The subagent gets what the local labeller gets
(labeler_v4.md, the τ² task reference, the system block without the one-call rule).

Contexts that already have a label from a Claude subagent writing pairs with them (`opus.jsonl`
of a pair batch) are skipped. A batch holds dialogues of one domain up to `--max-kib` of history
and `--max-contexts` contexts; a dialogue with more contexts is cut into runs of consecutive turns,
each shown up to its own latest turn, and a longer dialogue gets a batch of its own.

    python scripts/build_context_label_batch.py --split check
    python scripts/build_context_label_batch.py --split train
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup
from guardian.labeller import without_one_call_rule

ROOT = Path(__file__).resolve().parents[1]
MARKER = "⟦ASSISTANT⟧"


def labelled_by_opus(pairs: Path) -> set[str]:
    return {
        item["context_id"]
        for answers in pairs.glob("*/opus.jsonl")
        for item in journal.read(answers)
        if item.get("kind") == "natural"
    }


def history(prompt: str) -> str:
    return prompt[prompt.find(MARKER) :]


def write_batch(folder: Path, system: str, dialogues: list[pd.DataFrame], args: argparse.Namespace) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "labeller.md").write_text(args.labeller.read_text(encoding="utf-8"), encoding="utf-8")
    (folder / "system.md").write_text(system + "\n", encoding="utf-8")
    parts, ids = [], []
    for rows in dialogues:
        last = rows.iloc[-1]
        parts.append(f"# Диалог {last.task}\n\n## Эталон задачи (агент его не видел)\n\n{last.reference or 'нет'}\n")
        parts.append(f"## История (до хода {last.turn})\n\n```\n{history(last.prompt)}\n```\n")
        for row in rows.itertuples():
            ids.append(row.id)
            parts.append(
                f"## Контекст `{row.id}`\n\nРазмечаемый ход: ход {row.turn}. Агенту видна только история до маркера "
                f"`⟦ASSISTANT · ход {row.turn}⟧`, всё после него для этого контекста не существует.\n\n"
                f"```\n{row.response}\n```\n"
            )
    (folder / "batch.md").write_text("\n".join(parts), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    instruction = args.instruction.read_text(encoding="utf-8")
    (folder / "prompt.md").write_text(instruction.format(folder=folder.resolve(), n=len(ids)), encoding="utf-8")
    return len(ids)


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--split", required=True, choices=("train", "check"))
    ap.add_argument("--max-kib", type=int, default=160)
    ap.add_argument("--max-contexts", type=int, default=32)
    ap.add_argument("--labeller", type=Path, default=ROOT / "configs/prompts/labeler_v4.md")
    ap.add_argument("--instruction", type=Path, default=ROOT / "configs/prompts/context_labeller_v1.md")
    ap.add_argument("--pairs", type=Path, default=ROOT / "data/synthetic/pairs")
    ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic/opus_labels")
    args = ap.parse_args()

    frame = pd.read_parquet(args.contexts)
    frame = frame[(frame.split == args.split) & ~frame.id.isin(labelled_by_opus(args.pairs))]
    total = 0
    for domain, rows in frame.groupby("domain"):
        system = without_one_call_rule(rows.prompt.iloc[0][: rows.prompt.iloc[0].find(MARKER)].rstrip())
        dialogues = []
        for _, d in rows.groupby("task"):
            d = d.sort_values("turn")
            pieces = math.ceil(len(d) / args.max_contexts)
            step = math.ceil(len(d) / pieces)
            dialogues += [d.iloc[i : i + step] for i in range(0, len(d), step)]
        dialogues.sort(key=lambda d: -len(history(d.prompt.iloc[-1]).encode()))
        batches: list[list[pd.DataFrame]] = []
        size = 0
        for d in dialogues:
            kib = len(history(d.prompt.iloc[-1]).encode()) / 1024
            full = batches and sum(map(len, batches[-1])) + len(d) > args.max_contexts
            if not batches or size + kib > args.max_kib or full:
                batches.append([])
                size = 0
            batches[-1].append(d)
            size += kib
        for n, group in enumerate(batches, 1):
            name = f"{args.split}_{domain}_{n}"
            count = write_batch(args.out / name, system, group, args)
            kib = sum((args.out / name / f).stat().st_size for f in ("batch.md", "system.md", "labeller.md")) / 1024
            print(f"{name}: {count} contexts from {len(group)} dialogues, {kib:.0f} KiB")
            total += count
    print(f"{total} contexts")


if __name__ == "__main__":
    main()
