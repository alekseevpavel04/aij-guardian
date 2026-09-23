"""A random sample of accepted minimal pairs for a person to read, with a sheet to fill in.

Only pairs used for training are drawn (no copies of the agent's turn, no correct turn the blind
checker rejected), one pair per context, stratified by domain. Each pair shows the end of the
visible history, the correct turn, the error turn with the declared edit and both explanations.
The sheet has one line per pair for the verdict, so the error rate can be counted with its
denominator. The file stays in data/ (organiser dialogues).

    python scripts/review_sample.py --n 30
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
TAIL = 2500


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic/review/pairs_sample.md")
    args = ap.parse_args()

    frame = pd.concat(
        [pd.read_parquet(f) for f in (ROOT / "data/synthetic/pairs").glob("*_0[23]*/pairs.parquet")], ignore_index=True
    )
    rejected = {
        r["id"] for r in journal.read(ROOT / "data/interim/labels/opus_verify_pairs.jsonl") if r["score"] >= 0.5
    }
    copies = set(frame.id[frame["flags"].map(lambda f: "same_as_original" in list(f))])
    pairs = []
    for (cid, pair), g in frame.groupby(["context_id", "pair"]):
        correct, error = g[g.kind == "correct"], g[g.kind == "error"]
        if len(correct) == 1 and len(error) == 1 and not {correct.id.iloc[0]} & (rejected | copies):
            pairs.append((cid, pair, correct.iloc[0], error.iloc[0]))
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    by_domain: dict[str, list] = {}
    for p in pairs:
        by_domain.setdefault(p[2].domain, []).append(p)
    chosen, seen = [], set()
    while len(chosen) < args.n and any(by_domain.values()):
        for items in by_domain.values():
            while items and len(chosen) < args.n:
                p = items.pop()
                if p[0] not in seen:
                    seen.add(p[0])
                    chosen.append(p)
                    break

    lines = [
        "# Выборка пар на просмотр",
        "",
        f"{len(chosen)} пар из {len(pairs)} принятых, по одной на контекст, поровну по доменам (seed {args.seed}).",
        "Для каждой пары: верно ли, что правильный ход правильный, а в ошибочном ровно одна настоящая ошибка?",
        "",
        "## Лист",
        "",
        "| № | Пара | Правильный ход верен (да/нет) | Ошибка настоящая (да/нет) | Заметка |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {n} | `{c.id}` | | | |" for n, (_, _, c, _) in enumerate(chosen, 1)]
    for n, (cid, pair, c, e) in enumerate(chosen, 1):
        history = c.prompt[-TAIL:]
        lines += [
            "",
            f"## {n}. {c.domain}, пара {pair}, `{cid}`",
            "",
            f"Конец истории (последние {TAIL} символов):",
            "",
            "```",
            history,
            "```",
            "",
            "Правильный ход:",
            "",
            "```",
            c.response,
            "```",
            "",
            f"Почему верен (автор): {c.explanation}",
            "",
            f"Ошибочный ход ({e.error_type}):",
            "",
            "```",
            e.response,
            "```",
            "",
            f"Правка: `{e.edit_from}` → `{e.edit_to}`" if e.edit_from is not None else "Правка не объявлена (v3).",
            "",
            f"Ошибка (автор): {e.explanation}",
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = pd.Series([c.domain for _, _, c, _ in chosen]).value_counts().to_dict()
    print(json.dumps({"pairs": len(chosen), "accepted": len(pairs), "by_domain": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
