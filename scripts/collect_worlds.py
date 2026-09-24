"""Dialogues in new worlds written by Opus (configs/prompts/world_writer_v1.md), rendered into the competition format.

Each line of a batch's opus.jsonl is one invented dialogue: a policy, OpenAI tool definitions, the
messages before the turn under review, and that turn twice, correct and with one hallucination. Both
versions are appended to the messages and rendered with the layout of the data (guardian.render), so
the prompt is identical and only the last agent turn differs: the correct one becomes a row with
label 0, the other a row with label 1. A line that does not parse or render is counted and skipped.

    python scripts/collect_worlds.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from guardian.env import setup
from guardian.render import render_system, render_tools, render_trajectory

ROOT = Path(__file__).resolve().parents[1]


def as_message(turn: dict) -> dict:
    calls = []
    for c in turn.get("tool_calls") or []:
        f = c.get("function", c)
        args = f.get("arguments", {})
        calls.append(
            {
                "function": {
                    "name": f["name"],
                    "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
                }
            }
        )
    return {"role": "assistant", "content": turn.get("content") or "", "tool_calls": calls}


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data/synthetic/worlds")
    ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic/worlds/worlds_rows.parquet")
    args = ap.parse_args()

    rows, skipped = [], Counter()
    for answers in sorted(args.root.glob("*/opus.jsonl")):
        batch = answers.parent.name
        for k, line in enumerate(answers.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                system = render_system(d["policy"], render_tools(d["tools"]))
                history = [m for m in d["messages"] if m.get("role") in ("assistant", "user", "tool")]
                pair = {}
                for kind, label in (("correct", 0), ("error", 1)):
                    steps = render_trajectory([*history, as_message(d[kind])], system)
                    pair[kind] = (steps[-1], label)
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
                skipped[type(error).__name__] += 1
                continue
            if (
                pair["correct"][0].prompt != pair["error"][0].prompt
                or pair["correct"][0].response == pair["error"][0].response
            ):
                skipped["versions_do_not_differ_in_the_last_turn_only"] += 1
                continue
            for kind, (step, label) in pair.items():
                rows.append(
                    {
                        "id": f"world__{batch}__{k}__{kind}",
                        "prompt": step.prompt,
                        "response": step.response,
                        "label": label,
                        "world": d.get("world", ""),
                        "error_type": d.get("error_type", "") if label else "",
                        "explanation": d.get("explanation", "") if label else "",
                        "group": f"{batch}:{k}",
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(args.out, index=False)
    print(len(frame), "rows from", frame.group.nunique() if len(frame) else 0, "dialogues")
    if len(frame):
        print(frame[frame.label == 1].error_type.value_counts().to_dict())
        print(frame.world.value_counts().to_dict())
    print("skipped:", dict(skipped))


if __name__ == "__main__":
    main()
