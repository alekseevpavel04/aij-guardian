"""Turn a subagent's minimal pairs into rows, with automatic checks on every written turn.

The subagent answers with the text and the calls of a turn; the turn itself is rendered here in
the exact layout of the data (header, text, calls with sorted-key JSON), so it cannot drift from
the format. Each written turn is checked, and the flags are kept with the row rather than silently
dropping it, so a person can look at what was caught:

- `unknown_tool` / `schema`: a tool or argument outside the schema, unless that is the planted error;
- `value_not_in_context`: an argument value absent from the visible history, which for any other
  planted error than an invented value means the writer looked past the cut;
- `same_as_original`: the written turn repeats the agent's own turn;
- `rules_fire_on_correct`: a deterministic check fires on a turn written as correct;
- `edit_mismatch`: the error turn is not its pair's correct turn with the declared fragment replaced
  (only for writers that declare the edit; `edit_ratio` keeps how similar the two turns are anyway).

    python scripts/collect_pairs.py --batch airline_01
    python scripts/collect_pairs.py --batch banking_02a --answers opus.jsonl
"""

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup
from guardian.parsing import parse_available_tools, parse_transcript, system_turn, unknown_roles
from guardian.predictor import DEFAULT_CHECKS
from guardian.rules import Context, evidence, occurs_in_some_form

ROOT = Path(__file__).resolve().parents[1]
ERROR_TYPES = {
    "policy_violation",
    "wrong_calculation",
    "contradicts_tool_result",
    "premature_escalation",
    "claimed_unperformed_action",
    "fabricated_value",
    "stale_request",
    "repeated_failed_call",
    "arg_schema_violation",
    "unavailable_tool",
    "invalid_modification",
}
AUTHORS = {
    "sonnet.jsonl": "claude-sonnet",
    "opus.jsonl": "claude-opus",
}


def render_turn(turn: int, text: str, calls: list[dict]) -> str:
    lines = [f"⟦ASSISTANT · ход {turn}⟧"]
    if text.strip():
        lines.append(text.strip())
    for call in calls:
        arguments = json.dumps(call.get("arguments") or {}, ensure_ascii=False, sort_keys=True)
        lines.append(f"\t→ TOOL_CALL {call['name']}: {arguments}")
    return "\n".join(lines)


# arguments the agent composes itself (a handover summary, an arithmetic expression, a search query or
# pattern), not values it copies
FREE_TEXT = {"summary", "expression", "query", "pattern"}


def scalar_values(value: object) -> list[object]:
    if isinstance(value, dict):
        return [v for key, item in value.items() if key not in FREE_TEXT for v in scalar_values(item)]
    if isinstance(value, list):
        return [v for item in value for v in scalar_values(item)]
    return [value]


def checks(item: dict, context: str, response: str, original: str, tools: dict) -> list[str]:
    flags = []
    error_type = item.get("error_type")
    for call in item.get("calls") or []:
        spec = tools.get(call.get("name"))
        if spec is None:
            if error_type != "unavailable_tool":
                flags.append("unknown_tool")
            continue
        arguments = call.get("arguments") or {}
        if (set(arguments) - spec.top_level or spec.required - set(arguments)) and error_type != "arg_schema_violation":
            flags.append("schema")
        for value in scalar_values(arguments):
            if isinstance(value, bool) or value is None or (isinstance(value, str) and len(value.strip()) < 3):
                continue
            if not occurs_in_some_form(value, context) and error_type not in ("fabricated_value", "wrong_calculation"):
                flags.append(f"value_not_in_context:{value}")
    if response.strip() == original.strip():
        flags.append("same_as_original")
    if unknown_roles(parse_transcript(response)):
        flags.append("unparsed")
    if item["kind"] == "correct":
        found = evidence(context, response, Context.from_prompt(context))
        fired = [name for name in DEFAULT_CHECKS if name != "repeated_call" and found[name]]
        if fired:
            flags.append("rules_fire_on_correct:" + ",".join(fired))
    return flags


def replaced_once(correct: str, old: str, new: str, error: str) -> bool:
    start = correct.find(old)
    while start >= 0:
        if correct[:start] + new + correct[start + len(old) :] == error:
            return True
        start = correct.find(old, start + 1)
    return False


def twin_checks(frame: pd.DataFrame) -> None:
    frame["edit_ratio"] = None
    for _, group in frame.groupby(["context_id", "pair"]):
        correct, error = group[group.kind == "correct"], group[group.kind == "error"]
        if len(correct) != 1 or len(error) != 1:
            for i in group.index:
                frame.at[i, "flags"] = [*frame.at[i, "flags"], "no_twin"]
            continue
        c, e = correct.iloc[0], error.iloc[0]
        frame.at[e.name, "edit_ratio"] = round(SequenceMatcher(None, c.response, e.response, autojunk=False).ratio(), 3)
        old, new = e.edit_from, e.edit_to
        if old is None and new is None:
            continue
        if not old or not replaced_once(c.response, old, new or "", e.response):
            frame.at[e.name, "flags"] = [*frame.at[e.name, "flags"], "edit_mismatch" if old else "edit_unchecked"]


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--pairs", type=Path, default=ROOT / "data/synthetic/pairs")
    ap.add_argument("--answers", default="sonnet.jsonl", choices=sorted(AUTHORS))
    args = ap.parse_args()

    folder = args.pairs / args.batch
    manifest = {m["context_id"]: m for m in json.loads((folder / "manifest.json").read_text(encoding="utf-8"))}
    contexts = pd.read_parquet(args.contexts).set_index("id")
    items = journal.read(folder / args.answers)
    rows, naturals, problems = [], [], []
    for n, item in enumerate(items):
        cid = item.get("context_id")
        if cid not in manifest or item.get("kind") not in ("natural", "correct", "error"):
            problems.append(f"line {n + 1}: unknown context or kind")
            continue
        ctx = contexts.loc[cid]
        if item["kind"] == "natural":
            naturals.append(
                {"id": cid, "sonnet_label": item.get("label"), "sonnet_explanation": item.get("explanation")}
            )
            continue
        if item["kind"] == "error" and item.get("error_type") not in ERROR_TYPES:
            problems.append(f"line {n + 1}: error_type {item.get('error_type')!r}")
        tools = parse_available_tools(system_turn(parse_transcript(ctx.prompt)).text)
        response = render_turn(manifest[cid]["turn"], item.get("text") or "", item.get("calls") or [])
        rows.append(
            {
                "id": f"pair__{args.batch}__{cid.removeprefix('org__')}__{n}",
                "context_id": cid,
                "prompt": ctx.prompt,
                "response": response,
                "label": int(item["kind"] == "error"),
                "kind": item["kind"],
                "pair": item.get("pair"),
                "error_type": item.get("error_type") if item["kind"] == "error" else None,
                "difficulty": item.get("difficulty"),
                "explanation": item.get("explanation"),
                "edit_from": (item.get("edit") or {}).get("from"),
                "edit_to": (item.get("edit") or {}).get("to"),
                "flags": checks(item, ctx.prompt, response, ctx.response, tools),
                "domain": ctx.domain,
                "task": ctx.task,
                "split": ctx.split,
                "batch": args.batch,
                "author": AUTHORS[args.answers],
            }
        )
    per_context = Counter(item.get("context_id") for item in items)
    # writers up to v3 also label the original turn, one line more per context
    expected = 5 if naturals else 4
    problems += [
        f"{cid}: {per_context[cid]} lines, expected {expected}" for cid in manifest if per_context[cid] != expected
    ]
    frame = pd.DataFrame(rows)
    twin_checks(frame)
    frame.to_parquet(folder / "pairs.parquet", index=False)
    pd.DataFrame(naturals).to_parquet(folder / "natural_labels.parquet", index=False)
    flagged = frame[frame["flags"].map(len) > 0] if len(frame) else frame
    print(f"{len(items)} lines: {len(frame)} written turns, {len(naturals)} natural labels, {len(problems)} bad lines")
    for p in problems:
        print("  ", p)
    print(frame.groupby(["kind", "error_type"], dropna=False).size().to_string())
    print(f"{len(flagged)} written turns flagged:")
    for r in flagged.itertuples():
        print(f"   {r.id} [{r.kind}/{r.error_type}] {r.flags}")


if __name__ == "__main__":
    main()
