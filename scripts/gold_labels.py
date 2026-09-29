"""Labels from the τ² reference instead of a labeller, and how close they come to the organiser's.

A turn with tool calls is labelled 1 when one of the submission's rules fires, or when it makes a writing call (or
a transfer to a human, or a discoverable tool unlock) that the task's expected actions do not contain with the same
arguments; it is labelled 0 when it makes calls and none of that happens. Reading calls are not judged, and a turn
without calls gets no label: the reference says nothing about what an agent may say. Arguments are compared on the
reference's keys, numbers and strings alike as lower-case text, so key order and 1000 against "1000" do not count.

Compared with the organiser on the 46 labelled rows, and with v1s and Opus on the 384 organiser contexts.

    python scripts/gold_labels.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from guardian.env import setup
from guardian.parsing import parse_transcript
from guardian.rules import Context, evidence
from guardian.tau2_tasks import tasks, unique_task

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paired_variants import views  # noqa: E402
from turn_ruler import RULES  # noqa: E402

JUDGED = {"transfer_to_human_agents", "unlock_discoverable_agent_tool", "unlock_discoverable_user_tool"}


def writing_tools() -> dict[str, set[str]]:
    domains = json.loads((ROOT / "results/tau2_reference.json").read_text(encoding="utf-8"))["domains"]
    return {d: set(v["agent_write_tools"]) | JUDGED for d, v in domains.items()}


def canonical(value: object) -> object:
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, list):
        return sorted((canonical(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, str):
        try:  # call_discoverable_agent_tool passes its arguments as a JSON string
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return canonical(parsed)
        except ValueError:
            pass
        return value.strip().lower()
    return str(value).lower()


def matches(call_args: dict | None, expected: dict) -> bool:
    got = canonical(call_args or {})
    return all(v is None or got.get(k) == canonical(v) for k, v in expected.items())


def gold_label(example_id: str, prompt: str, response: str, writing: dict[str, set[str]]) -> tuple[int | None, list]:
    return label_against(unique_task(example_id), example_id.split("__")[0], prompt, response, writing)


def simulation_task(domain: str, task_id: str) -> dict | None:
    """A simulation keeps the full τ² task id, so its reference is never ambiguous."""
    return next((t for t in tasks(domain) if str(t["id"]) == str(task_id)), None)


def label_against(
    task: dict | None, domain: str, prompt: str, response: str, writing: dict[str, set[str]]
) -> tuple[int | None, list]:
    calls = [c for t in parse_transcript(response) for c in t.calls]
    if task is None or not calls:
        return None, []
    expected = (task.get("evaluation_criteria") or {}).get("actions") or []
    found = evidence(prompt, response, Context.from_prompt(prompt))
    reasons = [f"rule {k}" for k in RULES if found[k]]
    for c in calls:
        if c.name not in writing[domain]:
            continue
        same = [a for a in expected if a["name"] == c.name]
        if not same:
            reasons.append(f"{c.name} not in the reference")
        elif not any(matches(c.args, a["arguments"]) for a in same):
            reasons.append(f"{c.name} with other arguments")
    return int(bool(reasons)), reasons


def agreement(gold: pd.Series, other: pd.Series) -> dict:
    both = gold.notna() & other.notna()
    g, o = gold[both].astype(int), other[both].astype(int)
    return {
        "rows": int(both.sum()),
        "same": int((g == o).sum()),
        "gold_1_other_1": int(((g == 1) & (o == 1)).sum()),
        "gold_1_other_0": int(((g == 1) & (o == 0)).sum()),
        "gold_0_other_1": int(((g == 0) & (o == 1)).sum()),
        "gold_0_other_0": int(((g == 0) & (o == 0)).sum()),
    }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/gold_labels.json")
    args = ap.parse_args()
    writing = writing_tools()

    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet")
    labelled = [gold_label(r.id, r.prompt, r.response, writing) for r in valid.itertuples()]
    valid["gold"] = [g for g, _ in labelled]
    organiser_rows = [
        {"id": r.id, "organiser": int(r.label), "gold": None if pd.isna(r.gold) else int(r.gold), "reasons": reasons}
        for r, (_, reasons) in zip(valid.itertuples(), labelled, strict=True)
    ]

    contexts = pd.read_parquet(ROOT / "data/synthetic/organizer_contexts.parquet")
    contexts = contexts[contexts.split.isin(["train", "check"])].copy()
    labelled = [gold_label(r.id.removeprefix("org__"), r.prompt, r.response, writing) for r in contexts.itertuples()]
    contexts["gold"] = [g for g, _ in labelled]
    v = views()
    contexts["v1s"], contexts["opus"] = contexts.id.map(v["v1s"]), contexts.id.map(v["opus"])
    calls = contexts.response.str.contains("TOOL_CALL")

    result = {
        "organiser_46": {
            "rows_with_a_gold_label": int(valid.gold.notna().sum()),
            "vs_organiser": agreement(valid.gold, valid.label),
            "rows": organiser_rows,
        },
        "contexts": {
            "rows": len(contexts),
            "turns_with_calls": int(calls.sum()),
            "rows_with_a_gold_label": int(contexts.gold.notna().sum()),
            "gold_errors": int(contexts.gold.sum()),
            "vs_v1s": agreement(contexts.gold, contexts.v1s),
            "vs_opus": agreement(contexts.gold, contexts.opus),
            "v1s_errors_on_call_turns": int(contexts.v1s[calls].sum()),
            "opus_errors_on_call_turns": int(contexts.opus[calls].sum()),
        },
        # how often each labelling calls a turn an error, beside the test's share (130 of 548, from the leaderboard)
        "label_rates": {
            "test_errors": [130, 548],
            "organiser_46": {
                "errors": int(valid.label.sum()),
                "errors_on_call_turns": int(
                    (valid.label.astype(bool) & valid.response.str.contains("TOOL_CALL")).sum()
                ),
            },
            **{
                name: {
                    "errors": int(y.sum()),
                    "contexts": len(y),
                    "share": round(float(y.mean()), 3),
                    "errors_on_call_turns": int(y[calls].sum()),
                }
                for name, y in (
                    ("v1s", contexts.v1s.astype(int)),
                    ("opus", contexts.opus.astype(int)),
                    ("either", (contexts.v1s.astype(int) | contexts.opus.astype(int))),
                )
            },
        },
    }
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "rows" or isinstance(vv, int)} for k, v in result.items()},
            indent=1,
            ensure_ascii=False,
        )
    )
    for r in organiser_rows:
        if r["organiser"] != r["gold"]:
            print(r["id"], "organiser", r["organiser"], "gold", r["gold"], r["reasons"])


if __name__ == "__main__":
    main()
