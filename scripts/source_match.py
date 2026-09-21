"""Are the 46 labelled examples cut from the public τ²-bench leaderboard trajectories?

The leaderboard bucket (s3://sierra-tau-bench-public/submissions) holds full trajectories of every
submitted agent, in English. The dialogue in our data is Russian, but tool names, identifiers and
argument values are language-independent, so the ordered sequence of the agent's tool calls with
canonical arguments is a fingerprint that survives the change of language. An example matches a
simulation when the calls in its prompt are the first calls of that simulation; whether the calls
of its response follow is recorded separately, since a response rewritten to contain an error
would not.

A shared prefix alone proves little: on one task most agents start with the same lookups. So for
every example the script also records the longest prefix of its calls that any simulation of the
same task shares. A copied example shares all of them; one generated independently diverges.

    python scripts/source_match.py --trajectories data/external/tau2_leaderboard/submissions
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from guardian.env import setup
from guardian.parsing import parse_transcript

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")


def canonical(name: str, args: object) -> tuple[str, str]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return name, args
    return name, json.dumps(args, ensure_ascii=False, sort_keys=True)


def our_calls(text: str) -> list[tuple[str, str]]:
    return [
        canonical(c.name, c.args if c.args is not None else c.args_raw)
        for t in parse_transcript(text)
        if t.role == "ASSISTANT"
        for c in t.calls
    ]


def simulation_calls(messages: list[dict]) -> list[tuple[str, str]]:
    return [
        canonical(c["name"], c.get("arguments", {}))
        for m in messages
        if m.get("role") == "assistant"
        for c in m.get("tool_calls") or []
    ]


def task_key(task_id: str) -> str:
    """Telecom task ids like "[mms_issue]airplane_mode_on|..." appear in our ids without brackets,
    with "-" for "|" and cut at about 64 characters, so ids are compared on letters, digits and "_"
    and ours as a prefix."""
    return re.sub(r"[^a-z0-9_]", "", task_id.lower().removeprefix("task_"))


def shared_prefix(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data/raw/valid.parquet")
    ap.add_argument("--trajectories", type=Path, default=ROOT / "data/external/tau2_leaderboard/submissions")
    ap.add_argument("--out", type=Path, default=ROOT / "results/source_match.json")
    args = ap.parse_args()

    frame = pd.read_parquet(args.input)
    examples = []
    for row in frame.itertuples():
        domain, rest = row.id.split("__", 1)
        examples.append(
            {
                "id": row.id,
                "domain": domain,
                "task": task_key(rest.split("::")[0]),
                "label": int(row.label),
                "prompt_calls": our_calls(row.prompt),
                "response_calls": our_calls(row.response),
            }
        )

    files = sorted(args.trajectories.glob("*/trajectories/*.json"))
    searched: Counter = Counter()
    matches: dict[str, list[dict]] = {e["id"]: [] for e in examples}
    longest: dict[str, int] = {e["id"]: 0 for e in examples}
    same_task_simulations: Counter = Counter()
    for path in files:
        domain = next((d for d in DOMAINS if f"_{d}_" in path.name or path.name.startswith(f"{d}_")), None)
        if domain is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        simulations = data.get("simulations", []) if isinstance(data, dict) else []
        searched[domain] += len(simulations)
        candidates = [e for e in examples if e["domain"] == domain and e["prompt_calls"]]
        for sim in simulations:
            calls = simulation_calls(sim.get("messages", []))
            task = task_key(str(sim.get("task_id")))
            for e in candidates:
                same_task = task == e["task"] if e["domain"] != "telecom" else task.startswith(e["task"])
                if same_task:
                    same_task_simulations[e["id"]] += 1
                    whole = e["prompt_calls"] + e["response_calls"]
                    longest[e["id"]] = max(longest[e["id"]], shared_prefix(calls, whole))
                k = len(e["prompt_calls"])
                if calls[:k] != e["prompt_calls"]:
                    continue
                r = len(e["response_calls"])
                matches[e["id"]].append(
                    {
                        "submission": path.parent.parent.name,
                        "file": path.name,
                        "task_id": str(sim.get("task_id")),
                        "trial": sim.get("trial"),
                        "same_task_number": same_task,
                        "response_calls_follow": bool(r) and calls[k : k + r] == e["response_calls"],
                        "reward": (sim.get("reward_info") or {}).get("reward"),
                    }
                )

    per_example = []
    for e in examples:
        found = matches[e["id"]]
        per_example.append(
            {
                "id": e["id"],
                "label": e["label"],
                "prompt_calls": len(e["prompt_calls"]),
                "response_calls": len(e["response_calls"]),
                "searched": e["domain"] in searched,
                "same_task_simulations": same_task_simulations[e["id"]],
                "longest_shared_prefix": longest[e["id"]],
                "simulations_matching_prompt": len(found),
                "with_same_task_number": sum(m["same_task_number"] for m in found),
                "with_response_calls_following": sum(m["response_calls_follow"] for m in found),
                "submissions": sorted({m["submission"] for m in found}),
            }
        )

    searched_examples = [p for p in per_example if p["searched"] and p["prompt_calls"]]
    summary = {
        "files": len(files),
        "simulations_searched_by_domain": dict(searched),
        "examples_searched": len(searched_examples),
        "examples_without_calls_in_prompt": sum(p["prompt_calls"] == 0 for p in per_example if p["searched"]),
        "examples_matched": sum(p["simulations_matching_prompt"] > 0 for p in searched_examples),
        "examples_matched_with_same_task_number": sum(p["with_same_task_number"] > 0 for p in searched_examples),
        "matched_by_label": dict(
            Counter(str(p["label"]) for p in searched_examples if p["simulations_matching_prompt"] > 0)
        ),
        "examples_with_whole_call_sequence_found": sum(
            p["longest_shared_prefix"] == p["prompt_calls"] + p["response_calls"] for p in searched_examples
        ),
        # a short sequence is shared by chance (every agent starts with the same lookups), a long one is not
        "whole_sequence_found_by_length": {
            name: {
                "examples": len(group),
                "found": sum(p["longest_shared_prefix"] == p["prompt_calls"] + p["response_calls"] for p in group),
                "calls": sum(p["prompt_calls"] + p["response_calls"] for p in group),
                "calls_in_longest_shared_prefixes": sum(p["longest_shared_prefix"] for p in group),
            }
            for name, group in (
                ("under_8_calls", [p for p in searched_examples if p["prompt_calls"] + p["response_calls"] < 8]),
                ("8_calls_or_more", [p for p in searched_examples if p["prompt_calls"] + p["response_calls"] >= 8]),
            )
        },
        "matched_with_response_calls_following_by_label": dict(
            Counter(str(p["label"]) for p in searched_examples if p["with_response_calls_following"] > 0)
        ),
        "examples": per_example,
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, ensure_ascii=False, indent=2))
    for p in per_example:
        if p["searched"]:
            print(
                f"{p['id'][:40]:>40} label {p['label']} calls {p['prompt_calls']:>2}+{p['response_calls']} "
                f"task sims {p['same_task_simulations']:>3} longest shared {p['longest_shared_prefix']:>2} "
                f"sims {p['simulations_matching_prompt']:>3} same task {p['with_same_task_number']:>3} "
                f"response follows {p['with_response_calls_following']:>3} {p['submissions'][:3]}"
            )


if __name__ == "__main__":
    main()
