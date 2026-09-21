"""TrajErrBench, τ² subset, rendered into the competition transcript format.

Source: github.com/THU-KEG/TrajDebug, data/trajerrbench/en/tau2bench: 400 failed trajectories of
airline and retail (agents: DeepSeek-Reasoner, GPT-5, Qwen3-Max) with the earliest decisive error
step and its type annotated. The annotations are MIT; the trajectories come from τ²-bench (MIT).

The annotated step becomes a positive. Agent steps before it become negatives with
label_quality "weak": an earlier step can still be wrong without being the decisive error. Steps
after it are dropped, since what follows a decisive error is neither clean nor labelled.

The files carry no tool schemas. Their policies are the τ²-bench version our data uses (airline 7676
characters in both), so the [AVAILABLE TOOLS] block is taken from valid.parquet for the domain.

    python scripts/build_trajerr.py --repo D:/VScode_projects/TrajDebug
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

from guardian.env import setup
from guardian.parsing import (
    TOOLS_HEADER,
    format_recognised,
    parse_transcript,
    system_turn,
    unknown_roles,
    unparsed_markers,
)
from guardian.render import render_system, render_trajectory, tau2_user

ROOT = Path(__file__).resolve().parents[1]
CALL_LINE_RE = re.compile(r"^- (?P<name>[\w.\-]+)\((?P<args>.*)\)$")


def to_messages(raw: list[dict]) -> list[dict]:
    """TrajDebug's messages, with calls written as "[tool_calls]\\n- name({json})", as OpenAI messages."""
    out = []
    for m in raw:
        content = m.get("content") or ""
        if m["role"] == "assistant" and content.startswith("[tool_calls]"):
            calls = []
            for line in content.splitlines()[1:]:
                call = CALL_LINE_RE.match(line.strip())
                if call is None:
                    raise ValueError(f"step {m['step']}: unexpected tool call line {line!r}")
                calls.append({"function": {"name": call.group("name"), "arguments": call.group("args")}})
            out.append({"role": "assistant", "content": "", "tool_calls": calls, "step": m["step"]})
        else:
            out.append({"role": m["role"], "content": content, "step": m["step"]})
    # results carry no tool name; they follow the calls of the preceding agent message in order
    pending: list[str] = []
    for m in out:
        if m["role"] == "assistant":
            pending = [c["function"]["name"] for c in m.get("tool_calls", [])]
        elif m["role"] == "tool":
            m["name"] = pending.pop(0) if pending else "unknown"
    return out


def tools_blocks(valid: Path) -> dict[str, str]:
    blocks = {}
    for row in pd.read_parquet(valid).itertuples():
        text = system_turn(parse_transcript(row.prompt)).text
        blocks.setdefault(row.id.split("__")[0], text[text.find(TOOLS_HEADER) :])
    return blocks


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("D:/VScode_projects/TrajDebug"))
    ap.add_argument("--valid", type=Path, default=ROOT / "data/raw/valid.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "data/external/trajerr_tau2.parquet")
    ap.add_argument("--stats", type=Path, default=ROOT / "results/external_data.json")
    args = ap.parse_args()

    source = args.repo / "data/trajerrbench/en/tau2bench"
    commit = subprocess.run(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    tools = tools_blocks(args.valid)
    files = sorted(source.glob("*.json"))
    digest = hashlib.sha256()
    records, skipped = [], Counter()
    for path in files:
        raw = path.read_bytes()
        digest.update(raw)
        t = json.loads(raw)
        domain = path.name.split("_", 1)[0]
        annotation = t["metadata"]["annotation"]
        messages = to_messages(t["messages"])
        steps = render_trajectory(messages, render_system(messages[0]["content"], tools[domain]))
        by_step = {messages[s.message_index]["step"]: s for s in steps}
        critical = annotation["critical_error_step"]
        if critical not in by_step:
            skipped["critical_step_is_not_an_agent_step"] += 1
            continue
        group = f"{domain}:{tau2_user(messages[1:]) or path.stem}"
        agent = path.stem.split("_failed_", 1)[1].rsplit("_", 1)[0]  # airline_failed_<agent>_<uuid>
        for number, step in sorted(by_step.items()):
            if number > critical:
                break
            records.append(
                {
                    "id": f"trajerr__{domain}__{t['metadata']['task_id'].split('::')[1]}::t{step.turn}",
                    "prompt": step.prompt,
                    "response": step.response,
                    "weak_label": int(number == critical),
                    "source": "trajerrbench",
                    "source_label": (annotation["critical_error_type"] or "untyped")
                    if number == critical
                    else "before_critical",
                    "domain": domain,
                    "label_quality": "human_critical_step" if number == critical else "weak",
                    "first_error": number == critical,
                    "group": group,
                    "agent": agent,
                }
            )

    frame = pd.DataFrame(records)
    frame.to_parquet(args.out, index=False)
    parse_failures = sum(
        not format_recognised(r.prompt)
        or bool(unparsed_markers(r.prompt) or unparsed_markers(r.response))
        or bool(unknown_roles(parse_transcript(r.prompt)))
        for r in frame.itertuples()
    )
    positives = frame[frame.weak_label == 1]
    stats = json.loads(args.stats.read_text(encoding="utf-8")) if args.stats.exists() else {}
    stats["trajerrbench_tau2"] = {
        "source": "https://github.com/THU-KEG/TrajDebug",
        "revision": commit,
        "path": "data/trajerrbench/en/tau2bench",
        "files_sha256": digest.hexdigest(),
        "license": "annotations MIT; trajectories from tau2-bench (MIT)",
        "trajectories": len(files),
        "skipped": dict(skipped),
        "rows": len(frame),
        "positives": len(positives),
        "negatives_weak": int((frame.weak_label == 0).sum()),
        "positives_by_error_family": dict(Counter(v.split(".")[0] for v in positives.source_label).most_common()),
        "by_domain": {d: {"rows": len(g), "positives": int(g.weak_label.sum())} for d, g in frame.groupby("domain")},
        "by_agent": dict(Counter(positives.agent)),
        "groups": int(frame.group.nunique()),
        "rows_the_parser_does_not_fully_recognise": int(parse_failures),
        "prompt_chars_p50": int(frame.prompt.str.len().median()),
    }
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats["trajerrbench_tau2"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
