"""Reference facts from the τ²-bench source for the four domains of the data.

For every domain: which tools the agent and the user have in τ²-bench and whether each one reads or
writes (the `@is_tool(ToolType.X)` decorator), which of them the agent sees in the ⟦SYSTEM⟧ block of
the data, and the "current time" line of the policy. Write tools are the ones policy checks attach
to, so the list is taken from the benchmark code rather than guessed from names.

    python scripts/tau2_reference.py --tau2 D:/VScode_projects/tau2-bench
"""

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from guardian.env import setup
from guardian.parsing import parse_available_tools, parse_transcript, section, system_turn

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")
TIME_RE = re.compile(r"current time is ([^.\n]+)")
# banking_knowledge adds its retrieval tools through mixins; the data shows the all-tools variant
EXTRA_TOOL_FILES = {"banking_knowledge": ("retrieval_mixins.py",)}


def tool_types(path: Path) -> dict[str, str]:
    """{function name: READ | WRITE | GENERIC | THINK} for every method decorated with @is_tool."""
    out = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.FunctionDef):
            continue
        for d in node.decorator_list:
            if isinstance(d, ast.Call) and getattr(d.func, "id", None) == "is_tool" and d.args:
                out[node.name] = getattr(d.args[0], "attr", ast.unparse(d.args[0]))
    return out


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau2", type=Path, default=Path("D:/VScode_projects/tau2-bench"))
    ap.add_argument("--input", type=Path, default=ROOT / "data/raw/valid.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/tau2_reference.json")
    args = ap.parse_args()

    commit = subprocess.run(
        ["git", "-C", str(args.tau2), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    frame = pd.read_parquet(args.input)
    systems = {}
    for row in frame.itertuples():
        systems.setdefault(row.id.split("__")[0], system_turn(parse_transcript(row.prompt)).text)

    domains = {}
    for domain in DOMAINS:
        src = args.tau2 / "src/tau2/domains" / domain
        agent = tool_types(src / "tools.py")
        for extra in EXTRA_TOOL_FILES.get(domain, ()):
            agent |= tool_types(src / extra)
        user = tool_types(src / "user_tools.py") if (src / "user_tools.py").exists() else {}
        shown = set(parse_available_tools(systems[domain]))
        time = TIME_RE.search(section(systems[domain], "policy") or "")
        domains[domain] = {
            "agent_tools": dict(sorted(agent.items())),
            "agent_write_tools": sorted(n for n, t in agent.items() if t == "WRITE"),
            "user_tools": dict(sorted(user.items())),
            "shown_in_data": sorted(shown),
            "in_tau2_not_shown": sorted(set(agent) - shown),
            "shown_not_in_tau2": sorted(shown - set(agent)),
            "user_tools_shown_to_agent": sorted(set(user) & shown),
            "policy_current_time": time.group(1).strip() if time else None,
        }

    result = {"tau2_commit": commit, "input": str(args.input.name), "domains": domains}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for domain, d in domains.items():
        print(
            f"{domain}: {len(d['agent_tools'])} agent tools, {len(d['agent_write_tools'])} write, "
            f"{len(d['user_tools'])} user tools, shown {len(d['shown_in_data'])}, "
            f"not shown {d['in_tau2_not_shown']}, unknown to tau2 {d['shown_not_in_tau2']}, "
            f"time {d['policy_current_time']}"
        )


if __name__ == "__main__":
    main()
