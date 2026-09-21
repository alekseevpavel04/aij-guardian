"""The τ²-bench task behind a competition example, as privileged reference for a labeller.

Example ids name the task: `airline__21::t7` is task "21" of airline, turn 7. Telecom task ids like
"[mms_issue]airplane_mode_on|bad_network_preference[PERSONA:Hard]" appear in the data with the
brackets dropped, "|" as "-", ":" as "_", and cut at 81 characters. A cut id can match dozens of
tasks with different expected actions; such an example gets no reference rather than a wrong one.

The reference is what the organiser's labeller could plausibly see and the detector cannot: what
the task tests, what the user wants, and the actions the benchmark expects. It is never given to the
detector itself.
"""

import json
from functools import cache
from pathlib import Path

TAU2 = Path("D:/VScode_projects/tau2-bench/data/tau2/domains")
CUT = 81


def data_id(task_id: str) -> str:
    """A τ² task id as it appears in competition example ids, before the cut."""
    return task_id.replace("[", "").replace("]", "").replace("|", "-").replace(":", "_")


@cache
def tasks(domain: str, root: Path = TAU2) -> tuple[dict, ...]:
    return tuple(json.loads((root / domain / "tasks.json").read_text(encoding="utf-8")))


def candidates(example_id: str, root: Path = TAU2) -> list[dict]:
    domain, rest = example_id.split("__", 1)
    wanted = rest.split("::")[0]
    found = [t for t in tasks(domain, root) if data_id(t["id"]) == wanted]
    if not found and len(wanted) == CUT:
        found = [t for t in tasks(domain, root) if data_id(t["id"]).startswith(wanted)]
    return found


def unique_task(example_id: str, root: Path = TAU2) -> dict | None:
    """The task, or None when there is none or the candidates expect different actions."""
    found = candidates(example_id, root)
    expected = {json.dumps((t.get("evaluation_criteria") or {}).get("actions"), sort_keys=True) for t in found}
    return found[0] if found and len(expected) == 1 else None


def reference_text(task: dict) -> str:
    """The task as a compact English block: purpose, user scenario, expected actions and outcomes."""
    description = task.get("description") or {}
    scenario = (task.get("user_scenario") or {}).get("instructions") or {}
    if isinstance(scenario, str):
        scenario = {"instructions": scenario}
    criteria = task.get("evaluation_criteria") or {}
    lines = []
    if description.get("purpose"):
        lines.append(f"Purpose of the task: {description['purpose'].strip()}")
    for key in ("reason_for_call", "known_info", "unknown_info", "task_instructions", "instructions"):
        if scenario.get(key):
            lines.append(f"User {key.replace('_', ' ')}: {str(scenario[key]).strip()}")
    actions = criteria.get("actions") or []
    if actions:
        lines.append("Expected agent actions (in the reference solution):")
        for a in actions:
            who = f" [{a['requestor']}]" if a.get("requestor") and a["requestor"] != "assistant" else ""
            arguments = json.dumps(a.get("arguments") or {}, ensure_ascii=False, sort_keys=True)
            lines.append(f"- {a['name']}{who}: {arguments}")
    for key, title in (("nl_assertions", "Expected outcomes"), ("communicate_info", "Information to communicate")):
        if criteria.get(key):
            lines.append(f"{title}:")
            lines += [f"- {item}" for item in criteria[key]]
    return "\n".join(lines)
