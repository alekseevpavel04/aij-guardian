"""τ²-bench simulations turned into competition rows, the way the organiser's data looks.

What the data shows about the organiser's rendering, and what is copied here:

- the ⟦SYSTEM⟧ block is byte-identical within a domain, so it is taken from valid.parquet as is;
- the fixed τ² greeting is replaced by the Russian one (44 of 46 examples; 2 kept the English);
- the user's own device tools (telecom) never appear: every result in the data is the agent's, so
  user tool calls and their results are dropped and the user's text stays;
- an agent message is its text, then its calls, then their results (guardian/render.py).
"""

import json
from pathlib import Path

from guardian.parsing import parse_transcript, system_turn
from guardian.render import RenderedStep, render_trajectory

GREETING = "Здравствуйте! Чем могу помочь?"


def organiser_systems(valid: Path) -> dict[str, str]:
    """The ⟦SYSTEM⟧ turn of each domain exactly as valid.parquet has it."""
    import pandas as pd

    systems = {}
    for row in pd.read_parquet(valid).itertuples():
        text = system_turn(parse_transcript(row.prompt)).text
        systems.setdefault(row.id.split("__")[0], f"⟦SYSTEM⟧\n{text}")
    return systems


def to_messages(simulation: dict) -> list[dict]:
    """τ² messages as the OpenAI-style list render_trajectory reads, without the user's tool traffic."""
    out, first_agent = [], True
    for m in simulation.get("messages") or []:
        role = m.get("role")
        if role == "user":
            if (m.get("content") or "").strip():
                out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            content = m.get("content") or ""
            if first_agent and not m.get("tool_calls"):
                content = GREETING
            first_agent = False
            calls = [c for c in m.get("tool_calls") or [] if c.get("requestor", "assistant") == "assistant"]
            out.append({"role": "assistant", "content": content, "tool_calls": calls})
        elif role == "tool" and m.get("requestor", "assistant") == "assistant":
            content = m.get("content")
            out.append(
                {
                    "role": "tool",
                    "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
                    "tool_call_id": m.get("id"),
                    "error": bool(m.get("error")),
                }
            )
    return out


def steps(record: dict, systems: dict[str, str]) -> list[RenderedStep]:
    """Every agent turn of a stored simulation as (prompt, response) in the competition format."""
    return render_trajectory(to_messages(record["simulation"]), systems[record["domain"]])
