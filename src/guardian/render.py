"""Rendering of τ²-bench style trajectories into the transcript format of the competition data.

External labelled trajectories (AgentProcessBench, TrajErrBench) come as OpenAI-style message lists
with JSON-schema tool definitions. To measure our checks on them and to train on them, they have to
look like the data the platform sends, down to the layout the parser and the checks rely on:

- ⟦SYSTEM⟧ holds <instructions>, <policy> and [AVAILABLE TOOLS] in the layout of valid.parquet;
- the greeting is an unnumbered ⟦ASSISTANT⟧ turn, every later agent message is ⟦ASSISTANT · ход N⟧;
- an agent message is its text, then all its calls, then all their results, one line each;
- call arguments are serialised with sorted keys (as in the data), results are kept as returned.

Turns are separated by a blank line and the system block by two, as in valid.parquet.
"""

import json
from dataclasses import dataclass

AGENT_INSTRUCTION = (
    "You are a customer service agent that helps the user according to the <policy> provided below.\n"
    "In each turn you can either:\n"
    "- Send a message to the user.\n"
    "- Make a tool call.\n"
    "You cannot do both at the same time.\n"
    "\n"
    "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."
)  # tau2-bench src/tau2/agent/llm_agent.py; the competition data adds a paragraph about Russian


def _resolve(schema: dict, defs: dict) -> dict:
    """Follow $ref and take the first non-null branch of anyOf (pydantic's Optional and item unions)."""
    while True:
        if "$ref" in schema:
            schema = defs[schema["$ref"].rsplit("/", 1)[-1]]
        elif "anyOf" in schema:
            branches = [b for b in schema["anyOf"] if b.get("type") != "null"]
            merged = {k: v for k, v in schema.items() if k != "anyOf"}
            schema = {**branches[0], **merged} if branches else merged
        else:
            return schema


def _param_lines(properties: dict, required: set[str], defs: dict, depth: int) -> list[str]:
    lines = []
    for name, raw in properties.items():
        schema = _resolve(raw, defs)
        line = "    " * (depth + 1) + ("· " if depth else "") + f"{name}: {schema.get('type', 'string')}"
        line += "!" if name in required else ""
        if schema.get("enum"):
            line += f" [enum: {'|'.join(str(v) for v in schema['enum'])}]"
        description = raw.get("description") or schema.get("description")
        if description:
            line += f" — {description.replace(chr(10), ' ')}"  # parameter descriptions are one line in the data
        lines.append(line)
        nested = _resolve(schema.get("items", {}), defs) if schema.get("type") == "array" else schema
        if nested.get("properties") and nested is not raw:
            lines += _param_lines(nested["properties"], set(nested.get("required", [])), defs, depth + 1)
    return lines


def render_tools(tools: list[dict]) -> str:
    """OpenAI-style tool definitions as the [AVAILABLE TOOLS] block, in the order given."""
    out = ["[AVAILABLE TOOLS]"]
    for tool in (t.get("function", t) for t in tools):
        head, *rest = (tool.get("description") or "").split("\n")
        out.append(f"- {tool['name']} — {head}")
        out += [f"    {line}" for line in rest]
        params = tool.get("parameters") or {}
        out += _param_lines(params.get("properties", {}), set(params.get("required", [])), params.get("$defs", {}), 0)
    return "\n".join(out)


def render_system(policy: str, tools: list[dict], instruction: str = AGENT_INSTRUCTION) -> str:
    return (
        f"⟦SYSTEM⟧\n<instructions>\n{instruction}\n</instructions>\n<policy>\n{policy.strip()}\n</policy>\n\n"
        + render_tools(tools)
    )


@dataclass(frozen=True)
class RenderedStep:
    message_index: int  # index of the agent message in the source trajectory
    turn: int  # N of ⟦ASSISTANT · ход N⟧
    prompt: str
    response: str


def _call_line(call: dict) -> str:
    function = call.get("function", call)
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return f"\t→ TOOL_CALL {function['name']}: {args}"
    return f"\t→ TOOL_CALL {function['name']}: {json.dumps(args, ensure_ascii=False, sort_keys=True)}"


def _result_line(message: dict, name: str) -> str:
    content = message.get("content")
    body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    error = message.get("error") or body.startswith("Error")
    return f"\t← TOOL_RESPONSE {name}{' [ERROR]' if error else ''}: {body.replace(chr(10), ' ')}"


def render_trajectory(messages: list[dict], system: str) -> list[RenderedStep]:
    """One step per agent message after the greeting: the prompt is everything before it."""
    blocks = [system + "\n"]
    steps: list[RenderedStep] = []
    turn = 0
    names: dict[str, str] = {}
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m["role"]
        if role == "system":
            i += 1
            continue
        if role == "user":
            blocks.append(f"⟦USER⟧\n{(m.get('content') or '').strip()}")
            i += 1
            continue
        if role == "tool":  # a result with no agent message before it: attach to nothing, keep visible
            blocks.append(_result_line(m, m.get("name", "unknown")))
            i += 1
            continue

        greeting = turn == 0 and not m.get("tool_calls") and not any(b.startswith("⟦ASSISTANT") for b in blocks)
        header = "⟦ASSISTANT⟧" if greeting else f"⟦ASSISTANT · ход {turn + 1}⟧"
        lines = [header]
        if (m.get("content") or "").strip():
            lines.append(m["content"].strip())
        calls = m.get("tool_calls") or []
        lines += [_call_line(c) for c in calls]
        for c in calls:
            names[c.get("id", "")] = c.get("function", c)["name"]
        response = "\n".join(lines)
        results = []
        j = i + 1
        while j < len(messages) and messages[j]["role"] == "tool":
            t = messages[j]
            results.append(_result_line(t, t.get("name") or names.get(t.get("tool_call_id", ""), "unknown")))
            j += 1
        if not greeting:
            turn += 1
            steps.append(RenderedStep(i, turn, "\n\n".join(blocks), response))
        blocks.append("\n".join([response, *results]))
        i = j
    return steps
