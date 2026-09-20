"""Parsing of agent transcripts.

Format is taken from the competition rules (p. 3.7): `prompt` and `response` are single text blocks
split by markers in double brackets, and tool calls and tool results carry attributes:

    ⟦SYSTEM⟧ / ⟦USER⟧ / ⟦ASSISTANT⟧
    ⟦ASSISTANT_TOOL_CALL name="X"⟧      arguments below as JSON
    ⟦USER_TOOL_CALL name="X"⟧
    ⟦TOOL_RESULT name="X" requestor="assistant"⟧

The rules and the original task description disagree on the markers, and neither has been checked
against real data yet, so unknown markers are kept rather than dropped: `unknown_roles` reports
them instead of the parser silently losing content.
"""

import ast
import json
import re
from dataclasses import dataclass, field

KNOWN_ROLES = frozenset(
    {"SYSTEM", "USER", "ASSISTANT", "ASSISTANT_TOOL_CALL", "USER_TOOL_CALL", "TOOL_RESULT", "PREAMBLE"}
)
TOOL_CALL_ROLES = frozenset({"ASSISTANT_TOOL_CALL", "USER_TOOL_CALL"})

MARKER_RE = re.compile(r"⟦\s*([A-Z_]+)((?:\s+[a-z_]+\s*=\s*\"[^\"]*\")*)\s*⟧")
ATTR_RE = re.compile(r"([a-z_]+)\s*=\s*\"([^\"]*)\"")
BLOCK_RE = re.compile(r"<(prompt|response)>(.*?)</\1>", re.DOTALL)


@dataclass
class Turn:
    role: str
    text: str
    attrs: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str | None:
        """Tool name for tool calls and tool results."""
        return self.attrs.get("name")

    @property
    def payload(self) -> object | None:
        """Body parsed as a JSON value, or None when it is not JSON."""
        return parse_payload(self.text)


def parse_payload(text: str) -> object | None:
    body = text.strip()
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    try:
        # Tool results are sometimes printed as Python literals (single quotes, True / None).
        return ast.literal_eval(body)
    except (ValueError, SyntaxError):
        return None


def strip_block_tags(text: str) -> str:
    """Return the contents of a <prompt> or <response> wrapper, or the text itself if there is none."""
    match = BLOCK_RE.search(text)
    return match.group(2) if match else text


def parse_transcript(text: str) -> list[Turn]:
    text = strip_block_tags(text)
    turns: list[Turn] = []
    matches = list(MARKER_RE.finditer(text))

    if not matches:
        body = text.strip()
        return [Turn("PREAMBLE", body)] if body else []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        turns.append(Turn("PREAMBLE", preamble))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        attrs = dict(ATTR_RE.findall(m.group(2) or ""))
        turns.append(Turn(m.group(1), text[m.end() : end].strip(), attrs))
    return turns


def unknown_roles(turns: list[Turn]) -> set[str]:
    return {t.role for t in turns} - KNOWN_ROLES


def tool_calls(turns: list[Turn]) -> list[Turn]:
    return [t for t in turns if t.role in TOOL_CALL_ROLES]


def tool_results(turns: list[Turn]) -> list[Turn]:
    return [t for t in turns if t.role == "TOOL_RESULT"]


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    """Flatten nested JSON into {path: scalar}, so context values can be looked up by path."""
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for k, v in value.items():
            out |= flatten(v, f"{prefix}.{k}" if prefix else str(k))
        return out
    if isinstance(value, list):
        out = {}
        for i, v in enumerate(value):
            out |= flatten(v, f"{prefix}[{i}]")
        return out
    return {prefix: value}
