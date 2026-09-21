"""Parsing of agent transcripts.

The format in the data is the one on the competition's Data tab, not the one in the competition rules
PDF (which lists markers like ⟦ASSISTANT_TOOL_CALL name="X"⟧ that do not occur in valid.parquet):

    ⟦SYSTEM⟧            <instructions>, <policy>, [AVAILABLE TOOLS]
    ⟦USER⟧
    ⟦ASSISTANT⟧         greeting, no turn number
    ⟦ASSISTANT · ход N⟧  numbered agent turn, tool calls and results indented below it
        → TOOL_CALL <name>: <json>
        ← TOOL_RESPONSE <name>: <json or text>
        ← TOOL_RESPONSE <name> [ERROR]: <text>

Markers are matched on their own line and roles are kept verbatim, so a marker the parser does not
recognise shows up in `unparsed_markers` instead of being silently swallowed into a neighbour turn.

The rules PDF and the organiser baseline's prompt describe a second format, with calls and results
as markers of their own and the argument or result JSON on the lines below:

    ⟦ASSISTANT_TOOL_CALL name="X"⟧      (the baseline prompt writes ⟦TOOL_CALL name="X"⟧)
    ⟦USER_TOOL_CALL name="X"⟧
    ⟦TOOL_RESULT name="X" requestor="assistant"⟧

No example in valid.parquet uses it, but nobody has said which one the test uses, so
`normalize_format` rewrites it into the first before parsing and both give the same turns.
"""

import ast
import json
import re
from dataclasses import dataclass, field

ROLE_RE = re.compile(r"^⟦(?P<role>[A-Z_]+)(?P<rest>[^⟧\n]*)⟧[ \t]*$", re.MULTILINE)
MARKER_LIKE_RE = re.compile(r"⟦[^⟧\n]*⟧")
CALL_RE = re.compile(r"^[ \t]*→ TOOL_CALL (?P<name>[\w.\-]+):[ ]?(?P<args>.*)$", re.MULTILINE)
RESULT_RE = re.compile(
    r"^[ \t]*← TOOL_RESPONSE (?P<name>[\w.\-]+)(?P<error>[ \t]*\[ERROR\])?:[ ]?(?P<body>.*)$", re.MULTILINE
)
TURN_INDEX_RE = re.compile(r"(\d+)")
ALT_MARKER_RE = re.compile(
    r"^[ \t]*⟦(?P<kind>ASSISTANT_TOOL_CALL|USER_TOOL_CALL|TOOL_CALL|TOOL_RESULT)(?P<attrs>[^⟧\n]*)⟧[ \t]*$",
    re.MULTILINE,
)
ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
WRAPPER_RE = re.compile(r"^[ \t]*</?(?:prompt|response)>[ \t]*\n?", re.MULTILINE)

KNOWN_ROLES = frozenset({"SYSTEM", "USER", "ASSISTANT", "PREAMBLE"})
TOOLS_HEADER = "[AVAILABLE TOOLS]"
TOOL_RE = re.compile(r"^- (?P<name>[\w.\-]+) — (?P<description>.*)$")
PARAM_TYPES = ("string", "integer", "number", "boolean", "array", "object")
PARAM_RE = re.compile(
    r"^(?P<indent>[ \t]+)(?P<bullet>·[ ]*)?(?P<name>[\w.\-]+): (?P<type>" + "|".join(PARAM_TYPES) + r")"
    r"(?P<required>!)?(?:[ ]*\[enum:[ ]*(?P<enum>[^\]]*)\])?(?:[ ]*—[ ]*(?P<description>.*))?$"
)


@dataclass(frozen=True)
class ToolCall:
    name: str
    args_raw: str
    args: dict | None

    @property
    def parsed(self) -> bool:
        return self.args is not None


@dataclass(frozen=True)
class ToolResult:
    name: str
    body: str
    is_error: bool

    @property
    def value(self) -> object | None:
        return parse_payload(self.body)


@dataclass
class Turn:
    role: str
    index: int | None
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    required: bool
    enum: list[str] | None
    parent: str | None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict[str, Param]

    @property
    def required(self) -> set[str]:
        return {p.name for p in self.params.values() if p.required and p.parent is None}

    @property
    def top_level(self) -> set[str]:
        return {p.name for p in self.params.values() if p.parent is None}


def parse_payload(text: str) -> object | None:
    body = text.strip()
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    try:
        # Some tool results are printed as Python literals (single quotes, True / None).
        return ast.literal_eval(body)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def normalize_format(text: str) -> str:
    """Rewrite the marker-per-call format of the rules PDF into the format of the data.

    A call or result becomes an indented `→ TOOL_CALL` / `← TOOL_RESPONSE` line under the turn it
    belongs to. A user's own tool call opens a ⟦USER⟧ turn if it is not already in one, and a
    following agent call opens an ⟦ASSISTANT⟧ turn back, so that user-side calls are never counted
    as the agent's. Text in the data format passes through unchanged.
    """
    if not ALT_MARKER_RE.search(text) and not WRAPPER_RE.search(text):
        return text
    text = WRAPPER_RE.sub("", text)
    alt_kinds = {"ASSISTANT_TOOL_CALL", "USER_TOOL_CALL", "TOOL_CALL", "TOOL_RESULT"}
    boundaries = sorted(
        [(m, True) for m in ALT_MARKER_RE.finditer(text)]
        + [(m, False) for m in ROLE_RE.finditer(text) if m.group("role") not in alt_kinds],
        key=lambda b: b[0].start(),
    )
    out = [text[: boundaries[0][0].start()] if boundaries else text]
    role = None
    for i, (marker, is_alt) in enumerate(boundaries):
        body = text[marker.end() : boundaries[i + 1][0].start() if i + 1 < len(boundaries) else len(text)]
        if not is_alt:
            role = marker.group("role")
            out.append(marker.group(0) + body)
            continue
        attrs = dict(ATTR_RE.findall(marker.group("attrs")))
        name, kind = attrs.get("name", "unknown"), marker.group("kind")
        owner = "USER" if kind == "USER_TOOL_CALL" or attrs.get("requestor") == "user" else "ASSISTANT"
        if role != owner:
            out.append(f"\n⟦{owner}⟧\n")
            role = owner
        payload = _one_line(body)
        if kind == "TOOL_RESULT":
            error = attrs.get("error", "").lower() in ("true", "1") or payload.startswith("Error")
            out.append(f"\t← TOOL_RESPONSE {name}{' [ERROR]' if error else ''}: {payload}\n")
        else:
            out.append(f"\t→ TOOL_CALL {name}: {payload}\n")
    return "".join(out)


def _one_line(body: str) -> str:
    """JSON re-serialised on one line in the data's spacing, anything else with its lines joined."""
    body = body.strip()
    try:
        return json.dumps(json.loads(body), ensure_ascii=False)
    except json.JSONDecodeError:
        return " ".join(line.strip() for line in body.splitlines() if line.strip())


def parse_transcript(text: str) -> list[Turn]:
    text = normalize_format(text)
    markers = list(ROLE_RE.finditer(text))
    turns: list[Turn] = []

    preamble = text[: markers[0].start()] if markers else text
    if preamble.strip():
        turns.append(_make_turn("PREAMBLE", None, preamble))

    for i, m in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        index = TURN_INDEX_RE.search(m.group("rest"))
        turns.append(_make_turn(m.group("role"), int(index.group(1)) if index else None, text[m.end() : end]))
    return turns


def _make_turn(role: str, index: int | None, body: str) -> Turn:
    calls = [ToolCall(m.group("name"), m.group("args"), parse_args(m.group("args"))) for m in CALL_RE.finditer(body)]
    results = [ToolResult(m.group("name"), m.group("body"), bool(m.group("error"))) for m in RESULT_RE.finditer(body)]
    text = RESULT_RE.sub("", CALL_RE.sub("", body)).strip()
    return Turn(role, index, text, calls, results)


def parse_args(raw: str) -> dict | None:
    value = parse_payload(raw)
    return value if isinstance(value, dict) else None


def unparsed_markers(text: str) -> list[str]:
    """Marker-like substrings the parser did not turn into a turn: a format drift alarm."""
    text = normalize_format(text)
    consumed = {m.group(0).strip() for m in ROLE_RE.finditer(text)}
    return [m.group(0) for m in MARKER_LIKE_RE.finditer(text) if m.group(0) not in consumed]


def unknown_roles(turns: list[Turn]) -> set[str]:
    """Roles outside the four the data uses: the other half of the format drift alarm."""
    return {t.role for t in turns} - KNOWN_ROLES


def format_recognised(prompt: str) -> bool:
    """Whether the prompt has a system turn with a parseable tool list: without one no check can fire."""
    system = system_turn(parse_transcript(prompt))
    return system is not None and bool(parse_available_tools(system.text))


def system_turn(turns: list[Turn]) -> Turn | None:
    return next((t for t in turns if t.role == "SYSTEM"), None)


def section(text: str, tag: str) -> str | None:
    """Contents of <tag>…</tag>, used for <policy> and <instructions>.

    The opening tag is taken as the last one before the closing tag, because <instructions> mentions
    "according to the <policy> provided below" and a first-match search starts from that mention.
    """
    close = text.find(f"</{tag}>")
    if close < 0:
        return None
    opened = text.rfind(f"<{tag}>", 0, close)
    return text[opened + len(tag) + 2 : close].strip() if opened >= 0 else None


def parse_available_tools(system_text: str) -> dict[str, ToolSpec]:
    """Parse the [AVAILABLE TOOLS] block into tool schemas.

    Tool descriptions wrap onto indented continuation lines that look a lot like parameter lines,
    so a parameter is only recognised when its type is one of the types the block actually uses.
    """
    start = system_text.find(TOOLS_HEADER)
    if start < 0:
        return {}

    tools: dict[str, ToolSpec] = {}
    name = description = None
    params: dict[str, Param] = {}
    stack: list[tuple[int, str]] = []  # (indent, param name) for nested array/object members

    def flush() -> None:
        if name is not None:
            tools[name] = ToolSpec(name, description or "", params)

    for line in system_text[start + len(TOOLS_HEADER) :].splitlines():
        if not line.strip():
            continue
        tool = TOOL_RE.match(line)
        if tool:
            flush()
            name, description, params, stack = tool.group("name"), tool.group("description"), {}, []
            continue
        if name is None:
            continue
        param = PARAM_RE.match(line)
        if param:
            indent = len(param.group("indent").expandtabs(4))
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else None
            key = f"{parent}.{param.group('name')}" if parent else param.group("name")
            enum = [v.strip() for v in param.group("enum").split("|")] if param.group("enum") else None
            params[key] = Param(param.group("name"), param.group("type"), bool(param.group("required")), enum, parent)
            stack.append((indent, key))
    flush()
    return tools


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
