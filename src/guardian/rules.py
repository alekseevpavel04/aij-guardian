"""Deterministic checks of the response against the context.

Every check answers a question that can be settled by code, without a model: is the tool the agent
called on the list it was given, do its arguments match the declared schema, does an argument value
occur anywhere in the context. The research question of the project is how far this gets on its own.

Each check returns an int (0/1 or a count) so that the whole set is a feature vector, and the ones
that fire are kept as evidence: a check that cannot say why it fired is not worth keeping.
"""

import re
from dataclasses import dataclass, field

from guardian.parsing import (
    ToolCall,
    ToolSpec,
    Turn,
    normalize_format,
    parse_available_tools,
    parse_payload,
    parse_transcript,
    system_turn,
)

NUMBER_RE = re.compile(r"-?\d[\d 	., ]*\d|\d")
SHORT_VALUE = 3  # below this an argument value matches the context by accident
LONG_VALUE = 48  # above this the value is prose the agent wrote, not a value copied from context
# An agent writes "May 24" as 2024-05-24 by design, so a date is derived rather than copied, and on the
# external dev part dates were most of this check's firings on negatives (docs/EXPERIMENTS.md).
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# An airport code for a city the user named ("Seattle" as SEA) is knowledge, not a copied value.
AIRPORT_PARAMS = frozenset({"origin", "destination"})
AIRPORT_RE = re.compile(r"^[A-Z]{3}$")
FREE_TEXT_PARAMS = frozenset(
    {"summary", "expression", "query", "pattern", "reason", "message", "content", "text", "note", "description"}
)


@dataclass
class Context:
    """Everything the agent could see before the turn under review."""

    prompt: str
    tools: dict[str, ToolSpec]
    turns: list[Turn]
    prior_calls: set[tuple[str, str]] = field(default_factory=set)
    failed_calls: set[tuple[str, str]] = field(default_factory=set)

    @classmethod
    def from_prompt(cls, prompt: str) -> "Context":
        prompt = normalize_format(prompt)
        turns = parse_transcript(prompt)
        system = system_turn(turns)
        tools = parse_available_tools(system.text) if system else {}
        prior: set[tuple[str, str]] = set()
        failed: set[tuple[str, str]] = set()
        for turn in turns:
            errored = {r.name for r in turn.results if r.is_error}
            for call in turn.calls:
                key = (call.name, _canonical(call.args_raw))
                prior.add(key)
                if call.name in errored:
                    failed.add(key)
        return cls(prompt, tools, turns, prior, failed)


def _canonical(args_raw: str) -> str:
    return re.sub(r"\s+", "", args_raw)


def _values(args: dict | None, into_json_strings: bool = False) -> list[tuple[str, object]]:
    """Flat list of (key, scalar) pairs of a call's arguments, descending into lists and dicts.

    `into_json_strings` also descends into a string that is itself a JSON object.
    `call_discoverable_agent_tool` passes the wrapped tool's arguments that way, and that is where a
    fabricated id can hide, but those inner keys belong to the wrapped tool's schema and not to the
    wrapper's, so schema checks must not look at them.
    """
    out: list[tuple[str, object]] = []

    def walk(key: str, value: object) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(k, v)
        elif isinstance(value, list):
            for v in value:
                walk(key, v)
        elif into_json_strings and isinstance(value, str) and value.lstrip().startswith("{"):
            nested = parse_payload(value)
            walk(key, nested) if isinstance(nested, dict) else out.append((key, value))
        else:
            out.append((key, value))

    walk("", args or {})
    return out


def occurs_in(value: object, text: str, ignore_case: bool = False) -> bool:
    """Whether a value occurs in the text as a whole token.

    Token boundaries matter: a zip code "3019" that only occurs inside the email address
    "sofia.thomas3019@example.com" is not a zip code the agent read from the context.
    """
    literal = str(value).strip()
    if not literal:
        return True
    flags = re.IGNORECASE if ignore_case else 0
    return re.search(rf"(?<!\w){re.escape(literal)}(?!\w)", text, flags) is not None


def occurs_in_some_form(value: object, text: str) -> bool:
    """Whether a value occurs in the text as written or in a form the agent may legitimately normalise:
    an order id with its "#" added or dropped, 5.0 for 5, a name in another case."""
    literal = str(value).strip()
    forms = {literal, literal.lstrip("#")}
    if isinstance(value, float) and value.is_integer():
        forms.add(str(int(value)))
    return any(occurs_in(form, text, ignore_case=True) for form in forms)


def normalize_number(text: str) -> str | None:
    """Digits of a number, so that 1 234,50 / 1234.5 / 1,234.50 compare equal."""
    digits = re.sub(r"[^\d]", "", text)
    return digits.lstrip("0") or "0" if digits else None


def context_numbers(prompt: str) -> set[str]:
    return {n for m in NUMBER_RE.finditer(prompt) if (n := normalize_number(m.group(0)))}


def calls_unknown_tool(calls: list[ToolCall], ctx: Context) -> list[str]:
    """Tool called that is not in [AVAILABLE TOOLS]."""
    if not ctx.tools:
        return []
    return [c.name for c in calls if c.name not in ctx.tools]


def unknown_arguments(calls: list[ToolCall], ctx: Context) -> list[str]:
    """Top-level argument name absent from the tool schema.

    Keys inside array elements are not checked: τ²-bench declares those elements with
    additionalProperties allowed, so `flights[].price` is accepted by the environment and is not an
    error. On the external dev part every firing of the nested version on a correct step was such a key.
    """
    out = []
    for call in calls:
        spec = ctx.tools.get(call.name)
        if spec is None or call.args is None:
            continue
        out += [f"{call.name}.{k}" for k in call.args if k not in spec.top_level]
    return out


def missing_required_arguments(calls: list[ToolCall], ctx: Context) -> list[str]:
    """Required argument absent, at the top level or inside an array element."""
    out = []
    for call in calls:
        spec = ctx.tools.get(call.name)
        if spec is None or call.args is None:
            continue
        out += [f"{call.name}.{name}" for name in sorted(spec.required - set(call.args))]
        for parent in spec.top_level:
            required = {p.name for p in spec.params.values() if p.parent == parent and p.required}
            elements = call.args.get(parent)
            if not required or not isinstance(elements, list):
                continue
            for element in elements:
                if isinstance(element, dict):
                    out += [f"{call.name}.{parent}.{name}" for name in sorted(required - set(element))]
    return out


def enum_violations(calls: list[ToolCall], ctx: Context) -> list[str]:
    out = []
    for call in calls:
        spec = ctx.tools.get(call.name)
        if spec is None or call.args is None:
            continue
        by_name = {p.name: p for p in spec.params.values() if p.enum}
        for key, value in _values(call.args):
            param = by_name.get(key)
            if param and isinstance(value, str) and value not in param.enum:
                out.append(f"{call.name}.{key}={value}")
    return out


def repeated_failed_calls(calls: list[ToolCall], ctx: Context) -> list[str]:
    """Same tool with the same arguments as a call that already returned an error."""
    return [c.name for c in calls if (c.name, _canonical(c.args_raw)) in ctx.failed_calls]


def repeated_calls(calls: list[ToolCall], ctx: Context) -> list[str]:
    return [c.name for c in calls if (c.name, _canonical(c.args_raw)) in ctx.prior_calls]


def unsupported_argument_values(calls: list[ToolCall], ctx: Context) -> list[str]:
    """Argument value that does not occur in the context as a whole token.

    Free-text arguments (summaries, search queries, expressions) are written by the agent and are
    expected to be new text, so they are skipped, as is anything long enough or spaced enough to be
    prose. What is left is identifiers, codes, dates and amounts: values the agent is supposed to
    copy from what it was shown.
    """
    out = []
    for call in calls:
        spec = ctx.tools.get(call.name)
        for key, value in _values(call.args, into_json_strings=True):
            if key in FREE_TEXT_PARAMS or not isinstance(value, (str, int, float)) or isinstance(value, bool):
                continue
            literal = str(value).strip()
            if not SHORT_VALUE <= len(literal) <= LONG_VALUE or " " in literal or ISO_DATE_RE.match(literal):
                continue
            if key in AIRPORT_PARAMS and AIRPORT_RE.match(literal):
                continue
            param = next((p for p in (spec.params.values() if spec else []) if p.name == key), None)
            if param and param.enum:
                continue
            if not occurs_in_some_form(value, ctx.prompt):
                out.append(f"{call.name}.{key}={literal}")
    return out


def no_op_replacements(calls: list[ToolCall], ctx: Context) -> list[str]:
    """A replacement argument equal to the value it replaces.

    Modification tools take the old values in `x` and the new ones in `new_x`; passing the same value
    in both is not a modification, and the agent claims a change that does not happen
    (retail__36::t10). Positions are compared pairwise, since the arrays are aligned by index.
    """
    out = []
    for call in calls:
        args = call.args or {}
        for key, new in args.items():
            if not key.startswith("new_"):
                continue
            old = args.get(key[4:])
            if isinstance(new, list) and isinstance(old, list) and len(new) == len(old):
                out += [f"{call.name}.{key}[{i}]={v}" for i, (v, w) in enumerate(zip(old, new, strict=True)) if v == w]
            elif not isinstance(new, list) and old is not None and old == new:
                out.append(f"{call.name}.{key}={new}")
    return out


def unsupported_numbers(text: str, ctx: Context, known: set[str] | None = None) -> list[str]:
    """Numbers in the response text that do not occur in the context.

    Sums and differences of context numbers are legitimate aggregation, so a number is only flagged
    when it is neither present in the context nor a plain arithmetic combination of two of its
    numbers. This is the noisiest check here: dates, percentages and counts written out by the agent
    all land in it.
    """
    known = known if known is not None else context_numbers(ctx.prompt)
    out = []
    for m in NUMBER_RE.finditer(text):
        n = normalize_number(m.group(0))
        if n and n not in known:
            out.append(m.group(0).strip())
    return out


CHECKS = {
    "unknown_tool": calls_unknown_tool,
    "unknown_argument": unknown_arguments,
    "missing_required_argument": missing_required_arguments,
    "enum_violation": enum_violations,
    "repeated_failed_call": repeated_failed_calls,
    "repeated_call": repeated_calls,
    "unsupported_argument_value": unsupported_argument_values,
    "no_op_replacement": no_op_replacements,
}


def evidence(prompt: str, response: str, ctx: Context | None = None) -> dict[str, list[str]]:
    """Run every call-level check and return what each one found."""
    ctx = ctx or Context.from_prompt(prompt)
    calls = [c for turn in parse_transcript(response) for c in turn.calls]
    found = {name: check(calls, ctx) for name, check in CHECKS.items()}
    text = "\n".join(t.text for t in parse_transcript(response))
    found["unsupported_number"] = unsupported_numbers(text, ctx)
    return found


def features(prompt: str, response: str, ctx: Context | None = None) -> dict[str, int]:
    return {name: len(hits) for name, hits in evidence(prompt, response, ctx).items()}
