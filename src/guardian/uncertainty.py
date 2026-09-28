"""The uncertainty head's inputs: how probable the judge model found each token of the agent's turn, by part.

For every token of the turn the pass keeps its log-probability, the entropy of the distribution it was drawn
from and the gap between the two most probable tokens. Each token is also marked as part of a tool call's
argument value, an argument name, the call's tool name, the layout (turn header, call markers, JSON
punctuation) or the free text. The head reads 12 summaries of the whole turn and 6 for each part but the
layout: 36 numbers per row (T4.H10, results/turn_ruler.json).

The same functions serve the home pass (llama.dll, scripts/turn_tokens.py), the FP8 pass that trains the
final heads, and the submission, so a row gets the same numbers wherever its token probabilities come from.
"""

import re

import numpy as np

CALL_RE = re.compile(r"→ TOOL_CALL (\w+): (\{.*\})")
# a JSON value: a quoted string (escapes allowed) or a number, right after a colon
VALUE_RE = re.compile(r':\s*("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?)')
KEY_RE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:')
PARTS = ("value", "key", "name", "text")


def part_of_each_char(turn: str) -> list[str]:
    """For every character of the turn: 'value' (argument value), 'key' (argument name), 'name' (tool
    name), 'layout' (the turn header, the call markers, JSON punctuation) or 'text'.

    Layout surprises the model in correct turns as much as in wrong ones, so it is kept apart."""
    parts = ["text"] * len(turn)
    header = re.match(r"\s*⟦[^⟧]*⟧\s*", turn)
    if header:
        parts[: header.end()] = ["layout"] * header.end()
    for m in CALL_RE.finditer(turn):
        parts[m.start() : m.start(1)] = ["layout"] * (m.start(1) - m.start())
        parts[m.end(1) : m.start(2)] = ["layout"] * (m.start(2) - m.end(1))
        parts[m.start(1) : m.end(1)] = ["name"] * (m.end(1) - m.start(1))
        body = m.group(2)
        parts[m.start(2) : m.end(2)] = ["layout"] * len(body)
        for k in KEY_RE.finditer(body):
            start = m.start(2) + k.start(1)
            parts[start : start + len(k.group(1))] = ["key"] * len(k.group(1))
        for v in VALUE_RE.finditer(body):
            start = m.start(2) + v.start(1)
            parts[start : start + len(v.group(1))] = ["value"] * len(v.group(1))
    return parts


def token_parts(turn: str, spans: list[tuple[int, int]]) -> list[str]:
    """The part of each token of the turn, from the characters each token covers ((start, end) offsets into
    the turn): the first of value, key, name, text found among them, else layout."""
    chars = part_of_each_char(turn)
    out = []
    for start, end in spans:
        covered = set(chars[max(start, 0) : max(end, start + 1)]) or {"text"}
        out.append(next((k for k in PARTS if k in covered), "layout"))
    return out


def summary(r: dict) -> np.ndarray:
    lp, h, m = (np.asarray(r[k], dtype=float) for k in ("logp", "entropy", "margin"))
    top = np.sort(h)[-10:]
    return np.array(
        [
            lp.mean(),
            lp.min(),
            lp.std(),
            np.quantile(lp, 0.05),
            np.quantile(lp, 0.25),
            (lp < np.log(0.1)).mean(),
            (lp < np.log(0.01)).mean(),
            h.mean(),
            h.max(),
            top.mean(),
            m.mean(),
            np.log(len(lp)),
        ]
    )


def part_summary(r: dict, parts: list[str]) -> np.ndarray:
    """For argument values, argument names, tool names and text (layout left out): count, lowest and
    mean log-probability, share below 0.01, mean and highest entropy."""
    lp, h, kinds = np.asarray(r["logp"], dtype=float), np.asarray(r["entropy"], dtype=float), np.asarray(parts)
    out = []
    for part in PARTS:
        x, e = lp[kinds == part], h[kinds == part]
        out += (
            [np.log1p(len(x)), x.min(), x.mean(), (x < np.log(0.01)).mean(), e.mean(), e.max()] if len(x) else [0.0] * 6
        )
    return np.array(out)


def features(r: dict, parts: list[str]) -> np.ndarray:
    return np.concatenate([summary(r), part_summary(r, parts)])
