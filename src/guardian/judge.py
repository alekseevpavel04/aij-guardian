"""An LLM judge scored by the probability of its one-token verdict.

The organiser's baseline asks a chat model for a single character, "1" (hallucination) or "0", and
turns the log-probabilities of the two tokens into P("1"). The same scoring works against any
OpenAI-compatible server: llama-server locally, vLLM on the platform. The client uses only the
standard library, because `pip install .` on the platform runs without network and a dependency
that is not already installed would not arrive.
"""

import json
import math
import time
import urllib.request
from dataclasses import dataclass


def build_messages(system_prompt: str, prompt: str, response: str) -> list[dict]:
    user = f"<prompt>\n{prompt}\n</prompt>\n<response>\n{response}\n</response>"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def p_one(top: list[tuple[str, float]]) -> float | None:
    """P("1") from the top log-probabilities of the verdict position, renormalised over "1" and "0"."""

    def best(label: str) -> float | None:
        values = [lp for token, lp in top if token.strip() == label]
        return max(values) if values else None

    one, zero = best("1"), best("0")
    if one is None and zero is None:
        return None
    if zero is None:
        return 1.0
    if one is None:
        return 0.0
    m = max(one, zero)
    return math.exp(one - m) / (math.exp(one - m) + math.exp(zero - m))


@dataclass(frozen=True)
class Verdict:
    score: float | None
    prompt_tokens: int | None
    cached_tokens: int | None
    seconds: float


class JudgeClient:
    def __init__(self, base_url: str, model: str = "judge", timeout: float = 3600, thinking: bool = False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.thinking = thinking

    def score(self, messages: list[dict]) -> Verdict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout) as reply:
            data = json.loads(reply.read().decode("utf-8"))
        seconds = time.monotonic() - started
        content = (data["choices"][0].get("logprobs") or {}).get("content") or []
        score = None
        for position in content:
            if position["token"].strip() in ("0", "1"):
                score = p_one([(t["token"], t["logprob"]) for t in position["top_logprobs"]])
                break
        if score is None and content:
            score = p_one([(t["token"], t["logprob"]) for t in content[0]["top_logprobs"]])
        usage = data.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            cached = (data.get("timings") or {}).get("cache_n")
        return Verdict(score, usage.get("prompt_tokens"), cached, seconds)
