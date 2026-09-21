"""An LLM judge scored by the probability of its one-token verdict.

The organiser's baseline asks a chat model for a single character, "1" (hallucination) or "0", and
turns the log-probabilities of the two tokens into P("1"). This client does the same over the plain
completions endpoint of any OpenAI-compatible server (llama-server locally, vLLM on the platform),
with the prompt rendered in the model's own ChatML template. Rendering it here, instead of letting
the server do it, is what makes a reasoning budget possible on a server that has none:

1. the reasoning is generated after "<think>\\n" until the model closes it or the budget runs out;
2. "</think>" is then written by us and the verdict is read as one token after it.

Without reasoning the template's empty think block is written directly and only step 2 runs. If
the server rejects a plain completion, the client switches, for good, to the chat endpoint with
thinking disabled: the exact request of the organiser's baseline, which is known to work on the
platform. The client uses only the standard library: `pip install .` runs without network.
"""

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

CLOSE_THINK = "\n</think>\n\n"


def user_text(prompt: str, response: str) -> str:
    return f"<prompt>\n{prompt}\n</prompt>\n<response>\n{response}\n</response>"


def render(system_prompt: str, prompt: str, response: str, thinking: bool) -> str:
    """The chat as the Qwen3.6 template renders it with add_generation_prompt (tests/test_judge.py)."""
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_text(prompt, response)}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    return text if thinking else text + CLOSE_THINK


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


def first_top_logprobs(logprobs: dict | None) -> list[tuple[str, float]]:
    """Top candidates of the first generated token, in llama-server's or vLLM's layout."""
    if not logprobs:
        return []
    if logprobs.get("content"):  # llama-server: [{"token", "logprob", "top_logprobs": [...]}]
        return [(t["token"], t["logprob"]) for t in logprobs["content"][0]["top_logprobs"]]
    if logprobs.get("top_logprobs"):  # OpenAI legacy completions (vLLM): [{token: logprob}]
        return list(logprobs["top_logprobs"][0].items())
    return []


@dataclass(frozen=True)
class Verdict:
    score: float | None
    prompt_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int | None
    seconds: float


class JudgeClient:
    def __init__(
        self,
        base_url: str,
        model: str = "judge",
        timeout: float = 3600,
        thinking: bool = False,
        thinking_tokens: int = 1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.thinking = thinking
        self.thinking_tokens = thinking_tokens
        self.use_chat = False

    def _post(self, endpoint: str, body: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/{endpoint}",
            data=json.dumps({"model": self.model, "temperature": 0.0, **body}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as reply:
            return json.loads(reply.read().decode("utf-8"))

    def _complete(self, prompt: str, **params) -> dict:
        return self._post("completions", {"prompt": prompt, **params})

    def _chat_verdict(self, system_prompt: str, prompt: str, response: str) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text(prompt, response)},
        ]
        body = {
            "messages": messages,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = self._post("chat/completions", body)
        # chat logprobs have the llama-server layout, so the same reader handles them
        return {"choices": [{"logprobs": data["choices"][0].get("logprobs")}], "usage": data.get("usage")}

    def score(self, system_prompt: str, prompt: str, response: str) -> Verdict:
        started = time.monotonic()
        if self.use_chat and not self.thinking:
            data = self._chat_verdict(system_prompt, prompt, response)
            return self._verdict(data, None, started)
        text = render(system_prompt, prompt, response, self.thinking)
        reasoning_tokens = None
        if self.thinking:
            thought = self._complete(text, max_tokens=self.thinking_tokens, stop=["</think>"])
            text += thought["choices"][0]["text"].rstrip() + CLOSE_THINK
            reasoning_tokens = (thought.get("usage") or {}).get("completion_tokens")
        try:
            data = self._complete(text, max_tokens=1, logprobs=20)
        except urllib.error.HTTPError:
            if self.thinking:
                raise
            self.use_chat = True
            data = self._chat_verdict(system_prompt, prompt, response)
        return self._verdict(data, reasoning_tokens, started)

    def _verdict(self, data: dict, reasoning_tokens: int | None, started: float) -> Verdict:
        top = first_top_logprobs(data["choices"][0].get("logprobs"))
        usage = data.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        return Verdict(
            p_one(top) if top else None,
            usage.get("prompt_tokens"),
            cached,
            reasoning_tokens,
            time.monotonic() - started,
        )
