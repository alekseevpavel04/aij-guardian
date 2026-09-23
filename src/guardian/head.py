"""The linear head on the judge model's last-token hidden state, as it runs inside the submission.

The vector is the final normalised hidden state of the last token of the judge's own prompt
(guardian.judge.render without reasoning), the position where the judge would write its verdict.
Locally it comes from llama-server in embedding mode, on the platform from vLLM in pooling mode
(runner pooling, convert embed, LAST pooling, no L2 normalisation); both return that same state.
The head is a standardisation and a logistic regression exported by
scripts/export_head.py; only the order of the scores matters, since the submission labels the top
share of rows. Standard library only: `pip install .` runs without network.
"""

import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from guardian.judge import render


@dataclass
class Head:
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float

    @classmethod
    def load(cls, path: Path) -> "Head":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["mean"], data["scale"], data["coef"], data["intercept"])

    def score(self, vector: list[float]) -> float:
        if len(vector) != len(self.coef):
            raise ValueError(f"vector of {len(vector)} values for a head of {len(self.coef)}")
        z = self.intercept + sum(
            c * (v - m) / s for c, v, m, s in zip(self.coef, vector, self.mean, self.scale, strict=True)
        )
        return 1 / (1 + math.exp(-z)) if z > -700 else 0.0


def embed(url: str, model: str, text: str, timeout: float) -> list[float]:
    # the text already carries the chat template's special tokens, nothing is to be added to it
    body = json.dumps({"model": model, "input": text, "add_special_tokens": False}).encode()
    request = urllib.request.Request(f"{url}/embeddings", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.loads(reply.read())["data"][0]["embedding"]


@dataclass
class HeadRun:
    scores: list[float | None]
    errors: int = 0
    seconds: float = 0.0


def head_scores(
    rows: list[dict], head: Head, url: str, model: str, system_prompt: str, deadline: float, workers: int
) -> HeadRun:
    """Score rows until the monotonic `deadline`; rows not scored by then, or failed, stay None."""
    started = time.monotonic()
    scores: list[float | None] = [None] * len(rows)
    errors = 0

    def one(i: int) -> float:
        text = render(system_prompt, rows[i].get("prompt", ""), rows[i].get("response", ""), thinking=False)
        return head.score(embed(url, model, text, timeout=max(deadline - time.monotonic(), 1)))

    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(one, i): i for i in range(len(rows))}
    try:
        for future in as_completed(futures, timeout=max(deadline - time.monotonic(), 1)):
            i = futures[future]
            try:
                scores[i] = future.result()
            except Exception as error:  # one failed request must not cost the run; the row falls back to rules
                errors += 1
                if errors <= 5:
                    print(f"head row {i}: {type(error).__name__}: {error}", file=sys.stderr)
    except TimeoutError:
        print("head deadline reached, unscored rows fall back to the rules", file=sys.stderr)
    pool.shutdown(wait=False, cancel_futures=True)
    return HeadRun(scores, errors, time.monotonic() - started)
