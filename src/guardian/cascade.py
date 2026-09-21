"""Rules first, then the judge: the submission's decision for a whole test file.

A row is labelled 1 when a deterministic check fires (13 true of 20 fired on the public test) or when
the judge's score is among the top `share` of the scored rows. Ranking within the test file instead of
a fixed threshold is deliberate: the baseline, with this model, flagged 254 of 548 rows at P = 0.5
while 130 are positive, so the absolute scale shifts between data sets, and between the quantised
model used for development and the FP8 one on the platform. A share does not depend on that scale.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from guardian.judge import JudgeClient
from guardian.rules import Context, evidence


def top_share(scores: list[float | None], share: float) -> list[int]:
    """1 for the `share` of scored rows with the highest scores; unscored rows get 0. Ties at the
    cut go in, so the count can exceed the share by the size of the tie."""
    scored = sorted((s for s in scores if s is not None), reverse=True)
    count = round(share * len(scored))
    if count == 0:
        return [0] * len(scores)
    cut = scored[count - 1]
    return [int(s is not None and s >= cut) for s in scores]


def rule_labels(rows: list[dict], checks: tuple[str, ...]) -> list[int]:
    labels = []
    for row in rows:
        prompt, response = row.get("prompt", ""), row.get("response", "")
        found = evidence(prompt, response, Context.from_prompt(prompt))
        labels.append(int(any(found[name] for name in checks)))
    return labels


@dataclass
class JudgeRun:
    scores: list[float | None]
    errors: int = 0
    seconds: float = 0.0


def judge_scores(
    rows: list[dict],
    client: JudgeClient,
    system_prompt: str,
    deadline: float,
    workers: int,
    order: list[int] | None = None,
) -> JudgeRun:
    """Score rows with the judge, in `order` if given, until the monotonic `deadline`; the rest stay None."""
    started = time.monotonic()
    scores: list[float | None] = [None] * len(rows)
    errors = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {
        pool.submit(client.score, system_prompt, rows[i].get("prompt", ""), rows[i].get("response", "")): i
        for i in (order if order is not None else range(len(rows)))
    }
    try:
        for future in as_completed(futures, timeout=max(deadline - time.monotonic(), 1)):
            i = futures[future]
            try:
                scores[i] = future.result().score
            except Exception as error:  # one failed request must not cost the run; the row falls back to rules
                errors += 1
                if errors <= 5:
                    print(f"judge row {i}: {type(error).__name__}: {error}", file=sys.stderr)
    except TimeoutError:
        print("judge deadline reached, unscored rows fall back to the rules", file=sys.stderr)
    pool.shutdown(wait=False, cancel_futures=True)
    return JudgeRun(scores, errors, time.monotonic() - started)


def closest_to_cut(scores: list[float | None], share: float) -> list[int]:
    """Indices of scored rows ordered by how far their rank is from the top-`share` cut: the rows whose
    label a second, slower opinion is most likely to change come first."""
    ranked = sorted((i for i, s in enumerate(scores) if s is not None), key=lambda i: -scores[i])
    cut = round(share * len(ranked))
    return [i for _, i in sorted((abs(rank - cut), i) for rank, i in enumerate(ranked))]


def merge(first: list[float | None], second: list[float | None]) -> list[float | None]:
    """The mean of the quick and the reasoned score where both exist, the quick one elsewhere.

    Not the reasoned score alone: it sits near 0 or 1 and loses the order inside each group. Locally
    it helped on valid.parquet (AUROC 0.913 against 0.817 quick) and hurt on TrajErrBench (0.816
    against 0.942); the mean kept most of the first and lost little of the second (0.853 and 0.913).
    """
    return [a if b is None else b if a is None else (a + b) / 2 for a, b in zip(first, second, strict=True)]


def combine(rules: list[int], judge: list[int]) -> list[int]:
    return [int(a or b) for a, b in zip(rules, judge, strict=True)]
