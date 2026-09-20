"""The predictor that runs inside the submission.

Two strategies, selected by `model/config.json`:

- `constant`: always the same label. Not a model: it is how the packaging is tested and how the
  share of positives in the test set is measured, since a leaderboard row of an all-ones submission
  gives the test size directly (docs/DECISIONS.md).
- `rules`: fire when any of the deterministic checks finds something (guardian/rules.py).

Both are deterministic and need no weights.
"""

import json
from pathlib import Path

from guardian.rules import CHECKS, Context, evidence

DEFAULT_CHECKS = tuple(CHECKS)  # every call-level check; the number check is noise on its own


class Predictor:
    def __init__(self, strategy: str = "rules", label: int = 1, checks: tuple[str, ...] = DEFAULT_CHECKS):
        if strategy not in ("constant", "rules"):
            raise ValueError(f"unknown strategy {strategy!r}")
        self.strategy = strategy
        self.label = int(label)
        self.checks = tuple(checks)

    @classmethod
    def from_config(cls, path: str | Path) -> "Predictor":
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            strategy=config.get("strategy", "rules"),
            label=config.get("label", 1),
            checks=tuple(config.get("checks", DEFAULT_CHECKS)),
        )

    @property
    def default_label(self) -> int:
        """Label given to an example the predictor failed on, so one bad row cannot lose the run."""
        return self.label if self.strategy == "constant" else 0

    def predict(self, prompt: str, response: str) -> int:
        if self.strategy == "constant":
            return self.label
        found = evidence(prompt, response, Context.from_prompt(prompt))
        return int(any(found[name] for name in self.checks))
