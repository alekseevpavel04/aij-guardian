"""The final detector as it runs inside the submission: rules, head B, the uncertainty head, one threshold.

Head B is LightGBM on the judge model's last-token hidden state, trained three times on different shares of the
written pairs; its score is the sum of the three probabilities. The uncertainty head is a logistic regression on
36 summaries of the turn's token probabilities (guardian.uncertainty). Each head's score is turned into its rank
among the out-of-fold scores of the organiser's contexts, so a row's score does not depend on the other rows of
the file; the system score is the rank of head B plus half the rank of the uncertainty head, and a row is labelled
1 when a rule fires on it or its score reaches the threshold (scripts/train_final.py writes the
file this module reads).

LightGBM is not installed on the platform (pip runs without network), so its trees are exported as arrays and
walked here with numpy, which the platform has with vLLM.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Tree:
    """One tree of a LightGBM binary model in array form; node i is internal when left[i] >= 0."""

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    default_left: np.ndarray
    missing: np.ndarray  # 0 none, 1 zero, 2 NaN
    value: np.ndarray

    def raw(self, x: np.ndarray) -> np.ndarray:
        node = np.zeros(len(x), dtype=np.int64)
        rows = np.arange(len(x))
        while True:
            inner = self.left[node] >= 0
            if not inner.any():
                return self.value[node]
            n = node[inner]
            v = x[rows[inner], self.feature[n]]
            miss = (self.missing[n] == 2) & np.isnan(v) | (self.missing[n] == 1) & ((v == 0) | np.isnan(v))
            go_left = np.where(miss, self.default_left[n], v <= self.threshold[n])
            node[inner] = np.where(go_left, self.left[n], self.right[n])


def tree_arrays(structure: dict) -> dict:
    """A tree of `Booster.dump_model()` as the arrays of Tree (lists, for JSON)."""
    out = {k: [] for k in ("feature", "threshold", "left", "right", "default_left", "missing", "value")}

    def visit(node: dict) -> int:
        i = len(out["value"])
        for k in out:
            out[k].append(0)
        if "leaf_value" in node:
            out["left"][i] = out["right"][i] = -1
            out["value"][i] = float(node["leaf_value"])
            return i
        if node["decision_type"] != "<=":
            raise ValueError(f"unsupported split {node['decision_type']}")
        out["feature"][i] = int(node["split_feature"])
        out["threshold"][i] = float(node["threshold"])
        out["default_left"][i] = bool(node["default_left"])
        out["missing"][i] = {"None": 0, "Zero": 1, "NaN": 2}[node["missing_type"]]
        out["left"][i] = visit(node["left_child"])
        out["right"][i] = visit(node["right_child"])
        return i

    visit(structure)
    return out


@dataclass
class Boosted:
    """A standardisation and a LightGBM binary model: P(error) of each row."""

    mean: np.ndarray
    scale: np.ndarray
    trees: list[Tree]

    @classmethod
    def from_dict(cls, d: dict) -> "Boosted":
        trees = [
            Tree(
                np.asarray(t["feature"], dtype=np.int64),
                np.asarray(t["threshold"], dtype=np.float64),
                np.asarray(t["left"], dtype=np.int64),
                np.asarray(t["right"], dtype=np.int64),
                np.asarray(t["default_left"], dtype=bool),
                np.asarray(t["missing"], dtype=np.int64),
                np.asarray(t["value"], dtype=np.float64),
            )
            for t in d["trees"]
        ]
        return cls(np.asarray(d["mean"]), np.asarray(d["scale"]), trees)

    def proba(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return sigmoid(sum(t.raw(z) for t in self.trees))


@dataclass
class Linear:
    """A standardisation and a logistic regression."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def from_dict(cls, d: dict) -> "Linear":
        return cls(np.asarray(d["mean"]), np.asarray(d["scale"]), np.asarray(d["coef"]), float(d["intercept"]))

    def proba(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return sigmoid(z @ self.coef + self.intercept)


def rank_against(reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Share of the reference scores below each score (the reference sorted ascending)."""
    return np.searchsorted(reference, scores) / len(reference)


@dataclass
class Final:
    head_b: list[Boosted]
    head_u: Linear
    reference_b: np.ndarray
    reference_u: np.ndarray
    weight_u: float
    threshold: float
    # the judge's verdict read off the same last state: state . (output row of "1" - row of "0"), ranked against the
    # contexts' own judge scores; absent in the archives of submissions 7-9
    judge_direction: np.ndarray | None = None
    reference_j: np.ndarray | None = None
    weight_j: float = 0.0

    @classmethod
    def load(cls, path: Path) -> "Final":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, d: dict) -> "Final":
        return cls(
            [Boosted.from_dict(b) for b in d["head_b"]],
            Linear.from_dict(d["head_u"]),
            np.sort(np.asarray(d["reference_b"], dtype=np.float64)),
            np.sort(np.asarray(d["reference_u"], dtype=np.float64)),
            float(d["weight_u"]),
            float(d["threshold"]),
            np.asarray(d["judge_direction"], dtype=np.float64) if "judge_direction" in d else None,
            np.sort(np.asarray(d["reference_j"], dtype=np.float64)) if "reference_j" in d else None,
            float(d.get("weight_j", 0.0)),
        )

    def score(self, vectors: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
        """The system score of each row without the rules: vectors (n, hidden), uncertainty features (n, 36)."""
        b = sum(h.proba(vectors) for h in self.head_b)
        u = self.head_u.proba(uncertainty)
        score = rank_against(self.reference_b, b) + self.weight_u * rank_against(self.reference_u, u)
        if self.judge_direction is not None:
            score = score + self.weight_j * rank_against(self.reference_j, vectors @ self.judge_direction)
        return score

    def labels(self, scores: np.ndarray, fired: list[int]) -> list[int]:
        return [int(f or s >= self.threshold) for s, f in zip(scores, fired, strict=True)]
