"""The home ruler: heads compared out of fold on the organiser's contexts, with labels both labellers agree on.

The 46 labelled rows are a curated 50/50 sample (a tool call alone gives AUROC 0.913 on them) and ext_val
comes from other agents with other labelling rules; both picked the wrong head at least once against
the test. The organiser's 384 contexts are rows of the same kind as the test (the same dialogues and
agent, a history and turn N) with 16-22% positives. Here every recipe is trained on 4/5 of the 40
dialogues and scores the held-out fifth (written pairs only from the training dialogues), so every
context gets an out-of-fold score. The measure is AUROC on the contexts where v1s and Opus agree, so
neither teacher is favoured, and a recipe is compared with head B by a paired bootstrap over dialogues.

Checked against the test: it ranks head A far below B (-0.064 [-0.094; -0.029]; test AUROC about 0.66
against 0.76) and B without pairs level with B (test 0.500 against 0.513).

    python scripts/context_ruler.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from judge_eval import load  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import PAIR_VECTORS, TEACHERS, auroc, labels, vectors  # noqa: E402

C = 0.003
DRAWS = 3
FOLDS = 5
V3_V4 = [
    *("airline_02", "retail_02a", "retail_02b", "telecom_02a", "telecom_02b", "banking_02a", "banking_02b"),
    *(f"airline_03{s}" for s in "abcdef"),
    *("retail_03", "telecom_03a", "telecom_03b", "banking_03a", "banking_03b", "banking_03f"),
]


class Ruler:
    def __init__(self, contexts: pd.DataFrame, vecs: dict, pairs: pd.DataFrame, external: tuple | None) -> None:
        self.ctx = contexts[contexts.split.isin(["train", "check"])].reset_index(drop=True)
        self.vecs, self.pairs, self.external = vecs, pairs, external
        self.x = np.stack([vecs[i] for i in self.ctx.id])
        self.tasks = self.ctx.task.to_numpy()
        self.label = {}
        for teacher in TEACHERS:
            found = {}
            for split in ("train", "check"):
                found |= labels(TEACHERS[teacher].format(split=split))
            self.label[teacher] = self.ctx.id.map(found).to_numpy().astype(int)
        self.calls = self.ctx.response.str.contains("TOOL_CALL").to_numpy()
        self.agree = self.label["v1s"] == self.label["opus"]
        self.truth = self.label["v1s"]
        self.groups = [np.where(self.tasks == t)[0] for t in np.unique(self.tasks)]

    def scores(
        self, y: np.ndarray, pair_share: float, pair_filter=None, external_weight: float | None = None
    ) -> np.ndarray:
        out = np.zeros(len(self.ctx))
        for fit, held in GroupKFold(FOLDS).split(self.x, y, self.tasks):
            held_tasks = set(self.tasks[held])
            total = np.zeros(len(held))
            for seed in range(DRAWS if pair_share else 1):
                xs, ys = [self.x[fit]], [y[fit]]
                if pair_share:
                    p = self.pairs[~self.pairs.task.isin(held_tasks)]
                    if pair_filter is not None:
                        p = p[pair_filter(p)]
                    ids = np.array(sorted(p.context_id.unique()))
                    keep = np.random.default_rng(seed).choice(ids, round(pair_share * len(ids)), replace=False)
                    p = p[p.context_id.isin(keep)]
                    if len(p):
                        xs.append(np.stack([self.vecs[i] for i in p.id]))
                        ys.append(p.label.to_numpy())
                n_russian = sum(map(len, ys))
                weights = [np.ones(n_russian)]
                if external_weight is not None:
                    x_ext, y_ext = self.external
                    xs.append(x_ext)
                    ys.append(y_ext)
                    weights.append(np.full(len(y_ext), external_weight * n_russian / len(y_ext)))
                xf, yf, wf = np.vstack(xs), np.concatenate(ys), np.concatenate(weights)
                scaler = StandardScaler().fit(xf)
                model = LogisticRegression(C=C, max_iter=5000).fit(scaler.transform(xf), yf, sample_weight=wf)
                total += model.decision_function(scaler.transform(self.x[held]))
            out[held] = total
        return out

    def auroc_ci(self, s: np.ndarray, repeats: int = 1000) -> dict:
        own = [g[self.agree[g]] for g in self.groups if self.agree[g].any()]
        rng, values = np.random.default_rng(0), []
        for _ in range(repeats):
            idx = np.concatenate([own[k] for k in rng.integers(0, len(own), len(own))])
            if 0 < self.truth[idx].sum() < len(idx):
                values.append(auroc(self.truth[idx], s[idx]))
        return {
            "auroc": auroc(self.truth[self.agree], s[self.agree]),
            "ci95": [round(float(v), 3) for v in np.percentile(values, [2.5, 97.5])],
        }

    def paired(self, s: np.ndarray, base: np.ndarray, repeats: int = 1000) -> dict:
        own = [g[self.agree[g]] for g in self.groups if self.agree[g].any()]
        rng, diffs = np.random.default_rng(0), []
        for _ in range(repeats):
            idx = np.concatenate([own[k] for k in rng.integers(0, len(own), len(own))])
            if 0 < self.truth[idx].sum() < len(idx):
                diffs.append(auroc(self.truth[idx], s[idx]) - auroc(self.truth[idx], base[idx]))
        diffs = np.array(diffs)
        return {
            "difference": round(
                auroc(self.truth[self.agree], s[self.agree]) - auroc(self.truth[self.agree], base[self.agree]), 4
            ),
            "ci95": [round(float(v), 3) for v in np.percentile(diffs, [2.5, 97.5])],
            "share_above_zero": round(float((diffs > 0).mean()), 3),
        }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--extra-pairs", nargs="*", default=[], help="more pair batches, compared as their own variant")
    ap.add_argument("--out", type=Path, default=ROOT / "results/context_ruler.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org") | vectors("q4_last_extdev1000")
    for name in PAIR_VECTORS:
        if (ROOT / "data/interim/embeddings" / f"{name}.jsonl").exists():
            vecs |= vectors(name)
    dev = load("ext_dev", 1000, 0)
    external = (np.stack([vecs[i] for i in dev.id]), dev.label.to_numpy().astype(int))
    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), external)
    v1s, opus = ruler.label["v1s"], ruler.label["opus"]

    base = ruler.scores(v1s, 0.25)
    variants = {
        "head_a_opus_calls_only_25pct_pairs": ruler.scores(opus & ruler.calls, 0.25),
        "head_b0_v1s_no_pairs": ruler.scores(v1s, 0.0),
        "v1s_10pct_pairs": ruler.scores(v1s, 0.10),
        "v1s_50pct_pairs": ruler.scores(v1s, 0.50),
        "v1s_all_pairs": ruler.scores(v1s, 1.0),
        "opus_all_labels_25pct_pairs": ruler.scores(opus, 0.25),
        "v1s_25pct_pairs_plus_ext_dev_weight_0_5": ruler.scores(v1s, 0.25, external_weight=0.5),
    }
    if args.extra_pairs:
        extra = written(args.extra_pairs, contexts)
        ruler_extra = Ruler(contexts, vecs, pd.concat([ruler.pairs, extra]), external)
        variants["v1s_25pct_pairs_plus_" + "_".join(sorted({b.split("_")[1] for b in args.extra_pairs}))] = (
            ruler_extra.scores(v1s, 0.25)
        )
    result = {
        "rows": len(ruler.ctx),
        "agreed_rows": int(ruler.agree.sum()),
        "agreed_positives": int(ruler.truth[ruler.agree].sum()),
        "dialogues": len(ruler.groups),
        "head_b_v1s_25pct_pairs": ruler.auroc_ci(base),
        "variants": {
            name: {**ruler.auroc_ci(s), "minus_head_b": ruler.paired(s, base)} for name, s in variants.items()
        },
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
