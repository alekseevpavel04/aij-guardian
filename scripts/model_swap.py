"""Another model in place of Qwen3.6-35B-A3B as the source of features, on the organiser's contexts.

The submitted system reads two things from one pass of the model over a row: the last-token state (head B) and the
token probabilities of the agent's turn (the uncertainty head). Both come from scripts/embed_llama.py for either
model (`--chat` picks the template), so the two are compared on equal terms: the same 384 contexts, the same five
folds by dialogue as tau2ru_ruler.py, head B as LightGBM 300x7, the uncertainty head as a logistic regression on
the 36 token statistics, the system as ranks of B plus half the ranks of U with the rules on top. Simulations, τ²
and written pairs are left out: their rows have no features from the other model, and they add little next to the
contexts (+0.007 AUROC on the agreed contexts, EXPERIMENTS 28.09). The difference is paired, with a bootstrap over
dialogues (scripts/paired_variants.py), on the agreed contexts, v1s and Opus.

    python scripts/model_swap.py --a g4_tok --b q4_tok
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
from guardian.rules import Context, evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import FOLDS, Ruler  # noqa: E402
from head_combos import ranks  # noqa: E402
from head_models import lgbm  # noqa: E402
from paired_variants import compare  # noqa: E402
from train_head_ru import vectors  # noqa: E402
from turn_ruler import RULES  # noqa: E402
from uncertainty_source_grid import features  # noqa: E402

KEYS = ("system", "head_b", "head_u")


def scores(prefix: str, contexts: pd.DataFrame) -> dict[str, np.ndarray]:
    ruler = Ruler(contexts, vectors(f"{prefix}_final_org"), pd.DataFrame(), None)
    own = features(prefix)
    x, y, tasks = ruler.x, ruler.label["v1s"], ruler.tasks
    xu = np.stack([own[i] for i in ruler.ctx.id])
    fired = np.array(
        [
            any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
            for r in ruler.ctx.itertuples()
        ]
    )
    b, u = np.zeros(len(y)), np.zeros(len(y))
    for fit, held in GroupKFold(FOLDS).split(x, y, tasks):
        sc = StandardScaler().fit(x[fit])
        b[held] = lgbm().fit(sc.transform(x[fit]), y[fit]).predict_proba(sc.transform(x[held]))[:, 1]
        sc = StandardScaler().fit(xu[fit])
        m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(xu[fit]), y[fit])
        u[held] = m.predict_proba(sc.transform(xu[held]))[:, 1]
    return {
        "ids": ruler.ctx.id.to_numpy().astype(str),
        "system": ranks(b) + 0.5 * ranks(u) + 10 * fired,
        "head_b": b,
        "head_u": u,
    }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="g4_tok", help="token pass of the candidate model")
    ap.add_argument("--b", default="q4_tok", help="token pass of the model it would replace")
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "results/model_swap.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    names = {}
    for prefix in (args.a, args.b):
        found = scores(prefix, contexts)
        names[prefix] = f"model_swap_{prefix}"
        np.savez(ROOT / "results" / f"{names[prefix]}.scores.npz", **found)
    result = {"a": args.a, "b": args.b, "contexts": len(found["ids"]), "paired": {}}
    for key in KEYS:
        for view in ("agreed", "v1s", "opus"):
            result["paired"][f"{key}_{view}"] = compare(
                names[args.a], names[args.b], view, key=key, repeats=args.repeats
            )
    args.out.write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    for name, r in result["paired"].items():
        print(name, {k: v for k, v in r.items() if not isinstance(v, (list, dict))})


if __name__ == "__main__":
    main()
