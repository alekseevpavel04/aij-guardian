"""The uncertainty head trained with each new source, in every combination and all together, on the context ruler.

The head of turn_ruler.py learns from the organiser's contexts only. Here written error turns (pairs
v5 with a tool call, v6, v7; their correct side is the original context, already in the base) and
external rows (τ² steps
of ext_dev, invented worlds) that went through the token pass are added, fold by fold: a held-out
dialogue's written turns never train its fold, external rows weigh half of the Russian rows. Each
combination is measured alone and in the system (rules first, then the mean of ranks of head B0 and the
uncertainty head at weight 0.5). A source without a token pass is left out.

    python scripts/uncertainty_source_grid.py
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from guardian import journal
from guardian.env import setup
from guardian.rules import Context, evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import FOLDS, Ruler  # noqa: E402
from head_combos import ranks  # noqa: E402
from judge_eval import load  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import vectors  # noqa: E402
from turn_ruler import RULES, part_summary, summary  # noqa: E402

EMB = ROOT / "data/interim/embeddings"


def features(prefix: str) -> dict[str, np.ndarray]:
    stats = {r["id"]: r for r in journal.read(EMB / f"{prefix}_uncertainty_org.jsonl")}
    return {
        r["id"]: np.concatenate([summary(stats[r["id"]]), part_summary(stats[r["id"]], r["parts"])])
        for r in journal.read(EMB / f"{prefix}_parts_org.jsonl")
        if r["id"] in stats
    }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/uncertainty_source_grid.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    no_pairs = pd.DataFrame(columns=["id", "task", "context_id", "label"])
    ruler = Ruler(contexts, vectors("q4_last_org"), no_pairs, None)
    y, tasks = ruler.label["v1s"], ruler.tasks
    base_b0 = ruler.scores(y, 0.0)
    fired = np.array(
        [
            any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
            for r in ruler.ctx.itertuples()
        ]
    )
    submitted = ranks(base_b0) + 10 * fired
    own = features("q4_tok")
    x = np.stack([own[i] for i in ruler.ctx.id])

    sources: dict[str, tuple[str, np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for name, prefix, batches in (
        ("pairs_v5_calls", "q4_tok_v5", "*_05*"),
        ("pairs_v6", "q4_tok_v6", "*_06*"),
        ("pairs_v7", "q4_tok_v7", "*_07*"),
    ):
        if not (EMB / f"{prefix}_parts_org.jsonl").exists():
            continue
        f = features(prefix)
        rows = written(
            sorted(p.name for p in (ROOT / "data/synthetic/pairs").glob(batches)), contexts, ("train", "check")
        )
        rows = rows[rows.id.isin(f)]
        sources[name] = ("pairs", np.stack([f[i] for i in rows.id]), rows.label.to_numpy(), rows.task.to_numpy())
    if (EMB / "q4_tok_extdev_parts_org.jsonl").exists():
        f = features("q4_tok_extdev")
        labels = load("ext_dev", None, 0).set_index("id").label
        ids = [i for i in f if i in labels.index]
        sources["tau2_external"] = ("external", np.stack([f[i] for i in ids]), labels.loc[ids].to_numpy(), None)
    if (EMB / "q4_tok_worlds_parts_org.jsonl").exists():
        f = features("q4_tok_worlds")
        rows = pd.read_parquet(ROOT / "data/synthetic/worlds/worlds_rows.parquet")
        rows = rows[rows.id.isin(f)]
        sources["worlds"] = ("external", np.stack([f[i] for i in rows.id]), rows.label.to_numpy(), None)

    def score(chosen: tuple[str, ...]) -> np.ndarray:
        out = np.zeros(len(y))
        for fit, held in GroupKFold(FOLDS).split(x, y, tasks):
            held_tasks = set(tasks[held])
            xs, ys = [x[fit]], [y[fit]]
            for s in chosen:
                kind, xf, yf, t = sources[s]
                if kind == "pairs":
                    keep = ~np.isin(t, list(held_tasks))
                    xs.append(xf[keep])
                    ys.append(yf[keep])
            n = sum(map(len, ys))
            ws = [np.ones(n)]
            ext = [sources[s] for s in chosen if sources[s][0] == "external"]
            if ext:
                xe, ye = np.vstack([e[1] for e in ext]), np.concatenate([e[2] for e in ext])
                xs.append(xe)
                ys.append(ye)
                ws.append(np.full(len(ye), 0.5 * n / len(ye)))
            xa, ya, wa = np.vstack(xs), np.concatenate(ys).astype(int), np.concatenate(ws)
            sc = StandardScaler().fit(xa)
            m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(xa), ya, sample_weight=wa)
            out[held] = m.predict_proba(sc.transform(x[held]))[:, 1]
        return out

    ref = score(())
    result = {
        "sources": {k: {"rows": len(v[2]), "positives": int(v[2].sum())} for k, v in sources.items()},
        "contexts_only": {
            "alone": ruler.auroc_ci(ref)["auroc"],
            "system": ruler.paired(submitted + 0.5 * ranks(ref), submitted),
        },
        "variants": {},
    }
    names = sorted(sources)
    for combo in (c for r in range(1, len(names) + 1) for c in itertools.combinations(names, r)):
        s = score(combo)
        result["variants"]["+".join(combo)] = {
            "alone": ruler.auroc_ci(s)["auroc"],
            "system_minus_contexts_only_system": ruler.paired(submitted + 0.5 * ranks(s), submitted + 0.5 * ranks(ref)),
        }
        print("+".join(combo), json.dumps(result["variants"]["+".join(combo)]), flush=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
