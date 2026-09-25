"""Every new training source for head B, alone, in every combination and all together, on the context ruler.

Sources: written error turns in real contexts (pairs v5 "one error", v6 "errors in the text", v7
"critical errors"), dialogues in invented worlds (world_writer_v1, both versions of the last turn),
and agents of other environments (TrajDebug's ALFWorld, WebShop, GAIA, SWE-bench Pro). The written
turns of a held-out dialogue never train its fold. Every head has the base of head B (the contexts'
v1s labels and a quarter of the v3/v4 pairs); the external rows (the τ² steps, the worlds, the other
environments) together weigh half of the Russian rows. Each combination is measured with and without
the τ² steps, for the logistic head and LightGBM, alone and in the system (rules first, then the mean
of ranks of the head and the uncertainty head at weight 0.5). A source whose vectors are not there
yet is left out.

    python scripts/source_grid.py
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
from context_ruler import FOLDS, V3_V4, C, Ruler  # noqa: E402
from head_combos import ranks  # noqa: E402
from head_models import lgbm  # noqa: E402
from judge_eval import load  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import vectors  # noqa: E402
from turn_ruler import RULES, part_summary, summary  # noqa: E402

EMB = ROOT / "data/interim/embeddings"
PAIR_SOURCES = {
    "pairs_v5": (
        "q4_last_pairs_05",
        [
            f"{d}_05{s}"
            for d, ss in (("airline", "ab"), ("retail", "ab"), ("telecom", "abcde"), ("banking", "abc"))
            for s in ss
        ],
    ),
    "pairs_v6": ("q4_last_pairs_06", "*_06*"),
    "pairs_v7": ("q4_last_pairs_07", "*_07*"),
}
EXTERNAL_SOURCES = {
    "worlds": ("q4_last_worlds", ROOT / "data/synthetic/worlds/worlds_rows.parquet"),
    "trajdebug": ("q4_last_trajdebug_more", ROOT / "data/external/trajdebug_more.parquet"),
}


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/source_grid.json")
    ap.add_argument("--models", nargs="+", default=["logistic", "lightgbm"])
    ap.add_argument("--all-combos", action="store_true", help="every combination for LightGBM too")
    ap.add_argument("--only", nargs="*", default=None, help="measure only these combinations, e.g. pairs_v5+trajdebug")
    ap.add_argument("--first-seed", type=int, default=0, help="another draw of the quarter of pairs, to recheck a finding")
    ap.add_argument("--with-tau", choices=("both", "yes", "no"), default="both")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org") | vectors("q4_last_extdev_all")
    for name in ("q4_last_pairs_02", "q4_last_pairs_03"):
        vecs |= vectors(name)
    sources: dict[str, pd.DataFrame] = {}
    for name, (vec_name, batches) in PAIR_SOURCES.items():
        if not (EMB / f"{vec_name}.jsonl").exists():
            continue
        vecs |= vectors(vec_name)
        if isinstance(batches, str):
            batches = sorted(p.name for p in (ROOT / "data/synthetic/pairs").glob(batches))
        frame = written(batches, contexts, ("train", "check"))
        sources[name] = frame[frame.id.isin(vecs)].assign(kind_of_source="pairs")
    for name, (vec_name, path) in EXTERNAL_SOURCES.items():
        if not (EMB / f"{vec_name}.jsonl").exists() or not path.exists():
            continue
        vecs |= vectors(vec_name)
        frame = pd.read_parquet(path)
        sources[name] = frame[frame.id.isin(vecs)].assign(kind_of_source="external", task="")
    tau = load("ext_dev", None, 0)
    tau = tau[tau.id.isin(vecs)]

    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), None)
    y, x, tasks = ruler.label["v1s"], ruler.x, ruler.tasks
    base_pairs = ruler.pairs
    no_pairs = pd.DataFrame(columns=["id", "task", "context_id", "label"])
    stats = {r["id"]: r for r in journal.read(EMB / "q4_tok_uncertainty_org.jsonl")}
    feats = {
        r["id"]: np.concatenate([summary(stats[r["id"]]), part_summary(stats[r["id"]], r["parts"])])
        for r in journal.read(EMB / "q4_tok_parts_org.jsonl")
    }
    unc = Ruler(contexts, feats, no_pairs, None).scores(y, 0.0, make=lambda: LogisticRegression(C=1.0, max_iter=5000))
    fired = np.array(
        [
            any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
            for r in ruler.ctx.itertuples()
        ]
    )

    def score(chosen: tuple[str, ...], with_tau: bool, model: str) -> np.ndarray:
        out = np.zeros(len(y))
        for fit, held in GroupKFold(FOLDS).split(x, y, tasks):
            held_tasks = set(tasks[held])
            total = np.zeros(len(held))
            for seed in range(args.first_seed, args.first_seed + 3):
                p = base_pairs[~base_pairs.task.isin(held_tasks)]
                keep = np.random.default_rng(seed).choice(
                    sorted(p.context_id.unique()), round(0.25 * p.context_id.nunique()), replace=False
                )
                russian = [p[p.context_id.isin(keep)]] + [
                    sources[s][~sources[s].task.isin(held_tasks)]
                    for s in chosen
                    if sources[s].kind_of_source.iloc[0] == "pairs"
                ]
                ext = ([tau] if with_tau else []) + [
                    sources[s] for s in chosen if sources[s].kind_of_source.iloc[0] == "external"
                ]
                rus = pd.concat(russian)
                xs, ys = [x[fit], np.stack([vecs[i] for i in rus.id])], [y[fit], rus.label.to_numpy()]
                n = len(fit) + len(rus)
                ws = [np.ones(n)]
                if ext:
                    e = pd.concat(ext)
                    xs.append(np.stack([vecs[i] for i in e.id]))
                    ys.append(e.label.to_numpy())
                    ws.append(np.full(len(e), 0.5 * n / len(e)))
                xf, yf, wf = np.vstack(xs), np.concatenate(ys).astype(int), np.concatenate(ws)
                sc = StandardScaler().fit(xf)
                m = LogisticRegression(C=C, max_iter=5000) if model == "logistic" else lgbm()
                m.fit(sc.transform(xf), yf, sample_weight=wf)
                xh = sc.transform(x[held])
                total += m.decision_function(xh) if model == "logistic" else m.predict_proba(xh)[:, 1]
            out[held] = total
        return out

    def system(head: np.ndarray) -> np.ndarray:
        return ranks(head) + 0.5 * ranks(unc) + 10 * fired

    names = sorted(sources)
    result: dict = {
        "sources": {k: {"rows": len(v), "positives": int(v.label.sum())} for k, v in sources.items()},
        "runs": {},
    }
    references = {}
    for model in args.models:
        for with_tau in {"both": (True, False), "yes": (True,), "no": (False,)}[args.with_tau]:
            ref = score((), with_tau, model)
            references[(model, with_tau)] = ref
            tag = f"{model}_{'tau' if with_tau else 'no_tau'}"
            result["runs"][tag] = {
                "reference": {"alone": ruler.auroc_ci(ref)["auroc"], "system": ruler.auroc_ci(system(ref))["auroc"]},
                "variants": {},
            }
            combos = [c for r in range(1, len(names) + 1) for c in itertools.combinations(names, r)]
            if model == "lightgbm" and not args.all_combos:  # the slow model: single sources and all of them
                combos = [c for c in combos if len(c) in (1, len(names))]
            if args.only is not None:
                combos = [tuple(c.split("+")) for c in args.only]
            for combo in combos:
                s = score(combo, with_tau, model)
                result["runs"][tag]["variants"]["+".join(combo)] = {
                    "alone": ruler.auroc_ci(s)["auroc"],
                    "alone_minus_reference": ruler.paired(s, ref)["difference"],
                    "system_minus_reference_system": ruler.paired(system(s), system(ref)),
                }
                print(tag, "+".join(combo), json.dumps(result["runs"][tag]["variants"]["+".join(combo)]), flush=True)
            args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
