"""The rows of the Russian τ² simulations as training data for head B and the uncertainty head, on the context ruler.

The simulations (scripts/simulate_tau2.py, Qwen3-8B as agent and user) are labelled by two labellers
with the task reference: Opus (data/synthetic/opus_labels_tau2ru) and the local v1s labeller
(data/interim/labels/labeler_v1s_tau2ru.jsonl). A row gets a label only where the chosen mode has
one: `agreed` keeps rows where both labellers exist and agree, `opus` and `v1s` take one labeller.
A simulation of one of the organiser's tasks never trains the fold where that task is held out.

The reference is the candidate: rules first, then the mean of ranks of head B (LightGBM with a
quarter of the v3/v4 pairs and the τ² steps, three seeds) and the uncertainty head at weight 0.5.
The simulation rows are added to head B, to the uncertainty head, or to both, at weight 1 or 0.5 of
a natural row; the logistic head B is measured too. Before that, a short comparison of the rows
with the organiser's contexts: length, share of turns with a tool call, share of errors, mean log
probability of the turn's tokens.

    python scripts/tau2ru_ruler.py --labels agreed
"""

import argparse
import difflib
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
from guardian.tau2_tasks import CUT, data_id

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import FOLDS, V3_V4, C, Ruler  # noqa: E402
from head_combos import ranks  # noqa: E402
from head_models import lgbm  # noqa: E402
from judge_eval import load  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import at_share, auroc, vectors  # noqa: E402
from turn_ruler import RULES  # noqa: E402
from uncertainty_source_grid import features  # noqa: E402

EMB = ROOT / "data/interim/embeddings"
CJK = "[぀-ヿ㐀-鿿]"  # kana and CJK ideographs


def simulation_labels(
    mode: str, rows: pd.DataFrame, opus_dir: Path, v1s_name: str, only_double: bool = False
) -> pd.Series:
    opus = {r["id"]: int(r["label"]) for f in opus_dir.glob("*/opus.jsonl") for r in journal.read(f)}
    path = ROOT / "data/interim/labels" / f"{v1s_name}.jsonl"
    v1s = {r["id"]: int(r["score"] >= 0.5) for r in journal.read(path)} if path.exists() else {}
    if only_double:  # compare labellers on the same turns: only turns both have labelled
        opus = {i: v for i, v in opus.items() if i in v1s}
        v1s = {i: v for i, v in v1s.items() if i in opus}
    if mode == "opus":
        found = opus
    elif mode == "v1s":
        found = v1s
    elif mode == "both":  # a disagreement becomes 0.5 and enters twice, as 0 and as 1, at half weight
        found = {i: (opus[i] + v1s[i]) / 2 for i in opus if i in v1s}
    else:
        found = {i: opus[i] for i in opus if i in v1s and opus[i] == v1s[i]}
    return rows.id.map(found)


def mean_logp(prefix: str) -> dict[str, float]:
    stats = journal.read(EMB / f"{prefix}_uncertainty_org.jsonl")
    return {r["id"]: float(np.mean(r["logp"])) for r in stats if len(r["logp"])}


def repeated_turns(sim: pd.DataFrame) -> np.ndarray:
    """True for a turn whose text (without its turn marker) nearly repeats the previous turn of its dialogue.
    Compared once per id: with --labels both a disputed turn is in `sim` twice."""
    turns = sim.drop_duplicates("id")
    body = turns.response.str.replace(r"^⟦ASSISTANT · ход \d+⟧\s*", "", regex=True).str[:600]
    order = turns.assign(dialogue=turns.id.str.split("::").str[0], body=body).sort_values(["dialogue", "turn"])
    repeats = set()
    for _, d in order.groupby("dialogue"):
        ids, texts = d.id.tolist(), d.body.tolist()
        for i in range(1, len(texts)):
            if difflib.SequenceMatcher(None, texts[i - 1], texts[i]).ratio() > 0.9:
                repeats.add(ids[i])
    return sim.id.isin(repeats).to_numpy()


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--rows", type=Path, default=ROOT / "data/synthetic/tau2_ru_rows.parquet")
    ap.add_argument("--labels", choices=("agreed", "opus", "v1s", "both"), default="agreed")
    ap.add_argument("--only-double-labelled", action="store_true", help="only simulation turns both labellers labelled")
    ap.add_argument("--prefix", default="q4_tok_tau2ru", help="token pass of the rows; its final vectors feed head B")
    ap.add_argument("--opus-dir", type=Path, default=ROOT / "data/synthetic/opus_labels_tau2ru")
    ap.add_argument("--v1s", default="labeler_v1s_tau2ru")
    ap.add_argument("--tag", default="tau2ru", help="name of the result file")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the simulated dialogues to use")
    ap.add_argument("--seed", type=int, default=0, help="which random share, with --fraction")
    ap.add_argument("--quick", action="store_true", help="only the main variant (both heads, weight 1)")
    ap.add_argument("--balance", choices=("none", "match", "negatives", "positives"), default="none")
    ap.add_argument(
        "--negative-share", type=float, default=1.0, help="share of the correct simulation turns kept (all errors kept)"
    )
    ap.add_argument("--no-organiser-tasks", action="store_true", help="drop simulation rows of the organiser's tasks")
    ap.add_argument("--sim-domains", nargs="*", default=None, help="keep only the simulation rows of these domains")
    ap.add_argument(
        "--sim-total",
        type=float,
        default=None,
        help="the simulation rows weigh this many times the fold's own rows in all (default: 1 per row)",
    )
    ap.add_argument(
        "--drop-repeats", action="store_true", help="drop turns that nearly repeat the previous turn of their dialogue"
    )
    ap.add_argument("--context-labels", choices=("v1s", "agreed", "both", "v1s_duplicated"), default="v1s")
    ap.add_argument(
        "--unc-skip", nargs="*", default=[], help="row id tags (e.g. glm) the uncertainty head does not learn from"
    )
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        help="rows|prefix|opus_dir of one simulation set; repeat to pool several (overrides the three above)",
    )
    ap.add_argument(
        "--dialogue-weight",
        action="store_true",
        help="a simulated dialogue weighs as much in all as an organiser's one (8B dialogues loop to ~26 turns)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    suffix = (
        ("_dialogue_weight" if args.dialogue_weight else "")
        + ("" if args.balance == "none" else f"_{args.balance}")
        + ("" if args.negative_share == 1.0 else f"_neg{args.negative_share:g}")
        + ("_norepeats" if args.drop_repeats else "")
        + ("" if args.sim_total is None else f"_simtotal{args.sim_total:g}")
        + ("" if args.sim_domains is None else "_" + "_".join(args.sim_domains))
        + ("_no_organiser_tasks" if args.no_organiser_tasks else "")
    )
    out = args.out or ROOT / f"results/{args.tag}_ruler_{args.labels}{suffix}.json"

    contexts = pd.read_parquet(args.contexts)
    sources = [tuple(x.split("|")) for x in args.source] or [(str(args.rows), args.prefix, str(args.opus_dir))]
    frames, sim_vecs, sim_feats, sim_logp, dropped = [], {}, {}, {}, 0
    for rows_path, prefix, opus_dir in sources:
        part = pd.read_parquet(rows_path)
        # Qwen3.6 at Q4 now and then slips Chinese words into a Russian sentence; the organiser's data has
        # none, so such turns would teach a cue the test cannot have
        cjk = part.response.str.contains(CJK)
        dropped += int(cjk.sum())
        part = part[~cjk].copy()
        part["label"] = simulation_labels(args.labels, part, Path(opus_dir), args.v1s, args.only_double_labelled)
        frames.append(part[part.label.notna()])
        # the token pass's final vectors equal the server's last-token vectors (cosine >= 0.995 on the contexts)
        sim_vecs |= vectors(f"{prefix}_final_org")
        sim_feats |= features(prefix)
        sim_logp |= mean_logp(prefix)
    sim = pd.concat(frames)
    soft = sim.label == 0.5
    sim = pd.concat(
        [sim[~soft].assign(lw=1.0), sim[soft].assign(label=0, lw=0.5), sim[soft].assign(label=1, lw=0.5)],
        ignore_index=True,
    )
    sim["label"] = sim.label.astype(int)
    sim["task"] = [f"{d}__{data_id(t)}"[: len(d) + 2 + CUT] for d, t in zip(sim.domain, sim.task_id, strict=True)]

    vecs = vectors("q4_last_org") | vectors("q4_last_extdev_all") | sim_vecs
    for name in ("q4_last_pairs_02", "q4_last_pairs_03"):
        vecs |= vectors(name)
    own = features("q4_tok")
    sim = sim[sim.id.isin(vecs) & sim.id.isin(sim_feats)]
    if args.fraction < 1.0:  # a random share of the simulated dialogues, for the scale curve
        dialogues = sorted(sim.id.str.split("::").str[0].unique())
        keep = np.random.default_rng(args.seed).choice(dialogues, round(args.fraction * len(dialogues)), replace=False)
        sim = sim[sim.id.str.split("::").str[0].isin(set(keep))]
    if args.sim_domains is not None:
        sim = sim[sim.domain.isin(args.sim_domains)]
    if args.no_organiser_tasks:
        sim = sim[~sim.task.isin(set(contexts.task))]
    repeats = 0
    if args.drop_repeats:
        # a dialogue stuck in a loop (the user cannot be found, the agent refuses again and again) gives dozens of
        # near-copies of one turn, all with the same label; the organiser's dialogues are about half as long
        sim["repeat"] = repeated_turns(sim)
        repeats = int(sim.repeat.sum())
        sim = sim[~sim.repeat]
    if args.negative_share < 1.0:  # a seeded share of the correct turns, every error kept
        negatives = sim.index[sim.label == 0]
        rng = np.random.default_rng(args.seed)
        drop = rng.choice(len(negatives), round((1 - args.negative_share) * len(negatives)), replace=False)
        sim = sim.drop(index=negatives[drop])
    tau = load("ext_dev", None, 0)
    tau = tau[tau.id.isin(vecs)]
    turns = sim.groupby(sim.id.str.split("::").str[0]).id.transform("size")
    organiser_turns = contexts.groupby("task").id.size().mean()
    sim["w"] = (organiser_turns / turns if args.dialogue_weight else 1.0) * sim.lw

    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), None)
    y, x, tasks, pairs = ruler.label["v1s"], ruler.x, ruler.tasks, ruler.pairs
    xu = np.stack([own[i] for i in ruler.ctx.id])
    fired = np.array(
        [
            any(evidence(r.prompt, r.response, Context.from_prompt(r.prompt))[k] for k in RULES)
            for r in ruler.ctx.itertuples()
        ]
    )
    org_logp = mean_logp("q4_tok")
    result: dict = {
        "labels": args.labels,
        "balance": args.balance,
        "uncertainty_head_skips": args.unc_skip,
        "context_labels": args.context_labels,
        "sources": [list(x) for x in sources],
        "turns_dropped_for_chinese_words": dropped,
        "negative_share": args.negative_share,
        "sim_total": args.sim_total,
        "sim_domains": args.sim_domains,
        "turns_dropped_as_repeats": repeats,
        "rows": len(sim),
        "positives": int(sim.label.sum()),
        "dialogues": int(sim.id.str.split("::").str[0].nunique()),
        "rows_of_organiser_tasks": int(sim.task.isin(set(tasks)).sum()),
        "compared_with_contexts": {
            "share_of_errors": [round(float(sim.label.mean()), 3), round(float(y.mean()), 3)],
            "share_with_tool_call": [
                round(float(sim.response.str.contains("TOOL_CALL").mean()), 3),
                round(float(ruler.calls.mean()), 3),
            ],
            "median_response_chars": [int(sim.response.str.len().median()), int(ruler.ctx.response.str.len().median())],
            "median_prompt_chars": [int(sim.prompt.str.len().median()), int(ruler.ctx.prompt.str.len().median())],
            "mean_token_logp": [
                round(float(np.mean([sim_logp[i] for i in sim.id if i in sim_logp])), 3),
                round(float(np.mean([org_logp[i] for i in ruler.ctx.id if i in org_logp])), 3),
            ],
        },
        "variants": {},
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)

    target = float(y.mean())

    def balance(s: pd.DataFrame) -> np.ndarray:
        """Per-row factor on the simulation rows: as they are, their errors weighted up to the contexts'
        share of errors (total weight kept), only their correct turns, or only their errors."""
        lab = s.label.to_numpy()
        if args.balance == "match" and 0 < lab.mean() < 1:
            return np.where(lab == 1, target / lab.mean(), (1 - target) / (1 - lab.mean()))
        if args.balance == "negatives":
            return (lab == 0).astype(float)
        if args.balance == "positives":
            return (lab == 1).astype(float)
        return np.ones(len(s))

    agree = ruler.agree

    def sim_weights(s: pd.DataFrame, sim_weight: float, own: float) -> np.ndarray:
        """1 per simulation row by default; with --sim-total their sum is that many times the fold's own weight,
        so 2000 simulated turns cannot outweigh the ~300 contexts of a fold (stage 1, 27.09)."""
        w = sim_weight * s.w.to_numpy() * balance(s)
        if args.sim_total is not None and w.sum() > 0:
            w = w * args.sim_total * own / w.sum()
        return w

    def context_rows(fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The contexts of a fold as training rows, per --context-labels: the v1s labels; only contexts where
        v1s and Opus agree; or both labellers (a disagreement enters twice, as 0 and as 1, at half weight)."""
        if args.context_labels == "agreed":
            keep = fit[agree[fit]]
            return keep, y[keep], np.ones(len(keep))
        if args.context_labels == "both":
            same, split = fit[agree[fit]], fit[~agree[fit]]
            rows = np.concatenate([same, split, split])
            labels = np.concatenate([y[same], np.zeros(len(split), int), np.ones(len(split), int)])
            return rows, labels, np.concatenate([np.ones(len(same)), np.full(2 * len(split), 0.5)])
        if args.context_labels == "v1s_duplicated":  # the rows of "both", the labels of v1s: a check on duplication
            same, split = fit[agree[fit]], fit[~agree[fit]]
            rows = np.concatenate([same, split, split])
            return rows, y[rows], np.concatenate([np.ones(len(same)), np.full(2 * len(split), 0.5)])
        return fit, y[fit], np.ones(len(fit))

    def head_b(model: str, sim_weight: float) -> np.ndarray:
        scores = np.zeros(len(y))
        for fit, held in GroupKFold(FOLDS).split(x, y, tasks):
            held_tasks = set(tasks[held])
            s = sim[~sim.task.isin(held_tasks)]
            assert not s.task.isin(held_tasks).any(), "a held-out task's simulation in training"
            for seed in range(3):
                p = pairs[~pairs.task.isin(held_tasks)]
                keep = np.random.default_rng(seed).choice(
                    sorted(p.context_id.unique()), round(0.25 * p.context_id.nunique()), replace=False
                )
                p = p[p.context_id.isin(keep)]
                rows, labels, weights = context_rows(fit)
                xs = [x[rows], np.stack([vecs[i] for i in p.id])]
                ys = [labels, p.label.to_numpy()]
                ws = [weights, np.ones(len(p))]
                # τ² weighs half of the Russian rows without the simulations, so adding simulations does not
                # also raise the weight of τ² (it did before 26.09 evening)
                n = len(fit) + len(p)
                if sim_weight and len(s):
                    xs.append(np.stack([vecs[i] for i in s.id]))
                    ys.append(s.label.to_numpy())
                    ws.append(sim_weights(s, sim_weight, n))
                xs.append(np.stack([vecs[i] for i in tau.id]))
                ys.append(tau.label.to_numpy())
                ws.append(np.full(len(tau), 0.5 * n / len(tau)))
                xf, yf, wf = np.vstack(xs), np.concatenate(ys).astype(int), np.concatenate(ws)
                sc = StandardScaler().fit(xf)
                m = LogisticRegression(C=C, max_iter=5000) if model == "logistic" else lgbm()
                m.fit(sc.transform(xf), yf, sample_weight=wf)
                xh = sc.transform(x[held])
                scores[held] += m.decision_function(xh) if model == "logistic" else m.predict_proba(xh)[:, 1]
        return scores

    def uncertainty(sim_weight: float) -> np.ndarray:
        scores = np.zeros(len(y))
        for fit, held in GroupKFold(FOLDS).split(xu, y, tasks):
            s = sim[~sim.task.isin(set(tasks[held]))] if sim_weight else sim.iloc[:0]
            assert not s.task.isin(set(tasks[held])).any(), "a held-out task's simulation in training"
            # an agent of another family shifts the token probabilities; its rows can be kept out of this head
            s = s[~s.id.str.split("__").str[0].isin(args.unc_skip)]
            rows, labels, weights = context_rows(fit)
            xf = np.vstack([xu[rows], *([np.stack([sim_feats[i] for i in s.id])] if len(s) else [])])
            yf = np.concatenate([labels, s.label.to_numpy()]).astype(int)
            wf = np.concatenate([weights, sim_weights(s, sim_weight, float(weights.sum()))])
            sc = StandardScaler().fit(xf)
            m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(xf), yf, sample_weight=wf)
            scores[held] = m.predict_proba(sc.transform(xu[held]))[:, 1]
        return scores

    def system(b: np.ndarray, u: np.ndarray) -> np.ndarray:
        return ranks(b) + 0.5 * ranks(u) + 10 * fired

    b_ref, u_ref = head_b("lightgbm", 0.0), uncertainty(0.0)
    reference = system(b_ref, u_ref)
    result["candidate"] = {
        "head_b_alone": ruler.auroc_ci(b_ref)["auroc"],
        "uncertainty_alone": ruler.auroc_ci(u_ref)["auroc"],
        "system": ruler.auroc_ci(reference)["auroc"],
        "system_f1_at_test_share_all_contexts": {t: at_share(ruler.label[t], reference)["f1"] for t in ("v1s", "opus")},
    }
    logistic_ref = None if args.quick else system(head_b("logistic", 0.0), u_ref)

    def f1_at_share(truth: np.ndarray, s: np.ndarray) -> float:
        return at_share(truth, s)["f1"]

    def new_ruler(s: np.ndarray, base: np.ndarray) -> dict:
        """F1 when the test's share is flagged, on all contexts, per labeller, with its difference to the
        reference and a bootstrap over dialogues. The v1s view is the independent one: the simulation rows
        are labelled by Opus, so the Opus view may favour what Opus teaches."""
        out = {}
        rng = np.random.default_rng(0)
        for teacher in ("v1s", "opus"):
            truth = ruler.label[teacher]
            diffs = []
            for _ in range(1000):
                idx = np.concatenate([ruler.groups[k] for k in rng.integers(0, len(ruler.groups), len(ruler.groups))])
                diffs.append(f1_at_share(truth[idx], s[idx]) - f1_at_share(truth[idx], base[idx]))
            out[teacher] = {
                "f1": f1_at_share(truth, s),
                "f1_minus_reference": round(f1_at_share(truth, s) - f1_at_share(truth, base), 4),
                "ci95": [round(float(v), 3) for v in np.percentile(diffs, [2.5, 97.5])],
                "auroc": round(float(auroc(truth, s)), 4),
            }
        return out

    scores: dict[str, np.ndarray] = {}

    def report(name: str, b: np.ndarray, u: np.ndarray, against: np.ndarray = reference) -> None:
        result["variants"][name] = {
            "head_b_alone": ruler.auroc_ci(b)["auroc"],
            "uncertainty_alone": ruler.auroc_ci(u)["auroc"],
            "system_minus_reference": ruler.paired(system(b, u), against),
            "new_ruler_all_contexts": new_ruler(system(b, u), against),
        }
        print(name, json.dumps(result["variants"][name]), flush=True)
        scores[name] = system(b, u)

    if args.quick:  # the scale curve needs only the main variant
        report("both_w1.0", head_b("lightgbm", 1.0), uncertainty(1.0))
    else:
        for w in (1.0, 0.5):
            b, u = head_b("lightgbm", w), uncertainty(w)
            report(f"head_b_w{w}", b, u_ref)
            report(f"uncertainty_w{w}", b_ref, u)
            report(f"both_w{w}", b, u)
        report("logistic_head_b_w1.0_against_logistic", head_b("logistic", 1.0), u_ref, logistic_ref)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # out-of-fold system scores per context, for comparing against the submissions (ruler_vs_test.py)
    np.savez(out.with_suffix(".scores.npz"), ids=ruler.ctx.id.to_numpy().astype(str), reference=reference, **scores)


if __name__ == "__main__":
    main()
