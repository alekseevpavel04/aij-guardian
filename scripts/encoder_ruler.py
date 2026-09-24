"""A small Russian encoder fine-tuned on the same folds as the context ruler, alone and ranked with head B.

Every head so far reads the same vector of the judge model, so their mixtures add nothing. A model of
another kind errs differently, and in the AI-text detector project a weaker fine-tuned model still
helped a frozen probe in a mean of ranks. The encoder is deepvk/USER2-base (ModernBERT trained on
Russian, 8192 tokens, Apache-2.0). A context is far longer than its window, so the input keeps the start
of the prompt (the system prompt with the tool list), the end of the history and the turn itself. It
trains on each ruler fold's training dialogues (their contexts with v1s labels and a quarter of their
pairs, as head B) and scores the held-out dialogues; runs of two seeds are averaged, since training on
the card is not deterministic. Scores are written per fold and seed, so a stopped run continues.

    python scripts/encoder_ruler.py --max-len 4096
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from guardian import journal
from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import FOLDS, V3_V4, Ruler  # noqa: E402
from head_combos import ranks  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import vectors  # noqa: E402

MODEL = "deepvk/USER2-base"
HEAD_TOKENS = 768
TURN_TOKENS = 1024


def encode(tok, prompt: str, response: str, max_len: int) -> list[int]:
    """Start of the prompt, end of the prompt, the turn; the turn is cut first to TURN_TOKENS."""
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok("\n[ОТВЕТ АГЕНТА]\n" + response, add_special_tokens=False)["input_ids"][:TURN_TOKENS]
    room = max_len - 2 - len(r)
    if len(p) > room:
        sep = tok("\n[...]\n", add_special_tokens=False)["input_ids"]
        p = p[:HEAD_TOKENS] + sep + p[-(room - HEAD_TOKENS - len(sep)) :]
    return [tok.cls_token_id, *p, *r, tok.sep_token_id]


def train_and_score(train_ids, train_y, test_ids, seed: int, args) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(train_ids) // args.accumulate
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=max(steps, 1), pct_start=0.1)
    pos_weight = torch.tensor((len(train_y) - train_y.sum()) / max(train_y.sum(), 1), device="cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    step = 0
    for _ in range(args.epochs):
        for k, i in enumerate(rng.permutation(len(train_ids)), 1):
            x = torch.tensor([train_ids[i]], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(input_ids=x).logits[0, 0]
            loss = loss_fn(logit.float(), torch.tensor(float(train_y[i]), device="cuda")) / args.accumulate
            loss.backward()
            if k % args.accumulate == 0 and step < steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
    model.eval()
    out = []
    with torch.no_grad():
        for ids in test_ids:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(float(model(input_ids=torch.tensor([ids], device="cuda")).logits[0, 0]))
    del model, opt
    torch.cuda.empty_cache()
    return np.array(out)


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--accumulate", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--pair-share", type=float, default=0.25)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="probe: only these folds")
    ap.add_argument("--name", default="user2_base")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    vecs = vectors("q4_last_org")
    for name in ("q4_last_pairs_02", "q4_last_pairs_03"):
        vecs |= vectors(name)
    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), None)
    ctx, pairs, y = ruler.ctx, ruler.pairs, ruler.label["v1s"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = {r.id: encode(tok, r.prompt, r.response, args.max_len) for r in ctx.itertuples()}
    enc |= {r.id: encode(tok, r.prompt, r.response, args.max_len) for r in pairs.itertuples()}

    log = ROOT / "data/interim/encoder" / f"{args.name}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["fold"], r["seed"]) for r in journal.read(log)}
    for fold, (fit, held) in enumerate(GroupKFold(FOLDS).split(ruler.x, y, ruler.tasks)):
        if args.folds is not None and fold not in args.folds:
            continue
        held_tasks = set(ruler.tasks[held])
        p = pairs[~pairs.task.isin(held_tasks)]
        ids = np.array(sorted(p.context_id.unique()))
        for seed in args.seeds:
            if (fold, seed) in done:
                continue
            keep = np.random.default_rng(seed).choice(ids, round(args.pair_share * len(ids)), replace=False)
            q = p[p.context_id.isin(keep)]
            train_ids = [enc[i] for i in ctx.id.iloc[fit]] + [enc[i] for i in q.id]
            train_y = np.concatenate([y[fit], q.label.to_numpy()])
            t0 = time.monotonic()
            s = train_and_score(train_ids, train_y, [enc[i] for i in ctx.id.iloc[held]], seed, args)
            journal.append(
                log,
                {
                    "fold": fold,
                    "seed": seed,
                    "ids": list(ctx.id.iloc[held]),
                    "scores": s.tolist(),
                    "train_rows": len(train_ids),
                    "seconds": round(time.monotonic() - t0, 1),
                },
            )
            print(f"fold {fold} seed {seed}: {len(train_ids)} rows, {time.monotonic() - t0:.0f} s", flush=True)

    records = journal.read(log)
    if {(r["fold"], r["seed"]) for r in records} != {(f, s) for f in range(FOLDS) for s in args.seeds}:
        return
    by_seed = {s: {} for s in args.seeds}
    for r in records:
        by_seed[r["seed"]] |= dict(zip(r["ids"], r["scores"], strict=True))
    per_seed = {s: np.array([m[i] for i in ctx.id]) for s, m in by_seed.items()}
    encoder = sum(ranks(v) for v in per_seed.values())
    base = ruler.scores(y, 0.25)
    mixes = {
        "encoder_two_seeds": encoder,
        **{f"encoder_seed_{s}": v for s, v in per_seed.items()},
        "mean_rank_head_b_and_encoder": ranks(base) + ranks(encoder),
    }
    result = {
        "model": MODEL,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "seeds": args.seeds,
        "agreed_rows": int(ruler.agree.sum()),
        "agreed_positives": int(ruler.truth[ruler.agree].sum()),
        "head_b": ruler.auroc_ci(base),
        "variants": {k: {**ruler.auroc_ci(s), "minus_head_b": ruler.paired(s, base)} for k, s in mixes.items()},
    }
    out = ROOT / "results" / f"encoder_ruler_{args.name}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
