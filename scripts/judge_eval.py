"""Score labelled rows with an LLM judge served by an OpenAI-compatible server and measure it.

Rows are written to a jsonl as they are scored, so a run that stops can be resumed with the same
command, and a finished run recomputes its summary from the jsonl without calling the server
(which is how scripts/verify_numbers.py checks it). On an external part, whole trajectories are
sampled and scored in turn order: consecutive steps of one trajectory share almost the whole
prompt, and the server's prompt cache then computes only the new tokens. The summary reports AUROC
and F1 at the fixed threshold; a threshold chosen on the same rows is reported as optimistic and is
not a result.

    python scripts/judge_eval.py --data valid --system-prompt <file> --name baseline_prompt_q4
    python scripts/judge_eval.py --data ext_val --limit 300 --system-prompt <file> --name ...
"""

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from guardian.env import setup
from guardian.judge import JudgeClient
from guardian.metrics import best_threshold, binary_scores

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = (ROOT / "data/external/apb_tau2.parquet", ROOT / "data/external/trajerr_tau2.parquet")


def load(data: str, limit: int | None, seed: int, source: str | None = None) -> pd.DataFrame:
    if data == "valid":
        frame = pd.read_parquet(ROOT / "data/raw/valid.parquet")
        frame["domain"] = frame.id.str.split("__").str[0]
        frame["source"] = "valid"
        return frame[["id", "prompt", "response", "label", "domain", "source"]]
    frame = pd.concat([pd.read_parquet(p) for p in EXTERNAL], ignore_index=True)
    ids = set(json.loads((ROOT / "data/external/splits.json").read_text(encoding="utf-8"))[data])
    frame = frame[frame.id.isin(ids)].rename(columns={"weak_label": "label"})
    frame["trajectory"] = frame.id.str.split("::").str[0]
    frame["turn"] = frame.id.str.split("::t").str[1].astype(int)
    if limit:
        trajectories = sorted(frame.trajectory.unique())
        random.Random(seed).shuffle(trajectories)
        chosen, rows = [], 0
        for t in trajectories:
            if rows >= limit:
                break
            chosen.append(t)
            rows += int((frame.trajectory == t).sum())
        frame = frame[frame.trajectory.isin(chosen)]
    if source:  # after sampling, so the rows are the same as in a run over all sources
        frame = frame[frame.source == source]
    frame = frame.sort_values(["trajectory", "turn"])
    return frame[["id", "prompt", "response", "label", "domain", "source"]]


def summarise(scored: pd.DataFrame) -> dict:
    y = scored.label.to_numpy().astype(int)
    s = scored.score.to_numpy().astype(float)
    at_half = binary_scores(y, (s >= 0.5).astype(int))
    threshold, best_f1 = best_threshold(y, s)
    out = {
        "rows": len(scored),
        "positives": int(y.sum()),
        "auroc": round(float(roc_auc_score(y, s)), 4) if 0 < y.sum() < len(y) else None,
        "at_threshold_0_5": {
            "predicted_positive": at_half.predicted_positive,
            "true_positives": at_half.true_positives,
            "precision": round(at_half.precision, 4),
            "recall": round(at_half.recall, 4),
            "f1": round(at_half.f1, 4),
        },
        "optimistic_best_threshold": {"threshold": round(threshold, 4), "f1": round(best_f1, 4)},
        "mean_score_by_label": {str(k): round(float(s[y == k].mean()), 4) for k in (0, 1) if (y == k).any()},
    }
    for column in ("source", "domain"):
        out[f"auroc_by_{column}"] = {
            value: round(float(roc_auc_score(g.label, g.score)), 4)
            for value, g in scored.groupby(column)
            if 0 < g.label.sum() < len(g)
        }
    return out


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=("valid", "ext_dev", "ext_val"), default="valid")
    ap.add_argument("--limit", type=int, default=None, help="rows to sample, as whole trajectories")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default=None, help="keep one external source after sampling")
    ap.add_argument("--system-prompt", type=Path, required=True)
    ap.add_argument("--append-prompt", type=Path, default=None, help="text added after the system prompt")
    ap.add_argument("--server", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--thinking-tokens", type=int, default=1024)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-rows", type=int, default=None, help="stop after this many new rows (speed probes)")
    ap.add_argument("--out", type=Path, default=None, help="summary file, results/judge_<name>.json by default")
    args = ap.parse_args()

    frame = load(args.data, args.limit, args.seed, args.source)
    system_prompt = args.system_prompt.read_text(encoding="utf-8")
    if args.append_prompt:
        system_prompt = system_prompt.rstrip() + "\n\n" + args.append_prompt.read_text(encoding="utf-8").strip()
    log = ROOT / "data/interim/judge" / f"{args.name}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in log.read_text(encoding="utf-8").splitlines()} if log.exists() else set()
    client = JudgeClient(args.server, thinking=args.thinking, thinking_tokens=args.thinking_tokens)

    todo = frame[~frame.id.isin(done)]
    if args.max_rows:
        todo = todo.head(args.max_rows)
    started = time.monotonic()
    with log.open("a", encoding="utf-8") as f:
        for i, row in enumerate(todo.itertuples(), 1):
            verdict = client.score(system_prompt, row.prompt, row.response)
            record = {
                "id": row.id,
                "score": verdict.score,
                "prompt_tokens": verdict.prompt_tokens,
                "cached_tokens": verdict.cached_tokens,
                "completion_tokens": verdict.completion_tokens,
                "seconds": round(verdict.seconds, 2),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            elapsed = time.monotonic() - started
            print(
                f"{len(done) + i}/{len(frame)} {row.id[:48]:<48} tokens {verdict.prompt_tokens} "
                f"cached {verdict.cached_tokens} {verdict.seconds:.1f} s, eta "
                f"{elapsed / i * (len(todo) - i) / 60:.0f} min",
                flush=True,
            )

    records = pd.DataFrame([json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()])
    scored = frame.drop(columns=["prompt", "response"]).merge(records, on="id")
    if len(scored) < len(frame):
        print(f"{len(scored)} of {len(frame)} rows scored so far; no summary written")
        return
    missing = int(scored.score.isna().sum())
    scored = scored.dropna(subset=["score"])
    summary = {
        "name": args.name,
        "data": args.data,
        "limit": args.limit,
        "seed": args.seed,
        "source": args.source,
        "system_prompt": args.system_prompt.name,
        "append_prompt": args.append_prompt.name if args.append_prompt else None,
        "thinking": args.thinking,
        "thinking_tokens": args.thinking_tokens if args.thinking else None,
        "rows_without_a_verdict_token": missing,
        "mean_seconds_per_row": round(float(records.seconds.mean()), 2),
        "mean_prompt_tokens": round(float(records.prompt_tokens.mean())),
        **summarise(scored),
    }
    out = args.out or ROOT / "results" / f"judge_{args.name}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
