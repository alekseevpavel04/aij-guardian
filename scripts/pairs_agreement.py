"""How often the local labeller agrees with the labels a Sonnet subagent gave in minimal-pair batches.

Two checks, both against the local labeller (Qwen3.6 with the τ² reference and reasoning, threshold
0.5), which itself agrees with the organiser on 44 of 46 labelled rows:

- natural turns: Sonnet's label of the organiser agent's own turn against the labeller's;
- written turns: a turn Sonnet wrote as correct should get 0, a turn with a planted error 1.

Neither side is ground truth; a disagreement marks a pair to drop or to look at.

    python scripts/pairs_agreement.py --batches airline_01 retail_01 \
        --natural labeler_v2_pilot --written labeler_v2_pilot_written
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]


def scores(name: str) -> dict[str, float]:
    return {r["id"]: r["score"] for r in journal.read(ROOT / "data/interim/labels" / f"{name}.jsonl")}


def style(written: pd.DataFrame, originals: pd.Series) -> dict:
    """Mean length, calls and bold text of written turns by kind, next to the organiser's turns they replace.

    A detector could learn the author instead of the error if these differ between labels."""
    frame = written.assign(original=written.context_id.map(originals))
    out = {}
    for kind, g in frame.groupby("kind"):
        out[kind] = {
            "turns": len(g),
            "chars": round(float(g.response.str.len().mean()), 1),
            "calls": round(float(g.response.str.count("→ TOOL_CALL").mean()), 2),
            "bold_share": round(float(g.response.str.contains("**", regex=False).mean()), 2),
        }
    first = frame.drop_duplicates("context_id")
    out["original"] = {
        "turns": len(first),
        "chars": round(float(first.original.str.len().mean()), 1),
        "calls": round(float(first.original.str.count("→ TOOL_CALL").mean()), 2),
        "bold_share": round(float(first.original.str.contains("**", regex=False).mean()), 2),
    }
    return out


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", required=True)
    ap.add_argument("--natural", required=True, help="label journal of the organiser's turns")
    ap.add_argument("--written", required=True, help="label journal of the written turns")
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/pairs_agreement.json")
    args = ap.parse_args()

    folders = [ROOT / "data/synthetic/pairs" / b for b in args.batches]
    naturals = pd.concat(
        [pd.read_parquet(f / "natural_labels.parquet").assign(batch=f.name) for f in folders], ignore_index=True
    )
    written = pd.concat([pd.read_parquet(f / "pairs.parquet") for f in folders], ignore_index=True)
    originals = pd.read_parquet(args.contexts).set_index("id").response
    natural_scores, written_scores = scores(args.natural), scores(args.written)
    naturals["local"] = naturals.id.map(natural_scores)
    written["local"] = written.id.map(written_scores)
    missing = int(naturals.local.isna().sum() + written.local.isna().sum())
    if missing:
        raise SystemExit(f"{missing} turns have no local label yet")
    naturals["agree"] = (naturals.local >= 0.5).astype(int) == naturals.sonnet_label
    written["agree"] = (written.local >= 0.5).astype(int) == written.label

    def share(frame: pd.DataFrame) -> dict:
        return {"agree": int(frame.agree.sum()), "of": len(frame)}

    result = {
        "batches": args.batches,
        "labeller_journals": {"natural": args.natural, "written": args.written},
        "style_by_batch": {b: style(g, originals) for b, g in written.groupby("batch")},
        "natural": {
            **share(naturals),
            "sonnet_positive": int(naturals.sonnet_label.sum()),
            "local_positive": int((naturals.local >= 0.5).sum()),
            "by_batch": {b: share(g) for b, g in naturals.groupby("batch")},
            "disagreements": [
                {"id": r.id, "sonnet": int(r.sonnet_label), "local": round(r.local, 3)}
                for r in naturals[~naturals.agree].itertuples()
            ],
        },
        "written": {
            **share(written),
            "correct": share(written[written.kind == "correct"]),
            "error": share(written[written.kind == "error"]),
            "by_batch": {
                b: {k: share(g[g.kind == k]) for k in ("correct", "error")} for b, g in written.groupby("batch")
            },
            "error_by_type": {t: share(g) for t, g in written[written.kind == "error"].groupby("error_type")},
            "disagreements": [
                {"id": r.id, "kind": r.kind, "error_type": r.error_type, "local": round(r.local, 3)}
                for r in written[~written.agree].itertuples()
            ],
        },
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
