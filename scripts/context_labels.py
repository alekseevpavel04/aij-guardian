"""The Opus labels of the organiser's contexts, as label journals, and their agreement with the local labeller.

Sources: the labelling batches (scripts/build_context_label_batch.py, `opus.jsonl`) and the `natural`
lines of the pair batches written by Opus. Every manifest id must have exactly one label, otherwise
the script stops. The labels go to data/interim/labels/opus_v4_org_{train,check}.jsonl in the
layout of the local labeller's journals (`score` is the label), so the head scripts read either.

    python scripts/context_labels.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
LOCAL = {"train": "labeler_v1s_org_train", "check": "labeler_v1s_org_check"}


def opus_labels() -> pd.DataFrame:
    rows = []
    for folder in sorted((ROOT / "data/synthetic/opus_labels").iterdir()):
        expected = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        got = journal.read(folder / "opus.jsonl")
        ids = [g["id"] for g in got]
        if sorted(ids) != sorted(expected):
            raise SystemExit(f"{folder.name}: {len(ids)} labels for {len(expected)} contexts")
        rows += [
            {"id": g["id"], "opus": int(g["label"]), "explanation": g["explanation"], "source": f"label:{folder.name}"}
            for g in got
        ]
    for answers in sorted((ROOT / "data/synthetic/pairs").glob("*/opus.jsonl")):
        rows += [
            {
                "id": g["context_id"],
                "opus": int(g["label"]),
                "explanation": g["explanation"],
                "source": f"pairs:{answers.parent.name}",
            }
            for g in journal.read(answers)
            if g.get("kind") == "natural"
        ]
    frame = pd.DataFrame(rows)
    if frame.id.duplicated().any():
        raise SystemExit(f"labelled twice: {frame.id[frame.id.duplicated()].tolist()[:3]}")
    return frame


def counts(frame: pd.DataFrame) -> dict:
    both = frame.dropna(subset=["local"])
    return {
        "contexts": len(frame),
        "opus_positives": int(frame.opus.sum()),
        "with_local_label": len(both),
        "agree": int((both.opus == both.local).sum()),
        "opus_1_local_0": int(((both.opus == 1) & (both.local == 0)).sum()),
        "opus_0_local_1": int(((both.opus == 0) & (both.local == 1)).sum()),
    }


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/opus_context_labels.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)[["id", "domain", "split"]]
    frame = opus_labels().merge(contexts, on="id", how="left")
    local = {}
    for name in LOCAL.values():
        local |= {r["id"]: int(r["score"] >= 0.5) for r in journal.read(ROOT / "data/interim/labels" / f"{name}.jsonl")}
    frame["local"] = frame.id.map(local)

    result = {"labeller": "claude-opus, configs/prompts/labeler_v4.md", "splits": {}}
    for split in ("train", "check"):
        part = frame[frame.split == split]
        missing = set(contexts.id[contexts.split == split]) - set(part.id)
        result["splits"][split] = {
            **counts(part),
            "unlabelled": sorted(missing),
            "by_domain": {d: counts(g) for d, g in part.groupby("domain")},
        }
        log = ROOT / "data/interim/labels" / f"opus_v4_org_{split}.jsonl"
        log.unlink(missing_ok=True)
        for r in part.sort_values("id").itertuples():
            journal.append(log, {"id": r.id, "score": float(r.opus), "explanation": r.explanation, "source": r.source})
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({s: {k: v for k, v in r.items() if k != "by_domain"} for s, r in result["splits"].items()}, indent=2)
    )


if __name__ == "__main__":
    main()
