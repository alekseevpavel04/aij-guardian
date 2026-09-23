"""What the blind second labeller said about the written turns (scripts/build_verify_batch.py).

For turns written as correct, a label 1 from the checker rejects the turn. For the control turns,
error turns whose correct twin was not shown, a label 1 is a planted error found, so the share of
them found is the checker's attention on known errors. Every manifest id must have one answer.
The checker's labels go to data/interim/labels/opus_verify_pairs.jsonl for the head scripts.

    python scripts/pair_verification.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="v1")
    ap.add_argument("--out", type=Path, default=ROOT / "results/pair_verification.json")
    args = ap.parse_args()

    rows = []
    for folder in sorted((ROOT / "data/synthetic/verify").glob(f"{args.prefix}_*")):
        manifest = {m["id"]: m["role"] for m in json.loads((folder / "manifest.json").read_text(encoding="utf-8"))}
        answers = {a["id"]: a for a in journal.read(folder / "opus.jsonl")}
        if set(answers) != set(manifest):
            raise SystemExit(f"{folder.name}: {len(answers)} answers for {len(manifest)} turns")
        rows += [
            {"id": i, "role": role, "checker": int(answers[i]["label"]), "explanation": answers[i]["explanation"]}
            for i, role in manifest.items()
        ]
    frame = pd.DataFrame(rows)
    written = pd.concat(
        [pd.read_parquet(f) for f in (ROOT / "data/synthetic/pairs").glob("*/pairs.parquet")], ignore_index=True
    )
    frame = frame.merge(written[["id", "batch", "pair", "kind", "error_type", "domain"]], on="id")
    frame["writer"] = frame.batch.str.extract(r"_(0\d)")[0].map({"02": "v3", "03": "v4"})

    def summary(part: pd.DataFrame) -> dict:
        return {"turns": len(part), "checker_1": int(part.checker.sum())}

    checked, control = frame[frame.role == "check"], frame[frame.role == "control"]
    result = {
        "checker": "claude-opus, labeler_v4.md, blind to the intended label",
        "correct_turns": {
            **summary(checked),
            "by_writer_and_pair": {f"{w}_{p}": summary(g) for (w, p), g in checked.groupby(["writer", "pair"])},
            "by_domain": {d: summary(g) for d, g in checked.groupby("domain")},
        },
        "control_error_turns": {
            **summary(control),
            "by_error_type": {t: summary(g) for t, g in control.groupby("error_type")},
        },
    }
    log = ROOT / "data/interim/labels/opus_verify_pairs.jsonl"
    log.unlink(missing_ok=True)
    for r in frame.sort_values("id").itertuples():
        journal.append(log, {"id": r.id, "score": float(r.checker), "role": r.role, "explanation": r.explanation})
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
