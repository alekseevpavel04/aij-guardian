"""Recover the size of the positive class of the public test set from the published leaderboard.

Recall = TP / P, so P * recall must be an integer for every team, and precision then fixes how many
positives each team predicted. Metrics are published with 5 decimals, which is enough to pin P down
to a single family of values: the derivation fixes P only up to a multiple, so the smallest
consistent value is the lower bound, not a proven count.

Each candidate is validated by reconstructing F1 from the recovered integers and comparing it with
the published F1 of every team.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-5  # metrics are published with 5 decimals; convention (round or truncate) is unknown


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [
            {"team": r["team"], "f1": float(r["f1"]), "precision": float(r["precision"]), "recall": float(r["recall"])}
            for r in csv.DictReader(f)
        ]


def reconstruct(rows: list[dict], p: int) -> list[dict] | None:
    """Recover TP and predicted-positive counts for every team, or None if P is inconsistent."""
    out = []
    for r in rows:
        tp = round(r["recall"] * p)
        if abs(tp / p - r["recall"]) > TOL:
            return None
        predicted_positive = round(tp / r["precision"])
        if predicted_positive <= 0 or abs(tp / predicted_positive - r["precision"]) > TOL:
            return None
        f1 = 2 * tp / (p + predicted_positive)
        if abs(f1 - r["f1"]) > TOL:
            return None
        out.append(
            {
                "team": r["team"],
                "true_positives": tp,
                "predicted_positive": predicted_positive,
                "false_positives": predicted_positive - tp,
                "f1_reconstructed": round(f1, 5),
                "f1_leaderboard": r["f1"],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=ROOT / "results/leaderboard/public_2026-09-20.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results/leaderboard_forensics.json")
    ap.add_argument("--max-p", type=int, default=3000)
    args = ap.parse_args()

    rows = read_rows(args.snapshot)
    solutions = {p: recon for p in range(2, args.max_p + 1) if (recon := reconstruct(rows, p))}
    if not solutions:
        raise SystemExit("no positive-class size is consistent with the snapshot")

    p = min(solutions)
    recon = solutions[p]
    result = {
        "snapshot": str(args.snapshot.relative_to(ROOT)).replace("\\", "/"),
        "snapshot_date": args.snapshot.stem.split("_")[-1],
        "teams_in_snapshot": len(rows),
        "positives_minimal": p,
        "positives_candidates": sorted(solutions),
        "candidates_are_multiples_of_minimal": all(c % p == 0 for c in solutions),
        "min_test_size_lower_bound": max(r["predicted_positive"] for r in recon),
        "teams": recon,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"positives P = {p} (candidates up to {args.max_p}: {sorted(solutions)[:5]}...)")
    print(f"all {len(rows)} published F1 values reproduced from integer counts")
    for r in recon:
        print(f"{r['team']:>18}  TP={r['true_positives']:>4}  predicted 1 = {r['predicted_positive']:>4}")


if __name__ == "__main__":
    main()
