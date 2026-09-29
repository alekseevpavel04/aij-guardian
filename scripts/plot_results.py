"""The README figure: F1 of the ten submissions, and the test AUROC the leaderboard's top ten imply.

Both panels are drawn from committed results only: results/submissions.csv (F1 as the platform showed it) and
results/budget_model.json (TP and flags recovered from precision and recall, AUROC under the binormal model).

    python scripts/plot_results.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    "правила",
    "судья",
    "судья,\nрассуждение",
    "голова A",
    "голова B",
    "B без пар",
    "финал,\nпорог 1.26",
    "финал,\nпорог 1.12",
    "финал,\nметки τ²",
    "финал,\n+ судья",
]
CHOSEN = {5, 6, 10}  # DECISIONS 49


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs/img/results.png")
    args = ap.parse_args()

    with (ROOT / "results/submissions.csv").open(encoding="utf-8") as f:
        f1 = [float(r["f1"]) for r in csv.DictReader(f)]
    board = json.loads((ROOT / "results/budget_model.json").read_text(encoding="utf-8"))["leaderboard_2026_09_29"]
    test = json.loads((ROOT / "results/leaderboard_forensics.json").read_text(encoding="utf-8"))
    always_one = 2 * test["positives_minimal"] / (test["test_size_lower_bound"] + test["positives_minimal"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = ["#2b6cb0" if i + 1 in CHOSEN else "#a0aec0" for i in range(len(f1))]
    left.bar(range(1, len(f1) + 1), f1, color=colors)
    for i, v in enumerate(f1, 1):
        left.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    left.axhline(board[0]["f1"], color="#c53030", lw=1, ls="--")
    left.text(0.55, board[0]["f1"] + 0.008, f"1-е место {board[0]['f1']:.3f}", color="#c53030", fontsize=8)
    left.axhline(always_one, color="#718096", lw=1, ls=":")
    left.text(0.55, always_one + 0.008, f"«всегда 1» {always_one:.3f}", color="#718096", fontsize=8)
    left.set_xticks(range(1, len(f1) + 1), [f"{i}\n{s}" for i, s in enumerate(LABELS, 1)], fontsize=7)
    left.set_ylim(0, 0.68)
    left.set_ylabel("F1 на публичном тесте")
    left.set_title("Десять отправок (синие выбраны в финал)", fontsize=10)

    places = [r["place"] for r in board]
    right.barh(places, [r["auroc"] for r in board], color=["#2b6cb0" if r["place"] == 8 else "#cbd5e0" for r in board])
    for r in board:
        right.text(r["auroc"] + 0.003, r["place"], f"{r['auroc']:.3f}", va="center", fontsize=8)
    right.set_yticks(places, [f"{p} место" for p in places], fontsize=8)
    right.invert_yaxis()
    right.set_xlim(0.7, 0.9)
    right.set_xlabel("AUROC на тесте, оценка по точности и полноте")
    right.set_title("Первая десятка 29.09 (мы восьмые)", fontsize=10)

    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
