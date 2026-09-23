"""Every pair and triple of heads (scripts/head_models.py), combined in several ways, on the context ruler.

Scores are turned into ranks over the 384 contexts (a rank needs no labels, so the platform can do the
same on the test) or into z-scores, and combined by mean, min, max, median, product of ranks, a vote of
the top 30% broken by the mean rank, and a mean with head B counted twice. With thousands of
combinations the best one beats head B by luck alone, so the pick itself is checked: dialogues are split
in half many times, the best combination is chosen on one half and its gain over head B is measured on
the other. Only a gain that survives this is worth anything.

    python scripts/head_combos.py
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_head_ru import auroc  # noqa: E402

TOP_SHARE = 0.3
SPLITS = 200


def ranks(s: np.ndarray) -> np.ndarray:
    return (np.argsort(np.argsort(s)) + 0.5) / len(s)


def z(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / s.std()


OPERATORS = {
    "mean_z": lambda zs, rs: zs.mean(0),
    "mean_rank": lambda zs, rs: rs.mean(0),
    "min_rank": lambda zs, rs: rs.min(0),
    "max_rank": lambda zs, rs: rs.max(0),
    "median_rank": lambda zs, rs: np.median(rs, 0),
    "product_rank": lambda zs, rs: np.log(rs).sum(0),
    "max_z": lambda zs, rs: zs.max(0),
    "min_z": lambda zs, rs: zs.min(0),
    "vote_top30": lambda zs, rs: (rs >= 1 - TOP_SHARE).sum(0) + rs.mean(0),
    "mean_rank_first_twice": lambda zs, rs: (rs.sum(0) + rs[0]) / (len(rs) + 1),
}


def auroc_rows(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """AUROC of every row of s (combinations x contexts) at once, by the rank sum."""
    r = np.argsort(np.argsort(s, axis=1), axis=1) + 1
    pos = y.sum()
    return (r[:, y == 1].sum(1) - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=ROOT / "data/interim/head_models_scores.npz")
    ap.add_argument("--repeats", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=ROOT / "results/head_combos.json")
    args = ap.parse_args()

    data = np.load(args.scores)
    truth, agree, tasks = data["truth"], data["agree"], data["tasks"]
    names = [k for k in data.files if k not in ("truth", "agree", "tasks")]
    zs = {k: z(data[k]) for k in names}
    rs = {k: ranks(data[k]) for k in names}

    labels, rows = [], []
    for size in (2, 3):
        for combo in itertools.combinations(names, size):
            # head B first when it is in the combination, so "first twice" means B twice
            combo = tuple(sorted(combo, key=lambda k: k != "head_b"))
            zc, rc = np.stack([zs[k] for k in combo]), np.stack([rs[k] for k in combo])
            for op, f in OPERATORS.items():
                labels.append(f"{op}({'+'.join(combo)})")
                rows.append(f(zc, rc))
    for k in names:
        labels.append(k)
        rows.append(data[k])
    matrix = np.stack(rows)[:, agree]
    y, t = truth[agree], tasks[agree]
    base = labels.index("head_b")
    point = auroc_rows(y, matrix)
    order = np.argsort(-point)

    groups = [np.where(t == g)[0] for g in np.unique(t)]
    rng = np.random.default_rng(0)
    picked_gain = []
    for _ in range(SPLITS):
        half = rng.permutation(len(groups))
        a = np.concatenate([groups[i] for i in half[: len(groups) // 2]])
        b = np.concatenate([groups[i] for i in half[len(groups) // 2 :]])
        if not (0 < y[a].sum() < len(a) and 0 < y[b].sum() < len(b)):
            continue
        best = int(np.argmax(auroc_rows(y[a], matrix[:, a])))
        on_b = auroc_rows(y[b], matrix[[best, base]][:, b])
        picked_gain.append(on_b[0] - on_b[1])
    picked_gain = np.array(picked_gain)

    def paired(i: int) -> dict:
        boot, diffs = np.random.default_rng(0), []
        for _ in range(args.repeats):
            idx = np.concatenate([groups[k] for k in boot.integers(0, len(groups), len(groups))])
            if 0 < y[idx].sum() < len(idx):
                diffs.append(auroc(y[idx], matrix[i, idx]) - auroc(y[idx], matrix[base, idx]))
        diffs = np.array(diffs)
        return {
            "auroc": round(float(point[i]), 4),
            "minus_head_b": round(float(point[i] - point[base]), 4),
            "ci95": [round(float(v), 3) for v in np.percentile(diffs, [2.5, 97.5])],
            "share_above_zero": round(float((diffs > 0).mean()), 3),
        }

    best_by_operator = {}
    for op in OPERATORS:
        idx = [i for i in order if labels[i].startswith(op + "(")]
        best_by_operator[op] = {"combination": labels[idx[0]], **paired(idx[0])}
    result = {
        "combinations": len(labels) - len(names),
        "agreed_rows": int(agree.sum()),
        "agreed_positives": int(y.sum()),
        "head_b_auroc": round(float(point[base]), 4),
        "single_models": {k: round(float(point[labels.index(k)]), 4) for k in names},
        "top_20": [{"combination": labels[i], **paired(i)} for i in order[:20]],
        "best_by_operator": best_by_operator,
        "pick_on_half_measure_on_other_half": {
            "splits": len(picked_gain),
            "mean_gain_over_head_b": round(float(picked_gain.mean()), 4),
            "ci95": [round(float(v), 3) for v in np.percentile(picked_gain, [2.5, 97.5])],
            "share_above_zero": round(float((picked_gain > 0).mean()), 3),
        },
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
