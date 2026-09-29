"""How repeatable the FP8 pass is on an A100 (rental of 29.09): the same rows passed twice, and what that does to the
final heads' labels.

vLLM 0.19.1 with the FP8 Qwen3.6-35B-A3B on an A100 does not return the same states for the same text twice: not
between the training pass and the submission's pass, not between two positions of one text in the submission's
own pass, and not between two passes of one row at a time, compiled or eager. Compared here:

- one row at a time, twice (compiled; eager);
- the training pass against the submission's pass (scripts/predict.py on data/rental/timing.parquet);
- the texts that appear twice in the submission's pass;

by the cosine of the last-token vectors, the gap of the turn's token log-probabilities, and, for the last two, by
the final model's scores and labels (submission_work/final_fp8.json).

    python scripts/pass_repeatability.py --rental submission_work/rental
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from guardian import journal
from guardian.cascade import rule_labels
from guardian.extract import decode
from guardian.final import Final
from guardian.uncertainty import features

ROOT = Path(__file__).resolve().parents[1]


def load(folder: Path, prefix: str) -> tuple[dict, dict, dict]:
    vec = {r["id"]: decode(r["vector"]) for r in journal.read(folder / f"{prefix}_final_org.jsonl")}
    unc = {r["id"]: r for r in journal.read(folder / f"{prefix}_uncertainty_org.jsonl")}
    par = {r["id"]: r["parts"] for r in journal.read(folder / f"{prefix}_parts_org.jsonl")}
    return vec, unc, par


def gap(a: tuple, b: tuple, pairs: list[tuple[str, str]]) -> dict:
    cos = [float(a[0][i] @ b[0][j] / np.linalg.norm(a[0][i]) / np.linalg.norm(b[0][j])) for i, j in pairs]
    d = np.concatenate([np.abs(np.asarray(a[1][i]["logp"]) - np.asarray(b[1][j]["logp"])) for i, j in pairs])
    return {
        "rows": len(pairs),
        "identical_vectors": sum(np.array_equal(a[0][i], b[0][j]) for i, j in pairs),
        "vector_cosine_median": round(float(np.median(cos)), 5),
        "vector_cosine_min": round(min(cos), 5),
        "turn_tokens": len(d),
        "share_logp_gap_above_0.1": round(float((d > 0.1).mean()), 4),
        "share_logp_gap_above_1": round(float((d > 1).mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rental", type=Path, default=ROOT / "submission_work/rental")
    ap.add_argument("--submission-pass", default="guardian_final_7s7hlgq7", help="the folder predict.py printed")
    ap.add_argument("--out", type=Path, default=ROOT / "results/rental_repeatability.json")
    args = ap.parse_args()

    det = args.rental / "det"
    one = [(i, i) for i in sorted(load(det, "det_a")[0])]
    eager = [(i, i) for i in sorted(load(det, "eag_a")[0])]
    result = {
        "one_row_at_a_time_twice_compiled": gap(load(det, "det_a"), load(det, "det_b"), one),
        "one_row_at_a_time_twice_eager": gap(load(det, "eag_a"), load(det, "eag_b"), eager),
    }

    training = load(args.rental / "data/interim/embeddings", "fp8_contexts")
    submission = load(args.rental / args.submission_pass, "test")
    timing = pd.read_parquet(ROOT / "data/rental/timing.parquet")
    contexts = pd.read_parquet(ROOT / "data/rental/contexts.parquet")
    by_text = {(r.prompt, r.response): r.id for r in contexts.itertuples()}
    positions: dict[str, list[str]] = {}
    for i, r in enumerate(timing.itertuples()):  # predict.py names the rows by their position
        if (r.prompt, r.response) in by_text:
            positions.setdefault(by_text[(r.prompt, r.response)], []).append(str(i))
    ids = sorted(positions)
    twice = [c for c in ids if len(positions[c]) > 1]
    result["training_pass_vs_submission_pass"] = gap(training, submission, [(c, positions[c][0]) for c in ids])
    result["same_text_twice_in_the_submission_pass"] = gap(
        submission, submission, [(positions[c][0], positions[c][1]) for c in twice]
    )

    model = Final.load(ROOT / "submission_work/final_fp8.json")
    checks = tuple(json.loads((ROOT / "submission_work/final_fp8.json").read_text(encoding="utf-8"))["checks"])

    def scores(source: tuple, keys: list[str]) -> np.ndarray:
        x = np.stack([source[0][k] for k in keys])
        u = np.stack([features(source[1][k], source[2][k]) for k in keys])
        return model.score(x, u)

    fired = np.array(rule_labels(contexts.set_index("id").loc[ids].reset_index().to_dict("records"), checks))
    a, b = scores(training, ids), scores(submission, [positions[c][0] for c in ids])
    la, lb = fired | (a >= model.threshold), fired | (b >= model.threshold)
    result["training_pass_vs_submission_pass"] |= {
        "score_correlation": round(float(np.corrcoef(a, b)[0, 1]), 4),
        "labels_that_differ": int((la != lb).sum()),
    }
    fired_twice = np.array([fired[ids.index(c)] for c in twice])
    a, b = scores(submission, [positions[c][0] for c in twice]), scores(submission, [positions[c][1] for c in twice])
    result["same_text_twice_in_the_submission_pass"] |= {
        "score_correlation": round(float(np.corrcoef(a, b)[0, 1]), 4),
        "labels_that_differ": int(
            ((fired_twice | (a >= model.threshold)) != (fired_twice | (b >= model.threshold))).sum()
        ),
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
