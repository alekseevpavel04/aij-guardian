"""The submission's features of a row against the training pass's features of the same row, on the rental.

scripts/predict.py runs scripts/final_pass.py on the test-shaped rows (data/rental/timing.parquet, which holds the
384 contexts) and leaves its files in the folder it prints; the training pass wrote the contexts under their own
prefix. Same code, same engine, same weights: the vectors and token statistics should agree to float precision,
up to what batching different rows together changes in vLLM.

    python scripts/compare_passes.py --training fp8_contexts --submission-log logs/rental_timing.log \
        --rows data/rental/timing.parquet --contexts data/rental/contexts.parquet
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from guardian import journal
from guardian.extract import decode

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data/interim/embeddings"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", required=True, help="prefix of the training pass of the contexts")
    ap.add_argument("--submission-log", type=Path, required=True, help="predict.py's log, which names its folder")
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--contexts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results/rental_pass_consistency.json")
    args = ap.parse_args()

    folder = Path(re.search(r"pass files in (\S+)", args.submission_log.read_text(encoding="utf-8")).group(1))
    rows = pd.read_parquet(args.rows)
    contexts = pd.read_parquet(args.contexts)
    by_text = {(r.prompt, r.response): r.id for r in contexts.itertuples()}
    # predict.py names the rows by their position in the input file
    pairs = [
        (str(i), by_text[(r.prompt, r.response)])
        for i, r in enumerate(rows.itertuples())
        if (r.prompt, r.response) in by_text
    ]

    def load(directory: Path, prefix: str) -> tuple[dict, dict, dict]:
        vec = {r["id"]: decode(r["vector"]) for r in journal.read(directory / f"{prefix}_final_org.jsonl")}
        unc = {r["id"]: r for r in journal.read(directory / f"{prefix}_uncertainty_org.jsonl")}
        par = {r["id"]: r["parts"] for r in journal.read(directory / f"{prefix}_parts_org.jsonl")}
        return vec, unc, par

    sv, su, sp = load(folder, "test")
    tv, tu, tp = load(EMB, args.training)
    both = [(a, b) for a, b in pairs if a in sv and b in tv]
    cos = [float(sv[a] @ tv[b] / np.linalg.norm(sv[a]) / np.linalg.norm(tv[b])) for a, b in both]
    logp = [float(np.abs(np.asarray(su[a]["logp"]) - np.asarray(tu[b]["logp"])).max()) for a, b in both]
    result = {
        "rows_compared": len(both),
        "vector_cosine_min": round(min(cos), 6),
        "vector_cosine_median": round(float(np.median(cos)), 6),
        "turn_logp_max_abs_diff_median": round(float(np.median(logp)), 5),
        "turn_logp_max_abs_diff_max": round(max(logp), 5),
        "same_token_counts": sum(su[a]["turn_tokens"] == tu[b]["turn_tokens"] for a, b in both),
        "same_parts": sum(sp[a] == tp[b] for a, b in both),
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
