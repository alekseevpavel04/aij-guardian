"""The rows the rented A100 passes through the platform's engine (scripts/final_pass.py), as one folder to upload.

Everything the final heads learn from (tau2ru_ruler.py --export with the base set, DECISIONS 47): the organiser's
384 contexts, the written pairs of the v3/v4 batches, the τ² rows of ext_dev, the simulations of the base set
(strong955 + the banking sample), all rows of each so the ruler can filter them as at home. Plus 548 rows shaped
like the test for timing the submission: the 384 contexts, the 46 labelled rows of valid and 118 contexts again.

    python scripts/build_rental_rows.py --out data/rental
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import V3_V4  # noqa: E402
from judge_eval import load  # noqa: E402
from pairs_effect import written  # noqa: E402

SIMULATIONS = ("q36", "glm", "q36n", "kbank_s350")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/rental")
    args = ap.parse_args()

    contexts = pd.read_parquet(ROOT / "data/synthetic/organizer_contexts.parquet")
    sets = {
        "contexts": contexts[contexts.split.isin(["train", "check"])],
        "pairs": written(V3_V4, contexts, ("train", "check")),
        "tau": load("ext_dev", None, 0),
    }
    for name in SIMULATIONS:
        sets[f"sim_{name}"] = pd.read_parquet(ROOT / f"data/synthetic/tau2_ru_{name}_rows.parquet")
    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet")
    again = sets["contexts"].sample(548 - len(sets["contexts"]) - len(valid), random_state=0)
    timing = pd.concat([sets["contexts"], valid, again], ignore_index=True)[["prompt", "response"]]
    sets["timing"] = timing.assign(id=[f"timing_{i}" for i in range(len(timing))])

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, frame in sets.items():
        frame = frame[["id", "prompt", "response"]].drop_duplicates("id")
        frame.to_parquet(args.out / f"{name}.parquet", index=False)
        chars = frame.prompt.str.len() + frame.response.str.len()
        manifest[name] = {
            "rows": len(frame),
            "chars": int(chars.sum()),
            "max_chars": int(chars.max()),
            # 3.5 characters a token on the organiser's contexts (home token pass)
            "tokens_estimate": int(chars.sum() / 3.5),
        }
    manifest["total_tokens_estimate"] = int(sum(v["tokens_estimate"] for k, v in manifest.items() if k != "timing"))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    assert not np.isin(sets["timing"].id, sets["contexts"].id).any()


if __name__ == "__main__":
    main()
