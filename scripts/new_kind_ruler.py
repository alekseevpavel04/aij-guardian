"""Does a new kind of external data lift head B again? TrajDebug's other environments on the context ruler.

The τ²-bench external steps gave head B about +0.023 and saturated by 800 rows; the hypothesis is that
a new kind of data brings a new gain (EXPERIMENTS 25.09). The rows of build_trajdebug_more.py (ALFWorld,
WebShop, GAIA, SWE-bench Pro agents) are added alone, on top of all ext_dev rows, and one environment
at a time, always weighted to half the Russian rows in total, and in one variant at full weight.

    python scripts/new_kind_ruler.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from context_ruler import V3_V4, Ruler  # noqa: E402
from judge_eval import load  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import vectors  # noqa: E402


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--rows", type=Path, default=ROOT / "data/external/trajdebug_more.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/new_kind_ruler.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    ext_vecs, new_vecs = vectors("q4_last_extdev_all"), vectors("q4_last_trajdebug_more")
    vecs = vectors("q4_last_org") | ext_vecs | new_vecs
    for name in ("q4_last_pairs_02", "q4_last_pairs_03"):
        vecs |= vectors(name)
    tau = load("ext_dev", None, 0)
    tau = tau[tau.id.isin(ext_vecs)]
    new = pd.read_parquet(args.rows)
    new = new[new.id.isin(new_vecs)]

    def rows(*frames: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        f = pd.concat(frames)
        return np.stack([vecs[i] for i in f.id]), f.label.to_numpy().astype(int)

    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), None)
    y = ruler.label["v1s"]
    base = ruler.scores(y, 0.25)
    ruler.external = rows(tau)
    reference = ruler.scores(y, 0.25, external_weight=0.5)
    variants = {}
    ruler.external = rows(new)
    variants["new_kind_alone"] = ruler.scores(y, 0.25, external_weight=0.5)
    ruler.external = rows(tau, new)
    variants["tau2_plus_new_kind"] = ruler.scores(y, 0.25, external_weight=0.5)
    variants["tau2_plus_new_kind_weight_1"] = ruler.scores(y, 0.25, external_weight=1.0)
    for source in sorted(new.source.unique()):
        ruler.external = rows(tau, new[new.source == source])
        variants[f"tau2_plus_{source}"] = ruler.scores(y, 0.25, external_weight=0.5)
    result = {
        "new_rows": len(new),
        "new_positives": int(new.label.sum()),
        "by_source": new.groupby("source").label.agg(["size", "sum"]).to_dict(),
        "head_b": ruler.auroc_ci(base),
        "tau2_external": {**ruler.auroc_ci(reference), "minus_head_b": ruler.paired(reference, base)},
        "variants": {
            k: {
                **ruler.auroc_ci(s),
                "minus_head_b": ruler.paired(s, base)["difference"],
                "minus_tau2": ruler.paired(s, reference),
            }
            for k, s in variants.items()
        },
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
