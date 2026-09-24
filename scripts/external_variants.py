"""Which external rows, and in what form, help head B: a battery of variants on the context ruler.

Head B learns from the organiser's contexts (v1s labels, a quarter of the pairs) and, in the best
variant, from external τ²-bench steps weighted to half the Russian rows. The external parts carry more
than a label: agentprocessbench has human step labels and marks the first error of a trajectory;
trajerrbench has a typed error at the critical step, weak zeros before it and the agent's model.
The organiser counts only new errors of a turn, so positives that continue an earlier error may teach
the wrong thing. Every variant changes only the external rows (their subset or weights) and is
compared with head B and with all external rows at weight 0.5.

    python scripts/external_variants.py --vectors q4_last_extdev_all
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
from judge_eval import EXTERNAL, load  # noqa: E402
from pairs_effect import written  # noqa: E402
from train_head_ru import vectors  # noqa: E402


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--vectors", default="q4_last_extdev_all")
    ap.add_argument("--out", type=Path, default=ROOT / "results/external_variants.json")
    args = ap.parse_args()

    contexts = pd.read_parquet(args.contexts)
    ext_vecs = vectors(args.vectors)
    vecs = vectors("q4_last_org") | ext_vecs
    for name in ("q4_last_pairs_02", "q4_last_pairs_03"):
        vecs |= vectors(name)
    meta = pd.concat([pd.read_parquet(p) for p in EXTERNAL], ignore_index=True).set_index("id")
    d = load("ext_dev", None, 0)
    d = d[d.id.isin(ext_vecs)].reset_index(drop=True)
    for column in ("source_label", "label_quality", "first_error", "group", "agent"):
        d[column] = d.id.map(meta[column])
    d["turn"] = d.id.str.split("::t").str[1].astype(int)
    d["trajectory"] = d.id.str.split("::").str[0]
    y_ext = d.label.to_numpy().astype(int)
    x_ext = np.stack([vecs[i] for i in d.id])

    ruler = Ruler(contexts, vecs, written(V3_V4, contexts), None)
    v1s = ruler.label["v1s"]
    base = ruler.scores(v1s, 0.25)

    def run(mask: np.ndarray, weight: float = 0.5, row_weights: np.ndarray | None = None) -> np.ndarray:
        """External rows under `mask`; row_weights reshape their relative weights (mean 1 kept)."""
        x, y = x_ext[mask], y_ext[mask]
        if row_weights is None:
            ruler.external = (x, y)
            return ruler.scores(v1s, 0.25, external_weight=weight)
        # the ruler gives every external row the same weight; duplicating rows in proportion to the
        # weights approximates a weighted fit with integer copies
        w = row_weights[mask]
        copies = np.maximum(1, np.round(w / w.mean() * 2).astype(int))
        ruler.external = (np.repeat(x, copies, axis=0), np.repeat(y, copies))
        return ruler.scores(v1s, 0.25, external_weight=weight)

    everything = np.ones(len(d), bool)
    apb, trajerr = (d.source == "agentprocessbench").to_numpy(), (d.source == "trajerrbench").to_numpy()
    kind = d.source_label.fillna("").str.split(".").str[0].to_numpy()
    variants: dict[str, np.ndarray] = {}
    reference = run(everything)
    variants["all_rows_weight_1"] = run(everything, 1.0)
    variants["agentprocessbench_only"] = run(apb)
    variants["trajerrbench_only"] = run(trajerr)
    for domain in sorted(d.domain.unique()):
        variants[f"only_{domain}"] = run((d.domain == domain).to_numpy())
        variants[f"without_{domain}"] = run((d.domain != domain).to_numpy())
    # new errors only: agentprocessbench positives that are not the trajectory's first error continue it
    inherited = apb & (y_ext == 1) & ~d.first_error.fillna(False).to_numpy(bool)
    variants["without_inherited_apb_positives"] = run(~inherited)
    variants["without_weak_trajerr_zeros"] = run(~(trajerr & (d.label_quality == "weak").to_numpy()))
    for k in ("reason", "obs", "plan", "act", "verify"):
        variants[f"trajerr_positives_without_{k}"] = run(~(trajerr & (kind == k)))
    grounded = np.isin(
        d.source_label.to_numpy(),
        [
            "obs.MisreadOutput",
            "obs.IgnoreOutput",
            "obs.GroundingFail",
            "act.WrongTool",
            "act.ToolSchemaMismatch",
            "reason.InvalidInference",
        ],
    )
    variants["trajerr_positives_grounding_only"] = run(~(trajerr & (y_ext == 1) & ~grounded))
    for agent in sorted(d.agent.dropna().unique()):
        variants[f"trajerr_agent_{agent}_only"] = run(apb | (trajerr & (d.agent == agent).to_numpy()))
    positives_telecom = apb & (d.domain == "telecom").to_numpy() & (y_ext == 1)
    variants["apb_telecom_positives_half_weight"] = run(everything, row_weights=np.where(positives_telecom, 0.5, 1.0))
    sizes = d.groupby("trajectory").id.transform("size").to_numpy()
    variants["equal_weight_per_trajectory"] = run(everything, row_weights=1.0 / sizes)
    late = d.turn.to_numpy() > np.median(d.turn)
    variants["early_turns_only"] = run(~late)
    variants["late_turns_only"] = run(late)
    balanced = np.where(y_ext == 1, (y_ext == 0).sum() / max(y_ext.sum(), 1), 1.0)
    variants["balanced_classes"] = run(everything, row_weights=balanced)

    result = {
        "external_rows": len(d),
        "external_positives": int(y_ext.sum()),
        "inherited_apb_positives": int(inherited.sum()),
        "head_b": ruler.auroc_ci(base),
        "all_rows_weight_0_5": {**ruler.auroc_ci(reference), "minus_head_b": ruler.paired(reference, base)},
        "variants": {
            k: {
                **ruler.auroc_ci(s),
                "minus_head_b": ruler.paired(s, base)["difference"],
                "minus_all_rows": ruler.paired(s, reference),
            }
            for k, s in variants.items()
        },
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
