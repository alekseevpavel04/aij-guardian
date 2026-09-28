"""Which simulation data trains the final system: a fixed grid of tau2ru_ruler.py runs and one summary.

Every run pools labelled simulation sets (Opus labels), weighs them 1 against the fold's own rows in all and drops
looping turns, unless the variant says otherwise. Each is compared, paired over dialogues, with the system without
simulations and with the base set (strong955 + the banking sample) under three labellings of the organiser's
contexts: the contexts where v1s and Opus agree (the main one), v1s, Opus. Leave-one-out and per-domain runs are
also compared with the run on all sets.

The rule for moving off the base set was fixed before the runs: only a difference whose interval
on the agreed contexts lies above zero and whose sign v1s and Opus both share.

    python scripts/data_grid.py            # runs what is missing (about 2 min each), then the summary
    python scripts/data_grid.py --summary  # only the summary
    python scripts/data_grid.py --checks   # the checks of the ruler and of how the rows are fed (after the grid)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paired_variants import VIEWS, compare  # noqa: E402

S = "data/synthetic"
SOURCES = {
    "q36": f"{S}/tau2_ru_q36_rows.parquet|q4_tok_tau2q36|{S}/opus_labels_tau2q36",
    "glm": f"{S}/tau2_ru_glm_rows.parquet|q4_tok_tau2glm|{S}/opus_labels_tau2glm",
    "q36n": f"{S}/tau2_ru_q36n_rows.parquet|q4_tok_tau2q36n|{S}/opus_labels_tau2q36n",
    "kq36n": f"{S}/tau2_ru_kq36n_s1_rows.parquet|q4_tok_tau2kq36n|{S}/opus_labels_tau2kq36n",
    "kq36": f"{S}/tau2_ru_kq36_s1_rows.parquet|q4_tok_tau2kq36|{S}/opus_labels_tau2kq36",
    "kglm": f"{S}/tau2_ru_kglm_s300_rows.parquet|q4_tok_tau2kglm|{S}/opus_labels_tau2kglm",
    "korg": f"{S}/tau2_ru_korg_rows.parquet|q4_tok_tau2korg|{S}/opus_labels_tau2korg",
    "horg": f"{S}/tau2_ru_horg_rows.parquet|q4_tok_tau2horg|{S}/opus_labels_tau2horg",
    "kbank": f"{S}/tau2_ru_kbank_s350_rows.parquet|q4_tok_tau2kbank|{S}/opus_labels_tau2kbank",
}
STRONG = ["q36", "glm", "q36n"]
BASE = [*STRONG, "kbank"]
ALL = list(SOURCES)
DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")


def grid() -> dict[str, dict]:
    v = {
        "strong": {"sources": STRONG},
        "base": {"sources": BASE},
        "all": {"sources": ALL},
        "all_without_repeat_sets": {"sources": [s for s in ALL if s not in ("korg", "horg")]},
    }
    v |= {f"all_minus_{s}": {"sources": [x for x in ALL if x != s]} for s in ALL}
    v |= {f"only_{s}": {"sources": [s]} for s in ALL}
    v |= {f"all_{d}": {"sources": ALL, "extra": ["--sim-domains", d]} for d in DOMAINS}
    v |= {
        "base_simtotal0.5": {"sources": BASE, "sim_total": 0.5},
        "base_simtotal2": {"sources": BASE, "sim_total": 2.0},
        "base_neg0.5": {"sources": BASE, "extra": ["--negative-share", "0.5"]},
        "base_neg0.25": {"sources": BASE, "extra": ["--negative-share", "0.25"]},
        "base_with_repeats": {"sources": BASE, "drop_repeats": False},
        "base_context_labels_agreed": {"sources": BASE, "extra": ["--context-labels", "agreed"]},
    }
    return v


def checks() -> dict[str, dict]:
    """Checks that came after the grid: whether the gain holds on other splits into folds, and
    whether the rows are fed in the wrong way (token features computed on Kaggle, τ² weight next to the
    simulations, one labelling convention, rows unlike the organiser's, per-dialogue weight)."""
    v = {}
    for seed in (1, 2, 3):
        v[f"base_folds{seed}"] = {"sources": BASE, "extra": ["--fold-seed", str(seed)]}
        v[f"all_folds{seed}"] = {"sources": ALL, "extra": ["--fold-seed", str(seed)], "against": f"base_folds{seed}"}
    v |= {
        "base_uncertainty_home_features": {"sources": BASE, "unc_skip": ["glm", "kglm", "kbank"]},
        "base_tau0.25": {"sources": BASE, "extra": ["--tau-weight", "0.25"]},
        "base_tau0": {"sources": BASE, "extra": ["--tau-weight", "0"]},
        "base_context_labels_opus": {"sources": BASE, "extra": ["--context-labels", "opus"]},
        "base_context_labels_both": {"sources": BASE, "extra": ["--context-labels", "both"]},
        "base_importance": {"sources": BASE, "extra": ["--importance"]},
        "all_importance": {"sources": ALL, "extra": ["--importance"], "against": "all"},
        "base_dialogue_weight": {"sources": BASE, "extra": ["--dialogue-weight"]},
    }
    return v


def checks2() -> dict[str, dict]:
    """The second round of checks: how many written pairs, a larger head B for more rows, the
    organiser's domain mix, and the organiser's own contexts cut to a half or a quarter (the gain of the
    simulations when the real contexts are scarce says what they are worth)."""
    v = {
        "base_pairs0": {"sources": BASE, "extra": ["--pair-share", "0"]},
        "base_pairs0.5": {"sources": BASE, "extra": ["--pair-share", "0.5"]},
        "base_lgbm_big": {"sources": BASE, "extra": ["--lgbm-size", "big"]},
        "all_lgbm_big": {"sources": ALL, "extra": ["--lgbm-size", "big"], "against": "all"},
        "base_domain_match": {"sources": BASE, "extra": ["--domain-match"]},
        "all_domain_match": {"sources": ALL, "extra": ["--domain-match"], "against": "all"},
    }
    for fraction, seed in ((0.5, 0), (0.5, 1), (0.25, 0)):
        cut = ["--context-fraction", str(fraction), "--seed", str(seed)]
        v[f"base_contexts{fraction}_seed{seed}"] = {"sources": BASE, "extra": cut}
        v[f"all_contexts{fraction}_seed{seed}"] = {
            "sources": ALL,
            "extra": cut,
            "against": f"base_contexts{fraction}_seed{seed}",
        }
    return v


def checks3() -> dict[str, dict]:
    """A question of 28.09: do more simulation rows pay off only with a larger head B, as boosting did where
    the linear head did not? Four amounts of data (strong955, base, all without the repeat sets, all) at three
    sizes of LightGBM; each run is also compared with strong955 at the same size (the 300x7 runs are in the grid,
    600x15 on base and all in checks2)."""
    amounts = {
        "strong": STRONG,
        "base": BASE,
        "all_without_repeat_sets": [s for s in ALL if s not in ("korg", "horg")],
        "all": ALL,
    }
    v = {}
    for size in ("big", "huge"):
        for name, sources in amounts.items():
            v[f"{name}_lgbm_{size}"] = {"sources": sources, "extra": ["--lgbm-size", size]}
            if name != "strong":
                v[f"{name}_lgbm_{size}"]["against"] = f"strong_lgbm_{size}"
    return v


def run(name: str, spec: dict) -> None:
    out = ROOT / "results" / f"grid_{name}.json"
    if out.with_suffix(".scores.npz").exists():
        return
    cmd = [
        sys.executable,
        "scripts/tau2ru_ruler.py",
        "--labels",
        "opus",
        "--quick",
        "--unc-skip",
        *spec.get("unc_skip", ["glm", "kglm"]),
    ]
    cmd += ["--tag", f"grid_{name}", "--out", str(out), "--sim-total", str(spec.get("sim_total", 1.0))]
    cmd += ["--drop-repeats"] if spec.get("drop_repeats", True) else []
    for s in spec["sources"]:
        cmd += ["--source", SOURCES[s]]
    cmd += spec.get("extra", [])
    with open(ROOT / f"logs/grid_{name}.log", "w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, "done", flush=True)


def summary(variants: dict[str, dict]) -> dict:
    rows = {}
    for name, spec in variants.items():
        ruler = json.loads((ROOT / "results" / f"grid_{name}.json").read_text(encoding="utf-8"))
        against = {"reference": "reference", "base": "grid_base"}
        if name.startswith("all_") and "against" not in spec:
            against["all"] = "grid_all"
        if "against" in spec:
            against[spec["against"]] = f"grid_{spec['against']}"
        rows[name] = {
            "sources": spec["sources"],
            "settings": {k: spec[k] for k in ("sim_total", "drop_repeats", "extra", "unc_skip") if k in spec},
            "rows": ruler["rows"],
            "positives": ruler["positives"],
            "dialogues": ruler["dialogues"],
            "compared": {
                label: {
                    view: {k: r[k] for k in ("auroc_a", "auroc_b", "a_minus_b", "ci95", "share_above_zero")}
                    for view in VIEWS
                    for r in [compare(f"grid_{name}", b, view)]
                }
                for label, b in against.items()
                if f"grid_{name}" != b
            },
        }
        print(name, json.dumps(rows[name]["compared"].get("reference", {}).get("agreed")), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="only the summary of the runs already made")
    ap.add_argument("--checks", action="store_true", help="the checks after the grid instead of the grid")
    ap.add_argument("--checks2", action="store_true", help="the second round of checks")
    ap.add_argument("--checks3", action="store_true", help="amount of data against the size of head B")
    args = ap.parse_args()
    variants = checks3() if args.checks3 else checks2() if args.checks2 else checks() if args.checks else grid()
    if not args.summary:
        for name, spec in variants.items():
            run(name, spec)
    name = (
        "data_checks3"
        if args.checks3
        else "data_checks2"
        if args.checks2
        else "data_checks"
        if args.checks
        else "data_grid"
    )
    out = ROOT / f"results/{name}.json"
    out.write_text(json.dumps(summary(variants), indent=1) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
