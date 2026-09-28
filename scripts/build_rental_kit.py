"""Everything the rented A100 needs, as one zip: the code, the data the final heads are trained from, the rows to
pass, the weights fingerprint and the pinned requirements (docs/RENTAL.md).

The data list is what `tau2ru_ruler.py --export` opened when it ran at home (traced 28.09.2026), less the home
embeddings, which the rental computes again with the platform's engine. The kit holds the organiser's contexts,
so it goes only to a rented machine and is deleted there afterwards.

    python scripts/build_rental_rows.py && python scripts/build_rental_kit.py
"""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIMULATIONS = ("q36", "glm", "q36n", "kbank")
DATA = [
    "data/external/apb_tau2.parquet",
    "data/external/trajerr_tau2.parquet",
    "data/external/splits.json",
    "data/interim/labels/labeler_v1s_org_check.jsonl",
    "data/interim/labels/labeler_v1s_org_train.jsonl",
    "data/interim/labels/opus_v4_org_check.jsonl",
    "data/interim/labels/opus_v4_org_train.jsonl",
    "data/interim/labels/opus_verify_pairs.jsonl",
    "data/synthetic/organizer_contexts.parquet",
    "data/synthetic/tau2_ru_q36_rows.parquet",
    "data/synthetic/tau2_ru_glm_rows.parquet",
    "data/synthetic/tau2_ru_q36n_rows.parquet",
    "data/synthetic/tau2_ru_kbank_s350_rows.parquet",
    "data/raw/baseline/baseline/src/hallucination_detector/prompt.md",
    "results/weights_fingerprint.json",
    "configs/rental_requirements.txt",
    "pyproject.toml",
]


def files() -> list[Path]:
    out = [ROOT / p for p in DATA]
    out += sorted((ROOT / "data/synthetic/pairs").glob("*/pairs.parquet"))
    for name in SIMULATIONS:
        out += sorted((ROOT / f"data/synthetic/opus_labels_tau2{name}").glob("*/opus.jsonl"))
    out += sorted((ROOT / "data/rental").glob("*"))
    out += sorted(p for p in (ROOT / "src/guardian").glob("*.py"))
    out += sorted((ROOT / "scripts").glob("*.py"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "submission_work/rental_kit.zip")
    args = ap.parse_args()
    paths = files()
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, p.relative_to(ROOT).as_posix())
    print(f"{len(paths)} files, {args.out.stat().st_size / 2**20:.0f} MB -> {args.out}")


if __name__ == "__main__":
    main()
