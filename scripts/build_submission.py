"""Assemble the submission archive from the repository.

    python scripts/build_submission.py --strategy constant --label 1
    python scripts/build_submission.py --strategy rules

Builds `submission/` with the structure the rules require (pyproject.toml, model/, src/, scripts/)
and zips it. The archive is rebuilt from scratch every time, so what is sent is always what is in
the repository.
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("__init__.py", "io.py", "parsing.py", "predictor.py", "rules.py")

PYPROJECT = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "guardian"
version = "{version}"
description = "Guardrail for contextual hallucination detection in LLM agent turns"
requires-python = ">=3.10"
dependencies = ["pyarrow"]

[tool.setuptools.packages.find]
where = ["src"]
"""

README = """# Guardian of Truth

Detector of contextual hallucinations in an LLM agent turn.

Strategy: {strategy}

Run:

    python scripts/predict.py --input <test.csv> --output <predictions.csv>

The input is read as parquet when the file starts with the parquet magic bytes and as CSV
otherwise, regardless of the extension. No network access and no model weights are needed.
"""


def build(out_dir: Path, strategy: str, label: int, version: str) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "src/guardian").mkdir(parents=True)
    (out_dir / "scripts").mkdir()
    (out_dir / "model").mkdir()

    for module in MODULES:
        shutil.copy(ROOT / "src/guardian" / module, out_dir / "src/guardian" / module)
    shutil.copy(ROOT / "scripts/predict.py", out_dir / "scripts/predict.py")

    (out_dir / "pyproject.toml").write_text(PYPROJECT.format(version=version), encoding="utf-8")
    (out_dir / "README.md").write_text(README.format(strategy=strategy), encoding="utf-8")
    config = {"strategy": strategy, "label": label}
    (out_dir / "model/config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def zip_dir(out_dir: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, path.relative_to(out_dir.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=("constant", "rules"), default="rules")
    ap.add_argument("--label", type=int, default=1, help="label for the constant strategy")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--out", type=Path, default=ROOT / "submission")
    ap.add_argument("--archive", type=Path, default=None)
    args = ap.parse_args()

    build(args.out, args.strategy, args.label, args.version)
    name = f"guardian_{args.strategy}{'_' + str(args.label) if args.strategy == 'constant' else ''}.zip"
    archive = args.archive or ROOT / "submissions" / name
    zip_dir(args.out, archive)
    size = archive.stat().st_size
    print(f"{args.out} built, archive {archive} ({size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
