"""Assemble the submission archive from the repository.

    python scripts/build_submission.py --strategy constant --label 1
    python scripts/build_submission.py --strategy rules

Builds `submission/` with the structure the rules require (pyproject.toml, model/, src/, scripts/)
and zips it. The archive is rebuilt from scratch every time, so what is sent is always what is in
the repository.

Three details come from failures other participants reported in the task comments:

- the files sit at the root of the zip, without a top-level folder: the platform runs
  `<unpacked>/scripts/predict.py`;
- every entry carries explicit Unix permissions (755 for directories, 644 for files): a zip built
  on Windows can unpack on Linux into directories the runner cannot enter, which fails at stage 08;
- zip, not tar, and no directory named `metrics` (organizer's request, 08.09.2026).
"""

import argparse
import json
import shutil
import stat
import zipfile
from pathlib import Path

from guardian.predictor import DEFAULT_CHECKS

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("__init__.py", "io.py", "parsing.py", "predictor.py", "rules.py")

PYPROJECT = """[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "guardian"
version = "{version}"
description = "Guardrail for contextual hallucination detection in LLM agent turns"
requires-python = ">=3.10"
# No dependencies: pip runs without network on the platform. pyarrow is already there (the metric
# stage reads predictions with it), and predict.py falls back to CSV if it is not.
dependencies = []

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


def build(out_dir: Path, strategy: str, label: int, version: str, without: tuple[str, ...] = ()) -> None:
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
    if without:
        config["checks"] = [c for c in DEFAULT_CHECKS if c not in without]
    (out_dir / "model/config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


DIR_MODE = stat.S_IFDIR | 0o755
FILE_MODE = stat.S_IFREG | 0o644


def zip_dir(out_dir: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if "__pycache__" in path.parts:
                continue
            name = path.relative_to(out_dir).as_posix()
            info = zipfile.ZipInfo(name + "/" if path.is_dir() else name)
            info.create_system = 3  # Unix, so that external_attr is read as permission bits
            info.external_attr = (DIR_MODE if path.is_dir() else FILE_MODE) << 16
            if path.is_dir():
                zf.writestr(info, b"")
            else:
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, path.read_bytes())


def check_archive(archive: Path) -> list[str]:
    """Problems that would stop the platform before our code runs; empty when the archive is fine."""
    problems = []
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
    names = [i.filename for i in infos]
    for required in ("pyproject.toml", "scripts/predict.py", "model/config.json"):
        if required not in names:
            problems.append(f"{required} is not at the root of the archive")
    if any(n.split("/")[0] == "metrics" or "/metrics/" in f"/{n}" for n in names):
        problems.append("archive contains a directory named metrics")
    for info in infos:
        mode = info.external_attr >> 16
        needed = 0o755 if info.is_dir() else 0o644
        if info.create_system != 3 or mode & needed != needed:
            problems.append(f"{info.filename}: permissions {oct(mode)} from system {info.create_system}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=("constant", "rules"), default="rules")
    ap.add_argument("--label", type=int, default=1, help="label for the constant strategy")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--out", type=Path, default=ROOT / "submission")
    ap.add_argument("--archive", type=Path, default=None)
    ap.add_argument("--without", nargs="+", default=[], choices=DEFAULT_CHECKS, help="checks left out of the predictor")
    args = ap.parse_args()

    build(args.out, args.strategy, args.label, args.version, tuple(args.without))
    suffix = f"_{args.label}" if args.strategy == "constant" else "".join(f"_no_{c}" for c in args.without)
    name = f"guardian_{args.strategy}{suffix}.zip"
    archive = args.archive or ROOT / "submissions" / name
    zip_dir(args.out, archive)
    problems = check_archive(archive)
    if problems:
        raise SystemExit("archive is not valid:\n  " + "\n  ".join(problems))
    size = archive.stat().st_size
    print(f"{args.out} built, archive {archive} ({size / 1024:.1f} KiB), layout and permissions checked")


if __name__ == "__main__":
    main()
