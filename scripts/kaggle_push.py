"""Drive the Kaggle simulation sessions from here with the official Kaggle CLI (token in ~/.kaggle/kaggle.json).

    python scripts/kaggle_push.py kit                                  # new version of the private kit dataset
    python scripts/kaggle_push.py push --name aij-sim-run1 --agent q36n --tasks tasks_run1.json
    python scripts/kaggle_push.py push --name aij-sim-test --mode test
    python scripts/kaggle_push.py status --name aij-sim-run1
    python scripts/kaggle_push.py fetch --name aij-sim-run1             # output into data/kaggle/out/<name>

Everything stays private: the kit holds the agent instruction found in the organiser's data, and every push
checks that the notebook's metadata says private before it goes out.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = str(Path(sys.executable).parent / "kaggle")
OWNER = "alekseevpavel"
KIT_DATASET = f"{OWNER}/aij-sim-kit-zip"
ACCELERATOR = "NvidiaTeslaT4"
BASE = ROOT / "data/kaggle"


def run(*args: str) -> str:
    out = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8")
    print(out.stdout.strip() or out.stderr.strip())
    if out.returncode:
        raise SystemExit(out.returncode)
    return out.stdout


def notebook(mode: str, agent: str, tasks: str, stop_after: float) -> dict:
    code = (
        "import glob, subprocess, sys\n"
        "script = glob.glob('/kaggle/input/**/kaggle_sim.py', recursive=True)[0]\n"
        "print(script)\n"
        f"subprocess.run([sys.executable, script, '--mode', '{mode}', '--agent', '{agent}', "
        f"'--stop-after', '{stop_after}'" + (f", '--tasks', '{tasks}'" if mode == "run" else "") + "], check=True)\n"
    )
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def push(args: argparse.Namespace) -> None:
    folder = BASE / "kernels" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "notebook.ipynb").write_text(
        json.dumps(notebook(args.mode, args.agent, args.tasks, args.stop_after), indent=1), encoding="utf-8"
    )
    meta = {
        "id": f"{OWNER}/{args.name}",
        "title": args.name,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [KIT_DATASET],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    written = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    if written.get("is_private") is not True:
        raise SystemExit("refusing to push: the notebook is not private")
    run("kernels", "push", "-p", str(folder), "--accelerator", ACCELERATOR)


def kit(_: argparse.Namespace) -> None:
    folder = BASE / "kit_upload"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy(BASE / "aij-sim-kit.zip", folder / "aij-sim-kit.zip")
    (folder / "dataset-metadata.json").write_text(json.dumps({"id": KIT_DATASET}), encoding="utf-8")
    run("datasets", "version", "-p", str(folder), "-m", "kit update")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("kit", "push", "status", "fetch"))
    ap.add_argument("--name", default="aij-sim-test")
    ap.add_argument("--mode", choices=("test", "run"), default="run")
    ap.add_argument("--agent", default="q36n")
    ap.add_argument("--tasks", default="tasks_run1.json")
    ap.add_argument("--stop-after", type=float, default=7.9)
    args = ap.parse_args()
    if args.action == "kit":
        kit(args)
    elif args.action == "push":
        push(args)
    elif args.action == "status":
        run("kernels", "status", f"{OWNER}/{args.name}")
    else:
        out = BASE / "out" / args.name
        out.mkdir(parents=True, exist_ok=True)
        run("kernels", "output", f"{OWNER}/{args.name}", "-p", str(out))


if __name__ == "__main__":
    main()
