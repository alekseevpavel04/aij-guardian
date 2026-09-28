"""Drive the Kaggle simulation sessions from here with the official Kaggle CLI (token in ~/.kaggle/kaggle.json).
The datasets belong to the account of that token.

    python scripts/kaggle_push.py kit                                  # new version of the private kit dataset
    python scripts/kaggle_push.py push --name aij-sim-run1 --agent q36n --tasks tasks_run1.json
    python scripts/kaggle_push.py push --name aij-sim-run4 --binary-from aij-sim-run1  # reuse its build
    python scripts/kaggle_push.py push --name aij-sim-test --mode test
    python scripts/kaggle_push.py status --name aij-sim-run1
    python scripts/kaggle_push.py fetch --name aij-sim-run1             # output into data/kaggle/out/<name>
    python scripts/kaggle_push.py bin                                  # the llama.cpp build as a private dataset
    python scripts/kaggle_push.py kit --rental                         # the rental kit (build_rental_kit.py)
    python scripts/kaggle_push.py push --name aij-final-rehearsal --rehearsal  # kaggle_final_rehearsal.py

Everything stays private: the kit holds the agent instruction found in the organiser's data, and every push
checks that the notebook's metadata says private before it goes out.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = str(Path(sys.executable).parent / "kaggle")
BUILD = "llama_bin_b11052_sm75.tar.gz"


def account() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api.get_config_value("username")


OWNER = account()
KIT_DATASET = f"{OWNER}/aij-sim-kit-zip"
TOK_DATASET = f"{OWNER}/aij-tok-kit-zip"
BIN_DATASET = f"{OWNER}/aij-llama-bin"
RENTAL_DATASET = f"{OWNER}/aij-rental-kit"
ACCELERATOR = "NvidiaTeslaT4"
BASE = ROOT / "data/kaggle"


def run(*args: str) -> str:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # the notebook log is not in the console code page
    out = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", env=env)
    print(out.stdout.strip() or out.stderr.strip())
    if out.returncode:
        raise SystemExit(out.returncode)
    return out.stdout


def notebook(script: str, arguments: list[str]) -> dict:
    code = (
        "import glob, subprocess, sys\n"
        f"script = glob.glob('/kaggle/input/**/{script}', recursive=True)[0]\n"
        "print(script)\n"
        f"subprocess.run([sys.executable, script, *{arguments!r}], check=True)\n"
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
    if args.rehearsal:
        build = (ROOT / "submission_work/rental_kit_build.txt").read_text(encoding="utf-8").strip()
        script, dataset, arguments = "kaggle_final_rehearsal.py", RENTAL_DATASET, ["--expect-build", build]
    elif args.tok:
        script, dataset = "kaggle_tok.py", TOK_DATASET
        arguments = ["--rows", *args.tok, "--stop-after", str(args.stop_after)]
    else:
        script, dataset = "kaggle_sim.py", KIT_DATASET
        arguments = ["--mode", args.mode, "--agent", args.agent, "--stop-after", str(args.stop_after)]
        arguments += ["--tasks", args.tasks] if args.mode == "run" else []
        arguments += ["--user-temperature", str(args.user_temperature)] if args.user_temperature is not None else []
    (folder / "notebook.ipynb").write_text(json.dumps(notebook(script, arguments), indent=1), encoding="utf-8")
    meta = {
        "id": f"{OWNER}/{args.name}",
        "title": args.name,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset] + ([BIN_DATASET] if args.binary_dataset else []),
        "competition_sources": [],
        # an earlier session's output carries the built llama-server (kaggle_sim.py unpacks it)
        "kernel_sources": [f"{OWNER}/{args.binary_from}"] if args.binary_from else [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    written = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    if written.get("is_private") is not True:
        raise SystemExit("refusing to push: the notebook is not private")
    # the CLI exits 0 on a refused push too ("Maximum batch GPU session count of 2 reached")
    if "successfully pushed" not in run("kernels", "push", "-p", str(folder), "--accelerator", ACCELERATOR):
        raise SystemExit("push refused")


def upload(name: str, source: Path, dataset: str) -> None:
    """A new version of a private dataset, or the dataset itself the first time (the CLI creates datasets private
    unless told --public)."""
    folder = BASE / "kit_upload"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy(source, folder / name)
    meta = {"id": dataset, "title": dataset.split("/")[1], "licenses": [{"name": "other"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    exists = subprocess.run([KAGGLE, "datasets", "status", dataset], capture_output=True, text=True, env=env)
    if exists.returncode == 0 and "ready" in exists.stdout:
        run("datasets", "version", "-p", str(folder), "-m", "kit update")
    else:
        run("datasets", "create", "-p", str(folder))
    # a session pushed while the version is still being built runs the previous one; "ready" alone was not enough
    # (28.09, twice), so a minute more after it
    for _ in range(60):
        time.sleep(10)
        status = subprocess.run([KAGGLE, "datasets", "status", dataset], capture_output=True, text=True, env=env)
        if "ready" in status.stdout:
            time.sleep(60)
            return
    raise SystemExit(f"{dataset} is not ready after 10 minutes")


def kit(args: argparse.Namespace) -> None:
    if args.rental:
        upload("rental_kit.zip", ROOT / "submission_work/rental_kit.zip", RENTAL_DATASET)
        return
    name, dataset = ("aij-tok-kit.zip", TOK_DATASET) if args.tok_kit else ("aij-sim-kit.zip", KIT_DATASET)
    upload(name, BASE / name, dataset)


def binary(_: argparse.Namespace) -> None:
    """The build an earlier session left in its output (kaggle_sim.py packs it), as a dataset of its own."""
    upload(BUILD, next((BASE / "out").glob(f"*/{BUILD}")), BIN_DATASET)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("kit", "push", "status", "fetch", "bin"))
    ap.add_argument("--name", default="aij-sim-test")
    ap.add_argument("--mode", choices=("test", "run"), default="run")
    ap.add_argument("--agent", default="q36n")
    ap.add_argument("--tasks", default="tasks_run1.json")
    ap.add_argument("--stop-after", type=float, default=7.9)
    ap.add_argument("--user-temperature", type=float, default=None)
    ap.add_argument("--tok", nargs="+", default=None, help="rows_name:prefix pairs: a token pass session instead")
    ap.add_argument("--tok-kit", action="store_true", help="kit: the token pass kit")
    ap.add_argument("--rental", action="store_true", help="kit: the rental kit (holds organiser data, stays private)")
    ap.add_argument("--rehearsal", action="store_true", help="push: the rehearsal of the rental on a T4")
    ap.add_argument("--binary-dataset", action="store_true", help="attach the aij-llama-bin dataset")
    ap.add_argument("--binary-from", default=None, help="a finished session whose output has the llama-server build")
    args = ap.parse_args()
    if args.action == "kit":
        kit(args)
    elif args.action == "bin":
        binary(args)
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
