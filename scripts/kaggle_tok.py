"""The token pass (scripts/embed_llama.py) on a Kaggle session with two T4, where the whole model fits the cards.

At home the experts' weights stay in RAM and a simulation turn takes about 17 s; here the pass runs from the
tok kit (scripts/build_tok_kit.py) with the llama.cpp b11052 build another session left in its output
(kaggle_sim.py packs it), so nothing is compiled:

1. the build unpacked to the path it was built at, the agent's GGUF from Hugging Face (as at home);
2. first a check: organiser contexts whose home vectors exist, so the two machines can be compared;
3. then every rows file of the kit, each into its own journals in /kaggle/working/embeddings; the pass is
   stopped at --stop-after hours, and a row counts only once its statistics record is written, so a stop
   loses at most the row in progress.

    python /kaggle/input/aij-tok-kit/scripts/kaggle_tok.py --rows tau2_ru_kq36_s1_rows:q4_tok_tau2kq36
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
WORK = Path("/kaggle/working")
TMP = Path("/tmp/kaggle_tok")
LIB = Path("/tmp/kaggle_sim/llama.cpp/build/bin")
MODEL = ("ggml-org/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-Q4_K_M.gguf")
STARTED = time.time()


def log(text: str) -> None:
    line = f"[{time.time() - STARTED:7.0f} s] {text}"
    print(line, flush=True)
    with open(WORK / "kaggle_tok.log", "a", encoding="utf-8") as f:
        print(line, file=f)


def restore_library_links(folder: Path) -> None:
    """libllama.so -> libllama.so.0 -> libllama.so.0.4.1 are symlinks in the build; an unpacked dataset may keep
    only the full names, so the short ones are made again where missing."""
    for full in folder.glob("*.so.*.*"):
        base = full.name.split(".so.")[0] + ".so"
        for short in (base, base + "." + full.name.split(".so.")[1].split(".")[0]):
            if not (folder / short).exists():
                (folder / short).symlink_to(full.name)


def attach_build() -> None:
    """The llama.cpp build of another session: its output's tarball, or the same tarball uploaded as a dataset,
    which Kaggle unpacks (then the files come without the executable bit)."""
    tarball = sorted(Path("/kaggle/input").glob("**/llama_bin_b11052_sm75.tar.gz"))
    if tarball:
        subprocess.run(f"tar -xzf {tarball[0]} -C /", shell=True, check=True)
        return
    unpacked = sorted(Path("/kaggle/input").glob("**/llama.cpp/build/bin/llama-server"))
    if not unpacked:
        raise RuntimeError("no llama.cpp build attached (a kaggle_sim.py session output or aij-llama-bin)")
    shutil.copytree(unpacked[0].parent, LIB, dirs_exist_ok=True)
    subprocess.run(f"chmod +x {LIB}/*", shell=True, check=True)
    restore_library_links(LIB)


def token_pass(rows: str, prefix: str, gguf: Path, stop_at: float) -> None:
    command = [
        sys.executable,
        str(KIT / "scripts/embed_llama.py"),
        "--rows",
        str(KIT / "data/tok" / f"{rows}.parquet"),
        "--prefix",
        prefix,
        "--gguf",
        str(gguf),
        "--lib",
        str(LIB),
        "--all-on-gpu",
        "--threads",
        str(os.cpu_count()),
        "--out-dir",
        str(WORK / "embeddings"),
    ]
    # prepended: the image's own path holds libcuda.so.1 (/usr/local/nvidia/lib64)
    libraries = ":".join(filter(None, [str(LIB), os.environ.get("LD_LIBRARY_PATH"), "/usr/local/nvidia/lib64"]))
    env = {**os.environ, "PYTHONPATH": str(KIT / "src"), "LD_LIBRARY_PATH": libraries, "PYTHONIOENCODING": "utf-8"}
    log(f"{rows} -> {prefix}")
    with open(WORK / f"embed_{prefix}.log", "a", encoding="utf-8") as out:
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env)
        while process.poll() is None:
            if time.time() > stop_at:
                process.terminate()
                process.wait()
                log(f"time is up during {rows}")
                return
            time.sleep(5)
    log(f"{rows} exit {process.returncode}")
    if process.returncode:
        raise RuntimeError(f"the token pass of {rows} failed, see embed_{prefix}.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True, help="rows_name:prefix, in the order to run")
    ap.add_argument("--stop-after", type=float, default=11.0, help="hours from the session start")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
    attach_build()
    from huggingface_hub import hf_hub_download

    gguf = Path(hf_hub_download(*MODEL, local_dir=TMP / "models"))
    log(f"model {gguf}")
    stop_at = STARTED + args.stop_after * 3600
    token_pass("organiser_check_rows", "kcheck_q4_tok", gguf, stop_at)
    for item in args.rows:
        rows, prefix = item.split(":")
        token_pass(rows, prefix, gguf, stop_at)
        if time.time() > stop_at:
            break


if __name__ == "__main__":
    sys.exit(main())
