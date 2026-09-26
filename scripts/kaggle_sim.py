"""Russian τ² simulations on a Kaggle GPU session (2 x T4), so the home card is left to the token passes.

Runs from the kit dataset (scripts/build_kaggle_kit.py), with the system Python and only the standard library
plus huggingface_hub; everything else is installed in the session:

1. uv, a Python 3.12 environment, τ²-bench at the commit used at home (τ² needs Python 3.12+);
2. llama.cpp at the release used at home (b11052), built for the T4 (sm_75) with CUDA;
3. the agent model's GGUF from Hugging Face (the same files as at home);
4. llama-server with the whole model on the two cards and two slots, then scripts/simulate_tau2.py once per task,
   two at a time; each dialogue is written to its own file as soon as it ends;
5. no new dialogue starts after --stop-after hours from the start of the session; what is done is zipped into
   /kaggle/working with a summary (dialogues, seconds each, rewards), so a stop by the platform loses at most
   the dialogues in progress.

    python /kaggle/input/aij-sim-kit/scripts/kaggle_sim.py --mode test
    python /kaggle/input/aij-sim-kit/scripts/kaggle_sim.py --mode run --agent q36n --tasks tasks_run1.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
WORK = Path("/kaggle/working")
TMP = Path("/tmp/kaggle_sim")  # models and builds; /kaggle/working keeps only the results (20 GB limit)
TAU2_COMMIT = "b7ea907"
LLAMA_TAG = "b11052"
MODELS = {
    "q36n": ("ggml-org/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-Q4_K_M.gguf", ["--reasoning", "off"], False),
    "q36": (
        "ggml-org/Qwen3.6-35B-A3B-GGUF",
        "Qwen3.6-35B-A3B-Q4_K_M.gguf",
        ["--reasoning", "on", "--reasoning-budget", "1024"],
        True,
    ),
    "glm": ("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf", ["--reasoning", "off"], False),
}
STARTED = time.time()


def log(text: str) -> None:
    line = f"[{time.time() - STARTED:7.0f} s] {text}"
    print(line, flush=True)
    with open(WORK / "kaggle_sim.log", "a", encoding="utf-8") as f:
        print(line, file=f)


def sh(command: str, cwd: Path | None = None) -> None:
    log(f"$ {command}")
    subprocess.run(command, shell=True, check=True, cwd=cwd)


def setup_python() -> Path:
    env = TMP / "tau2env"
    if not (env / "bin/python").exists():
        sh("pip install -q uv")
        sh(f"git clone -q https://github.com/sierra-research/tau2-bench {TMP / 'tau2-bench'}")
        sh(f"git -C {TMP / 'tau2-bench'} checkout -q {TAU2_COMMIT}")
        sh(f"uv venv -q --python 3.12 {env}")
        sh(f"uv pip install -q --python {env / 'bin/python'} -e {TMP / 'tau2-bench'}")
    return env / "bin/python"


def cuda_driver_library() -> Path:
    """The CUDA driver library for linking. Kaggle's image has only libcuda.so.1 (no libcuda.so, no stub where
    CMake looks), so CMake cannot find CUDA::cuda_driver; a libcuda.so link next to the build fixes that."""
    for folder in (
        "/usr/local/cuda/lib64/stubs",
        "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/nvidia/lib64",
        "/usr/lib64-nvidia",
        "/usr/local/cuda/compat",
    ):
        for name in ("libcuda.so", "libcuda.so.1"):
            found = Path(folder) / name
            if found.exists():
                if name == "libcuda.so":
                    return found
                link = TMP / "cudalib/libcuda.so"
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(found)
                return link
    raise RuntimeError("no libcuda.so found")


def setup_llama() -> Path:
    server = TMP / "llama.cpp/build/bin/llama-server"
    if not server.exists():
        sh(
            f"git -c advice.detachedHead=false clone -q --depth 1 --branch {LLAMA_TAG} "
            f"https://github.com/ggml-org/llama.cpp {TMP / 'llama.cpp'}"
        )
        driver = cuda_driver_library()
        log(f"CUDA driver library: {driver}")
        sh(
            "cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release "
            f"-DCUDA_cuda_driver_LIBRARY={driver} -DCMAKE_EXE_LINKER_FLAGS=-L{driver.parent} "
            f"-DCMAKE_SHARED_LINKER_FLAGS=-L{driver.parent}",
            cwd=TMP / "llama.cpp",
        )
        sh(f"cmake --build build --target llama-server -j {os.cpu_count()}", cwd=TMP / "llama.cpp")
    return server


def setup_model(agent: str) -> Path:
    from huggingface_hub import hf_hub_download

    repo, name, _, _ = MODELS[agent]
    log(f"downloading {repo}/{name}")
    return Path(hf_hub_download(repo, name, local_dir=TMP / "models"))


def start_server(server: Path, model: Path, agent: str) -> subprocess.Popen:
    flags = MODELS[agent][2]
    command = [
        str(server),
        "--model",
        str(model),
        "--n-gpu-layers",
        "99",
        "--ctx-size",
        "65536",
        "--parallel",
        "2",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--port",
        "8081",
        "--alias",
        "agent",
        "--jinja",
        *flags,
    ]
    log("$ " + " ".join(command))
    server_log = open(WORK / "llama_server.log", "w")  # noqa: SIM115 - the server outlives this function
    process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            if b'"ok"' in urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=2).read():
                log("server up")
                return process
        except Exception:
            time.sleep(2)
    raise RuntimeError("llama-server did not come up, see llama_server.log")


def simulate(python: Path, agent: str, tasks: list[tuple[str, str]], out: Path, stop_at: float, max_steps: int) -> None:
    thinking = ["--thinking"] if MODELS[agent][3] else []
    running: list[tuple[str, str, float, subprocess.Popen]] = []
    queue = list(tasks)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    while queue or running:
        while queue and len(running) < 2 and time.time() < stop_at:
            domain, task = queue.pop(0)
            command = [
                str(python),
                str(KIT / "scripts/simulate_tau2.py"),
                "--domain",
                domain,
                "--tasks",
                task,
                "--model",
                "openai/agent",
                "--api-base",
                "http://127.0.0.1:8081/v1",
                "--max-steps",
                str(max_steps),
                "--out",
                str(out),
                *thinking,
            ]
            err = open(WORK / "simulate_errors.log", "a")  # noqa: SIM115 - held by the running process
            running.append(
                (domain, task, time.time(), subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=err, env=env))
            )
        if time.time() >= stop_at and queue:
            log(f"time is up: {len(queue)} tasks not started")
            queue = []
        for item in list(running):
            domain, task, began, process = item
            if process.poll() is not None:
                running.remove(item)
                log(f"{domain} {task[:60]} exit {process.returncode}, {time.time() - began:.0f} s")
        time.sleep(2)


def summarise(out: Path, agent: str) -> None:
    rows = []
    for path in sorted(out.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        sim = record["simulation"]
        rows.append(
            {
                "domain": record["domain"],
                "task_id": record["task_id"],
                "seconds": record["seconds"],
                "messages": len(sim.get("messages") or []),
                "reward": (sim.get("reward_info") or {}).get("reward"),
            }
        )
    summary = {
        "agent": agent,
        "dialogues": len(rows),
        "session_seconds": round(time.time() - STARTED),
        "mean_seconds_per_dialogue": round(sum(r["seconds"] for r in rows) / len(rows), 1) if rows else None,
        "per_dialogue": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    archive = shutil.make_archive(str(WORK / f"sims_{agent}"), "zip", out)
    log(f"{len(rows)} dialogues -> {archive}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("test", "run"), default="test")
    ap.add_argument("--agent", choices=tuple(MODELS), default="q36n")
    ap.add_argument("--tasks", default=None, help="task list JSON in the kit ({domain: [task ids]})")
    ap.add_argument("--stop-after", type=float, default=7.9, help="hours from the session start")
    ap.add_argument("--max-steps", type=int, default=80)
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
    python = setup_python()
    server = setup_llama()
    model = setup_model(args.agent)
    process = start_server(server, model, args.agent)
    out = WORK / f"sims_{args.agent}"
    if args.mode == "test":
        tasks, stop_at, max_steps = [("airline", "0"), ("airline", "1")], time.time() + 20 * 60, 30
    else:
        listed = json.loads((KIT / "tasks" / args.tasks).read_text(encoding="utf-8"))
        tasks = [(d, t) for d, t in listed]
        stop_at, max_steps = STARTED + args.stop_after * 3600, args.max_steps
    log(f"{len(tasks)} tasks, new ones start until {time.strftime('%H:%M', time.localtime(stop_at))}")
    try:
        simulate(python, args.agent, tasks, out, stop_at, max_steps)
    finally:
        process.terminate()
        summarise(out, args.agent)


if __name__ == "__main__":
    sys.exit(main())
