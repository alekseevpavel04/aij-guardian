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
LLAMA_BIN = f"llama_bin_{LLAMA_TAG}_sm75.tar.gz"
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
# banking_knowledge's KB_search_dense uses the organiser's model, qwen3-embedding-8b. It does not fit the cards next to
# the agent (even at ubatch 4096, 27.09), so the documents' embeddings come in the kit, made at home with the same
# model into τ²'s own cache (data/kb_embeddings_cache), and the server here embeds only the queries, on the CPU.
EMBEDDERS = [
    ("Qwen/Qwen3-Embedding-8B-GGUF", "Qwen3-Embedding-8B-Q4_K_M.gguf", ["--n-gpu-layers", "0"], "4096", False),
]
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
        # the versions of the home environment: an unpinned install left out websockets, which τ² imports
        requirements = KIT / "configs/tau2_requirements.txt"
        sh(f"uv pip install -q --python {env / 'bin/python'} -r {requirements} -e {TMP / 'tau2-bench'}")
    sh(f"{env / 'bin/python'} {KIT / 'scripts/simulate_tau2.py'} --help > /dev/null")  # imports τ² and our code
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


def restore_library_links(folder: Path) -> None:
    """libllama.so -> libllama.so.0 -> libllama.so.0.4.1 are symlinks in the build; an unpacked dataset may keep
    only the full names, so the short ones are made again where missing."""
    for full in folder.glob("*.so.*.*"):
        base = full.name.split(".so.")[0] + ".so"
        for short in (base, base + "." + full.name.split(".so.")[1].split(".")[0]):
            if not (folder / short).exists():
                (folder / short).symlink_to(full.name)


def setup_llama() -> Path:
    """The build takes about 45 minutes of a session, so each session leaves build/bin in its output and a later
    session with that output attached unpacks it instead. It goes back to the same path: the build-tree rpath
    of llama-server points at the shared libraries there."""
    server = TMP / "llama.cpp/build/bin/llama-server"
    cached = sorted(Path("/kaggle/input").glob(f"**/{LLAMA_BIN}"))
    unpacked = sorted(Path("/kaggle/input").glob("**/llama.cpp/build/bin/llama-server"))
    if not server.exists() and cached:
        log(f"unpacking {cached[0]}")
        sh(f"tar -xzf {cached[0]} -C /")
    elif not server.exists() and unpacked:  # the tarball uploaded as a dataset, which Kaggle unpacks
        log(f"copying {unpacked[0].parent}")
        shutil.copytree(unpacked[0].parent, server.parent, dirs_exist_ok=True)
        sh(f"chmod +x {server.parent}/*")
        restore_library_links(server.parent)
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
    if not (WORK / LLAMA_BIN).exists():
        sh(f"tar -czf {WORK / LLAMA_BIN} -C / {str(server.parent).lstrip('/')}")
    return server


def setup_model(agent: str) -> Path:
    from huggingface_hub import hf_hub_download

    repo, name, _, _ = MODELS[agent]
    log(f"downloading {repo}/{name}")
    return Path(hf_hub_download(repo, name, local_dir=TMP / "models"))


def start_server(server: Path, model: Path, agent: str, slot_tokens: int = 32768) -> subprocess.Popen:
    flags = MODELS[agent][2]
    command = [
        str(server),
        "--model",
        str(model),
        "--n-gpu-layers",
        "99",
        "--ctx-size",
        str(2 * slot_tokens),
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


def start_embedder(server: Path) -> tuple[subprocess.Popen, str]:
    from huggingface_hub import hf_hub_download

    for repo, name, flags, batch, gpu in EMBEDDERS:
        model = hf_hub_download(repo, name, local_dir=TMP / "models")
        command = [str(server), "--model", model, "--embedding", "--pooling", "last", "--ctx-size", batch]
        command += ["--parallel", "1", "--batch-size", batch, "--ubatch-size", batch, "--port", "8082", *flags]
        # with no layers on the cards llama.cpp still offloads batch operations to them, so hide them
        env = os.environ if gpu else {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        log("$ " + " ".join(command))
        server_log = open(WORK / "embed_server.log", "a")  # noqa: SIM115 - the server outlives this function
        process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, env=env)
        for _ in range(150):
            if process.poll() is not None:
                break
            try:
                if b'"ok"' in urllib.request.urlopen("http://127.0.0.1:8082/health", timeout=2).read():
                    log(f"embedder up: {name}")
                    return process, name
            except Exception:
                time.sleep(2)
        process.kill()
        log(f"embedder {name} did not come up")
    raise RuntimeError("no embedding server, see embed_server.log")


def simulate(
    python: Path,
    agent: str,
    tasks: list[tuple[str, str, int]],
    out: Path,
    stop_at: float,
    max_steps: int,
    user_temperature: float | None,
) -> None:
    thinking = ["--thinking"] if MODELS[agent][3] else []
    running: list[tuple[str, str, float, subprocess.Popen]] = []
    queue = list(tasks)
    exits: list[int] = []
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    while queue or running:
        while queue and len(running) < 2 and time.time() < stop_at:
            domain, task, trials = queue.pop(0)
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
                "--trials",
                str(trials),
                *thinking,
                *(["--user-temperature", str(user_temperature)] if user_temperature is not None else []),
            ]
            err = open(WORK / "simulate_errors.log", "a")  # noqa: SIM115 - held by the running process
            # the working folder holds τ²'s embeddings cache (data/.embeddings_cache), shared by all dialogues
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=err, env=env, cwd=TMP)
            running.append((domain, task, time.time(), process))
        if time.time() >= stop_at and queue:
            log(f"time is up: {len(queue)} tasks not started")
            queue = []
        for item in list(running):
            domain, task, began, process = item
            if process.poll() is not None:
                running.remove(item)
                exits.append(process.returncode)
                log(f"{domain} {task[:60]} exit {process.returncode}, {time.time() - began:.0f} s")
                if len(exits) == 4 and all(exits):
                    raise RuntimeError("the first four dialogues all failed, see simulate_errors.log")
        time.sleep(2)


def summarise(out: Path, agent: str, embedder: str | None = None) -> None:
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
        "embedder": embedder,
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
    ap.add_argument("--user-temperature", type=float, default=None)
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
    python = setup_python()
    server = setup_llama()
    model = setup_model(args.agent)
    out = WORK / f"sims_{args.agent}"
    if args.mode == "test":
        tasks, stop_at, max_steps = (
            [("airline", "0", 1), ("banking_knowledge", "task_001", 1)],
            time.time() + 25 * 60,
            30,
        )
    else:
        listed = json.loads((KIT / "tasks" / args.tasks).read_text(encoding="utf-8"))
        # an entry may carry a third field, the number of trials of that task (repeats with a sampling user)
        tasks = [(item[0], item[1], item[2] if len(item) > 2 else 1) for item in listed]
        stop_at, max_steps = STARTED + args.stop_after * 3600, args.max_steps
    banking = any(d == "banking_knowledge" for d, _, _ in tasks)
    # the organiser's banking prompts reach ~230k characters (knowledge base results in full), so banking needs far
    # longer slots than the 32k that fit the other domains (27.09: with 32k most banking dialogues failed)
    process = start_server(server, model, args.agent, 131072 if banking else 32768)
    embedder, embedder_name = None, None
    if banking:
        embedder, embedder_name = start_embedder(server)
        # the documents' embeddings from the kit; the warm-up must find them (a miss would embed 698 documents on
        # the CPU for hours), so it has a time limit
        shutil.copytree(KIT / "data/kb_embeddings_cache", TMP / "data/.embeddings_cache", dirs_exist_ok=True)
        warm = f"{python} {KIT / 'scripts/simulate_tau2.py'} --domain banking_knowledge --warm-knowledge"
        sh(f"cd {TMP} && timeout 1200 {warm} > {WORK / 'warm_knowledge.log'} 2>&1")
    log(f"{len(tasks)} tasks, new ones start until {time.strftime('%H:%M', time.localtime(stop_at))}")
    try:
        simulate(python, args.agent, tasks, out, stop_at, max_steps, args.user_temperature)
    finally:
        process.terminate()
        if embedder:
            embedder.terminate()
        summarise(out, args.agent, embedder_name)


if __name__ == "__main__":
    sys.exit(main())
