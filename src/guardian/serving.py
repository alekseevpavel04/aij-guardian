"""Starting and stopping the vLLM server that serves the judge inside the submission.

The platform image has vLLM installed (the organiser's baseline relies on it), so the model is served
by `python -m vllm.entrypoints.openai.api_server` from the unpacked archive and queried over HTTP by
guardian.judge. Every step has a deadline: a server that does not come up in time leaves the run to
the rules alone instead of failing it.
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def start_vllm(
    model_dir: Path, port: int, served_name: str, max_model_len: int, log: Path, pooling: bool = False
) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_dir),
        "--served-model-name",
        served_name,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        "0.90",
        "--enable-prefix-caching",
        # the checkpoint is a vision-language model; only its text part is needed
        "--language-model-only",
    ]
    if pooling:
        # the head reads the final normalised hidden state of the last token, not L2-normalised,
        # like llama-server with --pooling last --embd-normalize -1
        command += [
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--pooler-config",
            json.dumps({"pooling_type": "LAST", "use_activation": False}),
        ]
        command.remove("--enable-prefix-caching")
    handle = log.open("w", encoding="utf-8")
    return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)


def wait_ready(port: int, deadline: float, process: subprocess.Popen | None = None) -> bool:
    """Poll /v1/models until it answers or the monotonic `deadline` passes; False on a dead process."""
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as reply:
                if reply.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(3)
    return False


def stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
