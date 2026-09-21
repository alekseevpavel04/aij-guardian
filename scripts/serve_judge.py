"""Start llama-server for the local judge with the settings kept in a config file.

The settings live in configs/ so that every judge result can name the exact server it came from.
The server runs in the foreground; stop it with Ctrl+C.

    python scripts/serve_judge.py --set n-cpu-moe=30 reasoning=on
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/llama_server_qwen36.json")
    ap.add_argument("--set", nargs="*", default=[], help="overrides as flag=value, without the leading dashes")
    args = ap.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    options = dict(config["args"])
    for override in args.set:
        flag, value = override.split("=", 1)
        options[f"--{flag}"] = value
    command = [config["binary"], "--model", config["model"]]
    for flag, value in options.items():
        # an empty value is a switch without an argument, e.g. --set embeddings=
        command += [flag] if value == "" else [flag, str(value)]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
