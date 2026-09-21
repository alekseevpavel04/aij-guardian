"""Speed of the running llama-server on fixed requests, with the memory it occupies at that moment.

The same rows are sent every time with the prompt cache off, so prefill and generation speeds of
different server settings compare directly; the speeds are the server's own timings. Each run is
appended to data/interim/server_bench.jsonl together with the settings given on the command line.

    python scripts/serve_judge.py --set ctx-size=76000 ubatch-size=2048
    python scripts/bench_server.py --label "ctx 76000, ubatch 2048"
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup
from guardian.judge import render
from guardian.machine import measure

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PROMPT = ROOT / "data/raw/baseline/baseline/src/hallucination_detector/prompt.md"
# a middle-sized row and the longest one of the 46
ROWS = ("airline__10::t19", "banking_knowledge__task_080::t30")


def complete(server: str, prompt: str, tokens: int) -> dict:
    payload = {"prompt": prompt, "n_predict": tokens, "cache_prompt": False, "temperature": 0, "ignore_eos": True}
    request = urllib.request.Request(
        f"{server}/completion", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=3600) as reply:
        return json.loads(reply.read())["timings"]


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="the server settings, as they were started")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet").set_index("id")
    system_prompt = BASELINE_PROMPT.read_text(encoding="utf-8")
    runs = []
    for row_id in ROWS:
        row = valid.loc[row_id]
        timings = complete(args.server, render(system_prompt, row.prompt, row.response, thinking=False), args.tokens)
        runs.append(
            {
                "row": row_id,
                "prompt_tokens": timings["prompt_n"],
                "prefill_tokens_per_s": round(timings["prompt_per_second"], 1),
                "generation_tokens_per_s": round(timings["predicted_per_second"], 2),
            }
        )
        print(runs[-1], flush=True)
    memory = measure()
    record = {
        "label": args.label,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": runs,
        "gpu_used_gb": round(memory.gpu_used_gb, 2),
        "shared_gb_max": round(max(memory.shared_gb.values(), default=0.0), 2),
        "dedicated_gb_max": round(max(memory.dedicated_gb.values(), default=0.0), 2),
        "ram_free_gb": round(memory.ram_free_gb, 1),
    }
    journal.append(ROOT / "data/interim/server_bench.jsonl", record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
