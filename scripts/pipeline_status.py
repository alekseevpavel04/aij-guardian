"""Where the pseudo-test pipeline stands, and the command that continues each stage.

After the computer was switched off, run this first. Every stage is resumable: rerunning its command
skips finished work (simulations are one file each, labels and scores are durable journals).

    python scripts/pipeline_status.py
"""

import json
import urllib.request
from pathlib import Path

from guardian import journal
from guardian.machine import measure

ROOT = Path(__file__).resolve().parents[1]
TAU2_PYTHON = "D:/VScode_projects/tau2-bench/.venv/Scripts/python.exe"


def server_up(url: str = "http://127.0.0.1:8080/health") -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as reply:
            return json.loads(reply.read()).get("status") == "ok"
    except OSError:
        return False


def main() -> None:
    up = server_up()
    print(f"llama-server: {'up' if up else 'down'}")
    if not up:
        print("  start: python scripts/serve_judge.py")
    memory = measure()
    print(
        f"GPU {memory.gpu_used_gb:.1f} of {memory.gpu_total_gb:.1f} GB, "
        f"RAM free {memory.ram_free_gb:.1f} of {memory.ram_total_gb:.1f} GB"
    )
    for warning in memory.warnings():
        print(f"  WARNING: {warning}")

    sims = ROOT / "data/synthetic/tau2_ru"
    print("\nsimulations (scripts/simulate_tau2.py):")
    for domain in ("airline", "retail", "telecom"):
        done = len(list((sims / domain).glob("*.json"))) if (sims / domain).exists() else 0
        stray = len(list((sims / domain).glob("*.tmp"))) if (sims / domain).exists() else 0
        note = f", {stray} unfinished .tmp (safe to delete)" if stray else ""
        print(f"  {domain:<8} {done} finished{note}")
    print(f"  continue: {TAU2_PYTHON} scripts/simulate_tau2.py --domain <domain> [--tasks ...]")

    rows = ROOT / "data/synthetic/pseudo_rows.parquet"
    print(f"\nrows (scripts/build_pseudo_rows.py): {'built' if rows.exists() else 'not built'}")

    print("\nlabel journals (scripts/label_rows.py):")
    for log in sorted((ROOT / "data/interim/labels").glob("*.jsonl")):
        records = journal.read(log)
        print(f"  {log.stem:<40} {len(records)} labelled")
    print("\njudge journals (scripts/judge_eval.py):")
    for log in sorted((ROOT / "data/interim/judge").glob("*.jsonl")):
        print(f"  {log.stem:<50} {len(journal.read(log))} scored")


if __name__ == "__main__":
    main()
