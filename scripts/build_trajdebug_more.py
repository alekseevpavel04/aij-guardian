"""Other environments of the TrajDebug release, rendered into the competition format: a new kind of external data.

External τ²-bench steps helped head B, and the gain came from their being a new kind of data, not from
their number (results/external_variants.json, EXPERIMENTS 25.09): 800 rows gave nearly all of it. The
same release (github.com/THU-KEG/TrajDebug, MIT annotations; trajectories under the upstream terms)
has agents in other environments, each trajectory with its decisive error step annotated:
AgentDebugBench ALFWorld, WebShop and GAIA, and TrajErrBench SWE-bench Pro. Who&When is left out: its
multi-agent logs have no user turns and do not map onto an agent turn.

The decisive step becomes a positive and up to NEGATIVES earlier agent steps of the trajectory become
negatives with label_quality "weak", as in build_trajerr.py; steps after it are dropped. There are no
tool schemas for these environments, so the tool block says so, and the first message, which holds the
environment's instructions, stands in for the policy. Rows longer than MAX_CHARS are dropped: they
would not fit the embedding server's context.

    python scripts/build_trajdebug_more.py --repo D:/VScode_projects/TrajDebug
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from guardian.env import setup
from guardian.render import render_system, render_trajectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_trajerr import to_messages  # noqa: E402

SOURCES = {
    "alfworld": "data/agentdebugbench/alfworld",
    "webshop": "data/agentdebugbench/webshop",
    "gaia": "data/agentdebugbench/gaia",
    "swebenchpro": "data/trajerrbench/en/swebenchpro",
}
NO_TOOLS = "[AVAILABLE TOOLS]\n(the source gives no tool schemas; the agent's actions follow the instructions above)"
NEGATIVES = 3
MAX_CHARS = 180_000


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("D:/VScode_projects/TrajDebug"))
    ap.add_argument("--out", type=Path, default=ROOT / "data/external/trajdebug_more.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records, skipped = [], Counter()
    for source, folder in SOURCES.items():
        for path in sorted((args.repo / folder).glob("*.json")):
            t = json.loads(path.read_text(encoding="utf-8"))
            critical = t["metadata"]["annotation"].get("critical_error_step")
            if critical is None:
                skipped[f"{source}: no decisive step"] += 1
                continue
            try:
                messages = to_messages(t["messages"])
            except ValueError:
                skipped[f"{source}: unreadable tool call"] += 1
                continue
            first, rest = messages[0], messages[1:]
            steps = render_trajectory(rest, render_system(first["content"], NO_TOOLS))
            by_step = {rest[s.message_index]["step"]: s for s in steps}
            if critical not in by_step:
                skipped[f"{source}: decisive step is not an agent step"] += 1
                continue
            before = [n for n in sorted(by_step) if n < critical]
            chosen = [critical, *rng.sample(before, min(NEGATIVES, len(before)))]
            for number in chosen:
                step = by_step[number]
                if len(step.prompt) + len(step.response) > MAX_CHARS:
                    skipped[f"{source}: longer than {MAX_CHARS} characters"] += 1
                    continue
                records.append(
                    {
                        "id": f"trajdebug__{source}__{path.stem}::t{step.turn}",
                        "prompt": step.prompt,
                        "response": step.response,
                        "label": int(number == critical),
                        "source": source,
                        "label_quality": "human_critical_step" if number == critical else "weak",
                        "group": f"{source}:{path.stem}",
                    }
                )
    frame = pd.DataFrame(records)
    frame.to_parquet(args.out, index=False)
    print(frame.groupby("source").label.agg(["size", "sum"]).to_string())
    print(json.dumps(dict(skipped), indent=1))


if __name__ == "__main__":
    main()
