"""A batch of labelled rows for a Sonnet subagent acting as the privileged labeller, to compare with the local one.

The subagent gets what the local labeller gets (configs/prompts/labeler_v1.md, the τ² task reference,
the dialogue without the one-call rule) and never the organiser's label or explanation. To keep it
cheap, each domain's ⟦SYSTEM⟧ block goes to its own file once, and each row is shown as the dialogue
after that block plus the response. The exact instruction is written next to the batch as prompt.md.

    python scripts/build_label_batch.py --batch valid_10 --ids airline__8::t7,...
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from guardian.env import setup
from guardian.labeller import without_one_call_rule
from guardian.tau2_tasks import reference_text, unique_task

ROOT = Path(__file__).resolve().parents[1]
MARKER = "⟦USER⟧"


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated ids of valid.parquet")
    ap.add_argument("--labeller", type=Path, default=ROOT / "configs/prompts/labeler_v1.md")
    ap.add_argument("--instruction", type=Path, default=ROOT / "configs/prompts/sonnet_labeller_v1.md")
    ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic/sonnet_labels")
    args = ap.parse_args()

    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet").set_index("id")
    ids = args.ids.split(",")
    folder = (args.out / args.batch).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "labeller.md").write_text(args.labeller.read_text(encoding="utf-8"), encoding="utf-8")

    systems: dict[str, str] = {}
    parts = []
    for row_id in ids:
        row = valid.loc[row_id]
        domain = row_id.split("__")[0]
        prompt = without_one_call_rule(row.prompt)
        cut = prompt.index(MARKER)
        system = prompt[:cut].rstrip()
        if systems.setdefault(domain, system) != system:
            raise SystemExit(f"{row_id}: the {domain} system block differs from the first row of that domain")
        task = unique_task(row_id)
        reference = reference_text(task) if task else "нет (задача не восстанавливается однозначно)"
        parts.append(
            f"# Строка `{row_id}` (домен {domain}, системный блок в system_{domain}.md)\n\n"
            f"## Эталон задачи (агент его не видел)\n\n{reference}\n\n"
            f"## Диалог после системного блока\n\n```\n{prompt[cut:]}\n```\n\n"
            f"## Размечаемый ход\n\n```\n{row.response}\n```\n"
        )
    for domain, system in systems.items():
        (folder / f"system_{domain}.md").write_text(system + "\n", encoding="utf-8")
    (folder / "batch.md").write_text("\n".join(parts), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    system_list = ", ".join(f"`{folder}\\system_{d}.md`" for d in systems)
    (folder / "prompt.md").write_text(
        args.instruction.read_text(encoding="utf-8").format(folder=folder, systems=system_list, n=len(ids)),
        encoding="utf-8",
    )
    size = sum(f.stat().st_size for f in folder.glob("*.md") if f.name != "prompt.md")
    print(f"{len(ids)} rows, {len(systems)} domains, {size / 1024:.0f} KiB of files -> {folder}")


if __name__ == "__main__":
    main()
