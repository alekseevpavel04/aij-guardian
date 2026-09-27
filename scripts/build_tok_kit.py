"""The kit for a Kaggle token pass session (scripts/kaggle_tok.py): the pass's code, the judge's prompt, the rows.

Next to the rows to pass go 20 organiser contexts that already have home vectors (q4_tok), so the session's
vectors can be compared with the home ones before they are used. The rows carry the organiser's system blocks
and the prompt is the organiser's baseline prompt, so the Kaggle dataset must stay private.

    python scripts/build_tok_kit.py --rows tau2_ru_kq36_s1_rows
"""

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from guardian import journal

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "scripts/kaggle_tok.py",
    "scripts/embed_llama.py",
    "scripts/embed_rows.py",
    "scripts/judge_eval.py",
    "src/guardian/__init__.py",
    "src/guardian/env.py",
    "src/guardian/journal.py",
    "src/guardian/judge.py",
    "src/guardian/labeller.py",
    "src/guardian/metrics.py",
    "src/guardian/tau2_tasks.py",
    "data/raw/baseline/baseline/src/hallucination_detector/prompt.md",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True, help="names of data/synthetic/<name>.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "data/kaggle/aij-tok-kit.zip")
    args = ap.parse_args()

    home = {r["id"] for r in journal.read(ROOT / "data/interim/embeddings/q4_tok_final_org.jsonl")}
    contexts = pd.read_parquet(ROOT / "data/synthetic/organizer_contexts.parquet")
    check = contexts[contexts.id.isin(home)].sample(20, random_state=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(ROOT / name, name)
        z.writestr("data/tok/organiser_check_rows.parquet", check.to_parquet(index=False))
        for name in args.rows:
            z.write(ROOT / f"data/synthetic/{name}.parquet", f"data/tok/{name}.parquet")
    print(args.out, "check rows", len(check), "rows files", args.rows)


if __name__ == "__main__":
    main()
