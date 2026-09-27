"""The Russian knowledge base of τ²'s banking_knowledge, as the organiser's agent searched it.

The organiser's banking policy says "The knowledge base documents are written in Russian", and the search
results in his dialogues show translated titles and contents (codes such as FREEZE and product names such as
Platinum Rewards stay in English). τ² ships the documents in English. So:

- `batches`: the documents whose full translation appears in the organiser's valid dialogues are taken as he
  has them; the others go into batches for a Claude subagent, with three of his translations as examples;
- `assemble`: data/synthetic/banking_kb_ru/<id>.json, one per document, with `translated_by` saying whose
  translation it is ("organiser" or the subagent's model).

    python scripts/kb_translation.py batches
    python scripts/kb_translation.py assemble --model claude-sonnet-5
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = Path("D:/VScode_projects/tau2-bench/data/tau2/domains/banking_knowledge/documents")
OUT = ROOT / "data/synthetic/banking_kb_ru"
BATCHES = ROOT / "data/synthetic/kb_translation"
HIT = re.compile(
    r"\d+\. (.+?)\n\s+ID: (doc_[^\n]+)\n\s+Score: [\d.]+\n\s+Content: (.*?)"
    r"(?=\n\s+\d+\. .+?\n\s+ID: doc_|\n⟦|\n\t→|\n\t←|$)",
    re.S,
)
INSTRUCTION = """You translate documents of a bank's internal knowledge base from English into Russian, for a \
customer-service agent who searches them. Work only with the files named here and write one output file. Do not \
run code, do not browse, do not read other files.

Read `{folder}\\examples.json` first: three documents with the translation the knowledge base already has. Match \
that translation's style exactly:
- titles and all prose in natural, businesslike Russian; markdown structure (headings, lists, tables, bold) kept \
line for line;
- product and account names (e.g. Platinum Rewards, Blue Account), status and enum codes in capitals (ACTIVE, \
FREEZE), tool names, field names, IDs, file names and anything in backticks stay in English as they are;
- numbers, percentages, amounts and dates keep their digits and separators; "$3,000" becomes "3,000 долларов";
- nothing added, nothing left out, no notes from you.

Then translate every document in `{folder}\\source.json` and write `{folder}\\russian.json` with the Write tool: \
a JSON list, one object per document in the same order, {{"id": <same id>, "title": <Russian title>, "content": \
<Russian content>}}, UTF-8, valid JSON (escape quotes and newlines). {n} documents in total.

When done, reply with only the number of documents written."""


def english() -> dict[str, dict]:
    return {d["id"]: d for d in (json.loads(p.read_text(encoding="utf-8")) for p in DOCS.glob("*.json"))}


def organiser_translations(en: dict[str, dict]) -> dict[str, dict]:
    """Documents shown in full in the organiser's valid dialogues (a result can be cut short: its length must be
    within 0.9-1.6 of the English one, the range full translations fall in)."""
    valid = pd.read_parquet(ROOT / "data/raw/valid.parquet")
    found: dict[str, dict] = {}
    for prompt in valid[valid.id.str.startswith("banking")].prompt:
        for m in HIT.finditer(prompt):
            title, doc, content = m.group(1).strip(), m.group(2).strip(), re.sub(r"\n\t  ?", "\n", m.group(3)).strip()
            if doc in en and (doc not in found or len(content) > len(found[doc]["content"])):
                found[doc] = {"id": doc, "title": title, "content": content}
    return {k: v for k, v in found.items() if 0.9 <= len(v["content"]) / len(en[k]["content"]) <= 1.6}


def batches(args: argparse.Namespace) -> None:
    en = english()
    done = organiser_translations(en)
    examples = [done[k] | {"english_title": en[k]["title"], "english_content": en[k]["content"]} for k in sorted(done)]
    examples = sorted(examples, key=lambda e: len(e["english_content"]))[len(examples) // 2 :][:3]
    todo = sorted(set(en) - set(done))
    groups, size = [[]], 0
    for doc in todo:
        n = len(en[doc]["content"])
        if groups[-1] and size + n > args.max_chars:
            groups.append([])
            size = 0
        groups[-1].append(doc)
        size += n
    for k, group in enumerate(groups, 1):
        folder = BATCHES / f"kb_{k:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
        source = [{"id": d, "title": en[d]["title"], "content": en[d]["content"]} for d in group]
        (folder / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=1), encoding="utf-8")
        prompt = INSTRUCTION.format(folder=folder.resolve(), n=len(group))
        (folder / "prompt.md").write_text(prompt, encoding="utf-8")
    print(f"{len(done)} organiser translations, {len(todo)} documents in {len(groups)} batches")


def assemble(args: argparse.Namespace) -> None:
    en = english()
    OUT.mkdir(parents=True, exist_ok=True)
    docs = {k: v | {"translated_by": "organiser"} for k, v in organiser_translations(en).items()}
    for path in sorted(BATCHES.glob("kb_*/russian.json")):
        source = json.loads((path.parent / "source.json").read_text(encoding="utf-8"))
        russian = json.loads(path.read_text(encoding="utf-8"))
        if [d["id"] for d in russian] != [d["id"] for d in source]:
            raise SystemExit(f"{path}: ids differ from source.json")
        docs |= {
            d["id"]: {"id": d["id"], "title": d["title"], "content": d["content"], "translated_by": args.model}
            for d in russian
        }
    missing = sorted(set(en) - set(docs))
    if missing:
        raise SystemExit(f"{len(missing)} documents without a translation, e.g. {missing[:3]}")
    for doc in docs.values():
        (OUT / f"{doc['id']}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    by = pd.Series([d["translated_by"] for d in docs.values()]).value_counts().to_dict()
    print(f"{len(docs)} documents -> {OUT}", by)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("batches", "assemble"))
    ap.add_argument("--max-chars", type=int, default=30000)
    ap.add_argument("--model", default="claude-sonnet-5")
    args = ap.parse_args()
    batches(args) if args.action == "batches" else assemble(args)


if __name__ == "__main__":
    main()
