"""Does the judge model find a planted call error more surprising than the original turn? A paired check.

The v5 pairs put one error into a real agent turn (configs/prompts/pair_writer_v5.md); the correct
side is the original turn, whose token probabilities come from the pass over the organiser's contexts
(q4_tok), and the pass over the written error turns gives the other side (q4_tok_v5). For every pair
with a tool call the check compares, error against original: the lowest log-probability of the tokens
covering the edit (edit_to in the error turn, edit_from in the original), the lowest log-probability
of any argument value, and the score of the uncertainty head trained on the organiser's contexts. A
head can only learn from pairs what the model itself sees, so a small share of pairs where the error
is the more surprising side means pairs will not teach the uncertainty head much.

    python scripts/pair_uncertainty.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from guardian import journal
from guardian.env import setup
from guardian.judge import render

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from embed_llama import GGUF  # noqa: E402
from embed_rows import BASELINE_PROMPT  # noqa: E402
from train_head_ru import TEACHERS, labels  # noqa: E402
from turn_ruler import part_summary, summary  # noqa: E402
from turn_tokens import Vocab, part_of_each_char  # noqa: E402


def token_spans(vocab: Vocab, system_prompt: str, prompt: str, response: str) -> list[tuple[int, int]]:
    """Character span of each turn token, in the order of the log-probability record."""
    text = render(system_prompt, prompt, response, thinking=False)
    cut = text.rindex("<response>\n") + len("<response>\n")
    before = len(vocab.tokenize(text[:cut]))
    tokens = vocab.tokenize(text[: cut + len(response)])[before:]
    # byte offsets of each piece, mapped back to characters
    char_of_byte = [i for i, ch in enumerate(response) for _ in ch.encode()]
    spans, offset = [], 0
    for t in tokens:
        n = len(vocab.piece(t))
        lo = char_of_byte[min(offset, len(char_of_byte) - 1)]
        hi = char_of_byte[min(offset + n, len(char_of_byte)) - 1] + 1
        spans.append((lo, hi))
        offset += n
    return spans


def lowest_at(record: dict, spans: list[tuple[int, int]], lo: int, hi: int) -> float | None:
    hit = [record["logp"][k] for k, (a, b) in enumerate(spans) if a < hi and b > lo]
    return min(hit) if hit else None


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--pairs", type=Path, default=ROOT / "data/interim/pairs_v5_call_errors.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/pair_uncertainty.json")
    args = ap.parse_args()

    folder = ROOT / "data/interim/embeddings"
    org = {r["id"]: r for r in journal.read(folder / "q4_tok_uncertainty_org.jsonl")}
    org_parts = {r["id"]: r["parts"] for r in journal.read(folder / "q4_tok_parts_org.jsonl")}
    err = {r["id"]: r for r in journal.read(folder / "q4_tok_v5_uncertainty_org.jsonl")}
    contexts = pd.read_parquet(args.contexts).set_index("id")
    pairs = pd.read_parquet(args.pairs)
    pairs = pairs[pairs.id.isin(err)]
    system_prompt = BASELINE_PROMPT.read_text(encoding="utf-8")
    vocab = Vocab(GGUF)

    # the uncertainty head of turn_ruler.py, fitted on all the organiser's contexts (v1s labels)
    v1s = {}
    for split in ("train", "check"):
        v1s |= labels(TEACHERS["v1s"].format(split=split))
    ids = [i for i in org_parts if i in v1s]

    def features(record: dict, parts: list[str]) -> np.ndarray:
        return np.concatenate([summary(record), part_summary(record, parts)])

    x = np.stack([features(org[i], org_parts[i]) for i in ids])
    scaler = StandardScaler().fit(x)
    head = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(x), [v1s[i] for i in ids])

    rows = []
    for p in pairs.itertuples():
        original = contexts.loc[p.context_id]
        e_spans = token_spans(vocab, system_prompt, p.prompt, p.response)
        o_spans = token_spans(vocab, system_prompt, original.prompt, original.response)
        e_rec, o_rec = err[p.id], org[p.context_id]
        char_parts = part_of_each_char(p.response)
        e_parts = [
            next((k for k in ("value", "key", "name", "text") if k in set(char_parts[a:b])), "layout")
            for a, b in e_spans
        ]
        row = {"id": p.id, "error_type": p.error_type}
        at_e = p.response.find(p.edit_to) if isinstance(p.edit_to, str) and p.edit_to else -1
        at_o = original.response.find(p.edit_from) if isinstance(p.edit_from, str) and p.edit_from else -1
        if at_e >= 0 and at_o >= 0:
            row["edit_error"] = lowest_at(e_rec, e_spans, at_e, at_e + len(p.edit_to))
            row["edit_original"] = lowest_at(o_rec, o_spans, at_o, at_o + len(p.edit_from))
        values_e = [lp for lp, k in zip(e_rec["logp"], e_parts, strict=True) if k == "value"]
        values_o = [lp for lp, k in zip(o_rec["logp"], org_parts[p.context_id], strict=True) if k == "value"]
        if values_e and values_o:
            row["value_min_error"], row["value_min_original"] = min(values_e), min(values_o)
        f_e = features(e_rec, e_parts)[None]
        f_o = features(o_rec, org_parts[p.context_id])[None]
        row["head_error"] = float(head.predict_proba(scaler.transform(f_e))[0, 1])
        row["head_original"] = float(head.predict_proba(scaler.transform(f_o))[0, 1])
        rows.append(row)
    frame = pd.DataFrame(rows)

    def compare(a: str, b: str, lower_is_surprise: bool) -> dict:
        both = frame.dropna(subset=[a, b])
        diff = (both[b] - both[a]) if lower_is_surprise else (both[a] - both[b])
        return {
            "pairs": len(both),
            "error_more_surprising": f"{int((diff > 0).sum())} of {len(both)}",
            "median_difference": round(float(diff.median()), 3) if len(both) else None,
        }

    result = {
        "pairs_with_a_call_error": len(frame),
        "edit_span_logprob": compare("edit_error", "edit_original", True),
        "lowest_argument_value_logprob": compare("value_min_error", "value_min_original", True),
        "uncertainty_head_score": compare("head_error", "head_original", False),
        "by_error_type": {
            t: compare("head_error", "head_original", False) for t in sorted(frame.error_type.dropna().unique())
        },
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
