"""First look at the labelled file: format, class balance, trivial baselines.

Writes results/eda.json and results/trivial_baselines.json. Everything printed here is meant to be
read before any modelling: if a trivial predictor is already near the top of the leaderboard, that
is the main fact of the project.

    python scripts/eda.py --input data/raw/valid.parquet
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from guardian.env import setup
from guardian.io import read_rows
from guardian.metrics import binary_scores, bootstrap_ci
from guardian.parsing import (
    MARKER_LIKE_RE,
    TURN_INDEX_RE,
    parse_transcript,
    system_turn,
    unknown_roles,
    unparsed_markers,
)

ROOT = Path(__file__).resolve().parents[1]
ALT_MARKERS = ("→ TOOL_CALL", "← TOOL_RESPONSE", "[AVAILABLE TOOLS]", "⟦ASSISTANT_TOOL_CALL")


def domain_of(example_id: str) -> str:
    return example_id.split("__")[0]


def format_report(frame) -> dict:
    """What the transcripts actually look like: the rules PDF and the Data tab disagree about it."""
    markers, unexpected, unparsed, tool_names = Counter(), Counter(), Counter(), Counter()
    calls_in_response = []
    for row in frame.itertuples():
        for column, text in (("prompt", row.prompt), ("response", row.response)):
            markers.update(TURN_INDEX_RE.sub("N", m.group(0)) for m in MARKER_LIKE_RE.finditer(text))
            unparsed.update(unparsed_markers(text))
            turns = parse_transcript(text)
            unexpected.update(f"{column}:{role}" for role in unknown_roles(turns))
            if column == "response":
                calls = [c for t in turns for c in t.calls]
                tool_names.update(c.name for c in calls)
                calls_in_response.append(len(calls))
    return {
        "columns": list(frame.columns),
        "rows": len(frame),
        "markers": markers.most_common(20),
        "markers_documented_on_data_tab": {
            marker: int(frame.prompt.str.count(re.escape(marker)).sum()) for marker in ALT_MARKERS
        },
        "unparsed_markers": dict(unparsed),
        "unexpected_roles": dict(unexpected),
        "tools_called_in_response": tool_names.most_common(20),
        "calls_in_response": dict(sorted(Counter(calls_in_response).items())),
    }


def length_report(frame) -> dict:
    out = {}
    for col in ("prompt", "response"):
        chars = frame[col].str.len().to_numpy()
        out[col] = {
            "chars_mean": round(float(chars.mean())),
            "chars_p50": round(float(np.percentile(chars, 50))),
            "chars_p90": round(float(np.percentile(chars, 90))),
            "chars_max": int(chars.max()),
        }
    return out


def system_block_report(frame) -> dict:
    """Distinct ⟦SYSTEM⟧ blocks per domain: one each means the policy is a fixed text per domain."""
    blocks: dict[str, set[str]] = {}
    for row in frame.itertuples():
        system = system_turn(parse_transcript(row.prompt))
        blocks.setdefault(domain_of(row.id), set()).add(system.text if system else "")
    return {
        domain: {"distinct": len(texts), "chars": sorted(len(t) for t in texts)}
        for domain, texts in sorted(blocks.items())
    }


def trivial_baselines(y: np.ndarray, frame, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    response_len = frame["response"].str.len().to_numpy()
    predictors = {
        "always_1": np.ones_like(y),
        "always_0": np.zeros_like(y),
        "random_at_base_rate": (rng.random(len(y)) < float(y.mean())).astype(int),
        "long_response": (response_len > np.median(response_len)).astype(int),
        "response_has_tool_call": frame["response"].str.contains("→ TOOL_CALL", regex=False).astype(int).to_numpy(),
    }
    out = {}
    for name, pred in predictors.items():
        f1, lo, hi = bootstrap_ci(y, pred)
        s = binary_scores(y, pred)
        out[name] = {
            "f1": round(f1, 4),
            "f1_ci95": [round(lo, 4), round(hi, 4)],
            "precision": round(s.precision, 4),
            "recall": round(s.recall, 4),
            "predicted_positive": s.predicted_positive,
        }
    return out


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data/raw/valid.parquet")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    frame = pd.DataFrame(read_rows(args.input))
    missing = {"id", "prompt", "response", "label"} - set(frame.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}; found {list(frame.columns)}")

    y = frame["label"].to_numpy().astype(int)
    domains = frame["id"].map(domain_of).to_numpy()
    explanations = frame["explanation"].fillna("").astype(str).str.strip() if "explanation" in frame else None

    eda = {
        "input": str(args.input),
        "format": format_report(frame),
        "lengths": length_report(frame),
        "label_counts": {str(k): int(v) for k, v in sorted(Counter(y.tolist()).items())},
        "positive_rate": round(float(y.mean()), 4),
        "by_domain": {
            domain: {"rows": int((domains == domain).sum()), "positives": int(y[domains == domain].sum())}
            for domain in sorted(set(domains))
        },
        "explanation_non_empty_by_label": (
            {str(k): int(explanations[y == k].ne("").sum()) for k in (0, 1)} if explanations is not None else None
        ),
        "system_blocks_by_domain": system_block_report(frame),
        "duplicate_prompts": int(frame["prompt"].duplicated().sum()),
        "duplicate_prompt_response_pairs": int(frame.duplicated(subset=["prompt", "response"]).sum()),
    }
    baselines = trivial_baselines(y, frame)

    args.out_dir.mkdir(exist_ok=True)
    (args.out_dir / "eda.json").write_text(json.dumps(eda, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "trivial_baselines.json").write_text(
        json.dumps(baselines, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fmt = eda["format"]
    print(f"rows: {fmt['rows']}, positive rate: {eda['positive_rate']}, by domain: {eda['by_domain']}")
    print(f"markers: {fmt['markers'][:6]}")
    print(f"markers documented on the Data tab: {fmt['markers_documented_on_data_tab']}")
    print(f"unparsed markers: {fmt['unparsed_markers']}, unexpected roles: {fmt['unexpected_roles']}")
    print(f"distinct system blocks by domain: {eda['system_blocks_by_domain']}")
    print(f"explanation non-empty by label: {eda['explanation_non_empty_by_label']}")
    print(f"prompt length in chars: {eda['lengths']['prompt']}")
    for name, m in baselines.items():
        lo, hi = m["f1_ci95"]
        print(f"{name:>24}  F1 {m['f1']:.4f} [{lo:.4f}; {hi:.4f}]  P {m['precision']:.3f}  R {m['recall']:.3f}")


if __name__ == "__main__":
    main()
