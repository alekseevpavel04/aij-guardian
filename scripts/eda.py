"""First look at the public labelled file: format, class balance, trivial baselines.

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

from guardian.env import setup
from guardian.io import read_table
from guardian.metrics import binary_scores, bootstrap_ci
from guardian.parsing import parse_transcript, tool_calls, unknown_roles

ROOT = Path(__file__).resolve().parents[1]
MARKER_RE = re.compile(r"⟦[^⟧\n]{0,60}⟧")
ALT_MARKER_RE = re.compile(r"(→ TOOL_CALL|← TOOL_RESPONSE|\[AVAILABLE TOOLS\])")


def format_report(frame) -> dict:
    """Which marker convention the data actually uses: the rules describe one, the Data tab another."""
    text = "\n".join(frame["prompt"].head(200).tolist() + frame["response"].head(200).tolist())
    markers = Counter(m.group(0) for m in MARKER_RE.finditer(text))
    unknown = Counter()
    tool_names = Counter()
    for t in frame["prompt"].head(200):
        turns = parse_transcript(t)
        unknown.update(unknown_roles(turns))
        tool_names.update(c.name for c in tool_calls(turns) if c.name)
    return {
        "columns": list(frame.columns),
        "rows": len(frame),
        "bracket_markers_top": markers.most_common(20),
        "alt_markers_from_data_tab": Counter(m.group(0) for m in ALT_MARKER_RE.finditer(text)).most_common(),
        "unknown_roles": dict(unknown),
        "tool_names_top": tool_names.most_common(30),
        "prompt_has_block_tags": int(frame["prompt"].str.contains("<prompt>", regex=False).sum()),
        "response_has_block_tags": int(frame["response"].str.contains("<response>", regex=False).sum()),
    }


def length_report(frame) -> dict:
    out = {}
    for col in ("prompt", "response"):
        chars = frame[col].str.len().to_numpy()
        out[col] = {
            "chars_mean": float(chars.mean()),
            "chars_p50": float(np.percentile(chars, 50)),
            "chars_p90": float(np.percentile(chars, 90)),
            "chars_p99": float(np.percentile(chars, 99)),
            "chars_max": int(chars.max()),
        }
    return out


def trivial_baselines(y: np.ndarray, frame, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    rate = float(y.mean())
    response_len = frame["response"].str.len().to_numpy()

    predictors = {
        "always_1": np.ones_like(y),
        "always_0": np.zeros_like(y),
        "random_at_base_rate": (rng.random(len(y)) < rate).astype(int),
        "long_response": (response_len > np.median(response_len)).astype(int),
        "response_has_tool_call": frame["response"].str.contains("TOOL_CALL", regex=False).astype(int).to_numpy(),
    }

    out = {}
    for name, pred in predictors.items():
        f1, lo, hi = bootstrap_ci(y, pred)
        s = binary_scores(y, pred)
        out[name] = {
            "f1": round(f1, 5),
            "f1_ci95": [round(lo, 5), round(hi, 5)],
            "precision": round(s.precision, 5),
            "recall": round(s.recall, 5),
            "predicted_positive": s.predicted_positive,
        }
    return out


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data/raw/valid.parquet")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    frame = read_table(args.input)
    missing = {"id", "prompt", "response", "label"} - set(frame.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}; found {list(frame.columns)}")

    y = frame["label"].to_numpy().astype(int)
    eda = {
        "input": str(args.input),
        "format": format_report(frame),
        "lengths": length_report(frame),
        "label_counts": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
        "positive_rate": round(float(y.mean()), 5),
        "has_explanation": "explanation" in frame.columns,
        "explanation_non_empty": (
            int(frame["explanation"].astype(str).str.strip().ne("").sum()) if "explanation" in frame.columns else 0
        ),
        "duplicate_prompts": int(frame["prompt"].duplicated().sum()),
        "duplicate_prompt_response_pairs": int(frame.duplicated(subset=["prompt", "response"]).sum()),
    }
    baselines = trivial_baselines(y, frame)

    args.out_dir.mkdir(exist_ok=True)
    (args.out_dir / "eda.json").write_text(json.dumps(eda, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "trivial_baselines.json").write_text(
        json.dumps(baselines, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"rows: {eda['format']['rows']}, positive rate: {eda['positive_rate']}")
    print(f"markers: {eda['format']['bracket_markers_top'][:6]}")
    print(f"markers from the Data tab: {eda['format']['alt_markers_from_data_tab']}")
    print(f"unknown roles: {eda['format']['unknown_roles']}")
    for name, m in baselines.items():
        lo, hi = m["f1_ci95"]
        print(f"{name:>24}  F1 {m['f1']:.4f} [{lo:.4f}; {hi:.4f}]  P {m['precision']:.3f}  R {m['recall']:.3f}")


if __name__ == "__main__":
    main()
