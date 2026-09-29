"""The judge's verdict from the same pass as a third term of the final system (submission 10).

The last state of the pass sits where the organiser's judge prompt expects its one-token answer, so
state . (output row of "1" - output row of "0") is the judge's logit without reasoning, the signal of submission 2,
at no cost in platform time. Its rank against the same score on the 384 organiser contexts (FP8 features of the
rental) is added at weight 0.25; the threshold is then set to flag as many of the rental's 548 timing rows as
submission 8 did (149), so the test result compares with 0.4956 and 0.5 at about the same number of flags.

    python scripts/add_judge.py --final submission_work/final_fp8_gold.json --out submission_work/final_fp8_gold_judge.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

from guardian.final import Final

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from threshold_choice import timing_scores  # noqa: E402
from train_head_ru import vectors  # noqa: E402


def judge_direction(model_dir: Path) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    (one,), (zero,) = tokenizer.encode("1", add_special_tokens=False), tokenizer.encode("0", add_special_tokens=False)
    for path in sorted(glob.glob(str(model_dir / "*.safetensors"))):
        with safe_open(path, "pt") as f:
            if "lm_head.weight" in f.keys():
                rows = f.get_slice("lm_head.weight")
                return (rows[one : one + 1].float().numpy()[0] - rows[zero : zero + 1].float().numpy()[0]).astype(
                    np.float64
                )
    raise KeyError(f"no lm_head.weight in {model_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "data/raw/baseline/baseline/model")
    ap.add_argument("--context-vectors", default="fp8_contexts_final_org")
    ap.add_argument("--weight", type=float, default=0.25)
    ap.add_argument("--timing-flags", type=int, default=149)
    ap.add_argument("--rental", type=Path, default=ROOT / "submission_work/rental")
    args = ap.parse_args()

    d = judge_direction(args.model_dir)
    ids = np.load(args.final.with_suffix(".oof.npz"), allow_pickle=True)["ids"].astype(str)
    v = vectors(args.context_vectors)
    final = json.loads(args.final.read_text(encoding="utf-8"))
    final |= {
        "judge_direction": d.tolist(),
        "reference_j": sorted(float(v[i] @ d) for i in ids),
        "weight_j": args.weight,
    }
    args.out.write_text(json.dumps(final), encoding="utf-8")
    s, rules = timing_scores(args.rental, args.out)
    candidates = np.unique(s)
    flags = np.array([(rules | (s >= t)).sum() for t in candidates])
    final["threshold"] = float(candidates[np.flatnonzero(flags >= args.timing_flags).max()])
    args.out.write_text(json.dumps(final), encoding="utf-8")
    model = Final.from_dict(final)
    print(
        f"judge weight {args.weight}, threshold {model.threshold}, timing rows flagged "
        f"{int((rules | (s >= model.threshold)).sum())} of {len(s)}"
    )


if __name__ == "__main__":
    main()
