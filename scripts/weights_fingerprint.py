"""SHA-256 of a few tensors of a checkpoint, to check that a copy downloaded elsewhere holds the same weights.

The submission carries the organiser's FP8 checkpoint (data/raw/baseline/baseline/model); the rented A100
downloads Qwen/Qwen3.6-35B-A3B-FP8 from the Hub instead of 35 GB from home. The shards are named differently in
the two copies, so tensors are compared by name: the output matrix, the embeddings, the final norm, the linear
attention of the first layer and FP8 expert weights with their scales of the first and last layers.

    python scripts/weights_fingerprint.py --model-dir data/raw/baseline/baseline/model   # at home, to results/
    python scripts/weights_fingerprint.py --model-dir /models/qwen36 --check results/weights_fingerprint.json
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = (
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
    "model.language_model.layers.0.linear_attn.",
    "model.language_model.layers.0.mlp.experts.0.",
    "model.language_model.layers.39.mlp.shared_expert.",
)
PER_PREFIX = 6


def header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def fingerprint(model_dir: Path) -> dict[str, str]:
    where = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        meta, start = header(path)
        for name, info in meta.items():
            if name != "__metadata__":
                where[name] = (path, start + info["data_offsets"][0], start + info["data_offsets"][1])
    chosen = []
    for prefix in PREFIXES:
        chosen += sorted(n for n in where if n.startswith(prefix))[:PER_PREFIX]
    out = {}
    for name in chosen:
        path, a, b = where[name]
        with path.open("rb") as f:
            f.seek(a)
            out[name] = hashlib.sha256(f.read(b - a)).hexdigest()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--check", type=Path, default=None, help="a fingerprint to compare with instead of writing one")
    ap.add_argument("--out", type=Path, default=ROOT / "results/weights_fingerprint.json")
    args = ap.parse_args()

    got = fingerprint(args.model_dir)
    if args.check is None:
        args.out.write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
        print(f"{len(got)} tensors -> {args.out}")
        return
    want = json.loads(args.check.read_text(encoding="utf-8"))
    missing = sorted(set(want) - set(got))
    differ = sorted(n for n in want if n in got and got[n] != want[n])
    print(json.dumps({"tensors": len(want), "missing": missing, "differ": differ}, indent=1))
    if missing or differ:
        raise SystemExit("the weights are not the same")
    print("same weights")


if __name__ == "__main__":
    main()
