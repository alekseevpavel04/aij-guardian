"""What the organiser's baseline archive contains, measured from the unpacked files.

The baseline (baseline.tar.gz.00 ... .06 on the Data tab) is unpacked into data/raw/baseline/baseline,
which is not in git. This script records its size, the model and its license, and the contract its
predict.py implements, so that the write-up can cite numbers instead of impressions. The code
itself is not copied into the repository (rules, section 3.7).

    python scripts/baseline_inventory.py --hash
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=Path, default=ROOT / "data/raw/baseline")
    ap.add_argument("--unpacked", type=Path, default=ROOT / "data/raw/baseline/baseline")
    ap.add_argument("--hash", action="store_true", help="sha256 of every part (reads about 31 GB)")
    ap.add_argument("--out", type=Path, default=ROOT / "results/baseline_inventory.json")
    args = ap.parse_args()

    parts = sorted(args.parts.glob("baseline.tar.gz.[0-9][0-9]"))
    files = sorted(p for p in args.unpacked.rglob("*") if p.is_file())
    model = args.unpacked / "model"
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    card = (model / "README.md").read_text(encoding="utf-8")
    predict = (args.unpacked / "scripts/predict.py").read_text(encoding="utf-8")
    server = (args.unpacked / "src/hallucination_detector/server.py").read_text(encoding="utf-8")
    prompt = (args.unpacked / "src/hallucination_detector/prompt.md").read_text(encoding="utf-8")

    weights = [p for p in files if p.suffix == ".safetensors"]
    weights_bytes = sum(p.stat().st_size for p in weights)
    result = {
        "archive_parts": [
            {"name": p.name, "bytes": p.stat().st_size} | ({"sha256": sha256(p)} if args.hash else {}) for p in parts
        ],
        "archive_bytes": sum(p.stat().st_size for p in parts),
        "unpacked_files": len(files),
        "unpacked_bytes": sum(p.stat().st_size for p in files),
        "weights_files": len(weights),
        "weights_bytes": weights_bytes,
        "non_model_files": sorted(
            str(p.relative_to(args.unpacked)).replace("\\", "/") for p in files if model not in p.parents
        ),
        "model": {
            "base_model": re.search(r"base_model:\s*\n-\s*(\S+)", card).group(1),
            "license": re.search(r"^license:\s*(\S+)", card, re.MULTILINE).group(1),
            "architecture": config["architectures"][0],
            "quantization": config.get("quantization_config", {}).get("quant_method"),
            "layers": text["num_hidden_layers"],
            "experts": text["num_experts"],
            "experts_per_token": text["num_experts_per_tok"],
            "hidden_size": text["hidden_size"],
            "max_position_embeddings": text["max_position_embeddings"],
        },
        "predict_contract": {
            "wraps_input_in_prompt_response_tags": "<prompt>\\n{prompt}\\n</prompt>" in predict,
            "reads_parquet_by_extension_else_csv": 'read_parquet(p) if str(p).endswith(".parquet")' in predict,
            "writes_parquet_by_extension_else_csv": 'to_parquet(p, index=False) if str(p).endswith(".parquet")'
            in predict,
            "default_threshold": float(re.search(r'"--threshold", type=float, default=([\d.]+)', predict).group(1)),
            "default_concurrency": int(re.search(r'"--concurrency", type=int, default=(\d+)', predict).group(1)),
            "vllm_max_model_len": int(re.search(r"max_model_len=(\d+)", server).group(1)),
            "vllm_prefix_caching": "--enable-prefix-caching" in server,
            "vllm_language_model_only": "--language-model-only" in server,
        },
        "prompt_markers": sorted(set(re.findall(r"⟦([A-Z_]+)", prompt))),
        "local_run": {
            "ran": False,
            "why": f"the FP8 weights take {weights_bytes / GIB:.1f} GiB, the RTX 3070 has 8 GiB; "
            "vLLM, which predict.py starts, does not run on Windows",
        },
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "archive_parts"}, ensure_ascii=False, indent=2))
    print(
        f"archive {result['archive_bytes'] / GIB:.1f} GiB in {len(parts)} parts, "
        f"unpacked {result['unpacked_bytes'] / GIB:.1f} GiB"
    )


if __name__ == "__main__":
    main()
