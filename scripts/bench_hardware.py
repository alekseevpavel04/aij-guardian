"""Measure what one training and one evaluation cycle cost on this machine.

Two numbers decide how the project is planned: how fast an 8B model reads a long context (that is
the cost of one LLM judge pass and of one generated synthetic example) and how fast a 150M encoder
trains at the sequence lengths the contexts require.

The encoder is built from a config with random weights, so nothing is downloaded and the number is a
property of the architecture and the GPU, not of a checkpoint.

    python scripts/bench_hardware.py

Timings are not reproducible bit for bit, so results/hardware_bench.json is not re-checked by
scripts/verify_numbers.py; it records the machine instead.
"""

import argparse
import json
import platform
import re
import subprocess
import time
from pathlib import Path

from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
LLAMA_BENCH = Path(r"D:\VScode_projects\llama.cpp\b11052\llama-bench.exe")
GGUF = Path(r"D:\VScode_projects\llm-models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf")
ROW_RE = re.compile(r"\|\s*(pp|tg)(\d+)\s*\|\s*([\d.]+)\s*±")


def bench_llama(prompt_tokens: list[int]) -> dict:
    """Prefill and generation throughput of the local 8B model, via llama-bench."""
    if not (LLAMA_BENCH.exists() and GGUF.exists()):
        return {"available": False}
    command = [
        str(LLAMA_BENCH), "-m", str(GGUF), "-ngl", "99", "-r", "1",
        "-p", ",".join(str(t) for t in prompt_tokens), "-n", "64",
        "-ctk", "q8_0", "-ctv", "q8_0",
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    rows = {f"{kind}{size}": float(rate) for kind, size, rate in ROW_RE.findall(result.stdout)}
    return {"available": True, "model": GGUF.name, "kv_cache": "q8_0", "tokens_per_second": rows}


def bench_encoder(lengths: list[int], batches: list[int], max_tokens: int) -> dict:
    import torch
    from transformers import ModernBertConfig, ModernBertForSequenceClassification

    if not torch.cuda.is_available():
        return {"available": False}

    torch.backends.cuda.matmul.allow_tf32 = True
    config = ModernBertConfig(
        vocab_size=50368, hidden_size=768, num_hidden_layers=22, num_attention_heads=12,
        intermediate_size=1152, max_position_embeddings=8192, num_labels=2, reference_compile=False,
    )
    model = ModernBertForSequenceClassification(config).cuda().to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    out = {"available": True, "parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "steps": {}}
    for length in lengths:
        for batch in batches:
            key = f"seq{length}_batch{batch}"
            if length * batch > max_tokens:
                # 4096 x 4 spills past 8 GiB into shared memory at ~1 minute per step: measured once
                # by hand (58 s/step), not worth repeating on every run
                out["steps"][key] = {"skipped": f"more than {max_tokens} tokens per step"}
                continue
            try:
                ids = torch.randint(0, 50000, (batch, length), device="cuda")
                labels = torch.randint(0, 2, (batch,), device="cuda")

                def step(ids: torch.Tensor, labels: torch.Tensor) -> None:
                    loss = model(input_ids=ids, labels=labels).loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                for _ in range(2):
                    step(ids, labels)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for _ in range(4):
                    step(ids, labels)
                torch.cuda.synchronize()
                seconds = (time.perf_counter() - started) / 4
                out["steps"][key] = {
                    "ms_per_step": round(seconds * 1000),
                    "examples_per_second": round(batch / seconds, 1),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }
            except torch.cuda.OutOfMemoryError:
                out["steps"][key] = {"oom": True}
                torch.cuda.empty_cache()
    return out


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/hardware_bench.json")
    ap.add_argument("--prompt-tokens", type=int, nargs="+", default=[512, 4096, 12288])
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--max-tokens-per-step", type=int, default=8192)
    args = ap.parse_args()

    report = {
        "machine": {"gpu": "RTX 3070 8 GiB", "platform": platform.platform(), "python": platform.python_version()},
        "llm_q4": bench_llama(args.prompt_tokens),
        "encoder_training": bench_encoder(args.lengths, args.batches, args.max_tokens_per_step),
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rates = report["llm_q4"].get("tokens_per_second", {})
    for key, value in rates.items():
        print(f"{'8B Q4 ' + key:>22}: {value:.0f} tok/s")
    for key, m in report["encoder_training"].get("steps", {}).items():
        if "skipped" in m:
            print(f"{'encoder ' + key:>22}: skipped, {m['skipped']}")
        elif m.get("oom"):
            print(f"{'encoder ' + key:>22}: out of memory")
        else:
            print(f"{'encoder ' + key:>22}: {m['ms_per_step']:>6} ms/step, "
                  f"{m['examples_per_second']:>5} examples/s, peak {m['peak_gib']} GiB")


if __name__ == "__main__":
    main()
