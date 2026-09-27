"""States and token probabilities of the agent's turn from one pass, through llama.dll itself.

llama-server cannot do this: with `--pooling none` a row must fit one physical batch (a row is up to
65 000 tokens, the card holds 2048), and it gives no probabilities of prompt tokens. The library can:
the row goes in causal chunks of 2048 tokens and outputs are asked only at the positions needed. The
row is rendered as for embed_rows.py; the turn's span is found by tokenizing the text up to it and
after it. Kept per row, in the journal format of embed_rows.py (train_head_ru.vectors reads them):

- turn_mean, turn_last, history_end (the token before the turn) and final (the pooling-last position,
  a check against q4_last_org), the final hidden states after the output norm;
- token statistics in <prefix>_uncertainty_org.jsonl: for every token of the turn
  the log-probability the model gave it (read at the position before it), the entropy and the margin
  of the top two there, so any summary can be taken later without another pass.

At home the model loads as llama-server loads it (all layers on the card, the experts' weights in RAM); on
a Kaggle session with two T4 (scripts/kaggle_tok.py) the whole model fits the cards, which is several times
faster. The ctypes structs follow include/llama.h of release b11052 field by field, on Windows and Linux.

    python scripts/embed_llama.py --rows data/synthetic/organizer_contexts.parquet --prefix q4_tok --max-rows 3
"""

import argparse
import ctypes as C
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from guardian import journal
from guardian.env import setup
from guardian.judge import render

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from embed_rows import BASELINE_PROMPT, encode  # noqa: E402

LIB = Path("D:/VScode_projects/llama.cpp/b11052")
GGUF = Path("D:/VScode_projects/llm-models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf")
KINDS = ("turn_mean", "turn_last", "history_end", "final")


class BuftOverride(C.Structure):
    _fields_ = [("pattern", C.c_char_p), ("buft", C.c_void_p)]


class ModelParams(C.Structure):
    _fields_ = [
        ("devices", C.c_void_p),
        ("tensor_buft_overrides", C.POINTER(BuftOverride)),
        ("n_gpu_layers", C.c_int32),
        ("split_mode", C.c_int),
        ("load_mode", C.c_int),
        ("lazy_mode", C.c_int),
        ("main_gpu", C.c_int32),
        ("tensor_split", C.c_void_p),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
        ("kv_overrides", C.c_void_p),
        ("vocab_only", C.c_bool),
        ("check_tensors", C.c_bool),
        ("use_extra_bufts", C.c_bool),
        ("no_host", C.c_bool),
        ("no_alloc", C.c_bool),
        ("load_mtp", C.c_bool),
    ]


class ContextParams(C.Structure):
    _fields_ = [
        ("n_ctx", C.c_uint32),
        ("n_batch", C.c_uint32),
        ("n_ubatch", C.c_uint32),
        ("n_seq_max", C.c_uint32),
        ("n_rs_seq", C.c_uint32),
        ("n_outputs_max", C.c_uint32),
        ("n_outputs_max_per_seq", C.c_uint32),
        ("n_threads", C.c_int32),
        ("n_threads_batch", C.c_int32),
        ("ctx_type", C.c_int),
        ("rope_scaling_type", C.c_int),
        ("pooling_type", C.c_int),
        ("attention_type", C.c_int),
        ("flash_attn_type", C.c_int),
        ("rope_freq_base", C.c_float),
        ("rope_freq_scale", C.c_float),
        ("yarn_ext_factor", C.c_float),
        ("yarn_attn_factor", C.c_float),
        ("yarn_beta_fast", C.c_float),
        ("yarn_beta_slow", C.c_float),
        ("yarn_orig_ctx", C.c_uint32),
        ("defrag_thold", C.c_float),
        ("cb_eval", C.c_void_p),
        ("cb_eval_user_data", C.c_void_p),
        ("type_k", C.c_int),
        ("type_v", C.c_int),
        ("abort_callback", C.c_void_p),
        ("abort_callback_data", C.c_void_p),
        ("embeddings", C.c_bool),
        ("offload_kqv", C.c_bool),
        ("no_perf", C.c_bool),
        ("op_offload", C.c_bool),
        ("swa_full", C.c_bool),
        ("kv_unified", C.c_bool),
        ("samplers", C.c_void_p),
        ("n_samplers", C.c_size_t),
        ("ctx_other", C.c_void_p),
    ]


class Batch(C.Structure):
    _fields_ = [
        ("n_tokens", C.c_int32),
        ("token", C.POINTER(C.c_int32)),
        ("embd", C.POINTER(C.c_float)),
        ("pos", C.POINTER(C.c_int32)),
        ("n_seq_id", C.POINTER(C.c_int32)),
        ("seq_id", C.POINTER(C.POINTER(C.c_int32))),
        ("logits", C.POINTER(C.c_int8)),
    ]


class Llama:
    def __init__(
        self, gguf: Path, n_ctx: int, n_batch: int, threads: int, lib_dir: Path = LIB, experts_in_ram: bool = True
    ) -> None:
        if os.name == "nt":
            os.add_dll_directory(str(lib_dir))
            names = ("ggml.dll", "ggml-base.dll", "llama.dll")
        else:
            names = ("libggml.so", "libggml-base.so", "libllama.so")
        self.ggml = C.CDLL(str(lib_dir / names[0]))
        base = C.CDLL(str(lib_dir / names[1]))
        lib = self.lib = C.CDLL(str(lib_dir / names[2]))
        # the backends are separate DLLs, looked up next to the executable unless a folder is given
        self.ggml.ggml_backend_load_all_from_path(str(lib_dir).encode())
        lib.llama_backend_init()
        self.ggml.ggml_backend_dev_by_type.restype = C.c_void_p
        self.ggml.ggml_backend_dev_by_type.argtypes = [C.c_int]
        base.ggml_backend_dev_buffer_type.restype = C.c_void_p
        base.ggml_backend_dev_buffer_type.argtypes = [C.c_void_p]
        cpu = base.ggml_backend_dev_buffer_type(self.ggml.ggml_backend_dev_by_type(0))  # GGML_BACKEND_DEVICE_TYPE_CPU
        # as --n-cpu-moe for every layer: the experts' weights stay in RAM
        self.overrides = (BuftOverride * 2)(BuftOverride(rb"\.ffn_(up|down|gate|gate_up)_(ch|)exps", cpu))
        lib.llama_model_default_params.restype = ModelParams
        lib.llama_context_default_params.restype = ContextParams
        mp = lib.llama_model_default_params()
        mp.n_gpu_layers = 99
        if experts_in_ram:
            mp.tensor_buft_overrides = self.overrides
        lib.llama_model_load_from_file.restype = C.c_void_p
        lib.llama_model_load_from_file.argtypes = [C.c_char_p, ModelParams]
        self.model = lib.llama_model_load_from_file(str(gguf).encode(), mp)
        if not self.model:
            raise RuntimeError("model did not load")
        cp = lib.llama_context_default_params()
        cp.n_ctx, cp.n_batch, cp.n_ubatch = n_ctx, n_batch, n_batch
        cp.n_threads = cp.n_threads_batch = threads
        cp.embeddings = True
        cp.pooling_type = 0  # LLAMA_POOLING_TYPE_NONE
        cp.flash_attn_type = 1
        lib.llama_init_from_model.restype = C.c_void_p
        lib.llama_init_from_model.argtypes = [C.c_void_p, ContextParams]
        self.ctx = lib.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("context did not start")
        lib.llama_model_get_vocab.restype = C.c_void_p
        lib.llama_model_get_vocab.argtypes = [C.c_void_p]
        self.vocab = lib.llama_model_get_vocab(self.model)
        lib.llama_vocab_n_tokens.argtypes = [C.c_void_p]
        self.n_vocab = lib.llama_vocab_n_tokens(self.vocab)
        lib.llama_model_n_embd.argtypes = [C.c_void_p]
        self.n_embd = lib.llama_model_n_embd(self.model)
        lib.llama_tokenize.argtypes = [C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(C.c_int32), C.c_int32]
        lib.llama_tokenize.argtypes += [C.c_bool, C.c_bool]
        lib.llama_batch_init.restype = Batch
        lib.llama_batch_init.argtypes = [C.c_int32, C.c_int32, C.c_int32]
        lib.llama_decode.argtypes = [C.c_void_p, Batch]
        lib.llama_get_logits_ith.restype = C.POINTER(C.c_float)
        lib.llama_get_logits_ith.argtypes = [C.c_void_p, C.c_int32]
        lib.llama_get_embeddings_ith.restype = C.POINTER(C.c_float)
        lib.llama_get_embeddings_ith.argtypes = [C.c_void_p, C.c_int32]
        lib.llama_get_memory.restype = C.c_void_p
        lib.llama_get_memory.argtypes = [C.c_void_p]
        lib.llama_memory_clear.argtypes = [C.c_void_p, C.c_bool]
        self.n_batch = n_batch
        self.batch = lib.llama_batch_init(n_batch, 0, 1)

    def tokenize(self, text: str) -> list[int]:
        data = text.encode()
        out = (C.c_int32 * (len(data) + 16))()
        n = self.lib.llama_tokenize(self.vocab, data, len(data), out, len(out), False, True)
        if n < 0:
            raise RuntimeError("tokenize buffer too small")
        return list(out[:n])

    def run(self, tokens: list[int], wanted: set[int]):
        """Decode in chunks; yields (position, state, logits) for the wanted positions."""
        self.lib.llama_memory_clear(self.lib.llama_get_memory(self.ctx), True)
        b = self.batch
        for start in range(0, len(tokens), self.n_batch):
            chunk = tokens[start : start + self.n_batch]
            b.n_tokens = len(chunk)
            outputs = []
            for k, t in enumerate(chunk):
                b.token[k], b.pos[k], b.n_seq_id[k] = t, start + k, 1
                b.seq_id[k][0] = 0
                b.logits[k] = int(start + k in wanted)
                if start + k in wanted:
                    outputs.append(k)
            if self.lib.llama_decode(self.ctx, b) != 0:
                raise RuntimeError(f"decode failed at token {start}")
            for k in outputs:
                state = np.ctypeslib.as_array(self.lib.llama_get_embeddings_ith(self.ctx, k), (self.n_embd,)).copy()
                logits = np.ctypeslib.as_array(self.lib.llama_get_logits_ith(self.ctx, k), (self.n_vocab,))
                yield start + k, state, logits


def token_stats(logits: np.ndarray, next_token: int) -> tuple[float, float, float]:
    """Log-probability of the next token, entropy and top-two margin of the distribution."""
    z = logits.astype(np.float64)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    top2 = np.partition(p, -2)[-2:]
    entropy = float(-(p * np.log(np.clip(p, 1e-30, None))).sum())
    return float(np.log(max(p[next_token], 1e-30))), entropy, float(top2[1] - top2[0])


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs="+", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--gguf", type=Path, default=GGUF)
    ap.add_argument("--system-prompt", type=Path, default=BASELINE_PROMPT)
    ap.add_argument("--n-ctx", type=int, default=76000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--lib", type=Path, default=LIB, help="folder with the llama.cpp b11052 libraries")
    ap.add_argument("--all-on-gpu", action="store_true", help="the experts' weights on the cards too (two T4)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/interim/embeddings")
    args = ap.parse_args()

    frame = pd.concat(map(pd.read_parquet, args.rows))
    system_prompt = args.system_prompt.read_text(encoding="utf-8")
    folder = args.out_dir
    folder.mkdir(parents=True, exist_ok=True)
    logs = {k: folder / f"{args.prefix}_{k}_org.jsonl" for k in KINDS}
    stats_log = folder / f"{args.prefix}_uncertainty_org.jsonl"
    for log in [*logs.values(), stats_log]:
        journal.repair(log)
    done = {r["id"] for r in journal.read(stats_log)}
    todo = frame[~frame.id.isin(done)]
    if args.max_rows:
        todo = todo.head(args.max_rows)
    print(f"{len(frame)} rows, {len(done)} done before, {len(todo)} to go", flush=True)
    llama = Llama(args.gguf, args.n_ctx, args.batch, args.threads, args.lib, not args.all_on_gpu)
    started = time.monotonic()
    for i, row in enumerate(todo.itertuples(), 1):
        t0 = time.monotonic()
        text = render(system_prompt, row.prompt, row.response, thinking=False)
        cut = text.rindex("<response>\n") + len("<response>\n")
        tokens = llama.tokenize(text)
        before = len(llama.tokenize(text[:cut]))
        through = len(llama.tokenize(text[: cut + len(row.response)]))
        # the distribution for turn token j is read at position j - 1
        wanted = set(range(before - 1, through)) | {len(tokens) - 1}
        states, logp, entropy, margin = {}, [], [], []
        for pos, state, logits in llama.run(tokens, wanted):
            states[pos] = state
            if before - 1 <= pos < through - 1:
                lp, h, m = token_stats(logits, tokens[pos + 1])
                logp.append(lp)
                entropy.append(h)
                margin.append(m)
        vectors = {
            "turn_mean": np.mean([states[p] for p in range(before, through)], axis=0),
            "turn_last": states[through - 1],
            "history_end": states[before - 1],
            "final": states[len(tokens) - 1],
        }
        for kind in KINDS:
            journal.append(
                logs[kind],
                {"id": row.id, "vector": encode(vectors[kind]), "dim": int(vectors[kind].size)},
            )
        # written last: a row counts as done only when all its records are there
        journal.append(
            stats_log,
            {
                "id": row.id,
                "prompt_tokens": len(tokens),
                "turn_tokens": through - before,
                "logp": np.round(logp, 4).tolist(),
                "entropy": np.round(entropy, 4).tolist(),
                "margin": np.round(margin, 4).tolist(),
                "seconds": round(time.monotonic() - t0, 2),
            },
        )
        elapsed = time.monotonic() - started
        print(
            f"{len(done) + i}/{len(frame)} {row.id[:50]:<50} {len(tokens)} tokens, turn {through - before}, "
            f"{time.monotonic() - t0:.1f} s, eta {elapsed / i * (len(todo) - i) / 60:.0f} min",
            flush=True,
        )


if __name__ == "__main__":
    main()
