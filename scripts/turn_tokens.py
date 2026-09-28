"""Where in the agent's turn each token falls, so token probabilities can be read per part of the turn (T4.H10).

embed_llama.py keeps the log-probability of every token of the turn but not the token itself. The
same llama.dll, loaded with the vocabulary only (no weights, no card), tokenizes each row again the
same way and gives the text of every token; a token is then marked as part of a tool call's argument
value (a JSON string or number inside `→ TOOL_CALL name: {...}`), an argument name, the call's tool
name, the layout (turn header, call markers, JSON punctuation) or the free text. Summaries per part go
to results through turn_ruler.py.

    python scripts/turn_tokens.py --prefix q4_tok
"""

import argparse
import ctypes as C
import sys
from pathlib import Path

import pandas as pd

from guardian import journal
from guardian.env import setup
from guardian.judge import render
from guardian.uncertainty import part_of_each_char

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from embed_llama import GGUF, LIB, ModelParams  # noqa: E402
from embed_rows import BASELINE_PROMPT  # noqa: E402


class Vocab:
    def __init__(self, gguf: Path) -> None:
        import os

        os.add_dll_directory(str(LIB))
        C.CDLL(str(LIB / "ggml.dll")).ggml_backend_load_all_from_path(str(LIB).encode())
        lib = self.lib = C.CDLL(str(LIB / "llama.dll"))
        lib.llama_backend_init()
        lib.llama_model_default_params.restype = ModelParams
        mp = lib.llama_model_default_params()
        mp.vocab_only = True
        lib.llama_model_load_from_file.restype = C.c_void_p
        lib.llama_model_load_from_file.argtypes = [C.c_char_p, ModelParams]
        model = lib.llama_model_load_from_file(str(gguf).encode(), mp)
        lib.llama_model_get_vocab.restype = C.c_void_p
        lib.llama_model_get_vocab.argtypes = [C.c_void_p]
        self.vocab = lib.llama_model_get_vocab(model)
        lib.llama_tokenize.argtypes = [C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(C.c_int32), C.c_int32]
        lib.llama_tokenize.argtypes += [C.c_bool, C.c_bool]
        lib.llama_token_to_piece.argtypes = [C.c_void_p, C.c_int32, C.c_char_p, C.c_int32, C.c_int32, C.c_bool]

    def tokenize(self, text: str) -> list[int]:
        data = text.encode()
        out = (C.c_int32 * (len(data) + 16))()
        n = self.lib.llama_tokenize(self.vocab, data, len(data), out, len(out), False, True)
        return list(out[:n])

    def piece(self, token: int) -> bytes:
        buf = C.create_string_buffer(256)
        n = self.lib.llama_token_to_piece(self.vocab, token, buf, 256, 0, True)
        return buf.raw[:n]


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--prefix", default="q4_tok")
    args = ap.parse_args()

    frame = pd.read_parquet(args.contexts).set_index("id")
    system_prompt = BASELINE_PROMPT.read_text(encoding="utf-8")
    stats = {
        r["id"]: r for r in journal.read(ROOT / "data/interim/embeddings" / f"{args.prefix}_uncertainty_org.jsonl")
    }
    vocab = Vocab(GGUF)
    out = ROOT / "data/interim/embeddings" / f"{args.prefix}_parts_org.jsonl"
    out.unlink(missing_ok=True)
    mismatched = 0
    for rid, r in stats.items():
        row = frame.loc[rid]
        text = render(system_prompt, row.prompt, row.response, thinking=False)
        cut = text.rindex("<response>\n") + len("<response>\n")
        before = len(vocab.tokenize(text[:cut]))
        tokens = vocab.tokenize(text[: cut + len(row.response)])[before:]
        # the log-probability of token j of the turn sits at index j of the record
        pieces = [vocab.piece(t) for t in tokens]
        turn_bytes = row.response.encode()
        parts, offset = [], 0
        char_parts = part_of_each_char(row.response)
        byte_part = []
        for ch, part in zip(row.response, char_parts, strict=True):
            byte_part += [part] * len(ch.encode())
        for p in pieces:
            span = byte_part[offset : offset + len(p)] or ["text"]
            parts.append(next((k for k in ("value", "key", "name", "text") if k in span), "layout"))
            offset += len(p)
        if offset != len(turn_bytes) or len(parts) != len(r["logp"]):
            mismatched += 1
        journal.append(out, {"id": rid, "parts": parts})
    print(f"{len(stats)} rows, {mismatched} with a token count other than the pass's", flush=True)


if __name__ == "__main__":
    main()
