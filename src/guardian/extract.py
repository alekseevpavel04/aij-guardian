"""One pass of the judge model over a row inside vLLM: what head B and the uncertainty head read (guardian.final).

The row is rendered as the organiser's baseline judge prompt (guardian.judge.render, no reasoning) and passed once
through the model with vLLM's pooling runner in ALL mode, which returns the final normalised hidden state of every
token. From them:

- the last state is the vector of head B (the same state the LAST pooler gives);
- the states just before each token of the agent's turn, times the checkpoint's output matrix, give the model's
  distribution for that token: its log-probability, the entropy and the gap between the two most probable tokens
  (on a Kaggle T4 these log-probabilities matched vLLM's own prompt_logprobs, results/kaggle_token_probe.json).

Tokens come from the checkpoint's Hugging Face tokenizer; its turn boundaries and parts agree with the home pass
(llama.cpp) on all 385 organiser contexts. Runs only where vLLM, torch and transformers are
installed: the training pass on a rented A100 and the submission on the platform.
"""

import base64
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from guardian.judge import render
from guardian.uncertainty import token_parts


def encode(vector: np.ndarray) -> str:
    """A vector as base64 of float32, the format of the home embedding journals (scripts/embed_rows.py)."""
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode()


def decode(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32)


OUTPUT_MATRIX = ("lm_head.weight", "model.language_model.embed_tokens.weight", "model.embed_tokens.weight")


@dataclass
class Rendered:
    ids: list[int]
    turn: list[int]  # positions of the turn's tokens in ids
    spans: list[tuple[int, int]]  # their character offsets in the turn


def render_tokens(tokenizer, system_prompt: str, prompt: str, response: str) -> Rendered:
    text = render(system_prompt, prompt, response, thinking=False)
    cut = text.rindex("<response>\n") + len("<response>\n")
    end = cut + len(response)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    turn, spans = [], []
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a >= cut and b <= end and b > a:
            turn.append(i)
            spans.append((a - cut, b - cut))
    return Rendered(list(enc["input_ids"]), turn, spans)


def output_matrix(model_dir: Path):
    """The output matrix straight from the checkpoint's safetensors, in float32 on the card (2 GiB for Qwen3.6):
    logits rounded to bfloat16 move the log-probabilities by up to 0.01."""
    import torch
    from safetensors import safe_open

    for path in sorted(glob.glob(str(Path(model_dir) / "*.safetensors"))):
        with safe_open(path, "pt") as f:
            names = set(f.keys())
            for name in OUTPUT_MATRIX:
                if name in names:
                    return f.get_tensor(name).to(device="cuda", dtype=torch.float32)
    raise KeyError(f"no output matrix in {model_dir}")


def turn_stats(states, w, ids: list[int], turn: list[int], chunk: int = 256) -> dict[str, list[float]]:
    """Log-probability, entropy and top-two gap of each turn token; the state at position i predicts token i + 1."""
    import torch

    before = torch.as_tensor([p - 1 for p in turn])
    target = torch.as_tensor([ids[p] for p in turn], device="cuda")
    logp, entropy, margin = [], [], []
    for a in range(0, len(turn), chunk):
        s = torch.as_tensor(states)[before[a : a + chunk]].to(device="cuda", dtype=torch.float32)
        z = torch.log_softmax(s @ w.T, dim=-1)
        p = z.exp()
        top = torch.topk(p, 2, dim=-1).values
        logp += z[torch.arange(len(s), device="cuda"), target[a : a + chunk]].tolist()
        entropy += (-(p * z).sum(-1)).tolist()
        margin += (top[:, 0] - top[:, 1]).tolist()
    return {"logp": logp, "entropy": entropy, "margin": margin}


class Pass:
    """The judge model in vLLM's pooling runner (ALL) plus its output matrix and tokenizer."""

    def __init__(
        self,
        model_dir: Path,
        system_prompt: str,
        max_model_len: int,
        gpu_memory_utilization: float,
        engine: dict | None = None,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.system_prompt = system_prompt
        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.llm = LLM(
            model=str(model_dir),
            runner="pooling",
            convert="embed",
            pooler_config={"pooling_type": "ALL", "use_activation": False},
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            # the checkpoint is a vision-language model; only its text part is needed
            language_model_only=True,
            enable_prefix_caching=False,
            **(engine or {}),
        )
        self.w = output_matrix(model_dir)

    def run(self, rows: list[dict]) -> list[dict | None]:
        """One record per row (None when the row is longer than the model's window): the last-token vector,
        the turn's token statistics and parts, and the token counts."""
        import torch
        from vllm.inputs import TokensPrompt

        rendered = [render_tokens(self.tokenizer, self.system_prompt, r["prompt"], r["response"]) for r in rows]
        fits = [i for i, x in enumerate(rendered) if len(x.ids) <= self.max_model_len and x.turn and x.turn[0] > 0]
        outputs = self.llm.encode(
            [TokensPrompt(prompt_token_ids=rendered[i].ids) for i in fits], pooling_task="token_embed", use_tqdm=False
        )
        records: list[dict | None] = [None] * len(rows)
        for i, o in zip(fits, outputs, strict=True):
            x, states = rendered[i], torch.as_tensor(o.outputs.data)
            records[i] = {
                "id": rows[i]["id"],
                "vector": states[-1].float().cpu().numpy(),
                "prompt_tokens": len(x.ids),
                "turn_tokens": len(x.turn),
                **turn_stats(states, self.w, x.ids, x.turn),
                "parts": token_parts(rows[i]["response"], x.spans),
            }
        return records
