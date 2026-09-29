import math
from pathlib import Path

import pytest

from guardian.judge import first_top_logprobs, p_one, render, user_text

MODEL = Path(__file__).resolve().parents[1] / "data" / "raw" / "baseline" / "baseline" / "model"


def test_probability_of_one_is_renormalised_over_the_two_labels():
    top = [("1", math.log(0.3)), ("0", math.log(0.1)), ("The", math.log(0.6))]
    assert abs(p_one(top) - 0.75) < 1e-9


def test_one_label_missing_from_the_top_gives_a_certain_verdict():
    assert p_one([("0", -0.1), ("x", -3.0)]) == 0.0
    assert p_one([(" 1", -0.1)]) == 1.0
    assert p_one([("yes", -0.1)]) is None


def test_user_text_wraps_prompt_and_response_like_the_baseline():
    assert user_text("P", "R") == "<prompt>\nP\n</prompt>\n<response>\nR\n</response>"


def test_both_logprob_layouts_are_read():
    llama = {"content": [{"token": "1", "logprob": -0.1, "top_logprobs": [{"token": "1", "logprob": -0.1}]}]}
    vllm = {"tokens": ["1"], "top_logprobs": [{"1": -0.1, "0": -2.5}]}
    assert first_top_logprobs(llama) == [("1", -0.1)]
    assert first_top_logprobs(vllm) == [("1", -0.1), ("0", -2.5)]
    assert first_top_logprobs(None) == []


@pytest.mark.skipif(not MODEL.exists(), reason="needs the organiser's checkpoint for its chat template")
@pytest.mark.parametrize("thinking", [False, True])
def test_render_matches_the_models_chat_template(thinking):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": user_text("P", "R")}]
    expected = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )
    assert render("S", "P", "R", thinking) == expected


def test_client_falls_back_to_the_chat_endpoint_when_completions_are_rejected():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from guardian.judge import JudgeClient

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if self.path.endswith("/chat/completions"):
                top = [{"token": "1", "logprob": math.log(0.8)}, {"token": "0", "logprob": math.log(0.2)}]
                body = {"choices": [{"logprobs": {"content": [{"token": "1", "top_logprobs": top}]}}]}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = JudgeClient(f"http://127.0.0.1:{server.server_port}/v1")
        assert abs(client.score("S", "P", "R").score - 0.8) < 1e-9
        assert client.use_chat
        assert abs(client.score("S", "P", "R").score - 0.8) < 1e-9
    finally:
        server.shutdown()


def test_a_missing_reference_never_becomes_a_block():
    for missing in (None, "", "  ", float("nan")):
        assert "<reference>" not in user_text("P", "R", missing)
    assert user_text("P", "R", "gold").endswith("<reference>\ngold\n</reference>")


def test_verdict_tokens_are_read_through_markdown_wrappers_after_an_answer_prefix():
    assert p_one([("`", -0.1), ("`0", -0.5), ("1", -3.0)], markdown=True) < 0.5
    assert p_one([("**1", -0.2), ("0", -2.0)], markdown=True) > 0.5
    # the quick judge reads bare digits only, as in the submitted cascade
    assert p_one([("**1", -0.2), ("0", -2.0)]) == 0.0


GEMMA4 = Path("D:/VScode_projects/llm-models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")


@pytest.mark.skipif(not GEMMA4.exists(), reason="needs the Gemma 4 GGUF for its chat template")
def test_gemma4_render_matches_the_template_in_the_gguf():
    import json

    from gguf import GGUFReader
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    from guardian.judge import render_gemma4

    field = GGUFReader(GEMMA4).fields["tokenizer.chat_template"]
    template = bytes(field.parts[field.data[0]]).decode("utf-8")
    env = ImmutableSandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, **_: json.dumps(x, ensure_ascii=False)
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": user_text("P", "R")}]
    expected = env.from_string(template).render(
        messages=messages, add_generation_prompt=True, bos_token="<bos>", enable_thinking=False
    )
    assert render_gemma4("S", "P", "R") == expected
