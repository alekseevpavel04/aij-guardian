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
