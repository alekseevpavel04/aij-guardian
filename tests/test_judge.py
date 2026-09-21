import math

from guardian.judge import build_messages, p_one


def test_probability_of_one_is_renormalised_over_the_two_labels():
    top = [("1", math.log(0.3)), ("0", math.log(0.1)), ("The", math.log(0.6))]
    assert abs(p_one(top) - 0.75) < 1e-9


def test_one_label_missing_from_the_top_gives_a_certain_verdict():
    assert p_one([("0", -0.1), ("x", -3.0)]) == 0.0
    assert p_one([(" 1", -0.1)]) == 1.0
    assert p_one([("yes", -0.1)]) is None


def test_messages_wrap_prompt_and_response_like_the_baseline():
    messages = build_messages("sys", "P", "R")
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1]["content"] == "<prompt>\nP\n</prompt>\n<response>\nR\n</response>"
