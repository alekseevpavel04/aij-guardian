from pathlib import Path

import pytest

from conftest import response, to_rules_pdf_format
from guardian.parsing import (
    flatten,
    format_recognised,
    normalize_format,
    parse_available_tools,
    parse_transcript,
    section,
    system_turn,
    unknown_roles,
    unparsed_markers,
)


def test_roles_and_turn_numbers(prompt):
    turns = parse_transcript(prompt)
    assert [(t.role, t.index) for t in turns] == [
        ("SYSTEM", None),
        ("ASSISTANT", None),
        ("USER", None),
        ("ASSISTANT", 1),
        ("ASSISTANT", 2),
    ]


def test_every_marker_is_consumed(prompt):
    assert unparsed_markers(prompt) == []


def test_only_the_four_roles_of_the_data_are_expected(prompt):
    assert unknown_roles(parse_transcript(prompt)) == set()


def test_an_unknown_marker_shows_up_as_an_unexpected_role(prompt):
    stray = prompt + "\n⟦CRITIC⟧\nlooks fine"
    assert unknown_roles(parse_transcript(stray)) == {"CRITIC"}


def test_rules_pdf_format_parses_to_the_same_turns(prompt):
    pdf = to_rules_pdf_format(prompt, wrapper="prompt")
    assert "→ TOOL_CALL" not in pdf
    expected = [(t.role, t.text, t.calls, t.results) for t in parse_transcript(prompt)]
    assert [(t.role, t.text, t.calls, t.results) for t in parse_transcript(pdf)] == expected
    assert unparsed_markers(pdf) == []
    assert unknown_roles(parse_transcript(pdf)) == set()


def test_rules_pdf_format_error_result_without_error_attribute_is_flagged(prompt):
    turn = parse_transcript(to_rules_pdf_format(prompt))[4]
    assert turn.results[0].is_error is True


def test_user_tool_calls_are_not_the_agents():
    text = (
        "⟦ASSISTANT⟧\nПроверьте статус сети.\n"
        '⟦USER_TOOL_CALL name="check_network_status"⟧\n{}\n'
        '⟦TOOL_RESULT name="check_network_status" requestor="user"⟧\n{"signal": "none"}\n'
        '⟦ASSISTANT_TOOL_CALL name="get_details_by_id"⟧\n{"id": "L1001"}\n'
    )
    turns = parse_transcript(text)
    assert [(t.role, [c.name for c in t.calls]) for t in turns] == [
        ("ASSISTANT", []),
        ("USER", ["check_network_status"]),
        ("ASSISTANT", ["get_details_by_id"]),
    ]
    assert turns[1].results[0].value == {"signal": "none"}


def test_data_format_passes_through_unchanged(prompt):
    assert normalize_format(prompt) is prompt


def test_format_recognised(prompt):
    assert format_recognised(prompt)
    assert format_recognised(to_rules_pdf_format(prompt, wrapper="prompt"))
    assert not format_recognised("⟦USER⟧\nno system turn")


VALID = Path(__file__).resolve().parents[1] / "data" / "raw" / "valid.parquet"


@pytest.mark.skipif(not VALID.exists(), reason="valid.parquet is not in the repository")
def test_rules_pdf_format_gives_the_same_evidence_on_the_labelled_file():
    import pandas as pd

    from guardian.rules import CHECKS, evidence

    # unsupported_number is left out: it reads turn numbers ("ход 3") as context numbers, and the PDF
    # format has none, so it differs on retail__47::t12. It is not one of the predictor's checks.
    frame = pd.read_parquet(VALID)
    for row in frame.itertuples():
        pdf = evidence(to_rules_pdf_format(row.prompt, "prompt"), to_rules_pdf_format(row.response, "response"))
        data = evidence(row.prompt, row.response)
        assert {k: pdf[k] for k in CHECKS} == {k: data[k] for k in CHECKS}, row.id


def test_tool_calls_and_results(prompt):
    turn = parse_transcript(prompt)[3]
    assert [c.name for c in turn.calls] == ["get_user_details"]
    assert turn.calls[0].args == {"user_id": "sofia_kim_7287"}
    assert turn.results[0].value["reservations"] == ["OI5L9G"]
    assert turn.results[0].is_error is False


def test_error_results_are_flagged(prompt):
    turn = parse_transcript(prompt)[4]
    assert turn.results[0].is_error is True
    assert turn.results[0].value is None


def test_tool_lines_are_removed_from_turn_text(prompt):
    turn = parse_transcript(prompt)[3]
    assert "TOOL_CALL" not in turn.text
    assert "TOOL_RESPONSE" not in turn.text
    assert turn.text == ""


def test_response_with_text_and_a_call():
    body = 'Перевожу вас на оператора.\n\t→ TOOL_CALL transfer_to_human_agents: {"summary": "x"}'
    turns = parse_transcript(response(body))
    assert turns[0].text == "Перевожу вас на оператора."
    assert turns[0].calls[0].name == "transfer_to_human_agents"


def test_text_without_markers_is_kept():
    turns = parse_transcript("just text")
    assert (turns[0].role, turns[0].text) == ("PREAMBLE", "just text")
    assert parse_transcript("   ") == []


def test_policy_and_instructions(prompt):
    system = system_turn(parse_transcript(prompt))
    assert "The current time is 2024-05-15 15:00:00 EST." in section(system.text, "policy")
    assert "customer service agent" in section(system.text, "instructions")
    assert section(system.text, "missing") is None


def test_tool_schema(system_text):
    tools = parse_available_tools(system_text)
    assert set(tools) == {
        "get_user_details",
        "cancel_reservation",
        "update_reservation_flights",
        "transfer_to_human_agents",
    }
    update = tools["update_reservation_flights"]
    assert update.top_level == {"reservation_id", "cabin", "flights"}
    assert update.required == {"reservation_id", "cabin", "flights"}
    assert update.params["cabin"].enum == ["business", "economy", "basic_economy"]
    assert update.params["flights.flight_number"].parent == "flights"
    assert update.params["flights.date"].type == "string"


def test_wrapped_description_lines_are_not_parsed_as_parameters(system_text):
    update = parse_available_tools(system_text)["update_reservation_flights"]
    assert "Even if a flight segment is not changed" not in update.params
    assert all(":" not in name for name in update.params)


def test_no_tools_section():
    assert parse_available_tools("⟦SYSTEM⟧\nnothing here") == {}


def test_flatten_paths():
    payload = {"user": {"name": "Noah", "ids": [7, 8]}, "ok": True}
    assert flatten(payload) == {"user.name": "Noah", "user.ids[0]": 7, "user.ids[1]": 8, "ok": True}
