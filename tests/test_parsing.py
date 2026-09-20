from conftest import response
from guardian.parsing import (
    flatten,
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


def test_a_marker_of_the_rules_pdf_format_shows_up_as_an_unexpected_role(prompt):
    """The rules PDF documents ⟦ASSISTANT_TOOL_CALL name="X"⟧, which valid.parquet does not use."""
    stray = prompt + '\n⟦ASSISTANT_TOOL_CALL name="x"⟧\n{}'
    assert unparsed_markers(stray) == []
    assert unknown_roles(parse_transcript(stray)) == {"ASSISTANT_TOOL_CALL"}


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
