import pytest

from guardian.parsing import (
    Turn,
    flatten,
    parse_transcript,
    strip_block_tags,
    tool_calls,
    tool_results,
    unknown_roles,
)

TRANSCRIPT = """<prompt>
⟦SYSTEM⟧
You are an airline agent. <policy>Never cancel without confirmation.</policy>
⟦USER⟧
Cancel my flight please.
⟦ASSISTANT⟧
Let me look you up.
⟦ASSISTANT_TOOL_CALL name="get_user_details"⟧
{"user_id": "noah_muller_9847"}
⟦TOOL_RESULT name="get_user_details" requestor="assistant"⟧
{"user_id": "noah_muller_9847", "reservations": [{"id": "ABC123", "price": 512.5}]}
</prompt>"""


def test_roles_and_order():
    turns = parse_transcript(TRANSCRIPT)
    assert [t.role for t in turns] == ["SYSTEM", "USER", "ASSISTANT", "ASSISTANT_TOOL_CALL", "TOOL_RESULT"]
    assert unknown_roles(turns) == set()


def test_block_tags_stripped():
    assert "<prompt>" not in parse_transcript(TRANSCRIPT)[0].text
    assert strip_block_tags("no tags here") == "no tags here"


def test_attributes_and_payload():
    call = tool_calls(parse_transcript(TRANSCRIPT))[0]
    assert call.name == "get_user_details"
    assert call.payload == {"user_id": "noah_muller_9847"}

    result = tool_results(parse_transcript(TRANSCRIPT))[0]
    assert result.attrs == {"name": "get_user_details", "requestor": "assistant"}
    assert result.payload["reservations"][0]["price"] == 512.5


def test_text_content_keeps_policy_markup():
    system = parse_transcript(TRANSCRIPT)[0]
    assert "<policy>Never cancel without confirmation.</policy>" in system.text


def test_response_block():
    response = """<response>
⟦ASSISTANT⟧
Cancelled, refund is 512.50 USD.
⟦ASSISTANT_TOOL_CALL name="cancel_reservation"⟧
{"reservation_id": "ABC123"}
</response>"""
    turns = parse_transcript(response)
    assert len(turns) == 2
    assert tool_calls(turns)[0].payload == {"reservation_id": "ABC123"}


def test_text_before_first_marker_is_kept():
    turns = parse_transcript("stray header\n⟦USER⟧\nhi")
    assert turns[0] == Turn("PREAMBLE", "stray header")
    assert turns[1].role == "USER"


def test_text_without_markers_is_kept():
    assert parse_transcript("just text") == [Turn("PREAMBLE", "just text")]
    assert parse_transcript("   ") == []


def test_unknown_marker_is_reported_not_dropped():
    turns = parse_transcript("⟦TOOL_CALL⟧\nsomething")
    assert unknown_roles(turns) == {"TOOL_CALL"}
    assert turns[0].text == "something"


def test_user_tool_call_is_a_tool_call():
    turns = parse_transcript('⟦USER_TOOL_CALL name="send_photo"⟧\n{}')
    assert tool_calls(turns)[0].name == "send_photo"


def test_payload_of_non_json_body():
    turns = parse_transcript('⟦TOOL_RESULT name="ping" requestor="user"⟧\nError: connection refused')
    assert turns[0].payload is None


def test_payload_of_python_literal_body():
    turns = parse_transcript("⟦TOOL_RESULT name=\"f\" requestor=\"assistant\"⟧\n{'ok': True, 'x': None}")
    assert turns[0].payload == {"ok": True, "x": None}


def test_marker_like_text_inside_body_is_not_a_marker():
    turns = parse_transcript("⟦USER⟧\nthe agent printed ⟦not a marker⟧ in its answer")
    assert len(turns) == 1
    assert "⟦not a marker⟧" in turns[0].text


@pytest.mark.parametrize("blob", ["⟦ASSISTANT⟧", "⟦ ASSISTANT ⟧"])
def test_marker_spacing(blob):
    assert parse_transcript(f"{blob}\nhello")[0].role == "ASSISTANT"


def test_flatten_paths():
    payload = {"user": {"name": "Noah", "ids": [7, 8]}, "ok": True}
    assert flatten(payload) == {"user.name": "Noah", "user.ids[0]": 7, "user.ids[1]": 8, "ok": True}


def test_flatten_scalar_and_empty():
    assert flatten(5) == {"": 5}
    assert flatten({"a": {}}) == {}
