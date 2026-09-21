from guardian.parsing import parse_available_tools, parse_transcript, system_turn
from guardian.render import render_system, render_tools, render_trajectory

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_reservation_flights",
            "description": "Update the flight information of a reservation.",
            "parameters": {
                "$defs": {
                    "FlightInfo": {
                        "properties": {"flight_number": {"description": "Flight number.", "type": "string"}},
                        "required": ["flight_number"],
                        "type": "object",
                    }
                },
                "properties": {
                    "cabin": {"description": "Cabin.", "enum": ["business", "economy"], "type": "string"},
                    "flights": {
                        "description": "All flights,\nchanged or not.",
                        "items": {"anyOf": [{"$ref": "#/$defs/FlightInfo"}, {"type": "object"}]},
                        "type": "array",
                    },
                    "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Optional."},
                },
                "required": ["cabin", "flights"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human_agents",
            "description": "Transfer the user.\n\nOnly transfer if asked.",
            "parameters": {"properties": {"summary": {"type": "string"}}, "required": ["summary"]},
        },
    },
]

MESSAGES = [
    {"role": "system", "content": "# Policy\nBe careful."},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Change my flight."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "function": {
                    "name": "update_reservation_flights",
                    "arguments": '{"flights": [{"flight_number": "HAT1"}], "cabin": "economy"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "Error: not allowed"},
    {"role": "assistant", "content": "Sorry, I cannot change it."},
]


def test_tools_block_layout():
    assert render_tools(TOOLS).split("\n") == [
        "[AVAILABLE TOOLS]",
        "- update_reservation_flights — Update the flight information of a reservation.",
        "    cabin: string! [enum: business|economy] — Cabin.",
        "    flights: array! — All flights, changed or not.",
        "        · flight_number: string! — Flight number.",
        "    note: string — Optional.",
        "- transfer_to_human_agents — Transfer the user.",
        "    ",
        "    Only transfer if asked.",
        "    summary: string!",
    ]


def test_rendered_tools_parse_back_to_the_schema():
    tools = parse_available_tools(render_system("# Policy", TOOLS))
    flights = tools["update_reservation_flights"]
    assert flights.required == {"cabin", "flights"}
    assert flights.params["cabin"].enum == ["business", "economy"]
    assert flights.params["flights.flight_number"].parent == "flights"


def test_trajectory_steps():
    steps = render_trajectory(MESSAGES, render_system(MESSAGES[0]["content"], TOOLS))
    assert [(s.message_index, s.turn) for s in steps] == [(3, 1), (5, 2)]
    call = steps[0]
    assert call.response == (
        "⟦ASSISTANT · ход 1⟧\n"
        '\t→ TOOL_CALL update_reservation_flights: {"cabin": "economy", "flights": [{"flight_number": "HAT1"}]}'
    )
    assert "\n\n\n⟦ASSISTANT⟧\nHi! How can I help?\n\n⟦USER⟧\nChange my flight." in call.prompt
    turns = parse_transcript(steps[1].prompt)
    assert [t.role for t in turns] == ["SYSTEM", "ASSISTANT", "USER", "ASSISTANT"]
    assert turns[-1].results[0].is_error
    assert system_turn(turns).text.startswith("<instructions>")
