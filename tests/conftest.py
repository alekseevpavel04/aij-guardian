"""Fixtures shaped like the real transcripts in valid.parquet, small enough to read."""

import pytest

SYSTEM = """⟦SYSTEM⟧
<instructions>
You are a customer service agent. Always communicate with the user in Russian.
</instructions>
<policy>
# Airline Agent Policy
The current time is 2024-05-15 15:00:00 EST.
You must obtain explicit user confirmation before updating a booking.
</policy>

[AVAILABLE TOOLS]
- get_user_details — Get the details of a user, including their reservations.
    user_id: string! — The user ID, such as 'sara_doe_496'.
- cancel_reservation — Cancel the whole reservation.
    reservation_id: string! — The reservation ID, such as 'ZFA04Y'.
- update_reservation_flights — Update the flight information of a reservation.

    Even if a flight segment is not changed, it should still be included in the array.
    reservation_id: string! — The reservation ID, such as 'ZFA04Y'.
    cabin: string! [enum: business|economy|basic_economy] — The cabin class of the reservation
    flights: array! — An array of objects describing the ENTIRE new reservation.
        · flight_number: string! — Flight number, such as 'HAT001'.
        · date: string! — The date for the flight in the format 'YYYY-MM-DD'.
- transfer_to_human_agents — Transfer the user to a human agent.
    summary: string! — A summary of the user's issue.
"""

PROMPT = (
    SYSTEM
    + """
⟦ASSISTANT⟧
Здравствуйте! Чем могу помочь?

⟦USER⟧
Отмените моё бронирование.

⟦ASSISTANT · ход 1⟧
\t→ TOOL_CALL get_user_details: {"user_id": "sofia_kim_7287"}
\t← TOOL_RESPONSE get_user_details: {"user_id": "sofia_kim_7287", "reservations": ["OI5L9G"]}

⟦ASSISTANT · ход 2⟧
\t→ TOOL_CALL get_user_details: {"user_id": "unknown_person_1"}
\t← TOOL_RESPONSE get_user_details [ERROR]: Error: user not found
"""
)


@pytest.fixture
def prompt():
    return PROMPT


@pytest.fixture
def system_text():
    return SYSTEM


def response(body: str) -> str:
    return f"⟦ASSISTANT · ход 3⟧\n{body}"
