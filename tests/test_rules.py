"""Each check is pinned to the pattern it was written for, taken from the labelled file.

Cases are named after the example in valid.parquet whose explanation describes that pattern, so a
rewrite of a check can be compared with what it was supposed to catch.
"""

from conftest import response
from guardian.rules import Context, evidence, features, occurs_in


def check(prompt, body):
    return {k: v for k, v in evidence(prompt, response(body), Context.from_prompt(prompt)).items() if v}


def test_clean_response_fires_nothing(prompt):
    body = '\t→ TOOL_CALL cancel_reservation: {"reservation_id": "OI5L9G"}'
    assert check(prompt, body) == {}


def test_tool_outside_available_tools(prompt):
    """telecom: the agent calls check_network_status, a diagnostic the user runs on their device."""
    body = '\t→ TOOL_CALL check_network_status: {"line_id": "L1002"}'
    assert check(prompt, body)["unknown_tool"] == ["check_network_status"]


def test_argument_name_outside_the_schema(prompt):
    """airline__23::t10: payment objects passed with id / source instead of payment_id / amount."""
    body = '\t→ TOOL_CALL cancel_reservation: {"reservation_id": "OI5L9G", "source": "gift_card"}'
    assert check(prompt, body)["unknown_argument"] == ["cancel_reservation.source"]


def test_missing_required_argument_inside_an_array_element(prompt):
    body = (
        '\t→ TOOL_CALL update_reservation_flights: {"reservation_id": "OI5L9G", "cabin": "economy", '
        '"flights": [{"flight_number": "HAT001"}]}'
    )
    assert check(prompt, body)["missing_required_argument"] == ["update_reservation_flights.flights.date"]


def test_enum_violation(prompt):
    body = (
        '\t→ TOOL_CALL update_reservation_flights: {"reservation_id": "OI5L9G", "cabin": "first_class", '
        '"flights": [{"flight_number": "HAT001", "date": "2024-05-20"}]}'
    )
    assert check(prompt, body)["enum_violation"] == ["update_reservation_flights.cabin=first_class"]


def test_repeating_a_call_that_already_failed(prompt):
    """retail__48::t4: the same find_user_id_by_name_zip call that already returned User not found."""
    body = '\t→ TOOL_CALL get_user_details: {"user_id": "unknown_person_1"}'
    found = check(prompt, body)
    assert found["repeated_failed_call"] == ["get_user_details"]
    assert found["repeated_call"] == ["get_user_details"]


def test_repeating_a_call_that_succeeded_is_not_a_failed_repeat(prompt):
    body = '\t→ TOOL_CALL get_user_details: {"user_id": "sofia_kim_7287"}'
    found = check(prompt, body)
    assert "repeated_failed_call" not in found
    assert found["repeated_call"] == ["get_user_details"]


def test_whitespace_does_not_hide_a_repeat(prompt):
    body = '\t→ TOOL_CALL get_user_details: {"user_id":"unknown_person_1"}'
    assert check(prompt, body)["repeated_failed_call"] == ["get_user_details"]


def test_fabricated_identifier(prompt):
    """banking__task_068::t11: account_id passed as light_blue_account_id_placeholder."""
    body = '\t→ TOOL_CALL cancel_reservation: {"reservation_id": "light_blue_account_id_placeholder"}'
    assert check(prompt, body)["unsupported_argument_value"] == [
        "cancel_reservation.reservation_id=light_blue_account_id_placeholder"
    ]


def test_value_hidden_inside_another_token_does_not_count_as_support(prompt):
    """retail__106::t3: zip 3019 taken from the email sofia.thomas3019@example.com."""
    assert occurs_in("3019", "sofia.thomas3019@example.com") is False
    assert occurs_in("sofia_kim_7287", "user_id: sofia_kim_7287.") is True


def test_free_text_arguments_are_not_checked_against_the_context(prompt):
    body = '\t→ TOOL_CALL transfer_to_human_agents: {"summary": "Клиент хочет отменить бронь, требуется оператор"}'
    assert check(prompt, body) == {}


def test_arguments_of_a_wrapped_tool_are_not_checked_against_the_wrapper_schema(prompt):
    """call_discoverable_agent_tool passes another tool's arguments as a JSON string."""
    wrapped = '{\\"card_id\\": \\"OI5L9G\\"}'
    body = f'\t→ TOOL_CALL cancel_reservation: {{"reservation_id": "OI5L9G", "arguments": "{wrapped}"}}'
    found = check(prompt, body)
    assert found["unknown_argument"] == ["cancel_reservation.arguments"]
    assert "unsupported_argument_value" not in found


def test_fabricated_value_inside_a_wrapped_tool_is_still_caught(prompt):
    wrapped = '{\\"card_id\\": \\"NOSUCHID\\"}'
    body = f'\t→ TOOL_CALL cancel_reservation: {{"reservation_id": "OI5L9G", "arguments": "{wrapped}"}}'
    assert check(prompt, body)["unsupported_argument_value"] == ["cancel_reservation.card_id=NOSUCHID"]


def test_numbers_are_not_matched_across_line_breaks(prompt):
    found = check(prompt, "Заказы:\n2872451762\n2 штуки")["unsupported_number"]
    assert all("\n" not in n for n in found)


def test_features_are_counts(prompt):
    body = '\t→ TOOL_CALL check_network_status: {}\n\t→ TOOL_CALL check_sim_status: {}'
    assert features(prompt, response(body))["unknown_tool"] == 2


def test_replacement_equal_to_the_value_it_replaces(prompt):
    """retail__36::t10: two of five items passed with new_item_id equal to the old item_id."""
    body = (
        '\t→ TOOL_CALL cancel_reservation: {"reservation_id": "OI5L9G", '
        '"item_ids": ["A1", "B2"], "new_item_ids": ["A1", "C3"]}'
    )
    assert check(prompt, body)["no_op_replacement"] == ["cancel_reservation.new_item_ids[0]=A1"]


def test_a_real_replacement_does_not_fire(prompt):
    body = (
        '\t→ TOOL_CALL cancel_reservation: {"reservation_id": "OI5L9G", '
        '"item_ids": ["A1", "B2"], "new_item_ids": ["D4", "C3"]}'
    )
    assert "no_op_replacement" not in check(prompt, body)
