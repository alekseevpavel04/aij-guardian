from guardian.tau2_sim import GREETING, to_messages
from guardian.tau2_tasks import data_id


def test_task_ids_are_written_the_way_the_data_writes_them():
    task = "[mms_issue]airplane_mode_on|break_app_both_permissions[PERSONA:Hard]"
    assert data_id(task) == "mms_issueairplane_mode_on-break_app_both_permissionsPERSONA_Hard"
    assert data_id("21") == "21"
    assert data_id("task_005") == "task_005"


SIMULATION = {
    "messages": [
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "Интернет не работает."},
        {
            "role": "assistant",
            "content": "Проверю линию.",
            "tool_calls": [
                {"id": "a1", "name": "get_details_by_id", "arguments": {"id": "L1"}, "requestor": "assistant"}
            ],
        },
        {"role": "tool", "id": "a1", "content": '{"status": "Active"}', "requestor": "assistant", "error": False},
        {"role": "user", "content": None, "tool_calls": [{"id": "u1", "name": "check_status_bar", "arguments": {}}]},
        {"role": "tool", "id": "u1", "content": "No signal", "requestor": "user"},
        {"role": "user", "content": "Сигнала нет."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "a2", "name": "enable_roaming", "arguments": {"line_id": "L1"}}],
        },
        {"role": "tool", "id": "a2", "content": "Error: not allowed", "requestor": "assistant", "error": True},
    ]
}


def test_simulation_messages_keep_the_agents_traffic_and_drop_the_users_tools():
    messages = to_messages(SIMULATION)
    assert [m["role"] for m in messages] == ["assistant", "user", "assistant", "tool", "user", "assistant", "tool"]
    assert messages[0]["content"] == GREETING
    assert messages[3]["tool_call_id"] == "a1" and messages[3]["error"] is False
    assert messages[6]["error"] is True
    assert all("check_status_bar" not in str(m) for m in messages)


def test_rendered_simulation_names_results_and_marks_errors():
    from guardian.parsing import parse_transcript
    from guardian.render import render_trajectory

    steps = render_trajectory(to_messages(SIMULATION), "⟦SYSTEM⟧\n<policy>\nP\n</policy>")
    assert [s.turn for s in steps] == [1, 2]
    turns = parse_transcript(steps[1].prompt)
    agent = [t for t in turns if t.role == "ASSISTANT"]
    assert agent[0].text == GREETING
    assert agent[1].results[0].name == "get_details_by_id"
    assert steps[1].response == '⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL enable_roaming: {"line_id": "L1"}'
