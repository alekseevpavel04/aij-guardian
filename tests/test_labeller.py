from guardian.labeller import ONE_CALL_RULES, without_one_call_rule


def test_removes_both_sentences_and_their_paragraph_break():
    prompt = f"<instructions>\nBe helpful.\n\n{ONE_CALL_RULES[0]}\n\nDeny bad requests.\n</instructions>\n- {ONE_CALL_RULES[1]}"
    out = without_one_call_rule(prompt)
    assert "one tool call at a time" not in out
    assert "Be helpful.\n\nDeny bad requests." in out


def test_leaves_a_prompt_without_the_rule_unchanged():
    prompt = "⟦SYSTEM⟧\nNo such rule here.\n⟦USER⟧\nПривет"
    assert without_one_call_rule(prompt) == prompt
