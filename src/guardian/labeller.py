"""What the privileged labeller sees of a row, where it differs from what the detector sees.

The organiser's agent is told to make one tool call at a time and not to talk while calling, yet 12%
of its turns in the organiser's own dialogues (46 of 385) have several calls. The organiser's labels
do not count that as an error: none of the eight labelled turns with several calls names it, all
eight are positive for other reasons. Our labeller does count it, since the rule is written in the
agent's instructions, and a sentence in its own prompt saying otherwise did not outweigh that
(labeler_v3) or made it lenient on real errors (labeler_v2). So the labeller is shown the dialogue
without these two sentences; everything else stays as the agent saw it.
"""

ONE_CALL_RULES = (
    # τ² agent instruction
    "You should only make one tool call at a time, and if you make a tool call, you should not respond to the "
    "user simultaneously. If you respond to the user, you should not make a tool call at the same time.",
    # airline policy
    "You should at most make one tool call at a time, and if you take a tool call, you should not respond to the "
    "user at the same time. If you respond to the user, you should not make a tool call at the same time.",
)


def without_one_call_rule(prompt: str) -> str:
    for sentence in ONE_CALL_RULES:
        prompt = prompt.replace("\n\n" + sentence, "").replace(sentence, "")
    return prompt
