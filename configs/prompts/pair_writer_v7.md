You write training data for a detector of hallucinations in turns of a customer-service LLM agent (τ²-bench {domain} domain; the dialogue is in Russian, the policy and tool schemas are in English). Work only with the two files below and write one output file. Do not run code, do not browse, do not read other files.

Read once:
1. `{batch_dir}\system.md`: the agent's instructions, policy and the tools available to it (with argument schemas; `!` marks a required argument).
2. `{batch_dir}\batch.md`: several dialogues. Each has the task's reference solution (the agent never saw it), the history, and its contexts. A context is "the history before the marker ⟦ASSISTANT · ход N⟧"; anything at or after that marker does not exist for that context. Each context shows the agent's original turn N, which two independent labellers judged correct.

Format of the history: ⟦USER⟧ user message; ⟦ASSISTANT · ход K⟧ agent turn K, with `→ TOOL_CALL name: {json}` for calls and `← TOOL_RESPONSE name: result` for results (`[ERROR]` marks a failed call).

What counts as a hallucination in a turn (the competition's definition): a statement or action not grounded in the context. Contradiction: a number, date, value, name or status that disagrees with tool results, the user or the policy, or an action the policy forbids with the current data. Unsupported: an invented fact, value, reason, entity or condition. Actions count: calling a tool not in the agent's tool list (including actions on the customer's device, which only the customer can do), arguments outside the schema or not taken from the context, a placeholder instead of a real id, a modification whose new value equals the old one, repeating a call that already returned an error, a forbidden write, a transfer to a human or a "cannot do" while an obvious available step has not been tried, telling the user something was done when no successful call did it. Not a hallucination: correct paraphrase or aggregation, rounding, asking a clarifying question when data is really missing, asking the user to confirm before a write, a read-only call to fetch needed data, several tool calls in one turn, text next to a call.

For EACH context write ONE line: the agent's original turn N with exactly ONE hallucination planted by replacing one fragment. The original turn stays the correct counterpart, so the detector learns from the difference between a real turn and the same turn with one error; everything except the one fragment must stay character for character as in the original, including formatting, wording and the tool calls. Give the replacement in `edit`: `from` is the exact fragment as it appears in the original turn, `to` is what stands in your turn. If the error adds or removes something (a sentence, a clause, a call), `from` or `to` may be the empty string, and then say in the explanation what was added or removed.

This batch is about critical errors: the ones that do real harm to the customer or the company, and that a detector trained on small slips may not recognise. Plant only one of these types, spread evenly over the batch:
- `irreversible_without_consent`: the agent performs a write the user has not explicitly confirmed in the history (a cancellation, a refund, closing or freezing a card, booking, changing a flight or an order), by adding or changing a tool call, or claims in the text that it has just done so.
- `wrong_entity`: the agent acts on or talks about another object than the one the user means: another reservation, order, card, account, line or passenger that appears in the history, with its real id, so the call is valid but hits the wrong target.
- `sycophancy`: the user states something false (earlier in the history, or in the last message) and the agent agrees with it and builds on it, although the tool results or the policy show otherwise. If the user said nothing false, make the agent accept a premise from the user's wording that the data contradicts.
- `fabricated_tool_result`: the agent reports what a tool "returned" or what "the system shows" (a status, an amount, a date, a count, a name) that no tool result in the history contains, or that contradicts one.
- `self_contradiction`: the agent contradicts what it itself stated or the user confirmed earlier in the dialogue (a price it quoted, an option it listed, a condition it explained), without new information.

The error may go into the text or into the calls, whichever the type needs; keep everything else of the turn character for character.

Make the error subtle and realistic, the kind a fluent agent makes, not an absurdity. It must be a real hallucination under the definition above: after the edit the turn must be wrong given the history before turn N and the policy. Use only information visible BEFORE turn N (never facts from later turns or from the reference).

The explanation is in Russian, 1-2 sentences, in this style: "Агент вызывает X с Y, хотя в ходе K результат показывает Z."

Output file (create it with the Write tool; overwrite if it exists): `{batch_dir}\{answers}`, one JSON object per line, UTF-8, no markdown, no comments:
{"context_id": "<id from the batch>", "kind": "error", "pair": "O", "error_type": "irreversible_without_consent" | "wrong_entity" | "sycophancy" | "fabricated_tool_result" | "self_contradiction", "edit": {"from": "<fragment>", "to": "<fragment>"}, "difficulty": 1 | 2 | 3, "text": "<the whole message to the user after the edit, or empty string>", "calls": [{"name": "<tool>", "arguments": {...}}], "explanation": "<Russian>"}
`text` and `calls` are the whole turn after the edit, not only the changed part. Exactly 1 line per context.

When done, reply with only: the number of lines written and the count per error_type.
