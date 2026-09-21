You write training data for a detector of hallucinations in turns of a customer-service LLM agent (τ²-bench airline domain; the dialogue is in Russian, the policy and tool schemas are in English). Work only with the two files below and write one output file. Do not run code, do not browse, do not read other files.

Read once:
1. `data\synthetic\pairs\airline_01\system.md`: the agent's instructions, policy and the tools available to it (with argument schemas; `!` marks a required argument).
2. `data\synthetic\pairs\airline_01\batch.md`: 5 dialogues. Each has the task's reference solution (the agent never saw it), the history, and 2 contexts. A context is "the history before the marker ⟦ASSISTANT · ход N⟧"; anything at or after that marker does not exist for that context. Each context also shows the agent's original turn N.

Format of the history: ⟦USER⟧ user message; ⟦ASSISTANT · ход K⟧ agent turn K, with `→ TOOL_CALL name: {json}` for calls and `← TOOL_RESPONSE name: result` for results (`[ERROR]` marks a failed call).

What counts as a hallucination in a turn (the competition's definition): a statement or action not grounded in the context. Contradiction: a number, date, value, name or status that disagrees with tool results, the user or the policy, or an action the policy forbids with the current data. Unsupported: an invented fact, value, reason, entity or condition. Actions count: calling a tool not in the list, arguments outside the schema or not taken from the context, a forbidden write, a transfer to a human although the tools and policy allow solving it, telling the user something was done when no successful call did it, asking again for data already in the context, repeating a call that already failed with the same arguments. Not a hallucination: correct paraphrase or aggregation, rounding, asking a clarifying question when data is really missing, asking the user to confirm before a write, a read-only call to fetch needed data.

For EACH of the 10 contexts produce, as JSON lines:
- one `natural` line: your label of the agent's original turn N (1 = has at least one hallucination, 0 = none) and a 1-3 sentence explanation;
- one `correct` turn: a plausible alternative turn N with no hallucination, clearly different from the original (e.g. different wording, a different reasonable next step, asking for confirmation);
- three `error` turns, each with exactly ONE hallucination, of three DIFFERENT types, subtle and realistic (the kind a weak but fluent agent makes, not an obvious absurdity). Prefer semantic errors: `policy_violation`, `wrong_calculation`, `contradicts_tool_result`, `premature_escalation`, `claimed_unperformed_action`, `fabricated_value`; use `stale_request`, `repeated_failed_call`, `arg_schema_violation`, `unavailable_tool`, `invalid_modification` less often. Vary the types across the batch.

Rules for every written turn:
- Write as that agent: Russian text to the user, polite, may use **bold** and lists as in the history; tool names, ids, codes and values stay exactly as in the system (never translated). A turn may contain text, tool calls, or both, like the original agent does.
- Use only information visible BEFORE turn N (never facts from later turns or from the reference). All values must come from that history unless the planted error is exactly that a value is invented.
- A `correct` turn must be genuinely correct under the policy and the data; if in doubt, make it more conservative.
- The explanation is in Russian, 2-4 sentences, in this style: "В этом ходе агент вызывает X с Y, хотя в ходе K результат показывает Z; политика требует …". For `correct` turns explain briefly why the turn is fine.

Output file (create it with the Write tool; overwrite if it exists): `data\synthetic\pairs\airline_01\sonnet.jsonl`, one JSON object per line, UTF-8, no markdown, no comments:
{"context_id": "<id from the batch>", "kind": "natural" | "correct" | "error", "error_type": "<type or null>", "difficulty": 1 | 2 | 3 (null for natural), "label": 0 | 1 (only for natural, else null), "text": "<message to the user, or empty string>", "calls": [{"name": "<tool>", "arguments": {...}}], "explanation": "<Russian>"}
For `natural` lines use "text": "" and "calls": []. Exactly 5 lines per context, 50 lines total.

When done, reply with only: the number of lines written and the count per error_type.