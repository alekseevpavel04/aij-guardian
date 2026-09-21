You write training data for a detector of hallucinations in turns of a customer-service LLM agent (τ²-bench {domain} domain; the dialogue is in Russian, the policy and tool schemas are in English). Work only with the two files below and write one output file. Do not run code, do not browse, do not read other files.

Read once:
1. `{batch_dir}\system.md`: the agent's instructions, policy and the tools available to it (with argument schemas; `!` marks a required argument).
2. `{batch_dir}\batch.md`: several dialogues. Each has the task's reference solution (the agent never saw it), the history, and its contexts. A context is "the history before the marker ⟦ASSISTANT · ход N⟧"; anything at or after that marker does not exist for that context. Each context also shows the agent's original turn N.

Format of the history: ⟦USER⟧ user message; ⟦ASSISTANT · ход K⟧ agent turn K, with `→ TOOL_CALL name: {json}` for calls and `← TOOL_RESPONSE name: result` for results (`[ERROR]` marks a failed call).

What counts as a hallucination in a turn (the competition's definition): a statement or action not grounded in the context. Contradiction: a number, date, value, name or status that disagrees with tool results, the user or the policy, or an action the policy forbids with the current data. Unsupported: an invented fact, value, reason, entity or condition. Actions count: calling a tool not in the list, arguments outside the schema or not taken from the context, a forbidden write, a transfer to a human although the tools and policy allow solving it, telling the user something was done when no successful call did it. Not a hallucination: correct paraphrase or aggregation, rounding, asking a clarifying question when data is really missing, asking the user to confirm before a write, a read-only call to fetch needed data, several tool calls in one turn, text next to a call.

For EACH context produce, as JSON lines:
- one `natural` line: your label of the agent's original turn N (1 = has at least one hallucination, 0 = none) and a 1-3 sentence explanation;
- one `correct` turn: a turn N with no hallucination;
- three `error` turns, each with exactly ONE hallucination, of three DIFFERENT types, subtle and realistic (the kind a weak but fluent agent makes, not an obvious absurdity).

How to write the turns. The detector must learn the error, not who wrote the text, so written turns must look like the original agent's turns:
- Start from the original turn N (or, if you labelled it 1, from your own corrected version of it) and make the error by a MINIMAL edit: change one number, one id, one date, one condition, one tool name, one argument, one sentence. Keep everything else, including length, structure, lists, **bold** and the number of tool calls.
- The `correct` turn is the original rewritten with the same content and about the same length and formatting (different wording), or, if the original has an error, the original with that error fixed.
- Russian text to the user; tool names, ids, codes and values stay exactly as in the system (never translated).
- Use only information visible BEFORE turn N (never facts from later turns or from the reference). All values must come from that history unless the planted error is exactly that a value is invented.
- A `correct` turn must be genuinely correct under the policy and the data; if in doubt, make it more conservative.

Error types. Prefer semantic errors: `policy_violation`, `wrong_calculation`, `contradicts_tool_result`, `premature_escalation`, `claimed_unperformed_action`, `fabricated_value`; use `arg_schema_violation`, `unavailable_tool`, `invalid_modification`, `stale_request`, `repeated_failed_call` less often. Vary the types across the batch. Name the type precisely: an invented id, name or number is `fabricated_value` even inside a tool argument; `arg_schema_violation` is only a missing required argument, an argument absent from the schema or a value of the wrong shape. `stale_request` only when acting on or asking for something already settled changes the outcome (asking the user again for data already given, using a value the user has since changed), not a harmless re-check.

The explanation is in Russian, 2-4 sentences, in this style: "В этом ходе агент вызывает X с Y, хотя в ходе K результат показывает Z; политика требует …". For `correct` turns explain briefly why the turn is fine.

Output file (create it with the Write tool; overwrite if it exists): `{batch_dir}\sonnet.jsonl`, one JSON object per line, UTF-8, no markdown, no comments:
{"context_id": "<id from the batch>", "kind": "natural" | "correct" | "error", "error_type": "<type or null>", "difficulty": 1 | 2 | 3 (null for natural), "label": 0 | 1 (only for natural, else null), "text": "<message to the user, or empty string>", "calls": [{"name": "<tool>", "arguments": {...}}], "explanation": "<Russian>"}
For `natural` lines use "text": "" and "calls": []. Exactly 5 lines per context.

When done, reply with only: the number of lines written and the count per error_type.
