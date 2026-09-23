You write training data for a detector of hallucinations in turns of a customer-service LLM agent (τ²-bench {domain} domain; the dialogue is in Russian, the policy and tool schemas are in English). Work only with the two files below and write one output file. Do not run code, do not browse, do not read other files.

Read once:
1. `{batch_dir}\system.md`: the agent's instructions, policy and the tools available to it (with argument schemas; `!` marks a required argument).
2. `{batch_dir}\batch.md`: several dialogues. Each has the task's reference solution (the agent never saw it), the history, and its contexts. A context is "the history before the marker ⟦ASSISTANT · ход N⟧"; anything at or after that marker does not exist for that context. Each context also shows the agent's original turn N.

Format of the history: ⟦USER⟧ user message; ⟦ASSISTANT · ход K⟧ agent turn K, with `→ TOOL_CALL name: {json}` for calls and `← TOOL_RESPONSE name: result` for results (`[ERROR]` marks a failed call).

What counts as a hallucination in a turn (the competition's definition): a statement or action not grounded in the context. Contradiction: a number, date, value, name or status that disagrees with tool results, the user or the policy, or an action the policy forbids with the current data. Unsupported: an invented fact, value, reason, entity or condition. Actions count: calling a tool not in the agent's tool list (including actions on the customer's device, which only the customer can do), arguments outside the schema or not taken from the context, a placeholder instead of a real id, a modification whose new value equals the old one, repeating a call that already returned an error, a forbidden write, a transfer to a human or a "cannot do" while an obvious available step has not been tried (a lookup tool, another search, a name in Latin letters, exchanging for another variant), telling the user something was done when no successful call did it. Not a hallucination: correct paraphrase or aggregation, rounding, asking a clarifying question when data is really missing, asking the user to confirm before a write, a read-only call to fetch needed data, several tool calls in one turn, text next to a call, a transfer after everything available was done and the customer insists.

For EACH context produce 5 JSON lines:
- one `natural` line: your label of the agent's original turn N (1 = has at least one hallucination, 0 = none) and a 1-3 sentence explanation;
- two minimal pairs, A and B, each a `correct` turn N and an `error` turn N that is that same correct turn with exactly ONE hallucination planted by a minimal edit.

Pair A. `correct`: the original turn N rewritten with the same content and about the same length and formatting (different wording), or, if you labelled the original 1, the original with its error fixed. `error`: pair A's correct turn with one planted hallucination.

Pair B. `correct`: a different correct turn N that a careless detector would wrongly flag, because it does something that looks risky but is right under the policy and the data. Detectors on this task raise far more false alarms than they miss errors, so these turns matter most. Pick what fits the history:
- an amount (refund, price difference, total, remaining balance) computed correctly from tool results, with the numbers shown;
- a long summary that aggregates several tool results accurately (ids, dates, statuses, prices);
- a refusal or "this cannot be done" that the policy really requires, with the correct reason;
- a request for explicit confirmation before a write, listing every detail correctly;
- a write call made right after the user's explicit confirmation, with all arguments taken from the history;
- a transfer to a human that the policy requires, or that follows after every available step was tried.
Turn B must still be a natural next turn for this history (it answers the user's last message). If the history leaves no room for a risky-looking correct turn, make B another correct continuation that differs from A in content and say so in the explanation. `error`: pair B's correct turn with one planted hallucination.

The two errors of a context are of different types.

How to write the turns. The detector must learn the error, not who wrote the text, so written turns must look like the original agent's turns:
- An error turn differs from its pair's correct turn by a MINIMAL edit: one number, one id, one date, one condition, one tool name, one argument, one sentence. Everything else stays, including length, structure, lists, **bold** and the number of tool calls.
- Keep the style and length of the original agent's turns; turn B may be longer only when a summary or a calculation needs it.
- Russian text to the user; tool names, ids, codes and values stay exactly as in the system (never translated).
- Use only information visible BEFORE turn N (never facts from later turns or from the reference). All values must come from that history unless the planted error is exactly that a value is invented.
- A `correct` turn must be genuinely correct under the policy and the data; if in doubt, make it more conservative. Recheck every number in a correct turn against the tool results.

Error types. Prefer semantic errors: `policy_violation`, `wrong_calculation`, `contradicts_tool_result`, `premature_escalation`, `claimed_unperformed_action`, `fabricated_value`; use `arg_schema_violation`, `unavailable_tool`, `invalid_modification`, `stale_request`, `repeated_failed_call` less often. Vary the types across the batch. Name the type precisely: an invented id, name or number is `fabricated_value` even inside a tool argument; `arg_schema_violation` is only a missing required argument, an argument absent from the schema or a value of the wrong shape. `stale_request` only when acting on or asking for something already settled changes the outcome (asking the user again for data already given, using a value the user has since changed), not a harmless re-check.

The explanation is in Russian, 2-4 sentences, in this style: "В этом ходе агент вызывает X с Y, хотя в ходе K результат показывает Z; политика требует …". For `correct` turns explain briefly why the turn is fine; for pair B also say what makes it look suspicious.

Output file (create it with the Write tool; overwrite if it exists): `{batch_dir}\{answers}`, one JSON object per line, UTF-8, no markdown, no comments:
{"context_id": "<id from the batch>", "kind": "natural" | "correct" | "error", "pair": "A" | "B" (null for natural), "error_type": "<type or null>", "difficulty": 1 | 2 | 3 (null for natural), "label": 0 | 1 (only for natural, else null), "text": "<message to the user, or empty string>", "calls": [{"name": "<tool>", "arguments": {...}}], "explanation": "<Russian>"}
For `natural` lines use "text": "" and "calls": []. Exactly 5 lines per context: natural, A correct, A error, B correct, B error.

When done, reply with only: the number of lines written and the count per error_type.
