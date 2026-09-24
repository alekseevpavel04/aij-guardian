You write training data for a detector of hallucinations in turns of a customer-service LLM agent. The detector has only seen four domains (airline, retail, telecom, banking); to learn what a hallucination is in general, it needs agents in other worlds. Work only from this instruction and write one output file. Do not run code, do not browse, do not read any files.

Write {count} complete, self-consistent dialogues, half in the world "{world_a}" and half in the world "{world_b}". For each dialogue invent:

1. `policy`: the rules the agent must follow in this world, in English, 150-350 words, concrete and checkable (conditions for refunds or changes, what needs the customer's explicit confirmation, when to transfer to a human, limits, fees, identity checks). Vary the policy from dialogue to dialogue within a world.
2. `tools`: 4-8 tools the agent has, as OpenAI function definitions: {"name", "description", "parameters": {"type": "object", "properties": {...}, "required": [...]}}. Read tools and write tools. Some things only the customer can do (on their device, in person); those are not tools.
3. `messages`: the dialogue before the agent's turn under review, as OpenAI messages: the agent's greeting first (`{"role": "assistant", "content": "..."}`), then user messages, agent messages with `"tool_calls": [{"function": {"name": "...", "arguments": "<JSON string>"}}]`, and tool results `{"role": "tool", "name": "<tool>", "content": "<result, usually JSON>"}` right after the call that produced them. 3-8 agent turns with realistic ids, names, dates, amounts, statuses. The user and the agent speak Russian; policy, tool names, schemas and tool results stay in English, as in the detector's data.
4. `correct`: the agent's next turn, correct under the policy and the history: {"content": "<text to the user or empty>", "tool_calls": [...]}.
5. `error`: the same turn with exactly ONE hallucination: a statement or action not grounded in the history and the policy. Keep everything else as in `correct`; the difference should be one fragment, and the turn should read as fluent and plausible.
6. `error_type`, one of: `fabricated_value` (an id, amount, date or name that the history does not contain), `contradicts_tool_result`, `wrong_calculation`, `policy_violation`, `claimed_unperformed_action` (says something was done or checked with no successful call), `premature_escalation` (transfers or refuses while an obvious available step is untried), `unavailable_tool` (calls a tool that is not in the list, or one that only the customer can do), `arg_schema_violation`, `wrong_entity` (acts on another object from the history), `sycophancy` (agrees with a false claim of the user), `invented_rule`, `stale_request` (asks for something already given). Spread the types evenly; about half of the errors should be in the text to the user and half in tool calls.
7. `explanation`: 1-2 sentences in Russian: what is wrong and which part of the history shows it.

Make the dialogues differ: different customers, goals, complications (a failed call, a user who changes their mind, a partial refund, missing data). The hallucination must be detectable from the history and the policy alone.

Output file (create it with the Write tool; overwrite if it exists): `{batch_dir}\opus.jsonl`, one JSON object per line, UTF-8, no markdown, no comments:
{"world": "<world>", "policy": "...", "tools": [...], "messages": [...], "correct": {"content": "...", "tool_calls": [...]}, "error": {"content": "...", "tool_calls": [...]}, "error_type": "...", "explanation": "..."}
Exactly {count} lines.

When done, reply with only: the number of lines written and the count per error_type.
