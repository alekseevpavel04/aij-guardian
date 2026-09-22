You label turns of a customer-service LLM agent for a hallucination detector. Work only with the files below and write one output file. Do not run code, do not browse, do not read other files.

Read once: `{folder}\labeller.md`, the labelling instructions (in Russian). Follow them exactly; you play the labeller they describe. Then read the agent's instructions, policy and tools for each domain: {systems}. Then read `{folder}\batch.md`: for each row, the τ² task reference (the agent never saw it), the dialogue the agent saw after its system block, and the turn to label.

For each row decide 1 (the turn has at least one hallucination) or 0, following labeller.md. Think it through, then write the output file (create it with the Write tool; overwrite if it exists): `{folder}\sonnet.jsonl`, one JSON object per line, UTF-8, no markdown:
{{"id": "<row id>", "label": 0 | 1, "explanation": "<2-4 sentences in Russian: which statement or call is wrong and what in the context contradicts it, or why the turn is fine>"}}
Exactly one line per row, {n} lines in total.

When done, reply with only the number of lines written and how many are 1.
