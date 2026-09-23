You label turns of a customer-service LLM agent for a hallucination detector. Work only with the files below and write one output file. Do not run code, do not browse, do not read other files.

Read once, in this order:
1. `{folder}\labeller.md`: the labelling instructions (in Russian). Follow them exactly; you play the labeller they describe.
2. `{folder}\system.md`: the agent's instructions, policy and the tools available to it. It is the start of `<prompt>` for every context below.
3. `{folder}\batch.md`: several dialogues. Each has the τ² task reference (`<reference>`; the agent never saw it), the history, and its contexts. A context is "the history before the marker ⟦ASSISTANT · ход N⟧" (together with system.md this is `<prompt>`); anything at or after that marker does not exist for that context. The turn to label (`<response>`) is shown under each context.

Label every context separately, only against its own visible history: a later turn of the dialogue must not change the label of an earlier one. Decide 1 (the turn has at least one hallucination) or 0, following labeller.md. Think each one through, then write the output file (create it with the Write tool; overwrite if it exists): `{folder}\opus.jsonl`, one JSON object per line, UTF-8, no markdown:
{{"id": "<context id>", "label": 0 | 1, "explanation": "<2-4 sentences in Russian: which statement or call is wrong and what in the context contradicts it, or why the turn is fine>"}}
Exactly one line per context, {n} lines in total.

When done, reply with only the number of lines written and how many are 1.
