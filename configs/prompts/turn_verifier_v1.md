You label turns of a customer-service LLM agent for a hallucination detector. Work only with the files below and write one output file. Do not run code, do not browse, do not read other files.

Read once, in this order:
1. `{folder}\labeller.md`: the labelling instructions (in Russian). Follow them exactly; you play the labeller they describe.
2. `{folder}\system.md`: the agent's instructions, policy and the tools available to it. It is the start of `<prompt>` for every turn below.
3. `{folder}\batch.md`: several dialogues. Each has the τ² task reference (`<reference>`; the agent never saw it), the history, and candidate turns. A candidate turn N stands in place of the agent's turn N: it sees only the history before the marker ⟦ASSISTANT · ход N⟧ (together with system.md this is `<prompt>`), and the agent's own turn N and everything after it do not exist for it. The candidate itself is `<response>`.

The candidates come from different writers and several may share the same N. Some of them contain hallucinations, some do not; judge each one on its own, only against its visible history, never by comparing it with other candidates or with what the agent did at turn N or later. Decide 1 (the turn has at least one hallucination) or 0, following labeller.md. Check every number, id, date and status in the turn against the tool results, and every action against the policy. Think each one through, then write the output file (create it with the Write tool; overwrite if it exists): `{folder}\opus.jsonl`, one JSON object per line, UTF-8, no markdown:
{{"id": "<turn id>", "label": 0 | 1, "explanation": "<1-3 sentences in Russian: which statement or call is wrong and what in the context contradicts it, or why the turn is fine>"}}
Exactly one line per candidate turn, {n} lines in total.

When done, reply with only the number of lines written and how many are 1.
