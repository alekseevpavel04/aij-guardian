"""Which model wrote the organiser's agent turns: each candidate rewrites the same turns, compared with the originals.

For a fixed sample of the organiser's contexts (airline, retail, telecom), the dialogue before the turn
is parsed back into chat messages: the agent's system text (the ⟦SYSTEM⟧ block without the tool list,
which the agent got through the API), τ²'s own tool schemas of the domain, the user and agent turns,
tool calls and their results. The candidate served by llama-server answers at temperature 0 (τ²'s
default for the agent), without reasoning. The answer is rendered in the data's layout and stored next
to the organiser's turn; `--report` compares them per model: same kind of turn (text or call), same
first call, same call with the same arguments, the text's similarity (difflib ratio), exact match.

Runs under the tau2-bench interpreter, for the tool schemas:

    D:/VScode_projects/tau2-bench/.venv/Scripts/python.exe scripts/agent_fingerprint.py --name qwen3_8b \\
        --model agent8b --api-base http://127.0.0.1:8081/v1
    D:/VScode_projects/tau2-bench/.venv/Scripts/python.exe scripts/agent_fingerprint.py --report

Each answer is appended to data/interim/fingerprint/<name>.jsonl, so a rerun continues where it stopped.
"""

import argparse
import difflib
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardian import journal  # noqa: E402
from guardian.parsing import parse_transcript  # noqa: E402
from guardian.render import _call_line  # noqa: E402

OUT = ROOT / "data/interim/fingerprint"
DOMAINS = ("airline", "retail", "telecom")
SAMPLE = 40


def sample(contexts: Path) -> pd.DataFrame:
    frame = pd.read_parquet(contexts)
    frame = frame[frame.domain.isin(DOMAINS)]
    return frame.sample(SAMPLE, random_state=0).sort_values("id")


def dump_tools() -> None:
    from tau2.registry import registry

    schemas = {d: [t.openai_schema for t in registry.get_env_constructor(d)().get_tools()] for d in DOMAINS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tools.json").write_text(json.dumps(schemas, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def messages(prompt: str) -> list[dict]:
    turns = parse_transcript(prompt)
    system = next(t for t in turns if t.role == "SYSTEM").text
    system = system[: system.find("[AVAILABLE TOOLS]")].strip() if "[AVAILABLE TOOLS]" in system else system
    out = [{"role": "system", "content": system}]
    for t in turns:
        if t.role == "USER":
            out.append({"role": "user", "content": t.text})
        elif t.role == "ASSISTANT":
            message = {"role": "assistant", "content": t.text or None}
            if t.calls:
                message["tool_calls"] = [
                    {
                        "id": f"call_{t.index}_{k}",
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.args_raw},
                    }
                    for k, c in enumerate(t.calls)
                ]
            out.append(message)
            for k, r in enumerate(t.results):
                out.append({"role": "tool", "tool_call_id": f"call_{t.index}_{k}", "name": r.name, "content": r.body})
    return out


def rendered(message: dict) -> str:
    lines = [(message.get("content") or "").strip()] if (message.get("content") or "").strip() else []
    lines += [_call_line(c) for c in message.get("tool_calls") or []]
    return "\n".join(lines)


def body(response: str) -> str:
    return response.split("\n", 1)[1] if response.startswith("⟦") and "\n" in response else response


def generate(api_base: str, model: str, chat: list[dict], tools: list[dict]) -> dict:
    payload = {
        "model": model,
        "messages": chat,
        "tools": tools,
        "temperature": 0.0,
        "max_tokens": 1024,
        # the organiser's agent makes up to 9 calls in one turn, so a candidate may too
        "parallel_tool_calls": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{api_base}/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())["choices"][0]["message"]


def kind_and_calls(text: str) -> tuple[str, list[str], str]:
    calls = [line.strip() for line in text.splitlines() if line.strip().startswith("→ TOOL_CALL")]
    prose = "\n".join(line for line in text.splitlines() if not line.strip().startswith(("→", "←"))).strip()
    return ("call" if calls else "text"), calls, prose


def rows(contexts: Path) -> pd.DataFrame:
    """Every candidate's answers as competition rows (same prompt, the answer under the turn's own
    marker), for the judge's token pass: how expected each writer's text is to the judge model."""
    frame = pd.read_parquet(contexts).set_index("id")
    out = []
    for path in sorted(OUT.glob("*.jsonl")):
        for r in journal.read(path):
            if "error" in r or not r["generated"].strip():
                continue
            header = r["organiser"].split("\n", 1)[0]
            out.append(
                {
                    "id": f"fp_{path.stem}__{r['id']}",
                    "prompt": frame.loc[r["id"]].prompt,
                    "response": f"{header}\n{r['generated']}",
                    "candidate": path.stem,
                    "context_id": r["id"],
                }
            )
    return pd.DataFrame(out)


def mean_logp(prefix: str) -> dict[str, float]:
    path = ROOT / "data/interim/embeddings" / f"{prefix}_uncertainty_org.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: sum(r["logp"]) / len(r["logp"]) for r in journal.read(path) if r["logp"]}


def report() -> dict:
    result = {}
    fp, org = mean_logp("q4_tok_fp"), mean_logp("q4_tok")
    for path in sorted(OUT.glob("*.jsonl")):  # one file per candidate
        rows = [r for r in journal.read(path) if "error" not in r]
        same_kind = same_first_call = same_call = exact = 0
        ratios = []
        for r in rows:
            k1, c1, p1 = kind_and_calls(r["generated"])
            k2, c2, p2 = kind_and_calls(body(r["organiser"]))
            same_kind += k1 == k2
            if c1 and c2:
                same_first_call += c1[0].split(":")[0] == c2[0].split(":")[0]
                same_call += c1[0] == c2[0]
            if p1 and p2:
                ratios.append(difflib.SequenceMatcher(None, p1, p2).ratio())
            exact += r["generated"].strip() == body(r["organiser"]).strip()
        calls_both = sum(
            1 for r in rows if kind_and_calls(r["generated"])[1] and kind_and_calls(body(r["organiser"]))[1]
        )
        result[path.stem] = {
            "answered": len(rows),
            "errors": sum(1 for r in journal.read(path) if "error" in r),
            "same_kind": [same_kind, len(rows)],
            "same_first_call_name": [same_first_call, calls_both],
            "same_first_call_with_arguments": [same_call, calls_both],
            "text_similarity_mean": round(sum(ratios) / len(ratios), 3) if ratios else None,
            "text_pairs": len(ratios),
            "exact": [exact, len(rows)],
            "turns_with_several_calls": [
                sum(len(kind_and_calls(r["generated"])[1]) > 1 for r in rows),
                sum(len(kind_and_calls(body(r["organiser"]))[1]) > 1 for r in rows),
            ],
        }
        paired = [(fp[f"fp_{path.stem}__{r['id']}"], org[r["id"]]) for r in rows if f"fp_{path.stem}__{r['id']}" in fp]
        if paired:
            result[path.stem]["mean_token_logp_candidate_vs_organiser"] = [
                round(sum(a for a, _ in paired) / len(paired), 3),
                round(sum(b for _, b in paired) / len(paired), 3),
                len(paired),
            ]
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--name", help="candidate name, the output file's stem")
    ap.add_argument("--model", help="the server's model alias")
    ap.add_argument("--api-base", default="http://127.0.0.1:8081/v1")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dump-tools", action="store_true")
    ap.add_argument("--rows-out", type=Path, default=None, help="write every candidate's answers as rows and stop")
    args = ap.parse_args()

    if args.dump_tools:
        dump_tools()
        return
    if args.rows_out:
        frame = rows(args.contexts)
        frame.to_parquet(args.rows_out, index=False)
        print(len(frame), "rows", frame.candidate.value_counts().to_dict())
        return
    if args.report:
        result = report()
        (ROOT / "results/agent_fingerprint.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"{args.name}.jsonl"
    done = {r["id"] for r in journal.read(log)}
    schemas = json.loads((OUT / "tools.json").read_text(encoding="utf-8"))
    for row in sample(args.contexts).itertuples():
        if row.id in done:
            continue
        started = time.monotonic()
        try:
            message = generate(args.api_base, args.model, messages(row.prompt), schemas[row.domain])
            record = {"id": row.id, "generated": rendered(message), "organiser": row.response}
        except Exception as error:  # a model that cannot take tools is itself a finding; keep going
            record = {"id": row.id, "error": repr(error)[:500], "organiser": row.response}
        record["seconds"] = round(time.monotonic() - started, 1)
        journal.append(log, record)
        print(args.name, row.id, record["seconds"], "s", "error" if "error" in record else "", flush=True)


if __name__ == "__main__":
    main()
