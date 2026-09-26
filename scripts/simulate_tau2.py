"""Russian-language τ²-bench simulations with the local model as agent and as user.

Rebuilds the organiser's data source as closely as the data shows it: the agent gets the exact
instructions found in valid.parquet (τ²'s own plus a paragraph about answering in Russian), the user
simulator is told to write in Russian and keep identifiers as they are, and both are the model the
judge uses, served by llama-server without reasoning.

Runs under the tau2-bench interpreter (Python 3.12), not the project's:

    D:/VScode_projects/tau2-bench/.venv/Scripts/python.exe scripts/simulate_tau2.py --domain airline --tasks 0,1

Every finished simulation is written to its own file through a temporary name, and a rerun skips
the files that exist, so switching the computer off loses at most the simulation in progress.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tau2.agent.llm_agent as llm_agent  # noqa: E402
import tau2.evaluator.evaluator_nl_assertions as nl_assertions  # noqa: E402
import tau2.user.user_simulator as user_simulator  # noqa: E402
from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.runner.batch import run_single_task  # noqa: E402
from tau2.runner.helpers import get_tasks  # noqa: E402

from guardian.journal import write_atomic  # noqa: E402

RUSSIAN_USER = (
    "\n\n## Language\n"
    "Write every message to the agent in Russian. Keep names, emails, user ids, order and reservation "
    "ids, codes, amounts and dates exactly as they are given to you; do not translate or transliterate them."
)


def patch_language(instruction_file: Path) -> None:
    """The organiser's agent instructions, and a Russian-speaking user; both read at call time by τ²."""
    llm_agent.AGENT_INSTRUCTION = instruction_file.read_text(encoding="utf-8").strip()
    original = user_simulator.get_global_user_sim_guidelines

    def russian_guidelines(use_tools: bool = False) -> str:
        return original(use_tools) + RUSSIAN_USER

    user_simulator.get_global_user_sim_guidelines = russian_guidelines


def file_name(task_id: str, trial: int) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)
    return f"{safe}__trial{trial}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=("airline", "retail", "telecom"))
    ap.add_argument("--tasks", default=None, help="comma-separated task ids; all tasks of the domain if omitted")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--model", default="openai/judge")
    ap.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--instructions", type=Path, default=ROOT / "configs/agent_instruction_ru.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic/tau2_ru")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true", help="the agent reasons first (the server must allow it)")
    ap.add_argument(
        "--organiser-system",
        type=Path,
        default=None,
        help="JSON of the organiser's ⟦SYSTEM⟧ block per domain (with its tool list); the agent gets it verbatim",
    )
    args = ap.parse_args()

    patch_language(args.instructions)
    if args.organiser_system:
        system = json.loads(args.organiser_system.read_text(encoding="utf-8"))[args.domain]
        llm_agent.LLMAgent.system_prompt = property(lambda self: system)
    llm_args = {
        "api_base": args.api_base,
        "api_key": "local",
        "temperature": args.temperature,
        # the organiser's agent makes several calls in one turn; the model may, if it wants to
        "parallel_tool_calls": True,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    agent_args = {**llm_args, "extra_body": {"chat_template_kwargs": {"enable_thinking": args.thinking}}}
    # tasks graded by natural-language assertions call τ²'s default judge (an OpenAI model); the
    # reward is only a side signal here, so the same local model grades them
    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = args.model
    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = llm_args
    task_ids = args.tasks.split(",") if args.tasks else None
    config = TextRunConfig(
        domain=args.domain,
        task_ids=task_ids,
        llm_agent=args.model,
        llm_args_agent=agent_args,
        llm_user=args.model,
        llm_args_user=llm_args,
        max_steps=args.max_steps,
        num_trials=args.trials,
    )
    tasks = get_tasks(task_set_name=args.domain, task_ids=task_ids)
    out = args.out / args.domain
    out.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        for trial in range(args.trials):
            path = out / file_name(task.id, trial)
            if path.exists():
                continue
            started = time.monotonic()
            simulation = run_single_task(config, task, seed=trial)
            record = {
                "task_id": task.id,
                "domain": args.domain,
                "trial": trial,
                "model": args.model,
                "nl_assertions_judge": args.model,
                "parallel_tool_calls": True,
                "temperature": args.temperature,
                "agent_thinking": args.thinking,
                "organiser_system": bool(args.organiser_system),
                "seconds": round(time.monotonic() - started, 1),
                "simulation": json.loads(simulation.model_dump_json()),
            }
            write_atomic(path, json.dumps(record, ensure_ascii=False) + "\n")
            reward = (record["simulation"].get("reward_info") or {}).get("reward")
            messages = len(record["simulation"].get("messages") or [])
            print(f"{args.domain} {task.id} trial {trial}: {messages} messages, reward {reward}, {record['seconds']} s")


if __name__ == "__main__":
    main()
