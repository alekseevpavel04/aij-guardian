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
import tau2.orchestrator.orchestrator as orchestrator  # noqa: E402
import tau2.user.user_simulator as user_simulator  # noqa: E402
from tau2.data_model.message import AssistantMessage  # noqa: E402
from tau2.data_model.simulation import RewardInfo, TextRunConfig  # noqa: E402
from tau2.data_model.tasks import RewardType  # noqa: E402
from tau2.runner.batch import run_single_task  # noqa: E402
from tau2.runner.helpers import get_tasks  # noqa: E402

from guardian.journal import write_atomic  # noqa: E402
from guardian.tau2_sim import GREETING  # noqa: E402

RUSSIAN_USER = (
    "\n\n## Language\n"
    "Write every message to the agent in Russian. Keep names, emails, user ids, order and reservation "
    "ids, codes, amounts and dates exactly as they are given to you; do not translate or transliterate them."
)


def patch_language(instruction_file: Path) -> None:
    """The organiser's agent instructions, the Russian greeting of the data, and a Russian-speaking user;
    all read at call time by τ².

    τ² opens every dialogue with a fixed English greeting; a strong user model answers in the language it
    is addressed in, and the agent follows the user (Qwen3.6 did, 26.09), so the greeting is the one the
    organiser's data shows, and the language rule is repeated after the user's scenario as well."""
    llm_agent.AGENT_INSTRUCTION = instruction_file.read_text(encoding="utf-8").strip()
    orchestrator.DEFAULT_FIRST_AGENT_MESSAGE = AssistantMessage(role="assistant", content=GREETING, cost=0.0)
    original = user_simulator.get_global_user_sim_guidelines

    def russian_guidelines(use_tools: bool = False) -> str:
        return original(use_tools) + RUSSIAN_USER

    user_simulator.get_global_user_sim_guidelines = russian_guidelines
    user_simulator.SYSTEM_PROMPT = user_simulator.SYSTEM_PROMPT + RUSSIAN_USER


def organiser_banking(embed_base: str, knowledge: Path, policy: Path) -> str:
    """banking_knowledge as the organiser's system block shows it, on this τ² commit:

    - tools: base banking tools + KB_search_bm25 + KB_search_dense + grep (here "alltools" has a sandboxed
      shell instead of grep, which also needs Linux binaries), the variant registered as "organiser-alltools";
    - KB_search_dense embeds with qwen3-embedding-8b through OpenRouter there; here the same client talks to a
      local llama-server with a Qwen3 embedding model;
    - the knowledge base in Russian (scripts/kb_translation.py) and the organiser's policy text verbatim, which
      says so and describes these three tools."""
    import dataclasses

    from openai import OpenAI
    from tau2.domains.banking_knowledge import environment, retrieval
    from tau2.domains.banking_knowledge import retrieval_mixins as mixins
    from tau2.domains.banking_knowledge.data_model import Document, KnowledgeBase
    from tau2.domains.banking_knowledge.tools import KnowledgeTools
    from tau2.knowledge.embedders.openrouter_embedder import OpenRouterEmbedder

    original_embedder = OpenRouterEmbedder.__init__

    def local_embedder(self, *args, **kwargs):
        original_embedder(self, *args, **{**kwargs, "api_key": "local"})
        self.client = OpenAI(api_key="local", base_url=embed_base)

    OpenRouterEmbedder.__init__ = local_embedder

    class AllToolsGrep(
        mixins.KBSearchBm25AllToolsMixin, mixins.KBSearchDenseAllToolsMixin, mixins.GrepMixin, KnowledgeTools
    ):
        def __init__(self, db, bm25, dense, grep):
            super().__init__(db)
            self._kb_bm25_pipeline, self._kb_dense_pipeline, self._grep_pipeline = bm25, dense, grep

    name = "organiser-alltools"
    retrieval.RETRIEVAL_VARIANTS[name] = dataclasses.replace(
        retrieval.RETRIEVAL_VARIANTS["alltools-qwen"], name=name, shell=None, grep=retrieval.GrepSpec()
    )
    build_tools, text = retrieval.build_tools, policy.read_text(encoding="utf-8").strip()

    def tools(variant, db, knowledge_base, read_log_allowlist=None):
        if variant.name != name:
            return build_tools(variant, db, knowledge_base, read_log_allowlist=read_log_allowlist)
        pipeline = retrieval._create_kb_pipeline
        toolkit = AllToolsGrep(
            db,
            pipeline(variant.kb_search_bm25, knowledge_base),
            pipeline(variant.kb_search_dense, knowledge_base),
            retrieval._create_grep_pipeline(variant.grep, knowledge_base),
        )
        toolkit.set_read_log_allowlist(read_log_allowlist)
        return toolkit

    def russian_knowledge_base() -> KnowledgeBase:  # read as UTF-8 on Windows too, only τ²'s own fields
        docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(knowledge.glob("*.json"))]
        return KnowledgeBase(
            documents={d["id"]: Document(id=d["id"], title=d["title"], content=d["content"]) for d in docs}
        )

    environment.build_tools = tools
    environment.build_policy = lambda variant, knowledge_base, task=None: text
    environment.get_knowledge_base = russian_knowledge_base
    return name


def file_name(task_id: str, trial: int) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)
    return f"{safe}__trial{trial}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=("airline", "retail", "telecom", "banking_knowledge"))
    ap.add_argument("--tasks", default=None, help="comma-separated task ids; all tasks of the domain if omitted")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--model", default="openai/judge")
    ap.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--instructions", type=Path, default=ROOT / "configs/agent_instruction_ru.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "data/synthetic/tau2_ru")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--embed-base", default="http://127.0.0.1:8082/v1", help="embeddings server for banking_knowledge")
    ap.add_argument("--knowledge", type=Path, default=ROOT / "data/synthetic/banking_kb_ru")
    ap.add_argument("--warm-knowledge", action="store_true", help="only embed the knowledge base into the cache")
    ap.add_argument("--banking-policy", type=Path, default=ROOT / "configs/organiser_banking_policy.txt")
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
    if args.thinking:  # a runaway reasoning must not hold a turn forever; the server also caps the thinking
        agent_args["max_tokens"] = 4096
    # tasks graded by natural-language assertions call τ²'s default judge (an OpenAI model); the
    # reward is only a side signal here, so the same local model grades them
    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = args.model
    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = llm_args
    grade = nl_assertions.NLAssertionsEvaluator.calculate_reward.__func__

    def grade_or_note(cls, task, full_trajectory):
        # a local model may answer the grader in prose instead of JSON; the dialogue is what we need, so its
        # reward becomes 0 with a note instead of losing the simulation
        try:
            return grade(cls, task=task, full_trajectory=full_trajectory)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            return RewardInfo(
                reward=0.0,
                nl_assertions=[],
                info={"note": f"local NL grader answer unusable: {type(error).__name__}"},
                reward_breakdown={RewardType.NL_ASSERTION: 0.0},
            )

    nl_assertions.NLAssertionsEvaluator.calculate_reward = classmethod(grade_or_note)
    task_ids = args.tasks.split(",") if args.tasks else None
    banking = args.domain == "banking_knowledge"
    variant = organiser_banking(args.embed_base, args.knowledge, args.banking_policy) if banking else None
    if args.warm_knowledge:
        from tau2.domains.banking_knowledge.environment import get_environment

        get_environment(retrieval_variant=variant)
        return
    config = TextRunConfig(
        domain=args.domain,
        task_ids=task_ids,
        llm_agent=args.model,
        llm_args_agent=agent_args,
        llm_user=args.model,
        llm_args_user=llm_args,
        max_steps=args.max_steps,
        num_trials=args.trials,
        retrieval_config=variant,
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
                "retrieval": f"{variant}, Russian knowledge base, local embeddings" if banking else None,
                "greeting": GREETING,
                "seconds": round(time.monotonic() - started, 1),
                "simulation": json.loads(simulation.model_dump_json()),
            }
            write_atomic(path, json.dumps(record, ensure_ascii=False) + "\n")
            reward = (record["simulation"].get("reward_info") or {}).get("reward")
            messages = len(record["simulation"].get("messages") or [])
            print(f"{args.domain} {task.id} trial {trial}: {messages} messages, reward {reward}, {record['seconds']} s")


if __name__ == "__main__":
    main()
