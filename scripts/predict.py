"""Submission entrypoint: python scripts/predict.py --input <test.csv> --output <predictions.csv>

One output row per input row, in input order, label 0 or 1, written as parquet whatever the output
extension (see guardian/io.py). A failure on one example is written to stderr and that example gets
the default label; the run continues. This is a one-shot batch on a platform that gives no second
chance, not a habit to carry into the rest of the code.

With the `cascade` strategy the judge model in model/ is served by vLLM (guardian/serving.py) and
the whole file is labelled at once (guardian/cascade.py). GUARDIAN_JUDGE_URL points the run at a
server that is already up instead, which is how it is tested locally.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
# `pip install .` runs without network on the platform; if it fails or installs nothing, the
# package is still importable straight from the archive.
sys.path.insert(0, str(ROOT / "src"))

from guardian.io import main_argv, read_rows, write_predictions  # noqa: E402
from guardian.parsing import format_recognised  # noqa: E402
from guardian.predictor import Predictor  # noqa: E402

CONFIG = ROOT / "model" / "guardian.json"


def per_row(rows: list[dict], predictor: Predictor) -> tuple[list[int], int]:
    labels, failures = [], 0
    for i, row in enumerate(rows):
        try:
            labels.append(predictor.predict(row.get("prompt", ""), row.get("response", "")))
        except Exception as error:  # one bad example must not cost the whole run
            failures += 1
            print(f"example {row.get('id', i)}: {type(error).__name__}: {error}", file=sys.stderr)
            labels.append(predictor.default_label)
    return labels, failures


def cascade(rows: list[dict], config: dict, started: float) -> list[int]:
    from guardian.cascade import closest_to_cut, combine, judge_scores, merge, rule_labels, top_share
    from guardian.judge import JudgeClient
    from guardian.serving import start_vllm, stop, wait_ready

    rules = rule_labels(rows, tuple(config["checks"]))
    print(f"rules fired on {sum(rules)} of {len(rows)} rows", file=sys.stderr)

    url, process = os.environ.get("GUARDIAN_JUDGE_URL"), None
    if url is None:
        port = config.get("port", 8000)
        process = start_vllm(ROOT / "model", port, config["served_name"], config["max_model_len"], ROOT / "vllm.log")
        if not wait_ready(port, started + 60 * config["server_minutes"], process):
            print("vLLM did not come up in time: rules only", file=sys.stderr)
            stop(process)
            return rules
        url = f"http://127.0.0.1:{port}/v1"
    deadline = started + 60 * config["total_minutes"]
    prompt = (ROOT / "model" / config["prompt"]).read_text(encoding="utf-8")
    try:
        quick = JudgeClient(url, model=config["served_name"], timeout=60 * config["total_minutes"])
        run = judge_scores(rows, quick, prompt, deadline, config["workers"])
        scores = run.scores
        print(
            f"quick pass scored {sum(s is not None for s in scores)} of {len(rows)} rows in {run.seconds:.0f} s, "
            f"{run.errors} errors",
            file=sys.stderr,
        )
        if config.get("thinking_tokens"):
            # the reasoned pass runs only in the time left and starts where it can change the most labels
            reasoned = JudgeClient(
                url,
                model=config["served_name"],
                timeout=60 * config["total_minutes"],
                thinking=True,
                thinking_tokens=config["thinking_tokens"],
            )
            order = closest_to_cut(scores, config["share"])
            second = judge_scores(rows, reasoned, prompt, deadline, config["workers"], order)
            scores = merge(scores, second.scores)
            print(
                f"reasoned pass scored {sum(s is not None for s in second.scores)} of {len(rows)} rows in "
                f"{second.seconds:.0f} s, {second.errors} errors",
                file=sys.stderr,
            )
    finally:
        stop(process)
    judge = top_share(scores, config["share"])
    labels = combine(rules, judge)
    print(f"top {config['share']} of judge scores: {sum(judge)}; rules add {sum(labels) - sum(judge)}", file=sys.stderr)
    return labels


def main() -> None:
    started = time.monotonic()
    input_path, output_path = main_argv()
    config = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    rows = read_rows(input_path)
    ids = [row.get("id", str(i)) for i, row in enumerate(rows)]
    # A row with no system turn or no tool list is one no check can fire on: a share near 1 means
    # the test uses a transcript format the parser does not know.
    unrecognised = sum(not format_recognised(row.get("prompt", "")) for row in rows)

    failures = 0
    if config.get("strategy") == "cascade":
        try:
            labels = cascade(rows, config, started)
        except Exception as error:  # the judge must never cost the run: fall back to the rules
            print(f"cascade failed, rules only: {type(error).__name__}: {error}", file=sys.stderr)
            labels, failures = per_row(rows, Predictor(checks=tuple(config["checks"])))
    else:
        labels, failures = per_row(rows, Predictor.from_config(CONFIG) if CONFIG.exists() else Predictor())

    written = write_predictions(ids, labels, output_path)
    elapsed = time.monotonic() - started
    print(
        f"{len(rows)} rows, {sum(labels)} predicted 1, {failures} failed, {unrecognised} with unrecognised format, "
        f"{written} written, {elapsed:.1f} s ({elapsed / max(len(rows), 1) * 1000:.1f} ms per row)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
