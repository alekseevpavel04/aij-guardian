"""Submission entrypoint: python scripts/predict.py --input <test.csv> --output <predictions.csv>

One output row per input row, in input order, label 0 or 1, written as parquet whatever the output
extension (see guardian/io.py). A failure on one example is written to stderr and that example gets
the default label; the run continues. This is a one-shot batch on a platform that gives no second
chance, not a habit to carry into the rest of the code.

With the `cascade` strategy the judge model in model/ is served by vLLM (guardian/serving.py) and
the whole file is labelled at once (guardian/cascade.py). GUARDIAN_JUDGE_URL points the run at a
server that is already up instead, which is how it is tested locally.

With the `head` strategy the same model is served in pooling mode and a linear head scores its
last-token hidden state (guardian/head.py; GUARDIAN_EMBED_URL for a server that is already up). If
the pooling server does not come up or most of its requests fail, the run falls back to the cascade
with the cascade's own deadlines, so a head that fails early costs nothing against the known result;
with "fallback": "rules" it falls back to the rules alone, whose score on the test is known exactly.

With the `final` strategy (DECISIONS 46, 47) one pass of the same model per row (scripts/final_pass.py, in a
process of its own with a deadline) gives head B its last-token vector and the uncertainty head the turn's token
probabilities; each row is labelled 1 when a rule fires or its score reaches the threshold, on its own, never as
a share of the file (guardian.final). Rows the pass has not scored by the deadline get the rules' label.
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


def head(rows: list[dict], config: dict, started: float) -> list[int] | None:
    """Rules, then the head on the judge model's hidden state; None when it could not score the file."""
    from guardian.cascade import combine, rule_labels, top_share
    from guardian.head import Head, head_scores, to_home_scale
    from guardian.serving import start_vllm, stop, wait_ready

    rules = rule_labels(rows, tuple(config["checks"]))
    url, process = os.environ.get("GUARDIAN_EMBED_URL"), None
    if url is None:
        port = config.get("port", 8000)
        process = start_vllm(
            ROOT / "model", port, config["served_name"], config["max_model_len"], ROOT / "vllm_pooling.log", True
        )
        if not wait_ready(port, started + 60 * config["server_minutes"], process):
            print("vLLM in pooling mode did not come up in time", file=sys.stderr)
            stop(process)
            return None
        url = f"http://127.0.0.1:{port}/v1"
    prompt = (ROOT / "model" / config["prompt"]).read_text(encoding="utf-8")
    model = Head.load(ROOT / "model" / config["head"])
    deadline = started + 60 * config["head_minutes"]
    calibration = None
    try:
        if config.get("threshold") is not None:
            # the calibration rows first, through the same engine: they carry the home scores the platform
            # scores are mapped onto, so that each test row is then decided on its own
            reference = json.loads((ROOT / "model" / config["calibration"]).read_text(encoding="utf-8"))
            cal = head_scores(reference, model, url, config["served_name"], prompt, deadline, config["workers"])
            pairs = [(p, r["home_score"]) for p, r in zip(cal.scores, reference, strict=True) if p is not None]
            print(f"calibration scored {len(pairs)} of {len(reference)} rows", file=sys.stderr)
            if len(pairs) < 0.8 * len(reference):
                return None
            calibration = (sorted(p for p, _ in pairs), sorted(h for _, h in pairs))
        run = head_scores(rows, model, url, config["served_name"], prompt, deadline, config["workers"])
    finally:
        stop(process)
    scored = sum(s is not None for s in run.scores)
    print(f"head scored {scored} of {len(rows)} rows in {run.seconds:.0f} s, {run.errors} errors", file=sys.stderr)
    # most requests failing means the pooling path does not work, and the cascade may still have time;
    # rows left unscored at the deadline are only a lack of time, and the cascade would have even less
    if run.errors > config["max_errors"] * len(rows) or scored == 0:
        return None
    if calibration is not None:
        platform_ref, home_ref = calibration
        home = [None if s is None else to_home_scale(s, platform_ref, home_ref) for s in run.scores]
        flags = [int(h is not None and h >= config["threshold"]) for h in home]
        labels = combine(rules, flags)
        print(f"rules fired on {sum(rules)}; with the threshold {config['threshold']} {sum(labels)}", file=sys.stderr)
        return labels
    labels = combine(rules, top_share(run.scores, config["share"]))
    print(f"rules fired on {sum(rules)}; with the top {config['share']} of head scores {sum(labels)}", file=sys.stderr)
    return labels


def final(rows: list[dict], config: dict, started: float) -> list[int] | None:
    """Rules, then the final heads on one pass of the judge model; None when the pass scored no row."""
    import subprocess
    import tempfile

    import numpy as np

    from guardian import journal
    from guardian.cascade import combine, rule_labels
    from guardian.extract import decode
    from guardian.final import Final
    from guardian.serving import stop
    from guardian.uncertainty import features

    rules = rule_labels(rows, tuple(config["checks"]))
    print(f"rules fired on {sum(rules)} of {len(rows)} rows", file=sys.stderr)
    model = Final.load(ROOT / "model" / config["final"])
    work = Path(tempfile.mkdtemp(prefix="guardian_final_"))
    with (work / "rows.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            record = {"id": str(i), "prompt": row.get("prompt", ""), "response": row.get("response", "")}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    deadline = started + 60 * config["final_minutes"]
    worker = os.environ.get("GUARDIAN_FINAL_PASS", str(ROOT / "scripts" / "final_pass.py"))
    command = [
        sys.executable,
        worker,
        "--rows",
        str(work / "rows.jsonl"),
        "--prefix",
        "test",
        "--model-dir",
        str(ROOT / "model"),
        "--system-prompt",
        str(ROOT / "model" / config["prompt"]),
        "--out-dir",
        str(work),
        "--max-model-len",
        str(config["max_model_len"]),
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        # no new batch after this; a batch under way is cut off at the deadline
        "--stop-at",
        str(time.time() + deadline - time.monotonic() - 60 * config["batch_minutes"]),
    ]
    with (ROOT / "final_pass.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(5)
        stop(process)
    print(f"pass exit {process.returncode} after {time.monotonic() - started:.0f} s", file=sys.stderr)

    vectors = {r["id"]: decode(r["vector"]) for r in journal.read(work / "test_final_org.jsonl")}
    parts = {r["id"]: r["parts"] for r in journal.read(work / "test_parts_org.jsonl")}
    stats = {r["id"]: r for r in journal.read(work / "test_uncertainty_org.jsonl")}
    scored = [i for i in range(len(rows)) if str(i) in stats and str(i) in vectors and str(i) in parts]
    print(f"pass scored {len(scored)} of {len(rows)} rows", file=sys.stderr)
    if not scored:
        return None
    scores = model.score(
        np.stack([vectors[str(i)] for i in scored]),
        np.stack([features(stats[str(i)], parts[str(i)]) for i in scored]),
    )
    flags = [0] * len(rows)
    for i, s in zip(scored, scores, strict=True):
        flags[i] = int(s >= model.threshold)
    labels = combine(rules, flags)
    print(f"threshold {model.threshold}: {sum(flags)} flagged; with the rules {sum(labels)}", file=sys.stderr)
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
    labels = None
    if config.get("strategy") == "final":
        try:
            labels = final(rows, config, started)
        except Exception as error:  # the rules' score on the test is known; a failed pass must not cost more
            print(f"final failed: {type(error).__name__}: {error}", file=sys.stderr)
        if labels is None:
            print("falling back to the rules", file=sys.stderr)
            labels, failures = per_row(rows, Predictor(checks=tuple(config["checks"])))
    if config.get("strategy") == "head":
        try:
            labels = head(rows, config, started)
        except Exception as error:  # the head is the experiment, the cascade below is the known result
            print(f"head failed: {type(error).__name__}: {error}", file=sys.stderr)
        if labels is None and config.get("fallback") == "rules":
            # a probe archive: the rules alone score exactly as submission 1 did, so the leaderboard shows
            # without doubt that the head did not run
            print("falling back to the rules", file=sys.stderr)
            labels, failures = per_row(rows, Predictor(checks=tuple(config["checks"])))
        elif labels is None:
            print("falling back to the cascade", file=sys.stderr)
    if labels is None and config.get("strategy") in ("cascade", "head"):
        try:
            labels = cascade(rows, config, started)
        except Exception as error:  # the judge must never cost the run: fall back to the rules
            print(f"cascade failed, rules only: {type(error).__name__}: {error}", file=sys.stderr)
            labels, failures = per_row(rows, Predictor(checks=tuple(config["checks"])))
    elif labels is None:
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
