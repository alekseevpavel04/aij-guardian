"""A rehearsal of the rental on a Kaggle T4: the final strategy's code with the real vLLM 0.19.1, on Qwen3.5-0.8B
(the same model code as Qwen3.6-35B-A3B) instead of the 35 GB checkpoint.

What only a machine with vLLM can check, before paying for one:

1. scripts/final_pass.py on 40 organiser contexts: the engine starts, the ALL pooler returns every token's state,
   the output matrix is found, the journals are written; the tokens per second;
2. the turn's token log-probabilities of the pass against vLLM's own prompt_logprobs on the three shortest rows;
3. heads trained on those 40 rows (only to have a model of the right shape, their scores mean nothing), the
   submission folder built by scripts/build_submission.py, and scripts/predict.py run on 60 rows with the pass as
   its own process;
4. the submission's features of the 40 contexts against the pass's (scripts/compare_passes.py).

Everything goes to /kaggle/working/rehearsal.json and the logs next to it.

    python /kaggle/input/aij-rental-kit/scripts/kaggle_final_rehearsal.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KIT_SOURCE = Path(__file__).resolve().parents[1]
KIT = Path("/tmp/kit")
WORK = Path("/kaggle/working")
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MODEL = Path("/tmp/q35")
PROMPT = KIT / "data/raw/baseline/baseline/src/hallucination_detector/prompt.md"
EMB = KIT / "data/interim/embeddings"
# a T4 has no bfloat16 and no FlashAttention 2 (as in the probes, results/kaggle_token_probe.json)
ENGINE = {"dtype": "float16", "attention_config": {"backend": "TRITON_ATTN"}, "gdn_prefill_backend": "triton"}
MAX_LEN = 65536
STARTED = time.time()
RESULT: dict = {"model": MODEL_NAME}


def log(text: str) -> None:
    line = f"[{time.time() - STARTED:6.0f} s] {text}"
    print(line, flush=True)
    with open(WORK / "rehearsal.log", "a", encoding="utf-8") as f:
        print(line, file=f)


def run(name: str, command: list[str], env: dict | None = None) -> float:
    t0 = time.time()
    with open(WORK / f"{name}.log", "w", encoding="utf-8") as out:
        code = subprocess.call(command, cwd=KIT, stdout=out, stderr=subprocess.STDOUT, env=env)
    seconds = time.time() - t0
    log(f"{name}: exit {code}, {seconds:.0f} s")
    RESULT[f"{name}_seconds"] = round(seconds)
    if code:
        save()
        raise RuntimeError(f"{name} failed, see {name}.log")
    return seconds


def save() -> None:
    (WORK / "rehearsal.json").write_text(json.dumps(RESULT, indent=1) + "\n", encoding="utf-8")


def logprob_check(env: dict) -> None:
    """vLLM's prompt_logprobs of the three shortest contexts, in a process of its own (one engine per process)."""
    code = f"""
import json, sys
sys.path.insert(0, "{KIT}/src")
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from guardian.extract import render_tokens
rows = pd.read_parquet("{KIT}/data/rental/r_contexts.parquet")
rows = rows.assign(n=rows.prompt.str.len()).nsmallest(3, "n")
tok = AutoTokenizer.from_pretrained("{MODEL}")
prompt = open("{PROMPT}", encoding="utf-8").read()
rendered = [render_tokens(tok, prompt, r.prompt, r.response) for r in rows.itertuples()]
llm = LLM(model="{MODEL}", max_model_len={MAX_LEN}, gpu_memory_utilization=0.6, language_model_only=True,
          **{json.dumps(ENGINE)})
out = llm.generate([TokensPrompt(prompt_token_ids=x.ids) for x in rendered],
                   SamplingParams(max_tokens=1, prompt_logprobs=0))
res = {{}}
for r, x, o in zip(rows.itertuples(), rendered, out):
    res[r.id] = [o.prompt_logprobs[p][x.ids[p]].logprob for p in x.turn]
json.dump(res, open("{WORK}/vllm_prompt_logprobs.json", "w"))
"""
    run("logprob_check", [sys.executable, "-c", code], env)
    theirs = json.loads((WORK / "vllm_prompt_logprobs.json").read_text())
    ours = {}
    for line in (EMB / "r_ctx_uncertainty_org.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ours[r["id"]] = r["logp"]
    diffs = {i: max(abs(a - b) for a, b in zip(ours[i], theirs[i], strict=True)) for i in theirs}
    RESULT["turn_logp_ours_vs_vllm_max_abs_diff"] = {i: round(d, 4) for i, d in diffs.items()}


def synthetic_heads() -> Path:
    """Heads of the right shape trained on the 40 rehearsal contexts with their v1s labels."""
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sys.path.insert(0, str(KIT / "src"))
    from guardian import journal
    from guardian.extract import decode
    from guardian.final import tree_arrays
    from guardian.predictor import DEFAULT_CHECKS
    from guardian.uncertainty import features

    labels = {}
    for split in ("train", "check"):
        labels |= {
            r["id"]: int(r["score"] >= 0.5)
            for r in journal.read(KIT / f"data/interim/labels/labeler_v1s_org_{split}.jsonl")
        }
    vec = {r["id"]: decode(r["vector"]) for r in journal.read(EMB / "r_ctx_final_org.jsonl")}
    unc = {r["id"]: r for r in journal.read(EMB / "r_ctx_uncertainty_org.jsonl")}
    par = {r["id"]: r["parts"] for r in journal.read(EMB / "r_ctx_parts_org.jsonl")}
    ids = [i for i in vec if i in labels]
    x = np.stack([vec[i] for i in ids])
    xu = np.stack([features(unc[i], par[i]) for i in ids])
    y = np.array([labels[i] for i in ids])
    heads = []
    b = np.zeros(len(ids))
    for seed in range(3):
        sc = StandardScaler().fit(x)
        m = LGBMClassifier(n_estimators=50, num_leaves=4, min_child_samples=3, random_state=seed, verbose=-1)
        m.fit(sc.transform(x), y)
        b += m.predict_proba(sc.transform(x))[:, 1]
        dump = m.booster_.dump_model()["tree_info"]
        heads.append(
            {
                "mean": sc.mean_.tolist(),
                "scale": sc.scale_.tolist(),
                "trees": [tree_arrays(t["tree_structure"]) for t in dump],
            }
        )
    scu = StandardScaler().fit(xu)
    mu = LogisticRegression(max_iter=5000).fit(scu.transform(xu), y)
    u = mu.predict_proba(scu.transform(xu))[:, 1]
    final = {
        "head_b": heads,
        "head_u": {
            "mean": scu.mean_.tolist(),
            "scale": scu.scale_.tolist(),
            "coef": mu.coef_[0].tolist(),
            "intercept": float(mu.intercept_[0]),
        },
        "reference_b": sorted(b.tolist()),
        "reference_u": sorted(u.tolist()),
        "weight_u": 0.5,
        "threshold": 1.0,
        "checks": [c for c in DEFAULT_CHECKS if c != "repeated_call"],
    }
    path = KIT / "submission_work/rehearsal_final.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(final), encoding="utf-8")
    RESULT["synthetic_heads_rows"] = len(ids)
    return path


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copytree(KIT_SOURCE, KIT, dirs_exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(KIT / "src"), "PYTHONIOENCODING": "utf-8"}
    run("pip", [sys.executable, "-m", "pip", "install", "-q", "vllm==0.19.1"])
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME, local_dir=str(MODEL))
    log("model downloaded")

    import pandas as pd

    contexts = pd.read_parquet(KIT / "data/rental/contexts.parquet").sample(40, random_state=0)
    contexts.to_parquet(KIT / "data/rental/r_contexts.parquet", index=False)
    timing = pd.read_parquet(KIT / "data/rental/timing.parquet")
    valid_like = timing.iloc[384:404]  # the labelled rows of valid, not among the contexts
    pd.concat([contexts, valid_like]).to_parquet(KIT / "data/rental/r_timing.parquet", index=False)
    RESULT["rows"] = {"pass": len(contexts), "predict": len(contexts) + len(valid_like)}

    seconds = run(
        "final_pass",
        [
            sys.executable,
            "scripts/final_pass.py",
            "--job",
            "data/rental/r_contexts.parquet:r_ctx",
            "--model-dir",
            str(MODEL),
            "--system-prompt",
            str(PROMPT),
            "--out-dir",
            str(EMB),
            "--max-model-len",
            str(MAX_LEN),
            "--gpu-memory-utilization",
            "0.8",
            "--engine-json",
            json.dumps(ENGINE),
        ],
        env,
    )
    tokens = sum(
        json.loads(line)["prompt_tokens"] for line in (EMB / "r_ctx_uncertainty_org.jsonl").read_text().splitlines()
    )
    RESULT["pass_tokens"] = tokens
    RESULT["pass_tokens_per_second_with_engine_start"] = round(tokens / seconds)
    save()

    logprob_check(env)
    save()

    final = synthetic_heads()
    run(
        "build",
        [
            sys.executable,
            "scripts/build_submission.py",
            "--strategy",
            "final",
            "--final",
            str(final),
            "--prompt",
            str(PROMPT),
            "--out",
            "submission",
            "--no-zip",
        ],
        env,
    )
    for f in MODEL.iterdir():
        (KIT / "submission/model" / f.name).symlink_to(f)
    config_path = KIT / "submission/model/guardian.json"
    config = json.loads(config_path.read_text())
    config |= {"max_model_len": MAX_LEN, "gpu_memory_utilization": 0.8, "engine": ENGINE, "final_minutes": 20}
    config_path.write_text(json.dumps(config, indent=1))
    run(
        "predict",
        [
            sys.executable,
            "submission/scripts/predict.py",
            "--input",
            "data/rental/r_timing.parquet",
            "--output",
            str(WORK / "rehearsal_predictions.parquet"),
        ],
        env,
    )
    shutil.copy(KIT / "submission/final_pass.log", WORK / "predict_final_pass.log")
    predictions = pd.read_parquet(WORK / "rehearsal_predictions.parquet")
    RESULT["predictions"] = {"rows": len(predictions), "labelled_1": int(predictions.label.sum())}

    run(
        "compare",
        [
            sys.executable,
            "scripts/compare_passes.py",
            "--training",
            "r_ctx",
            "--submission-log",
            str(WORK / "predict.log"),
            "--rows",
            "data/rental/r_timing.parquet",
            "--contexts",
            "data/rental/r_contexts.parquet",
            "--out",
            str(WORK / "consistency.json"),
        ],
        env,
    )
    RESULT["consistency"] = json.loads((WORK / "consistency.json").read_text())
    save()
    log("rehearsal done")


if __name__ == "__main__":
    main()
