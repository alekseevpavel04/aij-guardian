#!/usr/bin/env bash
# The rented A100, step by step (docs/RENTAL.md). Run from the unpacked kit:  bash scripts/rental.sh <step>
#
#   guard        first of all: the instance stops itself after GUARD_HOURS (6) whatever happens; its disk stays
#   setup        pinned requirements (vLLM 0.19.1 like the platform)
#   model        Qwen/Qwen3.6-35B-A3B-FP8 from the Hub, then the fingerprint check against the organiser's copy
#   pass         the platform's engine over every training row, one engine for all files (resumable)
#   export       the ruler on the FP8 features, the final heads, the threshold, the shift study
#   archive      the submission folder with the weights linked in
#   timing       predict.py on 548 test-shaped rows, with the card's memory logged
#   compare      the submission's features of the contexts against the training pass's
#   pack         what goes home: features, results, the exported heads, logs
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-/workspace/qwen36-fp8}
PROMPT=data/raw/baseline/baseline/src/hallucination_detector/prompt.md
EMB=data/interim/embeddings
S=data/synthetic
GPU_UTIL=${GPU_UTIL:-0.85}   # 0.90 on a 40 GB card
mkdir -p logs results submission_work "$EMB"

case "${1:-}" in
guard)
    # vast.ai gives each instance its own id and a key that can manage it; stopping ends the GPU charge and keeps
    # the disk, so the results survive a night nobody watches
    if [ -z "${CONTAINER_ID:-}" ] || [ -z "${CONTAINER_API_KEY:-}" ]; then
        echo "no CONTAINER_ID / CONTAINER_API_KEY in this instance: stop it from home instead"
        exit 1
    fi
    $PY -m pip install -q vastai
    nohup bash -c "sleep ${GUARD_HOURS:-6}h; vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY"         > logs/guard.log 2>&1 &
    echo "the instance stops itself at $(date -d "+${GUARD_HOURS:-6} hours")"
    ;;
setup)
    $PY -m pip install -r configs/rental_requirements.txt 2>&1 | tail -3
    $PY -m pip install -e . 2>&1 | tail -1
    $PY -c "import vllm, lightgbm, sklearn; print('vllm', vllm.__version__, 'lightgbm', lightgbm.__version__, 'sklearn', sklearn.__version__)"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    ;;
model)
    $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-35B-A3B-FP8', local_dir='$MODEL')"
    $PY scripts/weights_fingerprint.py --model-dir "$MODEL" --check results/weights_fingerprint.json
    ;;
pass)
    # the contexts first: with them alone the ruler can already compare Q4 and FP8. With two cards the τ² rows
    # (24 of the 52 million tokens) go to the second one in a pass of their own.
    ARGS=(--model-dir "$MODEL" --system-prompt "$PROMPT" --out-dir "$EMB" --gpu-memory-utilization "$GPU_UTIL")
    FIRST=(data/rental/contexts.parquet:fp8_contexts data/rental/sim_q36.parquet:fp8_sim_q36
           data/rental/sim_glm.parquet:fp8_sim_glm data/rental/sim_q36n.parquet:fp8_sim_q36n
           data/rental/sim_kbank_s350.parquet:fp8_sim_kbank data/rental/pairs.parquet:fp8_pairs)
    if [ "$(nvidia-smi -L | wc -l)" -ge 2 ]; then
        CUDA_VISIBLE_DEVICES=1 $PY scripts/final_pass.py "${ARGS[@]}" --job data/rental/tau.parquet:fp8_tau             > logs/rental_pass_tau.log 2>&1 &
        TAU=$!
        CUDA_VISIBLE_DEVICES=0 $PY scripts/final_pass.py "${ARGS[@]}" --job "${FIRST[@]}" 2>&1 | tee -a logs/rental_pass.log
        wait $TAU
    else
        $PY scripts/final_pass.py "${ARGS[@]}" --job "${FIRST[@]}" data/rental/tau.parquet:fp8_tau 2>&1             | tee -a logs/rental_pass.log
    fi
    ;;
export)
    $PY scripts/tau2ru_ruler.py --labels opus --quick --unc-skip glm kglm --tag final_fp8 \
        --out results/final_fp8_ruler.json --export submission_work/final_fp8.json \
        --sim-total 1 --drop-repeats \
        --context-vectors fp8_contexts_final_org --context-features fp8_contexts \
        --tau-vectors fp8_tau_final_org --pair-vectors fp8_pairs_final_org \
        --source "$S/tau2_ru_q36_rows.parquet|fp8_sim_q36|$S/opus_labels_tau2q36" \
        --source "$S/tau2_ru_glm_rows.parquet|fp8_sim_glm|$S/opus_labels_tau2glm" \
        --source "$S/tau2_ru_q36n_rows.parquet|fp8_sim_q36n|$S/opus_labels_tau2q36n" \
        --source "$S/tau2_ru_kbank_s350_rows.parquet|fp8_sim_kbank|$S/opus_labels_tau2kbank" 2>&1 | tee logs/rental_export.log
    for view in agreed v1s opus; do
        $PY scripts/paired_variants.py --a final_fp8_ruler --b reference --teacher "$view"
    done
    $PY scripts/threshold_shift.py --oof submission_work/final_fp8.oof.npz --out results/threshold_shift_fp8.json
    $PY scripts/threshold_shift.py --oof submission_work/final_fp8.oof.npz --teacher opus \
        --out results/threshold_shift_fp8_opus.json
    ;;
archive)
    $PY scripts/build_submission.py --strategy final --final submission_work/final_fp8.json --prompt "$PROMPT" \
        --out submission --no-zip
    for f in "$MODEL"/*; do ln -sf "$f" "submission/model/$(basename "$f")"; done
    ls submission/model | head
    ;;
timing)
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 5 > logs/rental_gpu.csv &
    SMI=$!
    # one card, as on the platform
    CUDA_VISIBLE_DEVICES=0 $PY submission/scripts/predict.py --input data/rental/timing.parquet \
        --output submission_work/timing_predictions.parquet 2>&1 | tee logs/rental_timing.log
    kill $SMI
    cp submission/final_pass.log logs/rental_timing_pass.log
    ;;
compare)
    $PY scripts/compare_passes.py --training fp8_contexts --submission-log logs/rental_timing.log \
        --rows data/rental/timing.parquet --contexts data/rental/contexts.parquet
    ;;
pack)
    tar -czf rental_results.tgz "$EMB"/fp8_* results submission_work/final_fp8.json submission_work/final_fp8.oof.npz \
        submission_work/timing_predictions.parquet logs
    ls -la rental_results.tgz
    ;;
*)
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac
