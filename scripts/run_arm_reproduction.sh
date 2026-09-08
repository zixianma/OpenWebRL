#!/usr/bin/env bash
# Run on the assigned compute node after its GPUs are available.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/h200_env.sh
ARM_RUNTIME="$OPENWEBRL_RUNTIME_ROOT/arm-reproduction"
ARM_PYTHON="$ARM_RUNTIME/venv/bin/python"
ACTOR_PATH="/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT"
ARM_MODEL_ROOT="/gpfs/scrubbed/zixianma/checkpoints/web/arm"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARM_RUNTIME/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
ARM_NUM_GPUS="${ARM_NUM_GPUS:-2}"
ACTOR_GPU="${ACTOR_GPU:-0}"
if [[ "$ARM_NUM_GPUS" == 1 ]]; then
    ARM_GPU="${ARM_GPU:-0}"
else
    ARM_GPU="${ARM_GPU:-1}"
fi
ACTOR_MEMORY_FRACTION="${ACTOR_MEMORY_FRACTION:-0.4}"
ARM_JOB_ID="${ARM_JOB_ID:-${SLURM_JOB_ID:-}}"
if [[ -z "$ARM_JOB_ID" ]]; then
    echo "Set ARM_JOB_ID to your active allocation ID." >&2
    exit 1
fi
ARM_JOB_INFO="$(scontrol show job "$ARM_JOB_ID" -d -o)"
if [[ "$ARM_JOB_INFO" != *"UserId=$(id -un)("* || "$ARM_JOB_INFO" != *"JobState=RUNNING"* ]]; then
    echo "The requested allocation is not your running job." >&2
    exit 1
fi
if [[ "${SLURM_JOB_ID:-}" != "$ARM_JOB_ID" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "Launch inside srun --jobid=$ARM_JOB_ID --overlap --gres=gpu:h200:$ARM_NUM_GPUS bash scripts/run_arm_reproduction.sh" >&2
    exit 1
fi
if [[ "$ARM_NUM_GPUS" != 1 && "$ARM_NUM_GPUS" != 2 ]]; then
    echo "ARM_NUM_GPUS must be 1 or 2." >&2
    exit 1
fi
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" != "$ARM_NUM_GPUS" ]]; then
    echo "Visible devices do not match ARM_NUM_GPUS=$ARM_NUM_GPUS." >&2
    exit 1
fi
ARM_NODES="$(squeue -h -j "$ARM_JOB_ID" -o %N)"
if ! scontrol show hostnames "$ARM_NODES" | grep -qx "$(hostname -s)"; then
    echo "Run this script on the allocated compute node." >&2
    exit 1
fi
# This launcher is for free assigned GPUs. Sharing active training needs a
# separate memory plan; never terminate workloads to make room.
for gpu in "$ACTOR_GPU" "$ARM_GPU"; do
    GPU_PROCESSES="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
    if [[ -n "$GPU_PROCESSES" ]]; then
        echo "GPU $gpu is occupied; existing workloads were left running." >&2
        exit 1
    fi
done
mkdir -p "$OUTPUT_ROOT"
ACTOR_PID=""
ARM_PID=""
cleanup() {
    if [[ -n "$ARM_PID" ]]; then kill "$ARM_PID" 2>/dev/null || true; wait "$ARM_PID" 2>/dev/null || true; fi
    if [[ -n "$ACTOR_PID" ]]; then kill "$ACTOR_PID" 2>/dev/null || true; wait "$ACTOR_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_service() {
    local endpoint="$1" pid="$2" deadline=$((SECONDS+900))
    until curl --noproxy '*' -fs "$endpoint" >/dev/null; do
        if ! kill -0 "$pid" 2>/dev/null; then echo "Server exited; see logs in $OUTPUT_ROOT" >&2; return 1; fi
        if (( SECONDS > deadline )); then echo "Server startup timed out" >&2; return 1; fi
        sleep 2
    done
}
CUDA_VISIBLE_DEVICES="$ACTOR_GPU" "$OPENWEBRL_RUNTIME_ROOT/venv/bin/python" -m sglang.launch_server \
    --model-path "$ACTOR_PATH" --host 127.0.0.1 --port 19100 \
    --dtype bfloat16 --tp 1 --mem-fraction-static "$ACTOR_MEMORY_FRACTION" --context-length 32768 \
    --max-running-requests 24 --chunked-prefill-size 4096 --disable-cuda-graph \
    >"$OUTPUT_ROOT/actor.log" 2>&1 &
ACTOR_PID=$!
wait_service http://127.0.0.1:19100/health_generate "$ACTOR_PID"
for mode in baseline scalar selection; do
    if [[ "$mode" != baseline ]]; then
        if [[ "$mode" == scalar ]]; then
            model="$ARM_MODEL_ROOT/scalar/71c58489cd7cbaebcd74656df7ef11ba818661b8"
        else
            model="$ARM_MODEL_ROOT/selection/81b452d800d9f859687074f82680dd5257e02d89"
        fi
        CUDA_VISIBLE_DEVICES="$ARM_GPU" "$ARM_PYTHON" scripts/serve_arm.py \
            --mode "$mode" --model "$model" --base "$ACTOR_PATH" \
            >"$OUTPUT_ROOT/$mode-server.log" 2>&1 &
        ARM_PID=$!
        wait_service http://127.0.0.1:19101/health "$ARM_PID"
    fi
    "$ARM_PYTHON" -m openwebrl.arm_eval --mode "$mode" --output "$OUTPUT_ROOT/$mode" \
        --parallel "${N_PARALLEL:-4}" --task-indices "${TASK_INDICES:-}" \
        --seed "${EVAL_SEED:-42}" --env-file .env >"$OUTPUT_ROOT/$mode-eval.log" 2>&1
    if [[ -n "$ARM_PID" ]]; then
        kill "$ARM_PID"
        wait "$ARM_PID" 2>/dev/null || true
        ARM_PID=""
    fi
done
"$ARM_PYTHON" scripts/summarize_arm_reproduction.py "$OUTPUT_ROOT"
