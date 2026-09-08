#!/usr/bin/env bash
# Small real-web GRPO run. Requires a configured judge endpoint.
# This launcher never kills unrelated processes or writes shell environment dumps.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/h200_env.sh"
cd "$OPENWEBRL_REPO"
NUM_GPUS="${NUM_GPUS:-2}"
TP_SIZE="${TP_SIZE:-2}"
if (( NUM_GPUS < TP_SIZE || NUM_GPUS % TP_SIZE != 0 )); then
    echo 'NUM_GPUS must be a positive multiple of TP_SIZE.' >&2; exit 1
fi
export SLIME_BROWSER_ENV_MODE=local_process
export OPENWEBRL_CUDA_CACHE_LIMIT_GIB="${OPENWEBRL_CUDA_CACHE_LIMIT_GIB:-100}"
# Avoid staging the full model and optimizer in host memory during synchronous saves.
export OPENWEBRL_STREAMING_CHECKPOINT="${OPENWEBRL_STREAMING_CHECKPOINT:-1}"
export SLIME_BROWSER_LOCAL_PROCESS_PYTHON="$OPENWEBRL_RUNTIME_ROOT/venv/bin/python"
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${BROWSER_CONCURRENCY:-4}"
export SLIME_BROWSER_ROLLOUT_CONCURRENCY="$SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES"
export MODEL_ARGS_ROTARY_BASE=5000000
source "$OPENWEBRL_REPO/scripts/model_configs/qwen3-4B.sh"
CKPT="${HF_CHECKPOINT:-/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT}"
SAVE_DIR="${SAVE_DIR:-${SLIME_SAVE_DIR:-$OPENWEBRL_RUNTIME_ROOT/runs/browser-$(date -u +%Y%m%d_%H%M%S)}}"
JUDGE_API_MODE="${JUDGE_API_MODE:-served}"
if [[ "${DRY_RUN:-0}" != 1 && "$JUDGE_API_MODE" == served && -z "${JUDGE_API_BASE:-}" ]]; then
    echo 'Set JUDGE_API_BASE to your judge endpoint before launching real-web training.' >&2; exit 1
fi
ARGS=("${MODEL_ARGS[@]}"
  --hf-checkpoint "$CKPT" --load "${SLIME_LOAD_CHECKPOINT:-$CKPT}"
  --save "$SAVE_DIR" --save-interval "${SAVE_INTERVAL:-1}"
  --num-rollout "${NUM_ROLLOUT:-2}" --num-gpus-per-node "$NUM_GPUS"
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_GPUS" --colocate
  --tensor-model-parallel-size "$TP_SIZE" --sequence-parallel
  --pipeline-model-parallel-size 1 --context-parallel-size 1
  --moe-token-dispatcher-type alltoall --megatron-to-hf-mode bridge --train-backend megatron
  --prompt-data "${TRAIN_DATA:-openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet}"
  --input-key prompt --rollout-shuffle
  --custom-generate-function-path openwebrl.generate_browser.generate_turn_sample
  --custom-rm-path openwebrl.reward_browser.reward_func
  --custom-config-path "${BROWSER_TRAIN_CONFIG:-openwebrl/browser_training_config.yaml}"
  --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonempty_nonzero_std
  --max-steps "${BROWSER_MAX_STEPS:-3}" --context-num-screenshots 1 --judge-max-attached-imgs 3
  --judge-api-mode "$JUDGE_API_MODE" --judge-api-model "${JUDGE_MODEL:-gpt-4.1}"
  --judge-prompt-variant action_history --turn-history-reasoning-mode full
  --browser-response-format-mode browser_env --browser-include-tool-response 1
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-2}" --n-samples-per-prompt "${N_SAMPLES:-5}"
  --rollout-max-response-len "${RESPONSE_LEN:-1024}" --rollout-max-context-len "${CONTEXT_LEN:-8192}"
  --rollout-temperature 0.8 --micro-batch-size 1 --global-batch-size "${GLOBAL_BATCH_SIZE:-8}"
  --advantage-estimator grpo --ppo-epochs 2 --use-rollout-logprobs
  --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28
  --optimizer adam --lr "${LEARNING_RATE:-5e-7}" --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.4
  --sglang-server-concurrency "${SGLANG_CONCURRENCY:-4}" --sglang-max-running-requests "${SGLANG_CONCURRENCY:-4}" --sglang-chunked-prefill-size 4096
)
if [[ "${RECOMPUTE_ACTIVATIONS:-0}" == 1 ]]; then
    ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
fi
if [[ -n "${SLIME_CKPT_STEP:-}" ]]; then ARGS+=(--ckpt-step "$SLIME_CKPT_STEP"); fi
if [[ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" == 1 ]]; then ARGS+=(--override-opt-param-scheduler); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' python train.py "${ARGS[@]}" "$@"; printf '\n'; exit 0
fi
mkdir -p "$SAVE_DIR/run_config"
cp "$OPENWEBRL_RUNTIME_ROOT/logs/environment.freeze.txt" "$SAVE_DIR/run_config/"
cp "${BASH_SOURCE[0]}" "$SAVE_DIR/run_config/launcher.sh"
exec python train.py "${ARGS[@]}" "$@"
