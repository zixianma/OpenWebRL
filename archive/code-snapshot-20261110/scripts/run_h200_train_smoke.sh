#!/usr/bin/env bash
# Synthetic replay through the actual Megatron training/checkpoint pipeline.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/h200_env.sh"
export OPENWEBRL_RUNTIME_ROOT
export MODEL_ARGS_ROTARY_BASE=5000000
source "$OPENWEBRL_REPO/scripts/model_configs/qwen3-4B.sh"
python "$OPENWEBRL_REPO/scripts/prepare_h200_fixture.py"
exec python "$OPENWEBRL_REPO/train.py" "${MODEL_ARGS[@]}" \
  --hf-checkpoint /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT \
  --load /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT \
  --save "$OPENWEBRL_RUNTIME_ROOT/runs/megatron-smoke-${SLURM_JOB_ID:-manual}" \
  --save-interval 1 --num-rollout 2 --debug-train-only \
  --load-debug-rollout-data "$OPENWEBRL_RUNTIME_ROOT/runs/fixture/rollout.pt" \
  --prompt-data "$OPENWEBRL_RUNTIME_ROOT/runs/fixture/prompts.jsonl" --input-key prompt \
  --actor-num-nodes 1 --actor-num-gpus-per-node "${SMOKE_GPUS:-2}" \
  --tensor-model-parallel-size "${SMOKE_GPUS:-2}" --sequence-parallel \
  --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --moe-token-dispatcher-type alltoall \
  --megatron-to-hf-mode bridge --train-backend megatron \
  --micro-batch-size 1 --global-batch-size 4 \
  --rollout-batch-size 2 --n-samples-per-prompt 2 \
  --rollout-max-response-len 32 --rollout-max-context-len 1024 \
  --advantage-estimator grpo --ppo-epochs 1 --kl-coef 0 --kl-loss-coef 0 \
  --optimizer adam --lr 5e-7 --lr-decay-style constant \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash "$@"
