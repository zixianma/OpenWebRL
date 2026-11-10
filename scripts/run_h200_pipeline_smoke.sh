#!/usr/bin/env bash
# Real SGLang sampling + Megatron GRPO + weight handoff, synthetic text rewards.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/h200_env.sh"
export OPENWEBRL_RUNTIME_ROOT MODEL_ARGS_ROTARY_BASE=5000000
source "$OPENWEBRL_REPO/scripts/model_configs/qwen3-4B.sh"
python "$OPENWEBRL_REPO/scripts/prepare_h200_fixture.py"
exec python "$OPENWEBRL_REPO/train.py" "${MODEL_ARGS[@]}" \
  --hf-checkpoint /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT \
  --load /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT \
  --save "$OPENWEBRL_RUNTIME_ROOT/runs/pipeline-smoke-${SLURM_JOB_ID:-manual}" \
  --save-interval 1 --num-rollout 2 --num-gpus-per-node 2 \
  --actor-num-nodes 1 --actor-num-gpus-per-node 2 --colocate \
  --tensor-model-parallel-size 2 --sequence-parallel \
  --pipeline-model-parallel-size 1 --context-parallel-size 1 --moe-token-dispatcher-type alltoall \
  --megatron-to-hf-mode bridge --train-backend megatron \
  --prompt-data "$OPENWEBRL_RUNTIME_ROOT/runs/fixture/prompts.jsonl" --input-key prompt --apply-chat-template \
  --custom-rm-path scripts.h200_smoke_reward.reward \
  --micro-batch-size 1 --global-batch-size 4 --rollout-batch-size 2 --n-samples-per-prompt 2 \
  --rollout-max-response-len 32 --rollout-max-context-len 1024 \
  --advantage-estimator grpo --ppo-epochs 2 --use-rollout-logprobs --kl-coef 0 --kl-loss-coef 0 \
  --optimizer adam --lr 5e-7 --lr-decay-style constant \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.35 \
  --sglang-server-concurrency 4 --sglang-max-running-requests 4 \
  --sglang-chunked-prefill-size 1024 --sglang-disable-cuda-graph "$@"
