#!/usr/bin/env bash
# Actual browser turns and multimodal training; synthetic rewards, no judge call.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/h200_env.sh"
export SLIME_BROWSER_ENV_MODE=local_process
export SLIME_BROWSER_LOCAL_PROCESS_PYTHON="$OPENWEBRL_RUNTIME_ROOT/venv/bin/python"
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES=4 SLIME_BROWSER_ROLLOUT_CONCURRENCY=4
exec bash "$OPENWEBRL_REPO/scripts/run_h200_pipeline_smoke.sh" \
  --prompt-data "$OPENWEBRL_RUNTIME_ROOT/runs/fixture/browser-prompts.jsonl" \
  --custom-generate-function-path openwebrl.generate_browser.generate_turn_sample \
  --custom-config-path "$OPENWEBRL_REPO/openwebrl/browser_training_config.yaml" \
  --max-steps 1 --context-num-screenshots 1 --browser-response-format-mode browser_env \
  --rollout-max-response-len 256 --rollout-max-context-len 8192 \
  --save "$OPENWEBRL_RUNTIME_ROOT/runs/browser-fixture-${SLURM_JOB_ID:-manual}" "$@"
