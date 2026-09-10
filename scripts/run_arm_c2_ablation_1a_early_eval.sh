#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python \
  scripts/run_arm_c2_ablation_1a_early_eval.py \
  --job-id 285567 \
  --run-root /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-ablation-1a
