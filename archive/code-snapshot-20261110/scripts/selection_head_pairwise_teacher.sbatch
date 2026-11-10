#!/usr/bin/env bash
# Proposed allocation; submit only after approval: 1 H200, 8 CPUs, 120 GiB, 1 hour.
# Four matched frozen-feature fits: soft CE versus GPT-5.5 winner-pairwise,
# seeds 0 and 1. No API calls, browser sessions, or model fine-tuning.
# The batch process owns and awaits every worker.
#SBATCH --job-name=arm-pairwise-teacher
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-pairwise-teacher-20261014/logs/slurm-%j.out
set -euo pipefail

PAIRWISE_ROOT=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-pairwise-teacher-20261014
PAIRWISE_SOURCE=$PAIRWISE_ROOT/source
PAIRWISE_DATA=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008
PAIRWISE_PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python
export PYTHONPATH=$PAIRWISE_SOURCE
export HF_HOME=$PAIRWISE_ROOT/hf-cache
export XDG_CACHE_HOME=$PAIRWISE_ROOT/cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PAIRWISE_ROOT/logs" "$PAIRWISE_ROOT/results"
cd "$PAIRWISE_SOURCE"

for objective in soft_ce pairwise_teacher; do
    for seed in 0 1; do
        output="$PAIRWISE_ROOT/results/${objective}-seed${seed}"
        "$PAIRWISE_PY" scripts/train_selection_head.py \
            --panels "$PAIRWISE_DATA/panels/panels.jsonl" \
            --features "$PAIRWISE_DATA/features-piotr" \
            --output "$output" --device cuda:0 \
            --pointwise --pools action_mean,last \
            --objective "$objective" --seed "$seed" \
            > "$PAIRWISE_ROOT/logs/train-${objective}-seed${seed}-${SLURM_JOB_ID}.log" 2>&1

        "$PAIRWISE_PY" scripts/train_selection_head.py \
            --panels "$PAIRWISE_DATA/panels/panels.jsonl" \
            --features "$PAIRWISE_DATA/features-piotr" \
            --branch-panels "$PAIRWISE_DATA/branch124/panels.jsonl" \
            --branch-features "$PAIRWISE_DATA/features-branch124" \
            --branch-picks-only --load-head "$output/head.pt" \
            --output "$output/branch124" --device cuda:0 \
            > "$PAIRWISE_ROOT/logs/picks-${objective}-seed${seed}-${SLURM_JOB_ID}.log" 2>&1
    done
done

"$PAIRWISE_PY" - "$PAIRWISE_ROOT" "$SLURM_JOB_ID" <<'PY'
import json
from pathlib import Path
import sys
root, job_id = Path(sys.argv[1]), sys.argv[2]
expected = [(objective, seed) for objective in ('soft_ce', 'pairwise_teacher') for seed in (0, 1)]
for objective, seed in expected:
    directory = root / 'results' / f'{objective}-seed{seed}'
    metrics = json.loads((directory / 'metrics.json').read_text())
    picks = json.loads((directory / 'branch124' / 'metrics.json').read_text())['branch_head_picks']
    assert metrics['args']['objective'] == objective
    assert len(picks) == 124, (objective, seed, len(picks))
(root / 'logs' / f'complete-{job_id}.json').write_text(json.dumps({'job_id': job_id, 'fits': len(expected), 'branch124_picks_per_fit': 124}, indent=2))
PY
