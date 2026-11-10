#!/usr/bin/env bash
# Prepared only. Submit after exact approval of 1 H200, 8 CPUs, 120 GiB, 2h total.
# Offline frozen-actor replay and a token-Q head; no API calls or browsers.
# The batch process owns and awaits the single worker.
#SBATCH --job-name=arm-action-token-q
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-action-token-q-pilot-20261014/logs/slurm-%j.out
set -euo pipefail

PILOT_ROOT=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-action-token-q-pilot-20261014
PILOT_REPO=/gpfs/projects/krishna/zixianma/OpenWebRL
PILOT_PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python
export PYTHONPATH="$PILOT_REPO"
export HF_HOME="$PILOT_ROOT/hf-cache"
export XDG_CACHE_HOME="$PILOT_ROOT/cache"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PILOT_ROOT/logs" "$PILOT_ROOT/results" "$HF_HOME" "$XDG_CACHE_HOME"
cd "$PILOT_REPO"
sha256sum --check "$PILOT_ROOT/pilot-inputs.sha256"

"$PILOT_PY" scripts/run_action_token_q_pilot.py \
    --forks "$PILOT_ROOT/forks.jsonl" \
    --output "$PILOT_ROOT/results" \
    --model /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT \
    --chunk-size 25

"$PILOT_PY" - "$PILOT_ROOT" "$SLURM_JOB_ID" <<'PY'
import json
from pathlib import Path
import sys
root, job = Path(sys.argv[1]), sys.argv[2]
data = json.loads((root / 'results' / 'metrics.json').read_text())
assert data['train_panels'] == 506 and data['validation_panels'] == 74
assert data['branch124_observed_fork_outcomes']['inventory']['historical_states'] == 124
assert data['branch124_observed_fork_outcomes']['inventory']['states_with_observed_fork'] == 105
(root / 'logs' / f'complete-{job}.json').write_text(json.dumps({
    'job_id': job, 'train_panels': 506, 'validation_panels': 74,
    'historical_branch_states': 124, 'fork_scored_branch_states': 105,
    'status': 'verified_complete'}, indent=2) + '\n')
PY
