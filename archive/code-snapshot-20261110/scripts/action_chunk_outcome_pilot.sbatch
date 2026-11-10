#!/usr/bin/env bash
# Prepared only: a new paid allocation requires exact user approval.
# Frozen actor replay and offline tool-name value fit; no API/browser calls.
#SBATCH --job-name=arm-chunk-value
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-action-chunk-outcome-20261017-v2/logs/slurm-%j.out
set -euo pipefail

PILOT_ROOT=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-action-chunk-outcome-20261017-v2
PILOT_PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python
export HF_HOME="$PILOT_ROOT/hf-cache"
export XDG_CACHE_HOME="$PILOT_ROOT/cache"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PILOT_ROOT/logs" "$PILOT_ROOT/results" "$HF_HOME" "$XDG_CACHE_HOME"
sha256sum --check "$PILOT_ROOT/pilot-inputs.sha256"

"$PILOT_PY" "$PILOT_ROOT/source/run_action_chunk_outcome_pilot.py" \
    --panels "$PILOT_ROOT/panels.jsonl" \
    --manifest "$PILOT_ROOT/manifest.json" \
    --output "$PILOT_ROOT/results" \
    --model /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT \
    --batch-size 16

"$PILOT_PY" - "$PILOT_ROOT" "$SLURM_JOB_ID" <<'PY'
import json
from pathlib import Path
import sys
root, job = Path(sys.argv[1]), sys.argv[2]
manifest = json.loads((root / "manifest.json").read_text())
metrics = json.loads((root / "results" / "metrics.json").read_text())
assert metrics["states"] == manifest["counts"]["admitted_states"]
assert metrics["task_groups"] == manifest["counts"]["task_groups"]
(root / "logs" / f"complete-{job}.json").write_text(json.dumps({
    "job_id": job, "status": "verified_complete", "states": metrics["states"],
    "task_groups": metrics["task_groups"]}, indent=2) + "\n")
PY
