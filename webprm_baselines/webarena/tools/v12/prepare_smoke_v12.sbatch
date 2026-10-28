#!/usr/bin/env bash
#SBATCH --job-name=wa-v12-prep
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v12-tools/prepare-smoke-v12-%j.log
# CPU-only (no GPU) wrapper.
set -euo pipefail
export SLURM_JOB_ID
/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v12-tools/prepare_smoke_v12.sh
