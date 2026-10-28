#!/usr/bin/env bash
#SBATCH --job-name=om2w-rerun-j1
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@240
#SBATCH --export=ALL
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-om2w-table5-20261012/rerun300-j1-vlm/campaign-%j.out
set -euo pipefail
: "${1:?Provide exact approved request path}"
cd /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-om2w-table5-20261012/rerun300-j1-vlm/source
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_PROJECT=openwebrl-evals WANDB_MODE=disabled
export BASH_ENV=/dev/null ENV=/dev/null HISTFILE=/dev/null PYTHONPATH=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-om2w-table5-20261012/rerun300-j1-vlm/source
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python -B scripts/launch_webprm_om2w_table5.py --execute --proposal /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-om2w-table5-20261012/rerun300-j1-vlm/proposal.json --approval "$1"
