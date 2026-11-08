#!/bin/bash
#SBATCH --job-name=rejudge --account=vlm --partition=gpu-h200 --qos=normal --cpus-per-task=2 --mem=16G --time=12:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/deferjudge/rejudge-%j.log
# CPU only (no GPUs). Judges deferred records with each benchmark's native judge under a per-day cap (default $100/day).
# Usage: sbatch scripts/eval/rejudge.sbatch <daily_usd> <ABSOLUTE plan dir: <root>/<condition>/<benchmark>> ...
# The judge key comes from JUDGE_API_KEY or OPENAI_API_KEY in the repo's .env (never printed). See openwebrl/docs/EVAL.md.
REPO=${OPENWEBRL_REPO:-/gpfs/projects/krishna/zixianma/OpenWebRL}
source $REPO/scripts/h200_env.sh
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$REPO
cd $REPO
CAP=$1; shift
exec ${PYTHON:-/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python} $REPO/scripts/eval/crossbench_defer.py \
  rejudge --daily-usd $CAP --concurrency 4 --dirs "$@"
