#!/usr/bin/env bash
#SBATCH --job-name=wa-v9f-gitlab-smoke
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:40:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v9b-tools/gitlab/smoke-v9f-%j.log
set -u
export TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export PYTHONPATH=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/nltk-runtime/venv/lib/python3.12/site-packages:/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/launch-preparation-v9f-gitlab-a3/source
export NLTK_DATA=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/nltk-runtime/nltk_data
cd /tmp && /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python -B /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v9b-tools/gitlab/smoke_gitlab.py /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/launch-preparation-v9f-gitlab-a3 /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v9b-tools/gitlab/smoke-v9f-run-$SLURM_JOB_ID
