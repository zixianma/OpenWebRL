#!/usr/bin/env bash
#SBATCH --job-name=wa-v11-fuzzy-rescore
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/fuzzy-rescore-gpt41-v11/rescore-%j.log
export TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export PYTHONPATH=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/nltk-runtime/venv/lib/python3.12/site-packages NLTK_DATA=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/nltk-runtime/nltk_data
cd /tmp && /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python -B /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/fuzzy-rescore-gpt41-v11/rescore.py
