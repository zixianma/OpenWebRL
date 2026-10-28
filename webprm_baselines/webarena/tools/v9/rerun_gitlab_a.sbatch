#!/usr/bin/env bash
#SBATCH --job-name=wa-v9b-glab-recheck
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v9b-tools/recheck-gitlab-%j.log
export TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
for n in gitlab-a gitlab-a; do
export V9_PACKAGE=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/launch-preparation-v9b-$n PYTHONPATH=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/launch-preparation-v9b-$n/source WEBPRM_NATIVE_SOURCE_ROOT=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/launch-preparation-v9b-$n/source
echo "=== test_v9 $n"; cd /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v9b-tools && $PY -B test_v9.py 2>&1 | tail -25
done
echo "=== gitlab hosting validator tests (a)"; cd $V9_PACKAGE/source && $PY -B -m unittest tests.test_validate_webprm_gitlab_hosting 2>&1 | tail -30
