#!/usr/bin/env bash
#SBATCH --job-name=wa-v9b-tests
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v9b-tools/tests-%j.log
set -u
export TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021
export V9_PACKAGE=${V9_PACKAGE:-$D/launch-preparation-v9b-reddit-a}
export PYTHONPATH=$V9_PACKAGE/source WEBPRM_NATIVE_SOURCE_ROOT=$V9_PACKAGE/source
echo "=== v9 cell + audit-v5 replay"; cd $D/v9b-tools && $PY -B test_v9.py 2>&1 | grep -v "^  " | tail -40
echo "=== package unit suites"; cd $V9_PACKAGE/source && for t in tests/test_webprm_concurrent_broker.py tests/test_launch_webprm_table5_campaign.py tests/test_webprm_cms_site.py tests/test_run_webprm_table5_cms.py tests/test_run_webprm_table5_shopping.py tests/test_audit_webprm_table5_cms.py tests/test_webprm_cms_native.py; do echo "--- $t"; $PY -B -m pytest -q -x -p no:cacheprovider $t 2>&1 | tail -4; done
echo "=== new-tab on v3"; cd $D/native-browser-v3 && NATIVE_EPISODE_DIR=$D/native-browser-v3 $PY -B test_newtab.py 2>&1 | grep -v "^  " | tail -4
