#!/usr/bin/env bash
#SBATCH --job-name=wa-v9b-tests-gitlab
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v9b-tools/tests-gitlab-%j.log
set -u
export TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021
for n in gitlab-a gitlab-b; do
  export V9_PACKAGE=$D/launch-preparation-v9b-$n PYTHONPATH=$D/launch-preparation-v9b-$n/source WEBPRM_NATIVE_SOURCE_ROOT=$D/launch-preparation-v9b-$n/source
  L=$D/v9b-tools/tests-$n.log
  { echo "=== $n v9 cell + audit-v5 replay"; cd $D/v9b-tools && $PY -B test_v9.py 2>&1 | grep -v "^  " | tail -14
    echo "=== $n v9b broker n=20 end-to-end"; cd /tmp && $PY -B $D/v9b-tools/test_v9b_broker_n20.py 2>&1 | grep -E "^test|^Ran|^OK|FAILED|Error" | tail -6
    echo "=== $n audit-v5 controls"; cd $D/native-audit-v6 && $PY -B test_native_audit_v6.py 2>&1 | tail -3
    echo "=== $n package unit suites"; cd $V9_PACKAGE/source && for t in test_webprm_concurrent_broker test_webprm_campaign_queue test_webprm_cms_site test_webprm_cms_native test_webprm_cms_hosting test_audit_webprm_table5_cms test_run_webprm_table5_cms test_run_webprm_table5_shopping test_validate_webprm_gitlab_hosting test_webprm_gitlab_native test_webprm_gitlab_recovery; do echo "--- $t"; $PY -B -m unittest tests.$t 2>&1 | tail -3; done
  } > $L 2>&1
  echo "$n: $(grep -c '^OK' $L) OK, $(grep -c 'FAILED\|Error' $L) fail-lines"
done
