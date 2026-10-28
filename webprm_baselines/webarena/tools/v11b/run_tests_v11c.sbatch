#!/usr/bin/env bash
#SBATCH --job-name=wa-v11c-tests
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/tests-v11c-%j.log
# CPU-only (no GPU): zero-paid v11c regression tests per package. Usage: V11C_NAMES="shop-a ..." sbatch run_tests_v11c.sbatch
set -u
export TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021
T=$D/v11b-tools
for n in $V11C_NAMES; do
  export V11_PACKAGE=$D/launch-preparation-v11c-$n PYTHONPATH=$D/launch-preparation-v11c-$n/source WEBPRM_NATIVE_SOURCE_ROOT=$D/launch-preparation-v11c-$n/source
  L=$T/tests-v11c-$n.log
  suites="test_webprm_concurrent_broker test_webprm_campaign_queue test_webprm_cms_site test_webprm_cms_native test_webprm_cms_hosting test_audit_webprm_table5_cms test_run_webprm_table5_cms test_run_webprm_table5_shopping"
  case $n in gitlab*) suites="$suites test_validate_webprm_gitlab_hosting test_webprm_gitlab_native test_webprm_gitlab_recovery";; esac
  { echo "=== $n v11 cell + Piotr + no_search + audit-v8 replay"; cd $T && $PY -B test_v11c.py 2>&1 | grep -v "^  " | tail -16
    echo "=== $n v11 broker piotr route"; cd /tmp && $PY -B $T/test_v11_broker.py 2>&1 | grep -E "^test|^Ran|^OK|FAILED|Error" | tail -6
    echo "=== $n v9e status + same-cell (v11 routes)"; cd /tmp && $PY -B $T/test_v11_status.py 2>&1 | grep -E "^test|^Ran|^OK|FAILED|Error" | tail -5
    echo "=== $n v11b Piotr preflight regression"; cd /tmp && $PY -B $T/test_v11b_preflight.py 2>&1 | grep -E "^test|^Ran|^OK|FAILED|Error|probe" | tail -6
    echo "=== $n v11c audit membership regression"; cd /tmp && $PY -B $T/test_v11c_audit_membership.py 2>&1 | grep -E "^test|^Ran|^OK|FAILED|Error" | tail -6
    echo "=== $n package unit suites"; cd $V11_PACKAGE/source && for t in $suites; do echo "--- $t"; $PY -B -m unittest tests.$t 2>&1 | tail -3; done
  } > $L 2>&1
  echo "$n: $(grep -c '^OK' $L) OK, $(grep -c 'FAILED\|Error' $L) fail-lines"
done
