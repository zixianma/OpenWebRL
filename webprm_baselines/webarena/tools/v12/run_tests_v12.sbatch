#!/usr/bin/env bash
#SBATCH --job-name=wa-v12-tests
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v12-tools/tests-v12-%j.log
# CPU-only (no GPU): zero-paid v12 tests per package. Usage: V12_NAMES="reddit-a ..." sbatch run_tests_v12.sbatch
set -u
export TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 HF_HOME=/tmp/v12-hf-$SLURM_JOB_ID
export NLTK_DATA=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/nltk-runtime/nltk_data
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
VPY=/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021
T=$D/v12-tools
NL=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/nltk-runtime/venv/lib/python3.12/site-packages
for n in $V12_NAMES; do
  P=$D/launch-preparation-v12-$n
  export V12_PACKAGE=$P WEBPRM_NATIVE_SOURCE_ROOT=$P/source
  L=$T/tests-v12-$n.log
  suites="test_webprm_concurrent_broker test_webprm_campaign_queue test_webprm_cms_site test_webprm_cms_native test_webprm_cms_hosting test_audit_webprm_table5_cms test_run_webprm_table5_cms test_run_webprm_table5_shopping"
  case $n in gitlab*) suites="$suites test_validate_webprm_gitlab_hosting test_webprm_gitlab_native test_webprm_gitlab_recovery";; esac
  { echo "=== $n v12 cell (venv; launcher-like PYTHONPATH)"; cd /tmp && PYTHONPATH=$NL:$P/source $VPY -B $T/test_v12.py 2>&1 | grep -vE "^\s+File|^    " | tail -40
    echo "=== $n v12 launcher"; cd /tmp && PYTHONPATH=$P/source $PY -B $T/test_v12_launcher.py 2>&1 | tail -12
    echo "=== $n v12 server startup (r1 proxy regression)"; cd /tmp && PYTHONPATH=$P/source $PY -B $T/test_v12_server_startup.py 2>&1 | grep -E "^test|^Ran|^OK|FAILED|Error|passed|negative" | tail -8
    echo "=== $n package unit suites"; cd $P/source && for t in $suites; do echo "--- $t"; PYTHONPATH=$P/source $PY -B -m unittest tests.$t 2>&1 | tail -3; done
  } > $L 2>&1
  echo "$n: $(grep -c '^OK' $L) OK, $(grep -c 'FAILED\|Error' $L) fail-lines"
done
