#!/usr/bin/env bash
#SBATCH --job-name=wa-v11b-reaudit
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/reaudit-v9-%j.log
# CPU-only (no GPU). Usage: PKG=<package dir> sbatch reaudit_v9.sbatch   (only for packages whose job has finished)
set -u
export TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
# The package's evaluator runtime (proposal nltk_environment), exactly as the launcher gives its auditor.
eval $(python3 -c "import json,shlex;e=json.load(open('$PKG/proposal.json'))['nltk_environment'];print(' '.join('export %s=%s'%(k,shlex.quote(v)) for k,v in e.items()))")
WEBPRM_NATIVE_SOURCE_ROOT=$PKG/source $PY -B /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/reaudit_v9.py $PKG
