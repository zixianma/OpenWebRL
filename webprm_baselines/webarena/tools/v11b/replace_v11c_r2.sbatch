#!/usr/bin/env bash
#SBATCH --job-name=wa-v11c-prepare
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/replace-v11c-%j.log
# CPU-only (no GPU) wrapper so the package copy/build runs off the login node. Never submits GPU work.
# Usage: sbatch replace_v11c.sbatch NEW_NAME SITE SECONDS API_USD OLD_JOB_ID OLD_PACKAGE [EARLIER_PACKAGES...]
set -euo pipefail
/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/replace_v11c_r2.sh "$@"
