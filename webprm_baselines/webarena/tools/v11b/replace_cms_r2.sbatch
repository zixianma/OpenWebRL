#!/usr/bin/env bash
#SBATCH --job-name=wa-v11c-prepare-cms-r2
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/replace-cms-r2-%j.log
# CPU-only; serial (shared ledger). Never submits GPU work.
set -uo pipefail
P=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/launch-preparation
/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/replace_v11c_r2.sh cms-a-r2 shopping_admin 5400 15 393554 $P-v11b-cms-a $P-v11c-cms-a-r1
/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/replace_v11c_r2.sh cms-b-r2 shopping_admin 3000 8 393555 $P-v11b-cms-b $P-v11c-cms-b-r1
echo ALL_DONE
