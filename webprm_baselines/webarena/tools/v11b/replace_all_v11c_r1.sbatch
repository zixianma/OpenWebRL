#!/usr/bin/env bash
#SBATCH --job-name=wa-v11c-prepare-all
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/replace-all-v11c-%j.log
# CPU-only. Serially prepares r1 replacements for the 7 remaining v11b packages (serial: shared ledger). Never submits GPU work.
set -uo pipefail
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/replace_v11c.sh; P=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/launch-preparation-v11b
$R shop-a-r1 shopping 7200 15 392811 $P-shop-a
$R shop-b-r1 shopping 7200 15 392812 $P-shop-b
$R shop-c-r1 shopping 7200 15 392813 $P-shop-c
$R cms-a-r1 shopping_admin 7200 15 392814 $P-cms-a
$R cms-b-r1 shopping_admin 6000 12 392815 $P-cms-b
$R gitlab-a-r1 gitlab 3600 10 392817 $P-gitlab-a
$R gitlab-b-r1 gitlab 2700 5 392818 $P-gitlab-b
echo ALL_DONE
