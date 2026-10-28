#!/usr/bin/env bash
#SBATCH --job-name=wa-v11b-build
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/build-v11b-%j.log
# CPU-only (no GPU). Runs the v11b builder for every shard (stage 1, or stage 2 once its integration review exists).
set -u
export OMP_NUM_THREADS=1
T=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools
source $T/shards_v11b.sh
for s in "${V11B_SHARDS[@]}"; do set -- $s; echo "=== $1"; $T/build_one_v11b.sh "$@" 2>&1 | tail -14; done
