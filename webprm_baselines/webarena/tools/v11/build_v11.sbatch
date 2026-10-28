#!/usr/bin/env bash
#SBATCH --job-name=wa-v11-build
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11-tools/build-v11-%j.log
# CPU-only (no GPU). Runs the v11 builder for every shard (stage 1, or stage 2 once its integration review exists).
set -u
export OMP_NUM_THREADS=1
T=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11-tools
source $T/shards_v11.sh
for s in "${V11_SHARDS[@]}"; do set -- $s; echo "=== $1"; $T/build_one_v11.sh "$@" 2>&1 | tail -14; done
