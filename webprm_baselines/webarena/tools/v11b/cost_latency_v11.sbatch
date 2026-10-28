#!/usr/bin/env bash
#SBATCH --job-name=wa-v11-costlat
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/cost-latency-v11-%j.log
cd /tmp && python3 -B /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/cost_latency_v11.py /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v11b-tools/cost-latency-v11.json $(ls -d /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/launch-preparation-v11b-* /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/launch-preparation-v11c-*)
