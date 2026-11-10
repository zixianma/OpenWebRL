#!/usr/bin/env bash
#SBATCH --job-name=openwebrl-smoke
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
case "${OPENWEBRL_RUN_MODE:-fixture}" in
    fixture) exec bash scripts/run_h200_browser_fixture_smoke.sh ;;
    text) exec bash scripts/run_h200_pipeline_smoke.sh ;;
    replay) exec bash scripts/run_h200_train_smoke.sh ;;
    web) exec bash scripts/run_h200_browser.sh ;;
    *) echo 'OPENWEBRL_RUN_MODE must be fixture, text, replay, or web.' >&2; exit 1 ;;
esac
