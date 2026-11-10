#!/usr/bin/env bash
set -euo pipefail
ROOT="${OPENWEBRL_RUNTIME_ROOT:-/gpfs/scrubbed/zixianma/openwebrl-runtime}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/venv/bin:$ROOT/cuda/bin:$PATH"
export CUDA_HOME="$ROOT/cuda" OMP_NUM_THREADS=4
export CPATH="$ROOT/src/python-headers/Include:$ROOT/src/python-headers"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/torchinductor-cache"
export TRITON_CACHE_DIR="$ROOT/triton-cache"
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}"
exec python "$REPO/scripts/gpu_model_smoke.py" --checkpoint /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT --output "$ROOT/runs/hf-smoke-${SLURM_JOB_ID:-manual}"
