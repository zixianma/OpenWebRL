#!/usr/bin/env bash
# Source this file to use the runtime built for this repository.
OPENWEBRL_RUNTIME_ROOT="${OPENWEBRL_RUNTIME_ROOT:-/gpfs/scrubbed/zixianma/openwebrl-runtime}"
OPENWEBRL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep generated caches and temporary files off the small, snapshotted project
# fileset.  These locations are reproducible and may be purged independently of
# repository code and durable experiment outputs.
OPENWEBRL_CACHE_ROOT="${OPENWEBRL_CACHE_ROOT:-$OPENWEBRL_RUNTIME_ROOT/cache}"
OPENWEBRL_JOB_TMP="${SLURM_TMPDIR:-/tmp/${USER:-unknown}/${SLURM_JOB_ID:-interactive}}"
export HF_HOME="${HF_HOME:-$OPENWEBRL_CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$OPENWEBRL_CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$OPENWEBRL_CACHE_ROOT/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$OPENWEBRL_CACHE_ROOT/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$OPENWEBRL_CACHE_ROOT/uv}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$OPENWEBRL_CACHE_ROOT/wandb}"
export TMPDIR="${TMPDIR:-$OPENWEBRL_JOB_TMP}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$OPENWEBRL_JOB_TMP/pycache}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
    "$WANDB_CACHE_DIR" "$TMPDIR" "$PYTHONPYCACHEPREFIX"

# Warn before work starts when the shared project fileset is nearly full.  Set
# OPENWEBRL_ENFORCE_PROJECT_HEADROOM=1 in a submission preflight to fail closed.
OPENWEBRL_PROJECT_MIN_FREE_BYTES="${OPENWEBRL_PROJECT_MIN_FREE_BYTES:-53687091200}"
OPENWEBRL_PROJECT_AVAILABLE_BYTES="$(df -PB1 /gpfs/projects/krishna 2>/dev/null | awk 'NR==2 {print $4}')"
export OPENWEBRL_PROJECT_AVAILABLE_BYTES
if [[ "$OPENWEBRL_PROJECT_AVAILABLE_BYTES" =~ ^[0-9]+$ ]] && \
   (( OPENWEBRL_PROJECT_AVAILABLE_BYTES < OPENWEBRL_PROJECT_MIN_FREE_BYTES )); then
    printf 'WARNING: /gpfs/projects/krishna has only %.1f GiB free; caches and temp files are redirected off-project.\n' \
        "$((OPENWEBRL_PROJECT_AVAILABLE_BYTES / 1024 / 1024))e-3" >&2
    if [[ "${OPENWEBRL_ENFORCE_PROJECT_HEADROOM:-0}" == 1 ]]; then
        return 75 2>/dev/null || exit 75
    fi
fi
export PATH="$OPENWEBRL_RUNTIME_ROOT/venv/bin:$OPENWEBRL_RUNTIME_ROOT/cuda/bin:$PATH"
export CUDA_HOME="$OPENWEBRL_RUNTIME_ROOT/cuda"
export PYTHONPATH="$OPENWEBRL_REPO:$OPENWEBRL_RUNTIME_ROOT/src/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$OPENWEBRL_RUNTIME_ROOT/src/python-headers/Include:$OPENWEBRL_RUNTIME_ROOT/src/python-headers"
for owrl_include in "$OPENWEBRL_RUNTIME_ROOT"/venv/lib/python3.12/site-packages/nvidia/*/include; do
    export CPATH="$CPATH:$owrl_include"
done
export LD_LIBRARY_PATH="$OPENWEBRL_RUNTIME_ROOT/venv/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for owrl_library in "$OPENWEBRL_RUNTIME_ROOT"/venv/lib/python3.12/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="$owrl_library:$LD_LIBRARY_PATH"
done
export TORCHINDUCTOR_CACHE_DIR="$OPENWEBRL_RUNTIME_ROOT/torchinductor-cache"
export TRITON_CACHE_DIR="$OPENWEBRL_RUNTIME_ROOT/triton-cache"
export PLAYWRIGHT_BROWSERS_PATH="$OPENWEBRL_RUNTIME_ROOT/browsers"
export OMP_NUM_THREADS=4 MAX_JOBS=8 CUDA_DEVICE_MAX_CONNECTIONS=1
export SLIME_HF_VISION_ATTN_IMPL=sdpa
# Use the stdlib loop and its compatible child-watcher policy for browser
# subprocesses; uvloop cancellation has caused native rollout-worker aborts.
export SLIME_ASYNC_USE_STDLIB_LOOP=1
