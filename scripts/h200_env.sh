#!/usr/bin/env bash
# Source this file to use the runtime built for this repository.
OPENWEBRL_RUNTIME_ROOT="${OPENWEBRL_RUNTIME_ROOT:-/gpfs/scrubbed/zixianma/openwebrl-runtime}"
OPENWEBRL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
# SGLang installs a uvloop policy. A stdlib loop under that policy cannot
# spawn browser subprocesses on Python 3.12 (get_child_watcher is unsupported).
export SLIME_ASYNC_USE_STDLIB_LOOP=0
