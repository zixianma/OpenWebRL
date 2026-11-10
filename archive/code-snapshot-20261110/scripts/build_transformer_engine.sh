#!/usr/bin/env bash
set -euo pipefail
ROOT="${OPENWEBRL_RUNTIME_ROOT:-/gpfs/scrubbed/zixianma/openwebrl-runtime}"
export UV_CACHE_DIR="$ROOT/uv-cache" TMPDIR="$ROOT/tmp" MAX_JOBS=8 OMP_NUM_THREADS=4
export CUDA_HOME="$ROOT/cuda" NVTE_CUDA_ARCHS=90 NVTE_FRAMEWORK=pytorch NVTE_BUILD_THREADS_PER_JOB=2
mkdir -p "$CUDA_HOME"
mkdir -p "$ROOT/downloads"
if [[ ! -f "$ROOT/downloads/cuda-nvcc.tar.xz" ]]; then
    curl -fL https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz -o "$ROOT/downloads/cuda-nvcc.tar.xz"
fi
tar -xf "$ROOT/downloads/cuda-nvcc.tar.xz" -C "$CUDA_HOME" --strip-components=1
SITE="$ROOT/venv/lib/python3.12/site-packages"
cp -rsn "$SITE/nvidia/cuda_runtime/include/." "$CUDA_HOME/include/"
cp -rsn "$SITE/nvidia/cuda_cccl/include/." "$CUDA_HOME/include/"
mkdir -p "$CUDA_HOME/lib64"
cp -rsn "$SITE/nvidia/cuda_runtime/lib/." "$CUDA_HOME/lib64/"
export PATH="$ROOT/venv/bin:$CUDA_HOME/bin:$PATH"
export CUDNN_PATH="$SITE/nvidia/cudnn"
uv pip install --python "$ROOT/venv/bin/python" 'sglang-router==0.3.2' 'setuptools<80' 'packaging' 'ninja' 'pybind11'
bash "$(dirname "${BASH_SOURCE[0]}")/finish_transformer_engine.sh"
