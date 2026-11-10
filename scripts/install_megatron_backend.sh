#!/usr/bin/env bash
set -euo pipefail
ROOT="${OPENWEBRL_RUNTIME_ROOT:-/gpfs/scrubbed/zixianma/openwebrl-runtime}"
export UV_CACHE_DIR="$ROOT/uv-cache" TMPDIR="$ROOT/tmp" MAX_JOBS=8 OMP_NUM_THREADS=4
UV="$ROOT/venv/bin/uv"
PYTHON="$ROOT/venv/bin/python"
mkdir -p "$ROOT/src"
if [[ ! -d "$ROOT/src/Megatron-LM/.git" ]]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git "$ROOT/src/Megatron-LM"
    git -C "$ROOT/src/Megatron-LM" checkout 3714d81d418c9f1bca4594fc35f9e8289f652862
fi
if [[ "$(git -C "$ROOT/src/Megatron-LM" rev-parse HEAD)" != 3714d81d418c9f1bca4594fc35f9e8289f652862 ]]; then
    echo 'Megatron source does not match the tested commit.' >&2; exit 1
fi
"$UV" pip install --python "$PYTHON" --no-deps -e "$ROOT/src/Megatron-LM"
"$UV" pip install --python "$PYTHON" 'nvidia-cuda-nvcc-cu12==12.9.86' 'nvidia-cuda-cccl-cu12' 'nvidia-cuda-runtime-cu12==12.9.79' 'pybind11' 'cmake' 'ninja' 'psutil' 'nvidia-ml-py' 'nvidia-modelopt[torch]>=0.37' 'einops' 'dill' 'tensorstore' 'flask-restful' 'nvidia-resiliency-ext' 'nvtx' 'zarr<3'
"$UV" pip install --python "$PYTHON" --no-deps 'git+https://github.com/fzyzcjy/Megatron-Bridge.git@35b4ebfc486fb15dcc0273ceea804c3606be948a'
"$PYTHON" -m pip freeze > "$ROOT/logs/environment.freeze.txt"
