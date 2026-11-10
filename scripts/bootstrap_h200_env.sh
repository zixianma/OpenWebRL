#!/usr/bin/env bash
# Dedicated user-owned environment; no shared Python environments or containers.
set -euo pipefail
RUNTIME_ROOT="${OPENWEBRL_RUNTIME_ROOT:-/gpfs/scrubbed/zixianma/openwebrl-runtime}"
mkdir -p "$RUNTIME_ROOT/logs" "$RUNTIME_ROOT/uv-cache" "$RUNTIME_ROOT/tmp"
export UV_CACHE_DIR="$RUNTIME_ROOT/uv-cache"
export TMPDIR="$RUNTIME_ROOT/tmp"
export UV_CONCURRENT_DOWNLOADS=8
export UV_CONCURRENT_BUILDS=2
export MAX_JOBS=8
export OMP_NUM_THREADS=4
if [[ ! -x "$RUNTIME_ROOT/venv/bin/python" ]]; then
    python3.12 -m venv "$RUNTIME_ROOT/venv"
fi
PYTHON="$RUNTIME_ROOT/venv/bin/python"
"$PYTHON" -m pip install --disable-pip-version-check uv
UV="$RUNTIME_ROOT/venv/bin/uv"
"$UV" pip install --python "$PYTHON" 'torch==2.9.1' 'torchvision==0.24.1' 'torchaudio==2.9.1' --index-url https://download.pytorch.org/whl/cu129
"$UV" pip install --python "$PYTHON" 'sglang==0.5.6.post2' 'ray[default]' 'playwright==1.58.0' 'accelerate' 'loguru' 'omegaconf' 'qwen-vl-utils' 'tensorboard' 'wandb' 'httpx[http2]' 'pylatexenc' 'timeout-decorator' 'mcp[cli]' 'numpy<2' 'setuptools<81' wheel
# SGLang's runtime correctness check requires cuDNN >= 9.15. This intentionally
# supersedes torch 2.9.1's older dependency pin; the GPU smoke verifies this pair.
"$UV" pip install --python "$PYTHON" --no-deps 'nvidia-cudnn-cu12==9.16.0.29'
"$PYTHON" -m pip freeze > "$RUNTIME_ROOT/logs/environment.freeze.txt"
echo "Core environment installed: $RUNTIME_ROOT/venv"
