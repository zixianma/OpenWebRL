#!/usr/bin/env bash
set -euo pipefail
ROOT="${OPENWEBRL_RUNTIME_ROOT:-/gpfs/scrubbed/zixianma/openwebrl-runtime}"
export UV_CACHE_DIR="$ROOT/uv-cache" TMPDIR="$ROOT/tmp" MAX_JOBS=8 OMP_NUM_THREADS=4
export CUDA_HOME="$ROOT/cuda" NVTE_CUDA_ARCHS=90 NVTE_FRAMEWORK=pytorch NVTE_BUILD_THREADS_PER_JOB=2
SITE="$ROOT/venv/lib/python3.12/site-packages"
export PATH="$ROOT/venv/bin:$CUDA_HOME/bin:$PATH"
export CUDNN_PATH="$SITE/nvidia/cudnn"
mkdir -p "$ROOT/src/python-headers"
mkdir -p "$ROOT/downloads"
PYTHON_HEADER_VERSION="$($ROOT/venv/bin/python -c 'import platform; print(platform.python_version())')"
if [[ ! -f "$ROOT/downloads/Python-$PYTHON_HEADER_VERSION.tar.xz" ]]; then
    curl -fL "https://www.python.org/ftp/python/$PYTHON_HEADER_VERSION/Python-$PYTHON_HEADER_VERSION.tar.xz" -o "$ROOT/downloads/Python-$PYTHON_HEADER_VERSION.tar.xz"
fi
tar -xf "$ROOT/downloads/Python-$PYTHON_HEADER_VERSION.tar.xz" -C "$ROOT/src/python-headers" --strip-components=1
(cd "$ROOT/src/python-headers" && ./configure --without-ensurepip > "$ROOT/logs/python-headers-configure.log" 2>&1)
export CPATH="$ROOT/src/python-headers/Include:$ROOT/src/python-headers"
for directory in "$SITE"/nvidia/*/include; do
    CPATH="$CPATH:$directory"
done
export CPATH
uv pip install --python "$ROOT/venv/bin/python" --no-build-isolation 'transformer_engine[pytorch]==2.10.0' 'hydra-core==1.3.2'
uv pip install --python "$ROOT/venv/bin/python" --no-deps 'nvidia-cudnn-cu12==9.16.0.29'
"$ROOT/venv/bin/python" -m pip freeze > "$ROOT/logs/environment.freeze.txt"
