#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${KODAMA_BUILD_DIR:-$ROOT_DIR/build}"
ENABLE_CUDA="${KODAMA_ENABLE_CUDA:-OFF}"
ENV_DIR="${ENV_DIR:-}"
TMP_DIR="${TMPDIR:-/tmp}/kodama_float32_smoke"
SKIP_BUILD="${KODAMA_SMOKE_SKIP_BUILD:-0}"

if [[ "$ENABLE_CUDA" == "ON" && -n "$ENV_DIR" ]]; then
  export CONDA_PREFIX="$ENV_DIR"
  export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  cmake_args=(
    -S "$ROOT_DIR"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DKODAMA_ENABLE_CUDA="$ENABLE_CUDA"
  )

  if [[ "$ENABLE_CUDA" == "ON" && -n "$ENV_DIR" ]]; then
    cmake_args+=(
      -DCMAKE_PREFIX_PATH="$ENV_DIR"
      -DCMAKE_CXX_COMPILER="$ENV_DIR/bin/x86_64-conda-linux-gnu-c++"
    )
  fi

  cmake "${cmake_args[@]}"
  cmake --build "$BUILD_DIR" -j
  ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

mkdir -p "$TMP_DIR"
python3 - "$TMP_DIR/x_float32.bin" "$TMP_DIR/y_int32.bin" <<'PY'
import random
import struct
import sys

x_path, y_path = sys.argv[1], sys.argv[2]
random.seed(44)
n = 60
p = 6
x = []
y = []
for c in range(3):
    for _ in range(n // 3):
        y.append(10 + c)
        for j in range(p):
            signal = 2.5 if (j == c or j == c + 3) else -0.5
            x.append(signal + random.gauss(0.0, 0.25))

with open(x_path, "wb") as f:
    f.write(struct.pack("<" + "f" * len(x), *x))
with open(y_path, "wb") as f:
    f.write(struct.pack("<" + "i" * len(y), *y))
PY

cv_methods="KNNCV_CPU,PLSDACV_CPU,PLSLDACV_CPU"
core_methods="core_pls_lda_cpu,core_knn_cpu"
if [[ "$ENABLE_CUDA" == "ON" ]]; then
  cv_methods="KNNCV_CPU,KNNCV_CUDA,PLSDACV_CPU,PLSLDACV_CPU,PLSDACV_CUDA,PLSLDACV_CUDA"
  core_methods="core_pls_lda_cpu,core_knn_cpu,core_pls_lda_cuda,core_knn_cuda"
fi

KODAMA_INPUT_DTYPE=float32 \
KODAMA_BENCH_METHODS="$cv_methods" \
  "$BUILD_DIR/kodama_cv_sanity_bin" toy "$TMP_DIR/x_float32.bin" "$TMP_DIR/y_int32.bin" \
  60 6 "$TMP_DIR/cv.csv" 1 3 0 0

KODAMA_INPUT_DTYPE=float32 \
KODAMA_CORE_METHODS="$core_methods" \
KODAMA_CORE_INIT=kmeans \
KODAMA_CORE_K=3 \
KODAMA_CORE_KNN_K=5 \
  "$BUILD_DIR/kodama_core_sanity_bin" toy "$TMP_DIR/x_float32.bin" "$TMP_DIR/y_int32.bin" 60 6 2

echo "float32 smoke output: $TMP_DIR"
