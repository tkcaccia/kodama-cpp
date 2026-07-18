#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

set -euo pipefail

ROOT="${KODAMA_CPP_ROOT:-/mnt/sata_ssd/kodama-cpp}"
BUILD="${KODAMA_CPP_BUILD_DIR:-$ROOT/build-cuda}"
ENV_DIR="${ENV_DIR:-/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs}"
OUT="${KODAMA_WRAPPER_BENCH_OUT:-/mnt/sata_ssd/KODAMAopt/wrapper-benchmarks}"

export CONDA_PREFIX="$ENV_DIR"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
PRELOAD_LIBS=()
for lib in libgomp.so libopenblasp-r0.3.33.so libstdc++.so.6; do
  if [ -f "$ENV_DIR/lib/$lib" ]; then
    PRELOAD_LIBS+=("$ENV_DIR/lib/$lib")
  fi
done
if [ "${#PRELOAD_LIBS[@]}" -gt 0 ]; then
  PRELOAD_VALUE="$(IFS=:; echo "${PRELOAD_LIBS[*]}")"
  export LD_PRELOAD="$PRELOAD_VALUE${LD_PRELOAD:+:$LD_PRELOAD}"
fi

mkdir -p "$OUT"

if [ "${KODAMA_WRAPPER_SKIP_CORE:-0}" != "1" ]; then
  nvidia-smi
  cmake --build "$BUILD" -j
  ctest --test-dir "$BUILD" --output-on-failure
fi

R_LIBS_USER="$OUT/Rlib"
mkdir -p "$R_LIBS_USER"
export R_LIBS_USER
export KODAMA_CPP_ROOT="$ROOT"
export KODAMA_CPP_BUILD_DIR="$BUILD"
CUDA_LIB_DIRS=(
  "$ENV_DIR/lib"
  "$ENV_DIR/targets/x86_64-linux/lib"
  "/usr/local/cuda-13.0/targets/x86_64-linux/lib"
  "/usr/local/cuda/lib64"
)
CUDA_LINK_FLAGS=()
for lib in cuvs cublas cusolver curand cufft cudart cugraph_c; do
  if find "${CUDA_LIB_DIRS[@]}" -maxdepth 1 -name "lib${lib}.so*" -print -quit 2>/dev/null | grep -q .; then
    CUDA_LINK_FLAGS+=("-l${lib}")
  fi
done
if find "$ENV_DIR/lib" -maxdepth 1 -name "libfaiss_gpu.so*" -print -quit | grep -q .; then
  CUDA_LINK_FLAGS=("-lfaiss_gpu" "${CUDA_LINK_FLAGS[@]}")
fi
export KODAMA_R_CUDA_LIBS="${CUDA_LINK_FLAGS[*]}"

if [ "${KODAMA_WRAPPER_SKIP_R:-0}" != "1" ]; then
  R CMD INSTALL --preclean "$ROOT/split-repos/kodama-r"
  KODAMA_R_BENCH_OUT="$OUT/kodama-r" \
  KODAMA_BENCH_BACKENDS="${KODAMA_BENCH_BACKENDS:-cpu,cuda}" \
  KODAMA_BENCH_M="${KODAMA_BENCH_M:-20}" \
  KODAMA_BENCH_TCYCLE="${KODAMA_BENCH_TCYCLE:-20}" \
  KODAMA_BENCH_LANDMARKS="${KODAMA_BENCH_LANDMARKS:-100000}" \
  Rscript "$ROOT/split-repos/kodama-r/inst/benchmarks/run_br8100_merfish.R"
fi

if [ "${KODAMA_WRAPPER_SKIP_PY:-0}" != "1" ]; then
  PYTHON_BIN="${KODAMA_PYTHON_BIN:-$ENV_DIR/bin/python}"
  if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
  fi
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install pybind11 scikit-build-core pytest numpy pandas

  PY_EXTRA_ITEMS=("$ENV_DIR/lib/libfaiss.so")
  for lib in libfaiss_gpu.so libcuvs.so libcugraph_c.so; do
    if [ -f "$ENV_DIR/lib/$lib" ]; then
      PY_EXTRA_ITEMS+=("$ENV_DIR/lib/$lib")
    fi
  done
  for lib in libcublas.so libcusolver.so libcurand.so libcufft.so libcudart.so; do
    for dir in "${CUDA_LIB_DIRS[@]}"; do
      if [ -f "$dir/$lib" ]; then
        PY_EXTRA_ITEMS+=("$dir/$lib")
        break
      fi
    done
  done
  PY_EXTRA_LIBS="$(IFS=';'; echo "${PY_EXTRA_ITEMS[*]}")"
  "$PYTHON_BIN" -m pip install -v "$ROOT/split-repos/kodama-python" \
    --config-settings=cmake.define.KODAMA_CPP_ROOT="$ROOT" \
    --config-settings=cmake.define.KODAMA_CPP_BUILD_DIR="$BUILD" \
    --config-settings=cmake.define.KODAMA_CPP_EXTRA_LIBS="$PY_EXTRA_LIBS"

  KODAMA_PY_EXPORT_DIR="$OUT/exported-spatial" \
  Rscript "$ROOT/split-repos/kodama-python/scripts/export_spatial_rdata.R"

  "$PYTHON_BIN" "$ROOT/split-repos/kodama-python/benchmarks/run_br8100_merfish.py" \
    --input-dir "$OUT/exported-spatial" \
    --out "$OUT/kodama-python-br8100-merfish.csv" \
    --M "${KODAMA_BENCH_M:-20}" \
    --Tcycle "${KODAMA_BENCH_TCYCLE:-20}" \
    --landmarks "${KODAMA_BENCH_LANDMARKS:-100000}" \
    --backends "${KODAMA_BENCH_BACKENDS:-cpu,cuda}"
fi

echo "Wrapper benchmark outputs written to $OUT"
