#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

set -euo pipefail

ROOT="${KODAMA_CPP_ROOT:-/mnt/sata_ssd/kodama-cpp}"
BUILD="${KODAMA_CPP_BUILD_DIR:-/mnt/sata_ssd/kodama-cpp/build-cuda}"
ENV_DIR="${ENV_DIR:-/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs}"

export CONDA_PREFIX="$ENV_DIR"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export KODAMA_CPP_ROOT="$ROOT"
export KODAMA_CPP_BUILD_DIR="$BUILD"
export KODAMA_RCPP_REBUILD=TRUE

cd "$ROOT"
nvidia-smi
cmake --build "$BUILD" -j
ctest --test-dir "$BUILD" --output-on-failure
Rscript "$ROOT/tools/test_cpu_cuda_parity.R"
