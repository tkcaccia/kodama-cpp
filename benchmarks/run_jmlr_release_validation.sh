#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

set -euo pipefail

repo_root="${KODAMA_CPP_ROOT:-/mnt/sata_ssd/kodama-cpp}"
build_dir="${KODAMA_CPP_BUILD_DIR:-$repo_root/build-cuda}"
out_dir="${KODAMA_RELEASE_OUT:-/mnt/sata_ssd/kodama-cpp-benchmarks/jmlr-release-validation}"
env_dir="${ENV_DIR:-/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs}"
current_lib="$out_dir/R-current"
historical_lib="$out_dir/R-historical"
historical_tar="$out_dir/KODAMA_2.4.1.tar.gz"

export CONDA_PREFIX="$env_dir"
export LD_LIBRARY_PATH="$env_dir/lib:$env_dir/targets/x86_64-linux/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export KODAMA_CPP_ROOT="$repo_root"
export KODAMA_CPP_BUILD_DIR="$build_dir"
export KODAMA_RELEASE_OUT="$out_dir"
export KODAMA_CURRENT_R_LIB="$current_lib"
export KODAMA_HISTORICAL_R_LIB="$historical_lib"

mkdir -p "$out_dir" "$current_lib" "$historical_lib"

cmake --build "$build_dir" -j
ctest --test-dir "$build_dir" --output-on-failure

R CMD INSTALL --library="$current_lib" "$repo_root/split-repos/kodama-r"

if [[ ! -f "$historical_tar" ]]; then
  curl -fsSL \
    https://cran.r-project.org/src/contrib/Archive/KODAMA/KODAMA_2.4.1.tar.gz \
    -o "$historical_tar"
fi
if [[ ! -d "$historical_lib/KODAMA" ]]; then
  R CMD INSTALL --library="$historical_lib" "$historical_tar"
fi

Rscript "$repo_root/benchmarks/run_jmlr_release_validation.R"

if [[ "${KODAMA_RUN_SENSITIVITY:-1}" == "1" ]]; then
  export KODAMA_SENS_OUT="${KODAMA_SENS_OUT:-$out_dir/m-tcycle-sensitivity}"
  Rscript "$repo_root/benchmarks/run_jmlr_m_tcycle_sensitivity.R"
  if [[ "${KODAMA_RUN_ENSEMBLE_CONVERGENCE:-1}" == "1" ]]; then
    export KODAMA_CONVERGENCE_RESULT_DIR="${KODAMA_CONVERGENCE_RESULT_DIR:-$KODAMA_SENS_OUT/runs}"
    export KODAMA_CONVERGENCE_OUT="${KODAMA_CONVERGENCE_OUT:-$KODAMA_SENS_OUT}"
    Rscript "$repo_root/benchmarks/run_jmlr_ensemble_convergence.R"
  fi
fi

if [[ "${KODAMA_RUN_VISUALIZATION:-1}" == "1" ]]; then
  export KODAMA_VIS_BENCH_OUT="${KODAMA_VIS_BENCH_OUT:-$out_dir/visualization}"
  export KODAMA_VIS_DATASETS="${KODAMA_VIS_DATASETS:-MetRef,USPS,MNIST}"
  export KODAMA_VIS_CLASSIFIERS="${KODAMA_VIS_CLASSIFIERS:-knn,pls_lda}"
  export KODAMA_VIS_METHODS="${KODAMA_VIS_METHODS:-UMAP,opentsne}"
  export KODAMA_VIS_BACKEND="${KODAMA_VIS_BACKEND:-cuda}"
  export KODAMA_VIS_M="${KODAMA_VIS_M:-100}"
  export KODAMA_VIS_TCYCLE="${KODAMA_VIS_TCYCLE:-100}"
  export KODAMA_VIS_LANDMARKS="${KODAMA_VIS_LANDMARKS:-100000}"
  export KODAMA_VIS_GRAPH_NEIGHBORS="${KODAMA_VIS_GRAPH_NEIGHBORS:-100}"
  export KODAMA_VIS_KNN_K="${KODAMA_VIS_KNN_K:-30}"
  export KODAMA_VIS_EMBED_K="${KODAMA_VIS_EMBED_K:-30}"
  export KODAMA_VIS_PERPLEXITY="${KODAMA_VIS_PERPLEXITY:-30}"
  Rscript "$repo_root/tools/run_classic_vs_kodama_visualization_benchmark.R"
fi
