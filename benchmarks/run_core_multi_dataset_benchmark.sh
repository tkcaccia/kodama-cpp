#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 OUT_DIR [CYCLES]" >&2
  exit 2
fi

OUT_DIR=$1
CYCLES=${2:-20}

REPO_DIR=${KODAMA_REPO_DIR:-/mnt/sata_ssd/kodama-cpp}
DATA_ROOT=${KODAMA_DATA_ROOT:-/mnt/sata_ssd/fastEmbedR/Data}
BUILD_DIR=${KODAMA_BUILD_DIR:-/mnt/sata_ssd/kodama-cpp/build-cuda}
BIN=${KODAMA_CORE_BIN:-$BUILD_DIR/kodama_core_sanity_bin}
EXPORTER=${KODAMA_EXPORTER:-$REPO_DIR/benchmarks/export_rdata_dataset_to_bin.R}

SEEDS=${KODAMA_CORE_SEEDS:-"11 23 37 41 53 67 79 83 97 123"}
DATASETS=${KODAMA_CORE_DATASETS:-"USPS TabulaMuris MNIST MetRef COIL20"}
METHODS=${KODAMA_CORE_METHODS:-core_knn_cuda}
INIT_K=${KODAMA_CORE_INIT_K:-100}
KNN_K=${KODAMA_CORE_KNN_K:-10}
AUTO_COARSEN=${KODAMA_CORE_AUTO_COARSEN:-1}
INPUT_DTYPE=${KODAMA_INPUT_DTYPE:-float32}
MIN_SAMPLES=${KODAMA_MIN_SAMPLES:-500}

mkdir -p "$OUT_DIR"/{bin,results,logs,labels}

summary="$OUT_DIR/results/core_multi_dataset_results.csv"
manifest="$OUT_DIR/results/dataset_manifest.csv"
metadata="$OUT_DIR/results/run_metadata.txt"

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_dir=$REPO_DIR"
  echo "build_dir=$BUILD_DIR"
  echo "bin=$BIN"
  echo "datasets=$DATASETS"
  echo "methods=$METHODS"
  echo "seeds=$SEEDS"
  echo "cycles=$CYCLES"
  echo "init=kmeans"
  echo "init_k=$INIT_K"
  echo "knn_k=$KNN_K"
  echo "auto_coarsen=$AUTO_COARSEN"
  echo "input_dtype=$INPUT_DTYPE"
  echo "min_samples=$MIN_SAMPLES"
} > "$metadata"

echo "dataset,input_path,n,p,n_labels,x_file,x_float32_file,labels_file" > "$manifest"
header_written=0

for dataset in $DATASETS; do
  input="$DATA_ROOT/$dataset/$dataset.RData"
  if [[ ! -f "$input" ]]; then
    echo "Missing dataset: $input" >&2
    exit 1
  fi

  bin_dir="$OUT_DIR/bin/$dataset"
  mkdir -p "$bin_dir"
  meta_file="$bin_dir/${dataset}_metadata.csv"
  if [[ ! -f "$meta_file" ]]; then
    Rscript "$EXPORTER" "$dataset" "$input" "$bin_dir" > "$OUT_DIR/logs/${dataset}_export.log" 2>&1
  fi

  tail -n +2 "$meta_file" >> "$manifest"
  n=$(awk -F, 'NR==2 {gsub(/"/, "", $3); print $3}' "$meta_file")
  p=$(awk -F, 'NR==2 {gsub(/"/, "", $4); print $4}' "$meta_file")
  if (( n < MIN_SAMPLES )); then
    echo "Skipping $dataset: n=$n is below KODAMA_MIN_SAMPLES=$MIN_SAMPLES" | tee "$OUT_DIR/logs/${dataset}_skip.log"
    continue
  fi
  if [[ "$INPUT_DTYPE" == "float32" || "$INPUT_DTYPE" == "float" || "$INPUT_DTYPE" == "single" ]]; then
    x_file="$bin_dir/${dataset}_x_float32_rowmajor.bin"
  elif [[ "$INPUT_DTYPE" == "float64" || "$INPUT_DTYPE" == "double" ]]; then
    x_file="$bin_dir/${dataset}_x_double_rowmajor.bin"
  else
    echo "KODAMA_INPUT_DTYPE must be float32 or float64, got: $INPUT_DTYPE" >&2
    exit 2
  fi
  y_file="$bin_dir/${dataset}_labels_int32.bin"

  for seed in $SEEDS; do
    run_prefix="$OUT_DIR/labels/${dataset}_seed_${seed}"
    run_csv="$OUT_DIR/results/${dataset}_seed_${seed}.csv"
    log_file="$OUT_DIR/logs/${dataset}_seed_${seed}.log"
    err_file="$OUT_DIR/logs/${dataset}_seed_${seed}.err"

    KODAMA_CORE_INIT=kmeans \
    KODAMA_CORE_K="$INIT_K" \
    KODAMA_CORE_KNN_K="$KNN_K" \
    KODAMA_CORE_METHODS="$METHODS" \
    KODAMA_CORE_AUTO_COARSEN="$AUTO_COARSEN" \
    KODAMA_CORE_SEED="$seed" \
    KODAMA_CORE_WRITE_PREFIX="$run_prefix" \
    KODAMA_INPUT_DTYPE="$INPUT_DTYPE" \
      "$BIN" "$dataset" "$x_file" "$y_file" "$n" "$p" "$CYCLES" \
      > "$run_csv" 2> "$err_file"

    cat "$run_csv" >> "$log_file"

    if [[ $header_written -eq 0 ]]; then
      header=$(head -n 1 "$run_csv")
      printf 'dataset_config,seed,knn_k,cycles,%s\n' "$header" > "$summary"
      header_written=1
    fi

    tail -n +2 "$run_csv" | while IFS= read -r row; do
      printf '%s,%s,%s,%s,%s\n' "$dataset" "$seed" "$KNN_K" "$CYCLES" "$row" >> "$summary"
    done
  done
done

echo "$summary"
