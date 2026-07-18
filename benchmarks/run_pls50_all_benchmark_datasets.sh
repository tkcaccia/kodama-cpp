#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/sata_ssd/fastEmbedR/Data}"
OUT_ROOT="${OUT_ROOT:-/mnt/sata_ssd/kodama-cpp-benchmarks/pls50_all_datasets}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-cuda}"
ENV_DIR="${ENV_DIR:-/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs}"
N_THREADS="${N_THREADS:-4}"
PLS_MAX_COMPONENTS="${PLS_MAX_COMPONENTS:-50}"
IVF_NLIST="${IVF_NLIST:-0}"
IVF_NPROBE="${IVF_NPROBE:-0}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-600}"
INPUT_DTYPE="${KODAMA_INPUT_DTYPE:-float32}"
MIN_SAMPLES="${KODAMA_MIN_SAMPLES:-500}"

export CONDA_PREFIX="$ENV_DIR"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$N_THREADS"

mkdir -p "$OUT_ROOT/bin" "$OUT_ROOT/results" "$OUT_ROOT/logs"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DKODAMA_ENABLE_CUDA=ON \
  -DCMAKE_PREFIX_PATH="$ENV_DIR" \
  -DCMAKE_CXX_COMPILER="$ENV_DIR/bin/x86_64-conda-linux-gnu-c++"
cmake --build "$BUILD_DIR" -j

declare -A DATASETS=(
  [COIL20]="$DATA_ROOT/COIL20/COIL20.RData"
  [USPS]="$DATA_ROOT/USPS/USPS.RData"
  [FashionMNIST]="$DATA_ROOT/FashionMNIST/FashionMNIST.RData"
  [FlowRepository_FR-FCM-ZYRM_files]="$DATA_ROOT/FlowRepository_FR-FCM-ZYRM_files/van_unen_FR-FCM-ZYRM.RData"
  [flow18]="$DATA_ROOT/flow18/flow18.RData"
  [MNIST]="$DATA_ROOT/MNIST/MNIST.RData"
  [imagenet]="$DATA_ROOT/imagenet/imagenet.RData"
  [MetRef]="$DATA_ROOT/MetRef/MetRef.RData"
  [mass41]="$DATA_ROOT/mass41/mass41.RData"
  [TabulaMuris]="$DATA_ROOT/TabulaMuris/TabulaMuris.RData"
)

summary="$OUT_ROOT/results/pls50_all_datasets_summary.csv"
rm -f "$summary"

for dataset in COIL20 USPS FashionMNIST FlowRepository_FR-FCM-ZYRM_files flow18 MNIST imagenet MetRef mass41 TabulaMuris; do
  input="${DATASETS[$dataset]}"
  if [[ ! -f "$input" ]]; then
    echo "Skipping $dataset: missing $input" | tee "$OUT_ROOT/logs/${dataset}.log"
    continue
  fi

  echo "=== $dataset ===" | tee "$OUT_ROOT/logs/${dataset}.log"
  Rscript "$ROOT_DIR/benchmarks/export_rdata_dataset_to_bin.R" "$dataset" "$input" "$OUT_ROOT/bin" | tee -a "$OUT_ROOT/logs/${dataset}.log"

  meta="$OUT_ROOT/bin/${dataset}_metadata.csv"
  n="$(Rscript -e "m <- read.csv('$meta'); cat(m\$n[1])")"
  p="$(Rscript -e "m <- read.csv('$meta'); cat(m\$p[1])")"
  if (( n < MIN_SAMPLES )); then
    echo "Skipping $dataset: n=$n is below KODAMA_MIN_SAMPLES=$MIN_SAMPLES" | tee -a "$OUT_ROOT/logs/${dataset}.log"
    continue
  fi
  if [[ "$INPUT_DTYPE" == "float32" || "$INPUT_DTYPE" == "float" || "$INPUT_DTYPE" == "single" ]]; then
    x_file="$OUT_ROOT/bin/${dataset}_x_float32_rowmajor.bin"
  elif [[ "$INPUT_DTYPE" == "float64" || "$INPUT_DTYPE" == "double" ]]; then
    x_file="$OUT_ROOT/bin/${dataset}_x_double_rowmajor.bin"
  else
    echo "KODAMA_INPUT_DTYPE must be float32 or float64, got: $INPUT_DTYPE" | tee -a "$OUT_ROOT/logs/${dataset}.log"
    exit 2
  fi
  y_file="$OUT_ROOT/bin/${dataset}_labels_int32.bin"
  out_csv="$OUT_ROOT/results/${dataset}_pls50_cv_sanity.csv"
  rm -f "$out_csv"

  set +e
  KODAMA_INPUT_DTYPE="$INPUT_DTYPE" \
  /usr/bin/time -v timeout "$BENCH_TIMEOUT" "$BUILD_DIR/kodama_cv_sanity_bin" \
    "$dataset" "$x_file" "$y_file" "$n" "$p" "$out_csv" \
    "$N_THREADS" "$PLS_MAX_COMPONENTS" "$IVF_NLIST" "$IVF_NPROBE" \
    > "$OUT_ROOT/logs/${dataset}_run.stdout" \
    2> "$OUT_ROOT/logs/${dataset}_run.stderr"
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "Dataset $dataset exited with status $status; keeping partial rows if available." | tee -a "$OUT_ROOT/logs/${dataset}.log"
  fi

  if [[ ! -f "$out_csv" ]]; then
    echo "No result CSV produced for $dataset" | tee -a "$OUT_ROOT/logs/${dataset}.log"
  elif [[ ! -f "$summary" ]]; then
    cp "$out_csv" "$summary"
  else
    tail -n +2 "$out_csv" >> "$summary"
  fi
done

echo "Summary: $summary"
