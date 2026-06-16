#!/usr/bin/env bash
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
  x_file="$OUT_ROOT/bin/${dataset}_x_double_rowmajor.bin"
  y_file="$OUT_ROOT/bin/${dataset}_labels_int32.bin"
  out_csv="$OUT_ROOT/results/${dataset}_pls50_cv_sanity.csv"

  /usr/bin/time -v "$BUILD_DIR/kodama_cv_sanity_bin" \
    "$dataset" "$x_file" "$y_file" "$n" "$p" "$out_csv" \
    "$N_THREADS" "$PLS_MAX_COMPONENTS" "$IVF_NLIST" "$IVF_NPROBE" \
    > "$OUT_ROOT/logs/${dataset}_run.stdout" \
    2> "$OUT_ROOT/logs/${dataset}_run.stderr"

  if [[ ! -f "$summary" ]]; then
    cp "$out_csv" "$summary"
  else
    tail -n +2 "$out_csv" >> "$summary"
  fi
done

echo "Summary: $summary"
