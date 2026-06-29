# Benchmark Notes

The first benchmark target is to compare `KNNCV`, `PLSDACV`, and `PLSLDACV` on the datasets
used by fastEmbedR Benchmark #2 and #3.

Suggested local dataset layout:

```text
/mnt/sata_ssd/fastEmbedR/Data/<dataset>/<dataset>.RData
```

The C++ library does not read RData directly. R and Python wrappers should load
datasets, scale/preprocess them, convert to row-major numeric arrays, and call
the C++ API.

This directory includes two helper files for command-line benchmarking:

- `export_rdata_dataset_to_bin.R`: exports an RData file containing `$data` and
  `$labels` to row-major float64, row-major float32, and int32 label binaries.
- `run_pls50_all_benchmark_datasets.sh`: builds the CUDA target, exports each
  benchmark dataset, runs `KNNCV`, `PLSDACV`, and `PLSLDACV` with
  `PLS_MAX_COMPONENTS=50`, and writes one CSV per dataset plus a combined
  summary. `PLS_MAX_COMPONENTS` is the requested component count; the benchmark
  does not perform an internal best-component search. The benchmark uses the
  float32 matrix by default; set `KODAMA_INPUT_DTYPE=float64` to run from the
  double matrix instead.

On chiamaka, run:

```bash
cd /mnt/sata_ssd/kodama-cpp
bash benchmarks/run_pls50_all_benchmark_datasets.sh
```

For a fast float32 end-to-end smoke test, including the CUDA paths when
available, run:

```bash
cd /mnt/sata_ssd/kodama-cpp
ENV_DIR=/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs \
KODAMA_BUILD_DIR=/mnt/sata_ssd/kodama-cpp/build-cuda \
KODAMA_ENABLE_CUDA=ON \
  bash benchmarks/run_float32_smoke.sh
```

Default output:

```text
/mnt/sata_ssd/kodama-cpp-benchmarks/pls50_all_datasets/
  bin/
    <dataset>_x_double_rowmajor.bin
    <dataset>_x_float32_rowmajor.bin
    <dataset>_labels_int32.bin
    <dataset>_metadata.csv
    <dataset>_label_map.csv
  results/
    <dataset>_pls50_cv_sanity.csv
    pls50_all_datasets_summary.csv
  logs/
```

The script currently enumerates:

```text
COIL20
USPS
FashionMNIST
FlowRepository_FR-FCM-ZYRM_files
flow18
MNIST
imagenet
MetRef
mass41
TabulaMuris
```

For each dataset, report:

```text
dataset
n samples
p features
method: KNNCV, PLSDACV, or PLSLDACV
backend requested
backend used
folds
stratified
constraint policy
k for KNNCV
metric for KNNCV
index type for KNNCV
requested components for PLSDACV / PLSLDACV
evaluated components for PLSDACV / PLSLDACV
global accuracy
fold accuracy
confusion matrix
runtime
peak memory
```

The planned GPU benchmark should compare:

- FAISS CPU IVF-Flat
- FAISS GPU IVF-Flat cosine / inner product
- cuVS nearest-neighbour search where available in the CUDA build
