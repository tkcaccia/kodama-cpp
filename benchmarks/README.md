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
  `$labels` to row-major double and int32 label binaries.
- `run_pls50_all_benchmark_datasets.sh`: builds the CUDA target, exports each
  benchmark dataset, runs `KNNCV`, `PLSDACV`, and `PLSLDACV` with
  `PLS_MAX_COMPONENTS=50`, and writes one CSV per dataset plus a combined
  summary.

On chiamaka, run:

```bash
cd /mnt/sata_ssd/kodama-cpp
bash benchmarks/run_pls50_all_benchmark_datasets.sh
```

Default output:

```text
/mnt/sata_ssd/kodama-cpp-benchmarks/pls50_all_datasets/
  bin/
    <dataset>_x_double_rowmajor.bin
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
max components for PLSDACV / PLSLDACV
selected components for PLSDACV / PLSLDACV
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
