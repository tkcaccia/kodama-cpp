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

The backend benchmark should compare:

- package-owned CPU HNSW
- package-owned CUDA exact and recall-tuned IVF-Flat search
- native Metal exact and recall-tuned IVF-Flat search

External FAISS/cuVS results may be reported as historical baselines, but are
not build dependencies of the benchmarked library.

Build and run the maintained native-backend microbenchmark with:

```bash
cmake --build build --target kodama_native_backend_benchmark --parallel
./build/kodama_native_backend_benchmark
```

The graph rows report native HNSW with one and four threads, overlap between
those approximate graphs, and exact-neighbor recall when Metal is available.
The measurements used for the 2026-07-31 parallel-HNSW validation are retained
in `native_hnsw_parallel_20260731.csv`.

## Graph-storage lifecycle

Build the single-owner graph benchmark with:

```bash
cmake --build build --target kodama_graph_storage_lifecycle --parallel
/usr/bin/time -v ./build/kodama_graph_storage_lifecycle single 1000021 100
```

Use mode `legacy` to reproduce the former simultaneous `global_graph`,
`result.knn`, and `result.base_knn` allocations. On chiamaka, the flow18-sized
case (`n=1,000,021`, `k=100`) reduced retained graph capacity from
2,408,050,568 to 808,016,968 bytes and maximum RSS from 2,354,304 to
791,808 KiB. The graph ownership/copy lifecycle fell from 0.561 to 0.059
seconds; complete process wall time, including graph creation and the
deliberate 500 ms RSS observation hold, fell from 1.60 to 1.07 seconds. Raw
results are retained in `graph_storage_lifecycle_flow18_20260731.csv`.

## Device-resident KODAMA lifecycle

`device_residency_20260731.csv` records paired development measurements made
before removal of the legacy transfer switch. The default CUDA/Metal path now
uploads the full-data graph once and reuses per-worker landmark/fold,
classifier, projection, and voting allocations across `Tcycle` and `M`.
The CSV includes both positive and neutral rows and records whether CV
accuracy, ARI, and class counts matched. These single pairs validate the
systems change; use the release benchmark protocol for performance claims.
