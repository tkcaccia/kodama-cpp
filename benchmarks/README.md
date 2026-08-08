# Benchmark Notes

`batched_kmeans_20260809/` records a rejected CUDA experiment that moved the
resident coarse-k-means centroid update from CPU to per-lane CUDA streams. The
production implementation already uploads one immutable matrix and shares it
across M lanes (`kmeans_input_uploads=1`); the candidate did not improve the
four-dataset M=Tcycle=100 pipeline and slowed MetRef, so it was reverted.

`knn_transfer_20260809/` records the accepted compact-label and persistent
Metal-buffer experiment. `metal_metref_m100t100.csv` is the matched four-lane
Metal run; `cuda_knn_m100t100.csv` covers MetRef, Br8100, MERFISH, and flow18 at
M=Tcycle=100; `cuda_plslda_smoke.csv` verifies that the untouched PLS-LDA path
still matches between raw and reusable-graph inputs. The attempted pinned
CUDA staging path was rejected because full-pipeline timings were neutral or
slower at the tested effective landmark counts.

`cpu_pls_projection_loop_order_20260807.csv` and its `_summary` companion
compare the preserved and reordered CPU PLS projection loops on MetRef, USPS,
and COIL20. Each cell uses three fresh processes, five folds, 50 requested
components, seed 7, and four CPU workers; Metal is included as an unchanged
control. Accuracy and selected component counts are recorded with runtime.

The first benchmark target is to compare `KNNCV`, `PLSDACV`, and `PLSLDACV` on the datasets
used by fastEmbedR Benchmark #2 and #3.

Suggested local dataset layout:

```text
/mnt/sata_ssd/fastEmbedR/Data/<dataset>/<dataset>.RData
```

The C++ library does not read RData directly. R and Python wrappers should load
datasets, scale/preprocess them, convert to row-major numeric arrays, and call
the C++ API.

## Local CPU and Metal comparison modes

`run_local_jmlr_cpu_metal.R` accepts `KODAMA_LOCAL_GRAPH_MODE=native` for a
complete backend pipeline or `KODAMA_LOCAL_GRAPH_MODE=shared_cpu` to build one
CPU graph per dataset/seed and supply that exact graph to CPU and Metal
optimization. Do not combine these timing interpretations. Summarize a paired
screen with:

```sh
Rscript benchmarks/analyze_local_cpu_metal_paired.R INPUT.csv OUTPUT_DIRECTORY
```

The analyzer writes backend seed summaries and matched Metal-minus-CPU deltas;
three-seed engineering screens are described with medians and ranges, not
inferential p-values.

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

The accepted standalone Metal UMAP experiment is retained under
`cycle12_metal_umap_20260807/`. It separates optimizer-only timing with a shared
graph/PCA start from end-to-end backend-native graph, PCA, and UMAP timing on
MetRef and USPS, and retains a rejected CPU sort-scratch candidate as negative
evidence. The full protocol is in
`manuscript/JMLR_CYCLE12_METAL_UMAP_REPORT_2026-08-07.md`.

Build the accelerator graph benchmark with:

```bash
cmake --build build-cuda --target kodama_graph_backend_benchmark --parallel
./build-cuda/kodama_graph_backend_benchmark cuda auto 1000021 11 100 1234 data.float32
```

The accepted 2026-08-06 CUDA measurements are retained in
`cuda_graph_ivf_20260806.csv` and `cuda_graph_ivf_end_to_end_20260806.csv`.
On flow18 (`n=1,000,021`, `p=11`, `k=100`), recall-tuned auto-IVF reduced
graph construction from 296.482 to 26.846 seconds (11.04x). With `M=100`,
`Tcycle=100`, and 1,000 landmarks, total KNN/PLS-LDA runtime changed from
318.588/357.032 to 48.377/65.960 seconds. Best CV accuracy remained 1.000;
PLS-LDA ARI changed from 0.926970 to 0.924996.

The subsequent resident-output implementation is recorded in
`resident_ivf_flow18_20260806.csv`. It selected `nlist=1001` independently of
the 256-list `nprobe` bound, tuned `nprobe=64` to pilot recall 0.995313, and
wrote exact candidate distances directly into the device graph. Graph time was
25.938 seconds. Compatible graph-returning KNN/PLS-LDA calls required
47.358/66.135 seconds; labels-only calls, which skip the final host graph
materialization, required 46.945/64.987 seconds. CV accuracy and ARI were
unchanged relative to the host-materialized IVF rows.

## CUDA low-dimensional grid top-k

`cuda_spatial_grid_topk_20260806.csv` records the matched CUDA grid benchmark
for one million 2D points and `k=100`. Replacing per-thread insertion arrays
with warp-cooperative shared-memory bitonic merge-selection reduced graph time
from 3.617523 to a 1.481778-second repeated-run median, a 2.44x speedup. The
compiled kernel uses zero local memory. Exact index and float32-distance parity
against the CPU grid implementation is tested at `k=100` in both 2D and 3D;
the corresponding one-million-point 3D smoke graph required 4.538378 seconds.
`cuda_spatial_grid_real_20260806.csv` additionally records new-kernel medians
of 1.144280 seconds for the 28,317-by-3 MERFISH coordinates and 0.140986
seconds for the 13,938-by-2 Br8100 coordinates.

## R graph conversion and external handles

`run_r_graph_conversion_benchmark.R` builds one handle-backed graph and times
explicit matrix materialization. A matched temporary build retained the former
row-first conversion solely for comparison; that legacy function is not part
of the library. On flow18 (`n=1,000,021`, `k=100`, CUDA), median conversion
time decreased from 0.7359 s to 0.5095 s (1.44x). The native graph occupied
808,016,968 bytes. Materializing R integer indices plus double distances
created a 1,200,026,480-byte R object, whereas the handle-backed R object was
32,007,712 bytes, mostly its two stored PCA initialization matrices. Handle
creation itself took approximately 20 microseconds and copied no graph arrays.
Raw measurements are in `r_graph_conversion_flow18_20260806.csv`.

## Current CUDA dataset validation

`run_current_cuda_dataset_validation.R` runs KODAMA KNN and PLS-LDA with
`M=100`, `Tcycle=100`, one graph build, handle-backed graph output, and fuzzy
CUDA UMAP. It checks all 100 label vectors and both layouts for finite values
and records IVF metadata, timing, CV accuracy, ARI, class counts, and a sampled
truth-label silhouette. The 2026-08-06 validation used the isolated CUDA 13.2
build on chiamaka.

| Dataset | Classifier | Core seconds | Graph seconds | Pilot recall | Best CV | ARI | Truth silhouette, classic -> KODAMA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MetRef | KNN | 4.287 | 0.006 | exact | 1.000 | 0.047 | 0.129 -> 0.003 |
| MetRef | PLS-LDA | 175.497 | 0.006 | exact | 1.000 | 0.712 | 0.129 -> 0.780 |
| MERFISH | KNN | 30.145 | 0.182 | 0.9914 | 0.874 | 0.376 | 0.240 -> 0.280 |
| MERFISH | PLS-LDA | 252.050 | 0.182 | 0.9913 | 0.872 | 0.338 | 0.240 -> 0.263 |
| Br8100 | KNN | 15.680 | 0.109 | 0.9904 | 0.914 | 0.384 | 0.159 -> 0.271 |
| Br8100 | PLS-LDA | 178.046 | 0.109 | 0.9904 | 0.916 | 0.382 | 0.159 -> 0.313 |
| flow18 | KNN | 49.944 | 27.070 | 0.9953 | 1.000 | -0.003 | 0.367 -> 0.380 |
| flow18 | PLS-LDA | 89.921 | 27.164 | 0.9953 | 1.000 | 0.927 | 0.367 -> 0.407 |

The MetRef KNN row is retained as a negative result: at 100 evolution cycles,
maximum CV selection preferred a two-class solution and degraded external
quality. The other seven rows completed normally; this isolates the behavior
from CUDA graph selection because MetRef used exact search and the larger
datasets used recall-tuned IVF-Flat.

## Full-cycle local CPU4 and Metal controls

`run_local_jmlr_cpu_metal.R` also supports a `shared_cpu` mode that constructs
one CPU graph and supplies it unchanged to CPU4 and physical Metal. Current
full-cycle controls use `M=100`, `Tcycle=100`, seed 4, graph k 100, predictor k
30, 50 PLS components, splitting 100, the historical 75% landmark rule, and a
common CPU UMAP visualization.

On USPS, KNN optimization required 12.451 seconds on CPU4 and 7.888 seconds on
Metal; PLS-LDA required 707.057 and 1105.998 seconds. KODAMA truth-label
silhouettes were 0.4550/0.4555 for KNN and 0.4227/0.4174 for PLS-LDA, compared
with approximately 0.13 for classic UMAP. Raw CSV files and plots are in
`manuscript/jmlr_cycle16_usps_knn_m100_t100_20260808/` and
`manuscript/jmlr_cycle16_usps_pls_m100_t100_20260808/`.

ImageSegmentation (2,310 by 19) provides a third full-cycle control. CPU4/Metal
KNN optimization required 1.107/1.672 seconds and PLS-LDA required
12.557/125.108 seconds. KODAMA truth-label silhouettes were 0.5387/0.5396 for
KNN and 0.6418/0.6503 for PLS-LDA, compared with 0.2935/0.2956 for classic
UMAP. The requested 50 PLS components were correctly limited to the 19
mathematically feasible components. Results and plots are in
`manuscript/jmlr_cycle17_imagesegmentation_m100_t100_20260808/`.

In `shared_cpu` mode, backend-matched raw-data PCA initialization is disabled
inside `KODAMA.matrix`; the final common CPU visualization supplies the shared
CPU PCA start. In `native` mode the matrix call retains backend-matched PCA
initialization. This distinction prevents visualization setup from being
mistaken for optimizer execution.

These dirty-worktree measurements are engineering controls, not release
benchmarks. They supersede earlier smaller-cycle USPS screens for current-code
performance interpretation and show that backend speed is classifier- and
matrix-shape dependent. The frozen CPU4/CUDA matrix must repeat both native
and shared-graph modes from an immutable tag.

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

## Local PLS-LDA dimensional scaling

`run_local_plslda_dimension_scaling.R` measures PLSLDACV independently of graph
construction and KODAMA evolution. It supports nested predictor widths,
multiple seeds, CPU/Metal backends, an untimed Metal warm-up, and explicit
fold/component/thread settings. The accepted three-seed COIL20 rows are in
`plslda_dimension_scaling_coil20_fixed_20260807.csv`; MetRef and USPS anchors
use the corresponding dataset-named files. The fold-worker regression is in
`metal_pls_fold_epoch_fixed_coil20_20260807.csv`. Rejected score-copy and host
workspace experiments remain archived separately and are not compiled paths.

## Streamed CPU PLS-LDA sufficient statistics

`cpu_pls_streamed_sufficient_stats_20260807.csv` compares preserved baseline
and candidate executables on MetRef, USPS, COIL20, and TabulaMuris. Every row
uses float32 input, five folds, 50 fixed components, seed 7, four CPU workers,
and three fresh processes. The candidate projects one row at a time into class
sums and the latent Gram matrix. Accuracy and predictions match exactly.
TabulaMuris improved from 4.110068 to 2.794163 seconds (1.47x), while the median
reported peak-memory ratio was 0.587. The summarized medians are in
`cpu_pls_streamed_sufficient_stats_summary_20260807.csv`.

## Robust float32 SIMPLS power norm

`simpls_robust_power_norm_20260807.csv` records three fresh CPU and physical
Metal processes for MetRef, USPS, COIL20, TabulaMuris, and MNIST70k. The final
guard leaves the original norm result unchanged whenever its squared value is
float-representable and falls back to double accumulation only when required.
On MNIST70k, the former CPU path stopped at one component with accuracy
0.112529 and Metal failed; the accepted path evaluated all 50 requested
components with accuracy 0.866457 on CPU and 0.866286 on Metal. Median Metal
time was 2.295991 seconds. Ordinary CPU controls were exactly unchanged.

The rejected alternative that normalized after every power iteration is
retained under `simpls_power_normalization_rejected_20260807.csv`; it is not a
compiled path. `simpls_robust_power_norm_summary_20260807.csv` contains the
accepted medians.

## Resident Metal PLS-LDA sufficient statistics

`metal_pls_sufficient_stats_20260807.csv` records the Cycle-9 CPU4/Metal
PLSLDACV screen on MetRef, USPS, raw COIL20, TabulaMuris, and MNIST70k for
seeds 4, 17, and 42. Every row uses float32 input, five folds, 50 fixed
components, and four workers; deterministic prediction hashes make exact
matches visible. The summarized medians are in
`metal_pls_sufficient_stats_multiseed_summary_20260807.csv`.

`metal_pls_sufficient_stats_comparison_20260807.csv` compares three fresh
seed-7 candidate processes with the retained pre-change Cycle-7 executable.
The new path keeps `XW` resident, downloads only class sums and `T'T`, scores
validation rows on Metal, and returns only labels. Accuracy and component count
are unchanged in all five preserved-executable comparisons. TabulaMuris
improves by 3.65x and MNIST70k by 1.35x; MetRef is 0.94x and raw COIL20 0.99x,
so the small-data and predictor-dominated crossover is retained rather than
hidden.

`metal_pls_sufficient_stats_kodama_smoke_20260807.csv` is a non-confirmatory
`M=5`, `Tcycle=5` integration screen on MetRef, USPS, and TabulaMuris. Native
CPU4/Metal speedups are 1.92x, 3.74x, and 1.98x. Because native graph builders
differ, these rows prove integration only and are never pooled with the
shared-graph optimizer comparison or the frozen release study.

## Cycle 30 CPU micro-optimizations

`cpu_micro_20260809/` contains five-run evidence for the runtime-dispatched
x86-64 AVX2/FMA HNSW distance kernel, the per-worker graph-correction sorting
buffer, and the rejected software-prefetch candidate. On chiamaka's Intel
Core i7-13700, AVX2 reduced median HNSW time by 6.7% with one thread and 7.0%
with four threads; candidate single/four-thread graph overlap was 1.000 in all
five runs. Apple Clang already emits NEON for the portable loop. Reusing the
correction buffer preserved the exact output hash and had neutral wall time.

## MPS latent Gram follow-up

`cycle11_metal_mps_gram_20260807/` retains alternating-order baseline and
candidate rows for replacing the serial-per-component-pair Metal `T'T` kernel
with `MPSMatrixMultiplication`. Run
`Rscript benchmarks/analyze_metal_mps_gram.R` to regenerate the raw, seed-pair,
seed-summary, and seven-repeat CSV files. Across seeds 4, 17, and 42, median
within-pair speedups are 1.14x on TabulaMuris, 1.10x on MNIST70k, 1.04x on
USPS, 1.02x on raw COIL20, and 0.99x on tiny MetRef. All rows retain 50
components and no PLSLDACV accuracy decreases. The separate `M=5`,
`Tcycle=5` integration rows are in
`metal_mps_gram_kodama_smoke_20260807.csv`.

## Cycle 18 release-candidate controls

`manuscript/jmlr_cycle18_pageblocks_m100_t100_20260808/` retains the complete
PageBlocks CPU4/physical-Metal shared-graph run at `M=Tcycle=100`. It is an
intentional adverse-quality control: median CV accuracy is backend-consistent,
but classic and KODAMA truth-label silhouettes are negative. The same cycle
also validated a clean installed CMake consumer, an R source package with no
errors or warnings under `R CMD check --as-cran`, and an installed-core Python
wheel in a fresh environment. Direct accelerator coverage is indexed in
`docs/accelerator-entrypoint-validation.md`.

## Cycle 20 stochastic contracts and PenDigits

`manuscript/jmlr_cycle20_pendigits_m100_t100_20260808/` contains the full
PenDigits CPU4/physical-Metal shared-graph control at `M=Tcycle=100`. It is an
adverse visualization control: classic UMAP truth silhouette is 0.5116, while
all KODAMA cells are negative despite high internal CV accuracy. It also
records a PLS-LDA optimization crossover of 39.311 seconds on CPU4 versus
199.897 seconds on Metal. Preserve this row in release confirmation rather
than tuning it after inspection.

## Cycle 21 orchestration timing and COIL20

`manuscript/jmlr_cycle21_coil20_m100_t100_20260808/` contains paired CPU4 and
physical-Metal KNN runs on raw COIL20 at `M=Tcycle=100`. Metal reduced
optimization wall time from 193.377 to 80.753 seconds (2.39x), while median CV
accuracy was 0.99537 on both backends. Both corrected truth silhouettes were
below classic UMAP, so the rows are acceleration and adverse-quality evidence,
not a universal quality claim. Raw CPU4 PLS-LDA was stopped after 300 seconds
before the first four runs completed and is preserved in `censored_cells.csv`.

The driver now stores the wrapper's per-run `landmark_seconds` vector and its
sum, mean, and median. Its append helper unions CSV schemas so historical rows
remain readable when new stage fields are added. Use these columns in the
frozen CPU4/CUDA rerun rather than attributing landmark construction to the
classifier core.

## Cycle 22 binary UMAP and SatImage

`manuscript/jmlr_cycle22_satimage_m100_t100_20260808/` contains a complete
seed-4 SatImage control at `M=Tcycle=100`, 4,827 effective landmarks, one
shared CPU graph protocol, CPU4/physical Metal optimization, and both KNN and
raw PLS-LDA. Metal was 1.06x slower than CPU4 for KNN and 5.87x slower for
PLS-LDA. Both KODAMA truth-label UMAP silhouettes were below classic UMAP, so
the result is retained as a negative systems and quality control. Cycle 22 also
adds a deterministic test for the public binary UMAP graph route copied from
fastEmbedR; no production optimizer or KODAMA mathematics changed.
