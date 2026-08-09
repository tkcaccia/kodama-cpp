# Backend Validation

This file records implementation decisions that were tested rather than
accepted from timing intuition alone.

## Accepted

### Package-native CPU HNSW

The default CPU KNN path no longer requires FAISS. On MetRef (873 samples,
375 variables, five folds, `k=10`) it reproduced the reference accuracy of
0.827033. The dependency-free single-thread build had a five-run median of
11.145 s.

HNSW construction now completes a deterministic connected base seed of
`min(n, max(n_threads + 1, 2 * m_hnsw + 1))` points, then inserts the
remaining nodes concurrently with per-node locking. It computes candidate
distances in reusable batches, caches reciprocal-edge distances during
construction, and queries contiguous row batches in parallel. Vectors,
adjacency lists, counts, and temporary build distances use flat
float32/integer storage. On an Apple M3, the maintained synthetic
4000-by-32, `k=30` graph benchmark produced these three-run medians:

| Implementation | Threads | Seconds | Quality |
| --- | ---: | ---: | --- |
| Previous serial-insertion path | 4 | 7.248 | exact-reference recall 1.000 in the recorded run |
| Parallel construction and query | 1 | 6.722 | reference |
| Parallel construction and query | 4 | 1.791 | median overlap with one-thread graph 0.99966 |

The accepted path therefore scales by 3.75x from one to four threads and is
4.05x faster than the previous four-thread implementation. Across 100
repeated four-thread builds of a 600-by-24 problem with `k=20`, minimum and
mean brute-force recall were 0.99975 and 0.99987. The CPU regression test
requires both one-thread and four-thread recall of at least 0.99 against
brute-force neighbors, and the complete CPU suite passes ThreadSanitizer.

### Native Apple Metal exact KNN

On the same MetRef test, Metal exact KNN reproduced accuracy 0.827033 with a
five-run median of 0.026 s, about 425x faster than the dependency-free
single-thread CPU HNSW run.

### Native Apple Metal PLS-LDA

The Metal path uses float32 label-aware SIMPLS, MPS matrix multiplication,
and latent-space LDA. With 50 requested components on MetRef, CPU and Metal
both selected 50 components and produced accuracy 0.991982. Five-run medians
in the dependency-free build were 3.395 s for CPU and 1.054 s for Metal,
about 3.2x faster.

### Persistent Metal state and fused SIMPLS projection commands

The Metal device, command queue, compiled library, and pipeline states are
created once. Each SIMPLS component encodes `Xw` and `X'Xw` into one command
buffer. This removes one synchronization per component without changing the
SIMPLS equations; parity tests remained unchanged.

Device discovery first uses the system default and then enumerates all Metal
devices; the selected device is cached process-wide. The preprocessing backend
uses the same fallback. Independent KODAMA `M` workers own persistent Metal
command queues and float32 PLS workspaces, mirroring CUDA stream lanes. On an
Apple M3, three warm matched MetRef PLS-LDA runs (`M=20`, `Tcycle=10`, 50
components, four workers) reduced median total time from 7.390 to 7.014 seconds
(1.054x), with identical best CV accuracy (0.934351), ARI (0.737346), and 43
classes. On Br8100, four lanes reduced the optimization phase by 2.61x relative
to one lane with identical quality. Raw measurements are in
`benchmarks/metal_pls_worker_queues_20260807.csv`.

The automatic planner was subsequently corrected to model Metal's retained
fold matrices rather than CUDA's predictor Gram workspace. On raw COIL20
(1,440 by 16,384, `M=4`, `Tcycle=1`, 50 components), explicit one/two/three/four
lane optimization times were 61.578/38.685/47.883/40.856 seconds with identical
best and median CV accuracy and identical selected ARI. Timing varied with
thermal/run order, so lane selection is based on the device working-set budget,
not the fastest observed row. The final fold-aware planner selected three lanes
for COIL20 (992 MiB estimated per lane) and four for MetRef (142 MiB per lane).
Raw rows are in `benchmarks/metal_pls_fold_scheduler_coil20_20260807.csv`.

The full-cycle scheduler was then audited at `M=Tcycle=100`. Two independent
systems defects explained the earlier Metal slowdown. First, the resident
matrix cache retained only one fold matrix, so a five-fold traversal evicted
and uploaded the same immutable matrices repeatedly. It now keeps a bounded
16-entry least-recently-used set per worker. Second, the independent-`M`
scheduler was capped at four lanes regardless of the device working-set
budget. The planner can now select up to 32 lanes while still bounding the
aggregate estimate by `recommendedMaxWorkingSetSize`. No SIMPLS, LDA, fold,
proposal, temperature, or label-selection equation changed.

On MetRef, the corrected Metal PLS-LDA path selected 26 lanes and reduced the
complete `M=100`, `Tcycle=100` time from 631.415 to 312.029 seconds (2.02x).
The four-core CPU reference required 316.329 seconds. Best CV accuracy
(1.000000), selected ARI (0.860185), selected classes (32), median classes
(25), class range (16--35), and collapse rate (0) were unchanged. On Br8100,
the corrected Metal path selected 21 lanes and reduced PLS-LDA from 457.563 to
363.496 seconds (1.26x), with identical selected ARI (0.380316), selected
classes (3), median classes (4), class range (3--4), and best CV accuracy
(0.915822). Metal KNN used 25 lanes and improved from 89.234 to 62.954 seconds
(1.42x); median classes remained 3 and collapse rate remained zero.

An additional nested-parallelism audit found that each independent Metal `M`
lane inherited the automatic lane count as its fold-worker count. With 29 `M`
lanes and five folds this could create 145 host workers and as many
thread-local queues/workspaces during repeated CV. Metal lanes now traverse
their fixed folds sequentially; concurrency remains across independent `M`
runs. On BreastCancerDiagnostic (`569 x 30`, `M=Tcycle=100`) this reduced
PLS-LDA from 109.557 to 105.285 seconds with unchanged CV, ARI, class-count,
and collapse diagnostics. Four-core CPU required 6.990 seconds, documenting a
real low-dimensional accelerator crossover rather than hiding it.

The current native benchmark isolates the source: exact Metal graph
construction required 0.044813 seconds versus 0.249282 seconds for four-core
HNSW at recall 1.000, and Metal KNN CV required 0.035125 seconds versus
0.879167 seconds on CPU. The corresponding small PLS-LDA CV probe required
0.028136 seconds on Metal versus 0.018463 seconds on CPU. Thus graph
construction and KNN acceleration are healthy; repeated host-controlled
SIMPLS component dependencies dominate low-p PLS-LDA. A one-command,
single-threadgroup SIMPLS candidate passed parity after stable-norm correction
but was slower and was removed completely. Measurements are in
`benchmarks/metal_graph_pls_crossover_20260808.csv`.

### Native Metal spatial-grid graph construction

The graph dispatcher formerly excluded Metal from the exact 2D/3D grid path,
so spatial coordinates were sent through generic IVF-Flat even though the
grid rule had selected a spatial workload. Metal now implements the same
cell indexing, expansion, lower-bound stopping rule, deterministic
distance/index ordering, and self-exclusion contract as the CPU grid path.
The kernel keeps candidate selection on the device and returns squared
distances for the existing Euclidean conversion.

On Br8100 coordinates (`n=13,938`, `k=100`), spatial graph construction fell
from 0.08943 seconds on four CPU cores to 0.02719 seconds on Metal (3.29x),
with exact neighbor overlap 1.000. The complete 50-variable graph required
1.931 seconds with CPU HNSW and 0.808 seconds with recall-tuned Metal IVF
(2.39x), with mean row overlap 0.98382. A physical-Metal regression on a
257-by-2 matrix requires exact CPU/Metal indices and float32 distance parity.
Raw rows are in `benchmarks/metal_graph_construction_20260808.csv`.

For tiny graphs, the first Metal call includes one-time runtime pipeline
compilation. Three MetRef graph calls in one R process required 0.136, 0.023,
and 0.023 seconds wall time; the corresponding core graph times were 0.1065,
0.0194, and 0.0200 seconds. The four-core CPU graph required 0.1517 seconds.
Both first-use and warm timings are therefore reported, rather than treating
shader compilation as repeated graph work.

A separate short end-to-end MetRef check (`M=5`, `Tcycle=10`, four workers)
measured 4.101/0.178 seconds for CPU/Metal KNN and 3.057/2.110 seconds for
CPU/Metal PLS-LDA, corresponding to 23.0x and 1.45x total speedups. Fixed-label
kernel parity is covered by the native tests. Complete stochastic trajectories
are not expected to be label-identical because CPU HNSW versus Metal exact
graphs and float32 reduction order can redirect near-tied proposals. Raw rows
are in `benchmarks/metal_metref_backend_validation_20260807.csv`.

### Metal PLS-LDA fold residency and dimensional scaling

Sequential Metal PLS-LDA folds formerly could reuse a temporary host address
and therefore select a stale resident matrix. Fold-specific worker-local epochs
now make matrix identity independent of allocator address reuse. The native
Metal suite compares one-thread and four-thread predictions directly.

On COIL20 (1,440 samples), five-fold PLSLDACV with 20 requested components and
16,384 raw predictors returned accuracy 0.555556 with one fold worker before
the fix versus 0.959028 with four. After the fix, seeds 4, 17, and 42 matched
exactly between worker counts; median kernel time was 0.634 versus 0.406 seconds
(1.56x). Across nested widths 256, 1,024, 4,096, and 16,384, median four-core
CPU/Metal speedups were 1.71x, 5.19x, 13.82x, and 10.78x. Raw rows are in
`benchmarks/metal_pls_fold_epoch_fixed_coil20_20260807.csv` and
`benchmarks/plslda_dimension_scaling_coil20_fixed_20260807.csv`.

### KODAMA-resident CUDA and Metal state

The complete KODAMA accelerator lifecycle now owns one resident full-data
graph and one scratch lane per concurrent `M` worker. A lane retains
landmark/fold matrices, compact labels, projected labels, KNN voting arrays,
PLS weights, cross-products, latent projections, and LDA workspaces.
Landmark membership and fold contents change between independent runs, so
their existing allocations are refreshed once per `M`; they are not
reallocated or retransferred for every proposal cycle. Landmark projection
uploads only selected row indices and labels. Per-lane epoch tags avoid
clearing an `n`-element mask, and projection writes directly into a resident
run-major `M x n` matrix. Exact constrained-majority relabeling preserves the
CPU smallest-label tie rule on-device. Final dissimilarity consumes the same
matrix and downloads it once. `projection_sparse_uploads`,
`projection_full_downloads`, `result_row_uploads`, and
`result_matrix_downloads` expose this transfer contract.

The final CPU agreement pass transposes the independent-run label matrix once
from run-major to sample-major storage, then compares each retained edge over
contiguous labels. At `M=100`, `k=100`, and four threads, matched synthetic
tests improved from 0.3170 to 0.1031 seconds for 20,000 samples (3.08x) and
from 1.5520 to 0.6323 seconds for 100,000 samples (2.45x), with byte-identical
corrected graphs. The public Metal correction now executes natively rather
than falling through to CPU and is checked against CPU distances. Two CUDA
alternatives, a host-transposed layout and warp-ballot/popcount reduction,
were exact but 10.7% and 4.8% slower respectively at 100,000 samples, so both
were rejected and CUDA retains its run-major single-kernel implementation.
Raw matched rows are in `benchmarks/dissimilarity_layout_20260807.csv`.

CUDA label-aware `X'Y` originally centered class sums by reading and writing
the same matrix in one kernel. A two-kernel column-sum/centering sequence
removes that race. Class accumulation and tile reduction use fixed row order,
the random generator offset is reset with its seed, and every SIMPLS stream
owns a persistent cuBLAS workspace. On the five MetRef folds, resident and
ordinary fits and two repeated ordinary fits produced bitwise-identical
float32 PLS weights after this correction.

The following single paired engineering runs compare the former transfer path
with the now-default resident path. They are lifecycle checks rather than a
multi-dataset performance claim.

| Dataset | Backend | Classifier | M / Tcycle | Transfer (s) | Resident (s) | Speedup | Output |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| MetRef | CUDA | KNN | 10 / 10 | 0.4647 | 0.4464 | 1.04x | exact |
| MetRef | CUDA | PLS-LDA | 10 / 10 | 1.5000 | 1.4418 | 1.04x | exact |
| Br8100 | CUDA | KNN | 10 / 10 | 1.0995 | 1.1082 | 0.99x | exact |
| Br8100 | CUDA | PLS-LDA | 10 / 10 | 1.9756 | 1.8044 | 1.09x | exact |
| Br8100 | CUDA | PLS-LDA | 1 / 100 | 3.5861 | 3.2451 | 1.11x | exact |
| MetRef | Metal | KNN | 10 / 10 | 0.2079 | 0.1820 | 1.14x | exact |
| MetRef | Metal | PLS-LDA | 10 / 10 | 9.2716 | 8.9736 | 1.03x | exact |

Raw rows are retained in
`benchmarks/device_residency_20260731.csv`. The neutral Br8100 KNN row is
reported deliberately: at this scale its search and stochastic orchestration
dominate the saved transfers.

### Package-owned CUDA exact/IVF KNN and k-means

The CUDA path now builds from `src/native_cuda_backend.cu` using only CUDA
Toolkit libraries. It contains float32 exact KNN, signed-hash IVF-Flat,
GPU k-means, inverted-list construction, and deterministic exact-pilot recall
tuning. FAISS, cuVS, RAFT, and RMM headers and runtime libraries are absent.

A clean CUDA 13.2 build on an NVIDIA GeForce RTX 5060 Ti passed both CTest
suites. `ldd` inspection of the test executable found no FAISS, cuVS, RAFT,
or RMM soname. The fresh CMake cache contained no package option,
target, header, or library entry for those dependencies; the CUDA Toolkit was
located inside an environment whose directory name retains a legacy
`faissgpu-cuvs` label.

Spot checks using five-fold cosine KNNCV with `k=10` were:

| Dataset | CUDA path | Accuracy | Seconds | nlist | maximum nprobe |
| --- | --- | ---: | ---: | ---: | ---: |
| MNIST70k | package-owned IVF-Flat | 0.973857 | 4.233 | 237 | 32 |
| MetRef | package-owned IVF-Flat | 0.816724 | 0.195 | 27 | 20 |

The recorded former FAISS/cuVS rows were 0.973029 in 4.753 s on MNIST70k and
0.814433 in 0.297 s on MetRef. These are single-run implementation checks,
not a replacement for the repeated benchmark protocol used by the manuscript.
The accepted automatic rule begins at `2 * ceil(sqrt(nlist))` probes and then
increases the budget if the exact pilot does not meet its recall target.

For full-data KODAMA graph preparation, CUDA and Metal use exact search when
`n <= 5000` or `n * n * p <= 2e8`; larger matrices use a resident IVF-Flat
index tuned to a fixed 0.99 exact-pilot recall target. This dispatch affects
graph construction only and does not change folds, proposals, classifiers,
landmarking, or the KODAMA objective. The returned graph records the actual
index type, `nlist`, `nprobe`, and pilot recall.

Automatic list count and probe count are independent. CUDA selects
`nlist = min(4096, ceil(sqrt(n)))`; only `nprobe` is bounded at 256. Removing
the former accidental `nlist <= 256` coupling gives flow18 `nlist=1001`.
IVF self-search now writes exact candidate distances directly into the
resident KODAMA graph. The wrappers do not materialize host matrices by
default; request them explicitly with `return.graph=TRUE` or
`return_graph=True`.

For large 2D/3D Euclidean graphs, the CUDA grid path uses one warp per query
and four queries per block. Candidates from each visited cell are loaded in
batches and merged with the current exact top-k in shared memory using a
deterministic distance/index bitonic ordering. This preserves the existing
grid expansion and lower-bound stopping rule while eliminating the former
`k`-sized thread-local arrays. CUDA 13 resource inspection reports
`LOCAL:0`; dedicated tests require exact CPU parity at `k=100` in 2D and 3D.

The R wrapper uses a 64-by-64 blocked transpose between native row-major graph
arrays and R column-major matrices. On flow18 (`1,000,021 x 100` neighbors),
matched median conversion time was 0.5095 s, versus 0.7359 s for the former
row-first loops. `storage="handle"` bypasses conversion and exposes an owning
external pointer accepted directly by KODAMA matrix optimization,
UMAP/openTSNE, and clustering. This avoids the 1,200,026,480-byte materialized
R graph until explicitly requested.

| Dataset | Scope | Exact (s) | Auto-IVF (s) | Speedup | Quality check |
| --- | --- | ---: | ---: | ---: | --- |
| Br8100 | graph, 13,938 x 50, k=100 | 0.413 | 0.312 | 1.32x | pilot recall 0.990486 |
| flow18 | graph, 1,000,021 x 11, k=100 | 296.482 | 26.846 | 11.04x | pilot recall 0.995359 |
| flow18 | KODAMA KNN, M=100 T=100 L=1000 | 318.588 | 48.377 | 6.59x | CV 1.000; ARI -0.003185/0.030913 |
| flow18 | KODAMA PLS-LDA, M=100 T=100 L=1000 | 357.032 | 65.960 | 5.41x | CV 1.000; ARI 0.926970/0.924996 |

After eliminating the remaining full-graph download/re-upload, the same
flow18 run selected `nlist=1001`, `nprobe=64`, and pilot recall 0.995313.
The compatible materialized return took 47.358 s for KNN and 66.135 s for
PLS-LDA; labels-only calls took 46.945 and 64.987 s. KNN improved modestly,
whereas the materialized PLS-LDA difference is neutral timing noise. ARI
values, 0.030913 and 0.924996, exactly match the accepted host-materialized
IVF path. Raw rows are in
`benchmarks/resident_ivf_flow18_20260806.csv`.

The Metal dispatch is implemented under the same policy, but these flow18
measurements are CUDA-only and are not presented as Metal runtime evidence.

Landmark KNN extraction now queries the retained CPU HNSW or CUDA/Metal IVF
index with a global-to-landmark membership map. Candidate expansion continues
until every query has `knn.k` landmark-local neighbors; retained candidates
use exact float32 distances, and an exact filtered fallback handles a row that
cannot be completed within the IVF probe bound. A regression with 32 landmarks
among 8,192 rows requires no missing neighbors and at least 0.99 recall on CPU
and CUDA; the CUDA test passed on chiamaka. On flow18 with 1,000 landmarks,
CUDA selected `nlist=1001`, `nprobe=64`, and exact-pilot recall 0.995313. The
shared graph/index cost was 25.204 seconds and a matched `M=100`, `Tcycle=100`
KNN run took 69.669 seconds total. Its best-run ARI remained 0.000723, so this
test establishes graph completeness and index reuse, not downstream label
quality. The audit identified a separate parity boundary: graph-based CV masks
validation-fold neighbors after constructing a graph of width `knn.k`, whereas
raw-data CV queries `knn.k` eligible training neighbors for each fold. Exact
raw-data parity therefore requires fold-filtered retained-index queries, not a
wider benchmark-specific graph. Graph-only input still induces the landmark
subgraph because supplied indices and distances do not include a reusable
search index.

### Dataset-scoped execution context

`KODAMA.matrix` now creates one execution context for the complete call. It
references the immutable host float32 matrix, owns the retained HNSW/IVF graph,
records the shared PCA initialization, and owns one full-data k-means workspace
per accelerator scheduler lane. CUDA stores one device matrix and precomputed
row norms plus lane-local cuBLAS, centroid, assignment, score, and change
buffers. Metal stores one matrix buffer plus lane-local centroid, assignment,
accumulator, count, initialization, and parameter buffers. Existing CUDA and
Metal classifier workspaces remain lane/thread resident across Tcycles.

Only the invariant full-data coarse partition uses this context. Landmark
initialization and spatially jittered matrices vary by M and therefore retain
their independent k-means calls. Initial permutations and seeds are unchanged;
tests compare complete assignment vectors against the former one-shot calls for
multiple lanes. `KODAMAMatrixResult::kmeans_input_uploads` is required to equal
one for non-spatial CUDA and Metal matrix runs.

On flow18 (1,000,021 by 11), CUDA KNN with 1,000 landmarks and `Tcycle=100`
changed from 30.689 to 30.631 seconds at `M=10`, and from 69.669 to 69.323
seconds at `M=100`. The matched `M=100` result retained best CV accuracy 1.0,
best external ARI 0.000723, eight classes, `nlist=1001`, `nprobe=64`, and pilot
recall 0.995313. The 0.5% end-to-end gain shows that repeated 44 MB uploads and
allocation were real but not dominant on this dataset; graph construction and
the host-order-preserving centroid update remain the larger costs. No change to
landmark selection is claimed.

### Device-side IVF construction and explicit resident indexes

CUDA and Metal IVF construction no longer downloads assignments to construct
centroids or inverted lists. Assignment accumulation, empty-cluster repair,
list counting, prefix offsets, and row-ID scatter are accelerator kernels.
The move-only `ResidentIVFIndex` owns the training matrix, projected matrix,
centroids, offsets, and IDs until its destructor releases them. There is no
implicit process-global cache.

The focused benchmark used a deterministic 50,000 by 32 float32 matrix,
`k=30`, five timed searches after one warm-up, and fixed probe counts. It compared rebuilding before
every search with one build followed by repeated searches:

| Backend | Queries/call | nlist | nprobe | Rebuild each call (s) | Reused search (s) | Speedup | Neighbor overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Metal | 50,000 | 256 | 32 | 0.8695 | 0.8091 | 1.07x | 1.000 |
| CUDA | 50,000 | 1,024 | 64 | 0.1984 | 0.1765 | 1.12x | 1.000 |
| Metal | 1,000 | 256 | 32 | 0.1179 | 0.0306 | 3.86x | 1.000 |
| CUDA | 1,000 | 1,024 | 64 | 0.0318 | 0.0092 | 3.45x | 1.000 |

These are systems microbenchmarks, not dataset-level accuracy estimates. The
full self-search rows show that query work dominates once every sample needs
neighbors; the batched rows isolate the saved construction and training-data
transfer. Reproduce them with `kodama_resident_ivf_benchmark`.

### Portable installed target

The installed static Metal target exports framework names rather than local
SDK paths. A clean external CMake consumer configured, linked, and ran using
only `find_package(kodama-cpp CONFIG REQUIRED)`.

## Accepted As Explicit Option

### Recall-tuned Metal IVF-Flat

On MNIST10k (784 variables, five folds, `k=10`):

| Method | Accuracy | Seconds |
| --- | ---: | ---: |
| Metal exact | 0.9490 | 2.054 |
| Metal IVF, auto 0.999 recall target | 0.9490 | 1.662 |

The IVF path selected up to `nprobe=128` and preserved classification
accuracy while reducing CV time by about 1.24x. It remains explicit because
a short end-to-end KODAMA test (`M=2`, `Tcycle=2`) took 9.223 s with IVF and
8.964 s with exact search: repeated index training outweighed the cheaper
queries at this size.

## Rejected As Default

### Fully resident proposal evolution (2026-08-09)

A CUDA/Metal candidate kept KNN proposal generation, prediction transitions,
class counts, guarded-diversity scoring, many-to-one absorption, Metropolis
acceptance, and label updates on the accelerator for a complete M run. The
candidate used one persistent cooperative workgroup so that no sample-level
prediction vector crossed the device boundary during `Tcycle`.

The candidate was tested with the same evolution formulas on physical Metal
and CUDA. MetRef used `M=100`, `Tcycle=100`; Br8100, MERFISH, and flow18 used
`M=20`, `Tcycle=100` as a broader screen after the disposable Metal build hit
memory pressure at `M=100` on the larger inputs. All runs used seed 1234,
splitting 100 (300 for flow18), graph k 100, and predictor k 30.

The resident candidate was slower in all seven matched comparisons. Slowdown
was 7.39x (Metal) and 3.34x (CUDA) on MetRef, 2.84x and 7.93x on Br8100, and
2.56x and 6.48x on MERFISH. On flow18 CUDA it was 1.10x slower end to end;
excluding the approximately 26-second graph build, evolution was 1.40x
slower. Class-count summaries were generally stable, but ARI changes were not
consistently favorable. The transfer savings did not compensate for serial
proposal bookkeeping and loss of accelerator occupancy.

Following the common-method requirement for CUDA and Metal, the candidate was
removed from both backends rather than retained as a hidden path. Spatial
datasets in this engineering screen are internal regression tests and are not
part of the non-spatial manuscript. Evidence is in
`benchmarks/resident_proposal_evolution_20260809.csv`.

- Fusing PLS projection directly into LDA sufficient statistics was rejected
  for every backend. Although isolated CPU and CUDA kernels were competitive,
  the first full matched MetRef runs did not improve the accepted CPU/CUDA
  timings, and the Metal tiled implementation produced a seed-specific
  accuracy failure (`0.787` versus approximately `0.99`). Restoring full Metal
  latent scores recovered accuracy but serialized the automatic worker lanes
  and increased the `M=100`, `Tcycle=100` run to 2,518 seconds. Because CPU,
  CUDA, and Metal must retain one common PLS-LDA methodology, no backend keeps
  the experimental fusion path.
- Recall-tuned Metal IVF at a 0.99 target: MNIST10k accuracy was 0.9482 versus
  0.9490 exact, despite a larger speedup. The automatic target was raised to
  0.999.
- Automatic replacement of exact Metal search by IVF: dataset scale and the
  number of repeated fold indices determine whether IVF helps, so callers
  select it explicitly.
- Silent Metal-to-CPU graph clustering: Metal clustering now raises a clear
  error. CPU clustering remains available when explicitly requested.

## Current Metal Boundary

Matrix KODAMA KNN and PLS-LDA, their CV/core kernels, nearest-neighbor graph
construction, k-means, UMAP, and FFT-grid openTSNE are native Metal operations. The graph-only API
first constructs sparse spectral features on the CPU and reports that time
separately as `graph_feature_seconds`; subsequent KODAMA optimization can run
on Metal. Community clustering remains CPU-only.

Metal worker lanes are selected automatically when `n_threads=0`. The planner
uses the device recommended working-set size, subtracts current allocations,
budgets 70% of the remainder, accounts for all retained PLS CV-fold matrices,
and caps independent lanes at four. Explicit positive thread counts remain
authoritative. R and Python return `gpu_auto_workers`,
`gpu_scheduler_lanes`, and `gpu_worker_memory_estimate_mb`.

## Resident SIMPLS Gram Reuse (2026-08-07)

CUDA PLS-LDA forms each fixed cross-validation fold's float32 Gram matrix once
and reuses it throughout the independent `M` runs and `Tcycle` evaluations.
SIMPLS previously recomputed `Xr` followed by `X'Xr` for every component. The
CUDA replacement uses `X'Xr` directly and obtains the unchanged score norm
from `r'X'Xr`; label uploads are ordered against the persistent stream with an
event instead of a host synchronization. Metal does not allocate this
predictor-by-predictor Gram matrix: it retains fold matrices and persistent MPS
buffers and encodes the paired products in one command buffer per component.

Matched CUDA `M=100`, `Tcycle=100` R-wrapper runs gave:

| Dataset | Previous (s) | Current (s) | Speedup | ARI before/after | truth silhouette before/after |
|---|---:|---:|---:|---:|---:|
| MetRef | 175.497 | 164.441 | 1.07x | 0.712 / 0.768 | 0.780 / 0.837 |
| Br8100 | 178.046 | 165.061 | 1.08x | 0.382 / 0.390 | 0.3135 / 0.3136 |
| flow18 | 89.921 | 88.576 | 1.02x | 0.927 / 0.927 | 0.4068 / 0.4018 |

All native CUDA tests passed. The different float32 reduction order can alter
near-tied stochastic proposal decisions, so exact label-vector equality is not
claimed; CV accuracy and downstream quality were retained. The analogous Metal
implementation was compiled and executed on Apple M3. Its dedicated suite
passed native exact/IVF KNN, resident graph reuse, PCA, normalization, scaling,
and KODAMA KNN/PLS-LDA checks; matched timing is reported above.

## Regression Suites

- macOS CPU ThreadSanitizer: the complete C++ suite passed with parallel HNSW
  construction and querying enabled.
- macOS CPU without Metal/OpenMP: both C++ and float32 smoke suites passed.
- macOS Metal without OpenMP: Metal, C++, and float32 smoke suites passed.
- Linux CUDA 13 on chiamaka, without FAISS/cuVS/RAFT/RMM: both C++ and float32
  binary smoke suites passed in a fresh build.

## Direct public accelerator entry points

Cycle 18 added direct tests for every public CUDA and Metal family, including
PLS-LDA prediction, graph-supplied Core execution, raw-data and graph-input
KODAMA.matrix paths, UMAP, openTSNE, PCA, preprocessing, resident IVF, and graph
correction. The complete source-to-test map and the invariant checked by each
test are in `accelerator-entrypoint-validation.md`. A fresh CUDA 13.2 build on
an NVIDIA GeForce RTX 5060 Ti passed the selected standalone, public-API, and
float32 tests; the physical Apple M3 Metal suite passed locally.

The PageBlocks full-cycle shared-graph control at `M=Tcycle=100` retained close
CPU4/Metal median CV accuracy for both classifiers. It also exposed an
important crossover: Metal KNN was approximately neutral (2.722 versus 2.778
seconds), while Metal PLS-LDA was slower (118.383 versus 13.655 seconds).
These rows are retained as negative performance and visualization controls.

Cycle 20 extends the same shared-graph protocol to PenDigits (10,992 by 16).
CPU4/Metal KNN optimization required 4.397/4.925 seconds; PLS-LDA required
39.311/199.897 seconds. Median CV accuracy remained close within each
classifier, but every KODAMA truth silhouette was negative versus 0.5116 for
classic UMAP. This is retained as an adverse graph-correction and Metal-PLS
crossover control, not tuned away after inspection. The expanded fixed-seed,
anti-collapse, shake, and degenerate-PLS contracts pass the physical Metal
suite.

Cycle 21 adds COIL20 (1,440 by 16,384) as a high-dimensional KNN control at
`M=Tcycle=100`, 1,080 effective landmarks, and a CPU HNSW graph. CPU4/Metal
optimization wall time was 193.377/80.753 seconds, a 2.39x Metal speedup;
complete pipeline time was 200.950/88.315 seconds. Median CV accuracy was
0.99537 on both backends. The KODAMA truth silhouettes were nevertheless below
classic UMAP, so this is acceleration evidence only. Raw CPU4 PLS-LDA was
censored after 300 seconds before the first four runs completed. The R wrapper
now preserves per-run landmark timings and sum/mean/median aggregates so this
cost is no longer hidden inside the optimization wall time.

Cycle 22 added the first SatImage (6,435 by 36) `M=Tcycle=100`, seed-4,
shared-CPU-graph control. It exposed two lifecycle defects rather than a graph
or mathematical defect: a single resident fold slot caused repeated Metal
eviction, and four fixed independent-run lanes underused available memory.
The accepted correction uses a bounded 16-entry LRU fold cache and chooses run
concurrency from the device working-set budget. With the same graph and seed,
CPU4/Metal optimization times are now 2.963/2.442 seconds for KNN and
56.484/171.675 seconds for PLS-LDA. Metal KNN is 1.21x faster than CPU4 and
Metal PLS-LDA is 1.93x faster than its pre-fix path, but remains 3.04x slower
than CPU4 for this 36-variable regime. Median CV accuracy remains close
(0.99834/0.99772 for KNN and 0.94883/0.94893 for PLS-LDA), no run collapses,
and selected-label PLS-LDA ARI is 0.493/0.581. Truth-label UMAP silhouettes
remain below classic UMAP, so this is backend lifecycle and crossover evidence,
not a universal visualization gain.

Three follow-up PLS-LDA micro-optimizations were rejected. A single-threadgroup
fused projection kernel underutilized the GPU and was 4.62x slower on a short
SatImage probe. Replacing the explicit score norm by an algebraic identity was
faster on SatImage and USPS but 13.2% slower on MetRef and changed the float32
trajectory. Caching MPS matrix descriptors helped USPS but slowed MetRef and
SatImage. None remains in production code.
