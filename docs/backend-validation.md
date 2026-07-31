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

### KODAMA-resident CUDA and Metal state

The complete KODAMA accelerator lifecycle now owns one resident full-data
graph and one scratch lane per concurrent `M` worker. A lane retains
landmark/fold matrices, compact labels, projected labels, KNN voting arrays,
PLS weights, cross-products, latent projections, and LDA workspaces.
Landmark membership and fold contents change between independent runs, so
their existing allocations are refreshed once per `M`; they are not
reallocated or retransferred for every proposal cycle. Graph projection and
final all-run dissimilarity correction execute against the same resident graph
and download only the final corrected rows.

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
construction, and k-means are native Metal operations. The graph-only API
first constructs sparse spectral features on the CPU and reports that time
separately as `graph_feature_seconds`; subsequent KODAMA optimization can run
on Metal. Native Metal UMAP/openTSNE and community clustering are not claimed
by this implementation.

## Regression Suites

- macOS CPU ThreadSanitizer: the complete C++ suite passed with parallel HNSW
  construction and querying enabled.
- macOS CPU without Metal/OpenMP: both C++ and float32 smoke suites passed.
- macOS Metal without OpenMP: Metal, C++, and float32 smoke suites passed.
- Linux CUDA 13 on chiamaka, without FAISS/cuVS/RAFT/RMM: both C++ and float32
  binary smoke suites passed in a fresh build.
