# Backend Validation

This file records implementation decisions that were tested rather than
accepted from timing intuition alone.

## Accepted

### Package-native CPU HNSW

The default CPU KNN path no longer requires FAISS. On MetRef (873 samples,
375 variables, five folds, `k=10`) it reproduced the reference accuracy of
0.827033. The dependency-free single-thread build had a five-run median of
11.145 s.

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

- macOS CPU without Metal/OpenMP: both C++ and float32 smoke suites passed.
- macOS Metal without OpenMP: Metal, C++, and float32 smoke suites passed.
- Linux CUDA 13 on chiamaka, without FAISS/cuVS/RAFT/RMM: both C++ and float32
  binary smoke suites passed in a fresh build.
