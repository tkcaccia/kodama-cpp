# Changelog

All notable changes to kodama-cpp are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- Added standalone float32 `Normalization` and `Scaling` APIs with explicit
  CPU, CUDA, and Metal entry points. The five KODAMA normalization methods
  (PQN, sum, median, Euclidean-norm, and none) and five variable-scaling
  methods (none, centering, autoscaling, range, and Pareto) preserve the
  original train/test rules. CPU rows and columns run concurrently; CUDA and
  Metal compute statistics, deterministic PQN medians, and transforms on
  device without a CPU fallback. R and Python now call these same kernels.
- Made CUDA and Metal KODAMA state persistent across proposal cycles and
  independent `M` runs. The full-data graph is uploaded once; each worker lane
  reuses landmark/fold, label, projection, voting, SIMPLS, and LDA allocations,
  refreshing only contents that mathematically change. CUDA SIMPLS now uses a
  dedicated cuBLAS workspace per stream and a race-free fixed-order float32
  label cross-product. The former hidden nonresident KODAMA switch was removed.
- Replaced the three-copy KODAMA graph lifecycle (`global_graph`, `knn`, and
  `base_knn`) with one owned graph buffer. Final trimming, optional graph
  fusion, and KODAMA dissimilarity correction now operate in place before the
  graph is moved into the result. `apply_kodama_dissimilarity=false` retains
  that one base graph, and `KODAMADissimilarityInPlace` supports later lazy
  correction. R and Python serialize only `knn` and expose
  `knn_is_kodama_corrected` plus `graph_storage_bytes`.
- Parallelized package-owned CPU HNSW construction and querying. Concurrent
  insertion uses per-node locks, distance evaluation and candidate expansion
  use reusable batches, graph queries use contiguous row batches, and input,
  adjacency, and cached build distances remain contiguous float32 arrays.
  Four-thread graph construction retained at least 0.99 exact-neighbor recall
  and was 4.11x faster than the previous four-thread implementation on the
  maintained 4000-by-32 benchmark.
- Build the full-data KNN graph once before all independent `M` searches and
  expose `graph_builds` as a lifecycle diagnostic. `KODAMA.matrix` now also
  performs one native float32 PCA that produces and stores both UMAP and
  openTSNE initializations; R/Python visualization reuses the returned graph
  and matching initialization without repeating either calculation. A
  backend-compatible stored start takes precedence over raw-data
  recomputation, even when the raw matrix is supplied again.
- Added an explicit move-only `ResidentIVFIndex` for CUDA and Metal. IVF
  assignments, centroid accumulation and repair, list-prefix construction,
  and ID scatter stay on-device; trained inputs and indexes can be reused by
  subsequent searches without a hidden global cache.
- Replaced the per-run `landmarks`-center k-means with exact-quota landmark
  sampling: coarse `splitting` strata for matrix input and an automatic
  population-weighted grid for 2D/3D coordinate input.
- Replaced the CPU/CUDA UMAP and openTSNE implementation with the current
  fastEmbedR kernels, including direct float32 CSR graph construction, the
  current smooth-kNN bandwidth rule, pooled CUDA workspaces, and the symmetric
  binary and fuzzy UMAP graph modes.
- Made fuzzy weighting the default UMAP graph mode across the C++, R, and
  Python APIs; binary weighting remains explicitly selectable.
- Added a standalone float32 randomized PCA API with CPU, CUDA, and Metal
  entry points plus thin R and Python bindings.
- Aligned the public Python wrapper with the R API across function signatures,
  defaults, accepted choices, one-based best-run reporting, and result fields;
  dotted R names use direct snake_case spellings in Python.
- Complete the tagged release archive, checksum, and archival DOI.
- Complete the final historical-R, backend, and visualization validation on
  the release hardware.

## [0.1.0] - 2026-07-16

Initial standalone release candidate.

- Added float32 KNN and SIMPLS + PLS-LDA cross-validation kernels.
- Added independent KODAMA KNN and PLS-LDA label-optimization paths.
- Added matrix, matrix-plus-graph, and graph-only KODAMA entry points.
- Added package-owned CPU HNSW, CUDA exact/IVF-Flat, and Metal exact/IVF-Flat
  nearest-neighbor backends.
- Added CPU/CUDA UMAP and openTSNE visualization primitives.
- Added thin R and Python wrappers around the same typed C++ interface.
- Added strict backend reporting, public API compile-link tests, provenance
  records, SPDX checks, and retained third-party license texts.

[Unreleased]: https://github.com/tkcaccia/kodama-cpp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tkcaccia/kodama-cpp/releases/tag/v0.1.0
