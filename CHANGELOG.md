# Changelog

All notable changes to kodama-cpp are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- Restored the original multi-sample spatial contract across C++, R, and
  Python. `KODAMA.graph()` and `KODAMA.matrix()` now accept one `samples`
  identifier per row and separate coordinate 1 slide by slide before spatial
  graph and constraint construction. The centralized float32 transform is
  identical for CPU, CUDA, and Metal; one-level inputs are unchanged. Spatial
  landmark strata, singleton repair, and final optimization constraints are
  now sample-keyed, which guarantees that a constraint group cannot cross a
  slide boundary.
- Reduced accelerator KNN proposal overhead without changing voting. Active
  labels are encoded once as ordered compact class codes, dense code lookup is
  used when its table is no larger than the label vector, and prediction
  storage is reused across Tcycle. Metal retains its shared device buffers and
  now also reuses host prediction storage. A CUDA pinned-buffer candidate was rejected after
  matched M=Tcycle=100 tests showed no end-to-end benefit at the effective
  landmark sizes. Four-lane Metal MetRef KNN decreased from 2.055 to 1.550
  seconds with identical CV, ARI, and class summaries; matched CUDA changes
  across MetRef, Br8100, MERFISH, and flow18 remained within 0.8% of baseline.
- Added a shared opaque `KODAMAGraphHandle` PIMPL. `KODAMA.graph()` now returns
  an R external pointer or Python capsule by default, and C++/R/Python matrix
  calls reuse the same CPU/CUDA/Metal graph owner without a graph download and
  re-upload. One-based host indices and distances are created only by explicit
  materialization. Repeated calls reset immutable base graph state and are
  serialized per handle. At M=Tcycle=100, paired CUDA checks preserved all
  PLS-LDA summaries on MetRef, Br8100, and MERFISH and all KNN summaries on
  Br8100, MERFISH, and flow18. Flow18 split one 68.862-second handle pipeline
  into 26.974 seconds of reusable graph preparation and 41.888 seconds of
  matrix work, versus 68.595 seconds when built internally. Labels-only calls
  now omit unobservable final distance correction.
- Optimized R graph interchange with cache-blocked row-major/column-major
  conversion. Explicit materialization avoids the previous poorly ordered
  transpose, while handle-backed calls avoid index and double-distance R
  matrices entirely.
- Added a dataset-scoped execution context for KODAMA.matrix. The immutable
  float32 matrix, retained HNSW/IVF graph, PCA initialization, and one
  accelerator k-means workspace per M scheduler lane now survive across all M
  runs. Full-data coarse k-means uploads its input once; only initialization,
  centroids, assignments, and output labels change between runs. Landmark and
  spatially jittered k-means inputs remain transient because their contents
  change. CUDA and Metal regressions require assignment parity with the former
  one-shot path and exactly one invariant input upload.
- Fixed landmark KNN extraction for raw-matrix runs. CPU HNSW and CUDA/Metal
  IVF indexes are now retained after global graph construction and queried
  with a landmark membership map, expanding the candidate search until each
  row has `knn.k` landmark neighbors and retaining exact candidate distances.
  This replaces filtering a fixed-width global graph, which produced empty
  rows when landmarks were sparse. Graph-only input retains induced-subgraph
  semantics because a plain neighbor matrix has no searchable index.
- Made the CPU PLS-LDA fold path stream latent-score sufficient statistics
  instead of retaining complete training and validation score matrices. On
  TabulaMuris (100,102 by 50), five-fold 50-component PLSLDACV improved by
  1.47x and peak memory fell by 41%, with identical predictions and accuracy.
- Fixed a float32 range failure in the randomized SIMPLS power-vector norm.
  CPU and Metal preserve their original reduction whenever the squared norm is
  representable and use a double-accumulated norm only on overflow. MNIST70k
  now evaluates all 50 requested components: CPU accuracy changed from the
  erroneous one-component fallback at 0.112529 to 0.866457; Metal changed from
  a fit failure to 0.866286. A synthetic overflow regression runs on CPU and
  physical Metal.
- Reordered the row-major CPU PLS score projection for contiguous weight
  access. Three-dataset validation preserved accuracy and 50 selected
  components while improving median CPU runtime by 1.25x to 1.47x.
- Renamed inherited CUDA visualization entry points and runtime controls from
  `fastembedr_*` / `FASTEMBEDR_*` to KODAMA-owned names. Added a CTest source
  and linked-symbol audit so standalone kodama-cpp can coexist with fastEmbedR
  without process-level CUDA symbol or configuration collisions.
- Added checksum-based R-wrapper invalidation so changing the selected core
  archive or public headers rebuilds and relinks both Rcpp bridge objects,
  preventing stale CPU/CUDA/Metal development installs.
- Fixed Metal PLS-LDA resident fold identity when sequential temporary fold
  matrices reuse a host address. Each fold now receives a deterministic
  worker-local residency epoch. A native regression requires identical
  one-thread and four-thread predictions. On raw COIL20 with 16,384 predictors,
  the fix raised the erroneous one-thread five-fold accuracy from 0.555556 to
  the four-thread value 0.959028 without changing SIMPLS or LDA mathematics.
- Corrected Metal automatic M-run scheduling to estimate the buffers Metal
  actually retains instead of charging a CUDA-style predictor Gram matrix.
  PLS-LDA now accounts for one landmark matrix traversal per CV fold and uses
  the device recommended working-set budget; `n.cores=0` selected three lanes
  for raw COIL20 (16,384 variables) and four for MetRef (375 variables) on an
  8 GB Apple M3. Explicit worker counts remain authoritative. R and Python now
  expose the selected lanes, automatic-selection flag, and memory estimate.
- Vectorized the package-owned CPU k-means distance and norm kernels with
  AArch64 NEON and x86-64 SSE2, retaining the scalar fallback, seeds, centroid
  updates, tie-breaking, and iteration count. On COIL20 (1,440 by 16,384),
  four-core KODAMA-KNN at M=20/Tcycle=20 decreased from 188.579 to 42.421
  seconds (4.45x) for seed 4. Across paired seeds 4, 17, and 42, median speedup
  was 4.42x and median SIMD-minus-scalar changes were 0.00000 for best CV,
  +0.00324 for median CV, +0.00559 for best ARI, -0.00060 for truth-label
  silhouette, and +0.00365 for label silhouette. Because SIMD reassociates
  float32 reductions, exact stochastic trajectories are not claimed.
- Replaced CUDA grid-KNN's one-thread-per-query insertion top-k with an exact
  warp-cooperative shared-memory merge. Four warps process four queries per
  block, batch cell candidates, and retain deterministic distance/index order
  through bitonic merge-selection. The compiled kernel reports zero local
  memory; at one million 2D points and `k=100`, graph construction decreased
  from 3.618 to a 1.482-second repeated-run median (2.44x against the matched
  baseline), with exact CPU parity in dedicated 2D and 3D `k=100` tests.
- Added automatic accelerator dispatch for the full-data KODAMA graph. CUDA
  and Metal retain exact search for small matrices and otherwise use a
  resident IVF-Flat index tuned to a fixed 0.99 exact-pilot recall target.
  Graph results report the selected index, `nlist`, `nprobe`, and pilot recall.
  On flow18, CUDA graph construction fell from 296.482 to 26.846 seconds;
  M=100/Tcycle=100 KNN and PLS-LDA totals fell from 318.588/357.032 to
  48.377/65.960 seconds, with unchanged best CV accuracy and PLS-LDA ARI
  changing from 0.926970 to 0.924996.
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
- Added runtime-dispatched AVX2/FMA distance evaluation to package-owned CPU
  HNSW while retaining the auto-vectorized NEON/SSE portable fallback. On the
  maintained 4,000-by-32 benchmark, x86-64 medians improved by 6.7% with one
  thread and 7.0% with four threads, with single/four-thread graph overlap of
  1.000. CPU graph correction now reuses one sorting buffer per OpenMP worker.
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
