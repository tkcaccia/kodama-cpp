# R API Map

The R package in `split-repos/KODAMA` contains conversion, validation, and S3
convenience code together with a synchronized portable CPU build of the
standalone core. Optional developer builds can link CUDA or Metal
`libkodama_cpp`.

| R function | C++ entry point | Purpose |
|---|---|---|
| `KNNCV()` | `kodama::KNNCV` | Cross-validated KNN classification |
| `PLSLDACV()` | `kodama::PLSLDACV` | SIMPLS plus latent-space LDA CV |
| `CoreKNN()` | `kodama::CoreKNN` | Optimize one label vector with KNN |
| `CorePLSLDA()` | `kodama::CorePLSLDA` | Optimize one label vector with PLS-LDA |
| `KODAMA.matrix()` | `kodama::KODAMAMatrix` | Complete independent-run ensemble |
| `KODAMA.matrix.graph()` | `kodama::KODAMAMatrixFromGraph` | KODAMA from supplied indices/distances |
| `KODAMA.graph()` | `kodama::KODAMAGraph` | Build a reusable neighbor graph and backend-matched PCA starts |
| `KODAMA.pca()` | `kodama::PCA` | Float32 randomized PCA |
| `KODAMA.visualization()` | `kodama::KODAMAUMAP_*`, `kodama::KODAMAOpenTSNE_*`, and `kodama::KODAMAVisualizationPCAInit` | Embed a graph with backend-native raw-data initialization by default |
| `KODAMA.clustering()` | `kodama::KODAMAGraphCluster` | Random-walk graph clustering |

For CPU graph construction, `n.cores` controls both lock-protected parallel
HNSW insertion and batched parallel graph querying.
For CUDA or Metal matrix optimization, `n.cores = 0` enables automatic
independent-`M` lane selection from the device and backend-specific workspace
estimate. A positive value remains an explicit lane count. Results expose
`n.cores`, `gpu_auto_workers`, `gpu_scheduler_enabled`,
`gpu_scheduler_lanes`, and `gpu_worker_memory_estimate_mb`.

Use `help(package = "KODAMA")` for argument-level R documentation. The
wrapper preserves raw result fields such as `acc`, `res`, `knn`, and `timing`,
and adds `best_labels`, `best_run`, `class_counts`, and `parameters`.
Matrix results also preserve one `landmark_seconds` value per independent
`M` run plus occupied/represented-stratum diagnostics. `KODAMA.timing()`
reports landmark sum, mean, and median separately from optimization wall and
classifier-core sums.

For reproducible evaluation, report at least the classifier, backend, `M`,
`Tcycle`, landmark count, splitting, component count or KNN `k`, graph-neighbor
count, seed, thread count, and whether KODAMA dissimilarity correction was
applied. External labels must not be used to choose the best run.

`KODAMA.graph()` defaults to `storage = "handle"` and owns one native
CPU/CUDA/Metal float32 graph behind an R external pointer. It retains PCA
starts, metadata, and timings, but not the input matrix. Use
`storage = "matrix"` to explicitly materialize graph arrays. The handle is consumed
directly by matrix optimization, visualization, and clustering; use
`KODAMA.graph.materialize()` only when R matrices are explicitly needed. The
matrix conversion uses a cache-blocked row-major/column-major transpose.
External handles are process-local and must be materialized before an object is
saved for use in another R session.

For multi-slide coordinate data, pass `samples` together with `spatial` to
`KODAMA.graph()` or `KODAMA.matrix()`. Following the original KODAMA contract,
the first spatial coordinate of each sample/slide is offset in sorted sample
order before the spatial graph and constraints are constructed. A one-level
`samples` vector leaves the coordinates unchanged. The separation is performed
centrally in float32, so CPU, CUDA, and Metal consume identical coordinates.
KODAMA also composes spatial landmark strata and constraint IDs with the sample
identifier. This is a hard boundary: an optimization block never contains
spots from different slides, even if their original coordinate ranges overlap.
`KODAMA.matrix()` reserves `data` for raw features and
`graph` for a prepared KODAMA graph or bare `indices`/`distances` graph. Either
argument can be supplied alone, or both can be supplied together. Graph-only
PLS-LDA uses disclosed self-tuning Laplacian features; supplying `data` uses
the ordinary data-input PLS-LDA geometry.

With raw input, `KODAMA.matrix()` builds its full-data KNN graph once before the `M` loop and
returns it as `knn`; the per-run landmark and fold structures remain local to
their independent searches. `graph_builds` reports the number of full-data
feature-graph constructions (`1` for matrix input and `0` for a supplied
graph). Final graph compaction and KODAMA distance correction reuse that same
C++ storage, so the wrapper serializes only `knn`, not a duplicate
`base_knn`. `knn_is_kodama_corrected` records its state and
`graph_storage_bytes` reports the retained native graph capacity. Use
`return.graph = "handle"` to retain that graph without two R matrices;
`timing$r_graph_conversion_seconds` reports wrapper conversion cost.
When `return.graph = FALSE`, final distance correction is not evaluated because
no corrected graph is observable; label optimization and accuracy are unchanged.

`KODAMA.matrix(..., visual.init = TRUE)` is the default. Inside that same
native call, one float32 PCA produces both the UMAP and openTSNE starts, which
are stored under `visual_init`; `timing$visual_init_seconds` reports its cost.
`KODAMA.visualization()` uses the returned `knn` directly and applies the
following initialization precedence: explicit `init`; a stored initialization
whose backend matches the requested CPU/CUDA/Metal visualization backend; PCA
initialization from explicit `raw.data` (or from a direct matrix input); then
the graph-only fallback. The returned matrix records `initialization` and
`initialization_backend` attributes.
