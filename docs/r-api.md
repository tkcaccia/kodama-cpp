# R API Map

The R package in `split-repos/kodama-r` contains only conversion, validation,
and S3 convenience code. Numerical work remains in `libkodama_cpp`.

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

Use `help(package = "kodamaR")` for argument-level R documentation. The
wrapper preserves raw result fields such as `acc`, `res`, `knn`, and `timing`,
and adds `best_labels`, `best_run`, `class_counts`, and `parameters`.

For reproducible evaluation, report at least the classifier, backend, `M`,
`Tcycle`, landmark count, splitting, component count or KNN `k`, graph-neighbor
count, seed, thread count, and whether KODAMA dissimilarity correction was
applied. External labels must not be used to choose the best run.

`KODAMA.graph()` stores graph arrays, PCA starts, metadata, and timings, but not
the input matrix. `KODAMA.matrix()` accepts a prepared KODAMA graph alone, a
prepared or bare graph plus `raw.data`, a bare `indices`/`distances` graph, or a
raw matrix. Graph-only PLS-LDA uses disclosed self-tuning Laplacian features;
supplying `raw.data` uses the ordinary data-input PLS-LDA geometry.

With raw input, `KODAMA.matrix()` builds its full-data KNN graph once before the `M` loop and
returns it as `knn`; the per-run landmark and fold structures remain local to
their independent searches. `graph_builds` reports the number of full-data
feature-graph constructions (`1` for matrix input and `0` for a supplied
graph). Final graph compaction and KODAMA distance correction reuse that same
C++ storage, so the wrapper serializes only `knn`, not a duplicate
`base_knn`. `knn_is_kodama_corrected` records its state and
`graph_storage_bytes` reports the retained native graph capacity.

`KODAMA.matrix(..., visual.init = TRUE)` is the default. Inside that same
native call, one float32 PCA produces both the UMAP and openTSNE starts, which
are stored under `visual_init`; `timing$visual_init_seconds` reports its cost.
`KODAMA.visualization()` uses the returned `knn` directly and applies the
following initialization precedence: explicit `init`; a stored initialization
whose backend matches the requested CPU/CUDA visualization backend; PCA
initialization from explicit `raw.data` (or from a direct matrix input); then
the graph-only fallback. The returned matrix records `initialization` and
`initialization_backend` attributes.
