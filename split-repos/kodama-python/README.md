# kodama-python

Thin Python wrapper for the standalone `kodama-cpp` C++/CUDA/Metal library.

The Python package exposes the complete public numerical surface of the
standalone core: KODAMA optimization, KNN and PLS-LDA CV, general SIMPLS,
randomized SVD, float32 PCA, preprocessing, graph utilities, UMAP, and
openTSNE. Numerical work remains in C++/CUDA/Metal.

## R/Python API Parity

The public Python functions mirror the R wrapper. Dotted R argument names use
the direct Python spelling with underscores, for example `n.cores` becomes
`n_cores`, `graph.neighbors` becomes `graph_neighbors`, and `raw.data` becomes
`raw_data`. Defaults, accepted choices, label encoding, one-based `best_run`,
and returned KODAMA fields are shared across wrappers.

| R | Python |
|---|---|
| `KNNCV()` | `kodama.KNNCV()` |
| `PLSLDACV()` | `kodama.PLSLDACV()` |
| `CoreKNN()` | `kodama.CoreKNN()` |
| `CorePLSLDA()` | `kodama.CorePLSLDA()` |
| `KODAMA.matrix()` | `kodama.KODAMA.matrix()` or `kodama.matrix()` |
| `KODAMA.matrix.graph()` | `kodama.KODAMA.matrix_graph()` or `kodama.matrix_graph()` |
| `KODAMA.graph()` | `kodama.KODAMA.graph()` or `kodama.graph()` |
| `KODAMA.pca()` | `kodama.KODAMA.pca()` or `kodama.kodama_pca()` |
| `KODAMA.visualization()` | `kodama.KODAMA.visualization()` or `kodama.visualization()` |
| `KODAMA.clustering()` | `kodama.KODAMA.clustering()` or `kodama.clustering()` |
| `KODAMA.timing()` | `kodama.KODAMA.timing()` or `kodama.timing()` |
| `KODAMA.diagnostics()` | `kodama.KODAMA.diagnostics()` or `kodama.diagnostics()` |

Python additionally exposes the broader standalone-core functions that are
intentionally maintained in separate R packages:

| Core capability | Python |
|---|---|
| General SIMPLS | `kodama.pls()` |
| Randomized SVD | `kodama.rsvd()` |
| PCA | `kodama.pca()` / `kodama.PCA()` |
| UMAP | `kodama.umap()` |
| openTSNE | `kodama.opentsne()` |

The compiled `_core` module is an implementation detail and is not part of the
public wrapper API.

For `backend="cpu"`, `n_cores` controls both native HNSW construction and
graph querying. Parallel builds are approximate; regression tests require at
least 0.99 recall against exact neighbors rather than bitwise equality with a
one-thread graph.
For `backend="cuda"` or `backend="metal"`, `n_cores=0` enables automatic
independent-`M` lane selection from available device resources and the
backend-specific workspace estimate. Positive values remain explicit. Matrix
results mirror R with `n_cores`, `gpu_auto_workers`,
`gpu_scheduler_enabled`, `gpu_scheduler_lanes`, and
`gpu_worker_memory_estimate_mb`.

## Development Install

Build `kodama-cpp` first:

```sh
cmake -S ../kodama-cpp -B ../kodama-cpp/build -DKODAMA_ENABLE_CUDA=OFF
cmake --build ../kodama-cpp/build -j
```

Then install this package:

```sh
python -m pip install -v . \
  --config-settings=cmake.define.KODAMA_CPP_ROOT=/path/to/kodama-cpp \
  --config-settings=cmake.define.KODAMA_CPP_BUILD_DIR=/path/to/kodama-cpp/build
```

To build a release wheel against an installed core, point CMake at the core's
package directory without replacing scikit-build's complete prefix path:

```sh
cmake --install /path/to/kodama-cpp/build --prefix /path/to/kodama-install
CMAKE_ARGS='-Dkodama-cpp_DIR=/path/to/kodama-install/lib/cmake/kodama-cpp' \
  python -m pip wheel . --no-deps --wheel-dir dist
```

Using `kodama-cpp_DIR` is intentional. Replacing `CMAKE_PREFIX_PATH` can hide
the isolated build environment's `pybind11` CMake package.

When linking a static `kodama-cpp`, pass any required external libraries through
`KODAMA_CPP_EXTRA_LIBS` or prefer a conda-forge installation of `kodama-cpp`.

On Linux CUDA systems, run Python with the same CUDA Toolkit and OpenMP
libraries used to build `kodama-cpp`. An interactive shell can use:

```sh
export CONDA_PREFIX=/path/to/cuda-env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so:$CONDA_PREFIX/lib/libstdc++.so.6"
```

Check the runtime with:

```python
import kodama
kodama.diagnostics()
```

## Recommended Workflow

```python
import numpy as np
import kodama

rng = np.random.default_rng(4)
x = np.vstack([
    rng.normal(1.0, 0.20, size=(250, 20)),
    rng.normal(3.0, 0.20, size=(250, 20)),
]).astype(np.float32)
labels = np.repeat(np.arange(2), 250)

normalized = kodama.normalization(x, method="pqn", backend="cpu", n_cores=4)
scaled = kodama.scaling(
    normalized["newXtrain"], method="autoscaling", backend="cpu", n_cores=4
)
analysis_x = scaled["newXtrain"]
pc = kodama.PCA(analysis_x, ncomp=20, backend="cpu")
assert pc["scores"].shape == (500, 20)

response = np.column_stack((analysis_x[:, 0], analysis_x[:, 1]))
pls_model = kodama.pls(analysis_x, response, ncomp=5, backend="cpu")
decomposition = kodama.rsvd(analysis_x, ncomp=20, backend="cpu")

coordinates = np.column_stack((np.linspace(0, 1, x.shape[0]), np.zeros(x.shape[0])))
slide_ids = np.zeros(x.shape[0], dtype=np.int32)
svg = kodama.spatial_feature_selection(
    x, coordinates, samples=slide_ids,
    n_cores=4,
)
top_features = svg["features"][:100]

prepared = kodama.graph(analysis_x, k=30, backend="cpu")
assert "data" not in prepared

kk = kodama.matrix(
    data=analysis_x,
    graph=prepared,
    classifier="knn",
    backend="cpu",
    M=10,
    Tcycle=20,
    knn_k=30,
    n_cores=4,
    return_graph=True,
)

kodama.timing(kk)
kodama_labels = kk.best_labels
um = kodama.visualization(
    kk,
    "UMAP",
    k=30,
    backend="cpu",
)
clu = kodama.clustering(um, n_iterations=10, random_walk_steps=4)
```

## AnnData, Squidpy, and SpatialData

The object adapters use the standard AnnData layout without importing the
single-cell ecosystem at module import time:

- expression is read from `adata.X` or a named `adata.layers` entry;
- coordinates are read from `adata.obsm["spatial"]` by default;
- slide identities can be read from a named `adata.obs` column;
- prepared Scanpy/Squidpy graphs are read from `adata.obsp`, preferring
  `spatial_distances`, `distances`, `spatial_connectivities`, or
  `connectivities`;
- KODAMA graphs and matrix results are stored in `adata.uns`, labels in
  `adata.obs`, embeddings in `adata.obsm`, and spatial-feature statistics in
  `adata.var`.

For SpatialData, pass `table_key` when more than one AnnData table exists:

```python
import kodama

kodama.RunKODAMAgraph(
    adata, spatial_key="spatial", sample_key="slide", backend="cpu"
)
kodama.RunKODAMAmatrix(
    adata, graph_key="kodama_graph", classifier="pls_lda",
    M=100, Tcycle=100, backend="cpu"
)
kodama.RunKODAMAvisualization(
    adata, result_key="kodama", embedding_key="X_kodama_umap",
    method="UMAP", backend="cpu"
)

squidpy_graph = kodama.graph_from_anndata(adata, k=100)
spatialdata_copy = kodama.RunKODAMAgraph(
    sdata, table_key="table", sample_key="slide", copy=True
)
```

Install optional object dependencies with `pip install .[singlecell]` or
`pip install .[spatial]`. The adapters are duck typed and therefore remain
usable with compatible versions of AnnData, Squidpy, and SpatialData.

`kodama.matrix()` returns a `KodamaMatrixResult`, which is still a normal
dictionary with the raw C++ fields (`res`, `acc`, `knn`, `timing`). Convenience
properties expose `best_labels`, `best_run`, `class_counts`, and `parameters`.
UMAP uses fuzzy graph weighting by default; pass `graph_mode="binary"` only
when binary graph compatibility is required.

`kodama.graph()` contains graph arrays and backend-matched PCA starts, but does
not retain the input array. `kodama.matrix()` reserves `data` for raw features
and `graph` for a prepared graph or bare `indices`/`distances` dictionary.
Either can be supplied alone, or both can be supplied together. Graph-only
PLS-LDA uses self-tuning Laplacian features; providing `data` uses the ordinary
data-input PLS-LDA geometry.

For multiple slides, pass one identifier per row to both stages:

```python
prepared = kodama.graph(x, spatial=xy, samples=slide_id)
fit = kodama.matrix(
    data=x, graph=prepared, spatial=xy, samples=slide_id,
    classifier="knn",
)
```

The first spatial coordinate is separated slide by slide using the original
KODAMA rule before spatial neighbors and constraints are constructed. A
single-level identifier vector leaves coordinates unchanged. Spatial landmark
strata and optimization constraints are also keyed by slide, so no constraint
group can contain rows from different slides.

The native `kodama.matrix()` call builds the full-data graph once before all
`M` searches. By default it also performs one backend-native float32 PCA and
stores separately scaled UMAP and openTSNE starts. `kodama.visualization()`
reuses the returned graph and start without another graph or PCA calculation.
Only `knn` is serialized; the former duplicate `base_knn` payload has been
removed. Inspect `graph_builds`, `knn_is_kodama_corrected`,
`graph_storage_bytes`, and `timing["visual_init_seconds"]` when profiling.
`KODAMA.graph()` returns a native capsule by default. Call
`kodama.graph_materialize(graph)` only when NumPy index and distance arrays are
needed; the capsule can be passed directly to `KODAMA.matrix()` and does not
retain the raw input matrix. The resident result graph is not materialized by
default. Set `return_graph=True`
when graph matrices or a later visualization call are required.
With `return_graph=False`, the labels-only path omits the otherwise-unused
final graph-distance correction.
Visualization prefers explicit `init`, then a stored initialization whose
backend matches the selected CPU/CUDA/Metal backend, then explicit `raw_data`.
UMAP and openTSNE support CPU, CUDA, and Metal. Pass the raw matrix when
changing backend:

```python
tsne_cpu = kodama.visualization(
    kk,
    "opentsne",
    raw_data=x,
    perplexity=30,
    backend="cpu",
)
```

The returned NumPy-compatible embedding exposes `initialization`,
`initialization_backend`, `backend`, `optimizer`, `runtime_seconds`,
`graph_edges`, and `graph_max_weight`. The graph fields describe the fuzzy
UMAP graph; they are zero for openTSNE because its sparse probabilities are a
different object.

## Benchmark

The maintained benchmark accepts a non-spatial NPZ file containing a float32
`data` matrix. Additional arrays such as `labels` are ignored by the timing
driver and are never passed to KODAMA:

```python
np.savez_compressed("MetRef.npz", data=x.astype(np.float32), labels=labels)
```

Then run:

```sh
python benchmarks/run_nonspatial.py MetRef.npz \
  --name MetRef --backends cpu,cuda --M 100 --Tcycle 100
```

The CSV reports `KODAMA.graph`, KNN/PLS-LDA `KODAMA.matrix`, UMAP, openTSNE,
and complete pipeline wall times separately for every requested backend.

For population-genetic samples with repeated collection latitude/longitude,
use `spatial_mode="population"`. The coordinate regularization is repeated
independently in every `M` run and precedes spatial constraint clustering;
`spatial_mode="standard"` remains the default. A genetics-only run without
`spatial` should be reported as the primary control.
