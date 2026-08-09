# kodama-python

Thin Python wrapper for the standalone `kodama-cpp` C++/CUDA/Metal library.

The Python package exposes KODAMA optimization, float32 PCA, graph utilities,
and UMAP/openTSNE while keeping numerical work in the standalone C++ core.

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

prepared = kodama.graph(analysis_x, k=30, backend="cpu")
assert "data" not in prepared

kk = kodama.matrix(
    data=analysis_x,
    graph=prepared,
    classifier="knn",
    backend="cpu",
    M=10,
    Tcycle=20,
    knn_k=10,
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
