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
import kodama

pc = kodama.PCA(x, ncomp=20, backend="cpu")
pc["scores"].shape

prepared = kodama.graph(x, k=30, backend="cuda")
assert "data" not in prepared

kk = kodama.matrix(
    data=x,
    graph=prepared,
    spatial=spatial,
    classifier="knn",
    backend="cuda",
    M=100,
    Tcycle=20,
    knn_k=30,
    n_cores=4,
)

kodama.timing(kk)
labels = kk.best_labels
um = kodama.visualization(
    kk,
    "UMAP",
    k=30,
    backend="cuda",
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
Visualization prefers explicit `init`, then a stored initialization whose
backend matches the selected CPU/CUDA backend, then explicit `raw_data`. Pass
the raw matrix when changing backend:

```python
tsne_cpu = kodama.visualization(
    kk,
    "opentsne",
    raw_data=x,
    perplexity=30,
    backend="cpu",
)
```

The returned NumPy-compatible embedding exposes `initialization` and
`initialization_backend` attributes reporting the selected path.

## Benchmark

First export RData datasets to CSV:

```sh
Rscript scripts/export_spatial_rdata.R
```

Then run:

```sh
python benchmarks/run_br8100_merfish.py --input-dir exported-spatial
```
