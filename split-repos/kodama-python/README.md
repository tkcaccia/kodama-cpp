# kodama-python

Thin Python wrapper for the standalone `kodama-cpp` C++/CUDA library.

The Python package exposes `kodama.matrix()` and keeps all numerical work in
the C++/CUDA core.

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

kk = kodama.matrix(
    x,
    spatial=spatial,
    classifier="knn",
    backend="cuda",
    M=100,
    Tcycle=20,
    knn_k=30,
)

kodama.timing(kk)
labels = kk.best_labels
um = kodama.visualization(kk, "UMAP", k=30, backend="cuda")
clu = kodama.clustering(um, n_iterations=10, random_walk_steps=4)
```

`kodama.matrix()` returns a `KodamaMatrixResult`, which is still a normal
dictionary with the raw C++ fields (`res`, `acc`, `knn`, `timing`). Convenience
properties expose `best_labels`, `best_run`, `class_counts`, and `parameters`.

## Benchmark

First export RData datasets to CSV:

```sh
Rscript scripts/export_spatial_rdata.R
```

Then run:

```sh
python benchmarks/run_br8100_merfish.py --input-dir exported-spatial
```
