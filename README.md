# kodama-cpp

Standalone C++17 kernels for KODAMA optimization, cross-validation, graph
construction, embedding, and clustering. The core does not depend on R or
Python; thin wrappers live in `tkcaccia/kodama-r` and
`tkcaccia/kodama-python`.

## Backends

| Backend | Required libraries | Native functionality |
| --- | --- | --- |
| CPU | C++17 standard library | float32 HNSW KNN, KODAMA KNN/PLS-LDA, SIMPLS/LDA, graph, UMAP/openTSNE, clustering |
| CPU + OpenMP | OpenMP compiler runtime | Optional parallel CPU loops |
| Apple Metal | Foundation, Metal, MetalPerformanceShaders | exact and IVF-Flat KNN, k-means, KODAMA KNN/PLS-LDA, label-aware SIMPLS/LDA |
| CUDA | CUDA toolkit, FAISS GPU, RAPIDS cuVS; cuGraph optional | CUDA KNN, k-means, float32 label-aware SIMPLS/LDA, KODAMA KNN/PLS-LDA, UMAP/openTSNE; optional CUDA clustering |

The default CPU build does not link FAISS, cuVS, fastEmbedR, fastPLS, R, or
Python. Apple Metal uses only system frameworks. CUDA is opt-in and currently
links installed FAISS GPU and RAPIDS cuVS, matching fastEmbedR's native CUDA
build model; fastEmbedR does not vendor those libraries either. Backend names
are strict: an unavailable accelerator raises an error rather than silently
running CPU code under a GPU label.

## Public Scope

- `KNNCV`, `KNNCV_CPU`, `KNNCV_CUDA`, `KNNCV_METAL`
- `PLSLDACV`, `PLSLDACV_CPU`, `PLSLDACV_CUDA`, `PLSLDACV_METAL`
- `CoreKNN_*` and `CorePLSLDA_*`
- `KODAMAMatrix_*`, including matrix+graph and graph-input variants
- KNN graph construction, UMAP/openTSNE, Louvain/Leiden/random-walk clustering

KODAMA optimization exposes only the KNN and PLS-LDA classifiers. Numerical
data matrices, PLS/LDA workspaces, and accelerator buffers use float32;
public timing and summary statistics use double precision.

PLS-cKNN is not part of the library API. The CUDA SIMPLS adapter is a native
float32 implementation and does not require Armadillo or a legacy double
fallback.

Metal currently covers nearest-neighbor search, k-means, cross-validation,
and the matrix-based KODAMA KNN/PLS-LDA optimization. Graph-only spectral
feature construction is performed on the CPU before Metal optimization.
UMAP/openTSNE and community clustering remain explicit CPU/CUDA operations;
requesting unavailable Metal clustering raises an error instead of silently
falling back to the CPU.

## Build

Dependency-free CPU build:

```sh
cmake -S . -B build \
  -DKODAMA_ENABLE_CUDA=OFF \
  -DKODAMA_ENABLE_FAISS=OFF \
  -DKODAMA_ENABLE_METAL=OFF \
  -DKODAMA_ENABLE_OPENMP=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Apple Metal build:

```sh
cmake -S . -B build-metal \
  -DKODAMA_ENABLE_METAL=ON \
  -DKODAMA_ENABLE_CUDA=OFF \
  -DKODAMA_ENABLE_FAISS=OFF
cmake --build build-metal -j
ctest --test-dir build-metal --output-on-failure
```

CUDA build, after activating the environment containing FAISS/cuVS:

```sh
cmake -S . -B build-cuda \
  -DKODAMA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc"
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

`KODAMA_ENABLE_OPENMP` defaults to off on macOS and on elsewhere. It remains
optional so a base R/Python package can compile without a Homebrew OpenMP
dependency.

## Metal KNN

`MetalExact` is the default and preserves exact neighbors for `k <= 128`.
`MetalIVFFlat` is an explicit large-input alternative ported from the
fastEmbedR IVF organization. With `ivf_nprobe = 0`, a deterministic exact
pilot selects `nprobe` against a 0.999 recall target; callers can provide
`ivf_nlist` and `ivf_nprobe` explicitly.

```cpp
kodama::KNNOptions options;
options.backend = kodama::Backend::Metal;
options.index_type = kodama::KNNIndexType::MetalIVFFlat;
options.k = 30;

auto result = kodama::KNNCV_METAL(x, labels, {}, options);
```

IVF remains opt-in because index training can outweigh search savings on
small and medium KODAMA runs. See
[`docs/backend-validation.md`](docs/backend-validation.md) for acceptance
results.

## Install

```sh
cmake --install build --prefix /path/to/kodama-cpp-install
```

The install exports a relocatable CMake package:

```cmake
find_package(kodama-cpp CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE kodama::kodama_cpp)
```

Shared builds are enabled with `-DBUILD_SHARED_LIBS=ON`. They are generally
the most convenient artifacts for R and Python binary wrappers.

## Provenance

The native CPU and Metal work was adapted from
[`tkcaccia/fastEmbedR`](https://github.com/tkcaccia/fastEmbedR) commit
`c98b9ecd124d442f20f849ce1be7f5bd4c13d0db`; Metal SIMPLS/LDA was adapted
from [`tkcaccia/fastPLS`](https://github.com/tkcaccia/fastPLS) commit
`ef4aa0e4ea663a097fb3c10f6a4ce9f2884a278f`. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and `licenses/`.

## License

MIT. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
