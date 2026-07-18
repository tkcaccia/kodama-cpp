# kodama-cpp

Standalone C++17 kernels for KODAMA optimization, cross-validation, graph
construction, embedding, and clustering. The core does not depend on R or
Python; thin wrappers live in `tkcaccia/kodama-r` and
`tkcaccia/kodama-python`.

The current source-compatible release candidate is `0.1.0`. See
[`API_STABILITY.md`](API_STABILITY.md), [`CHANGELOG.md`](CHANGELOG.md), and
[`CITATION.cff`](CITATION.cff). Release artifacts are created only from a clean
tagged tree using [`tools/make_release_archive.sh`](tools/make_release_archive.sh).

## Backends

| Backend | Required libraries | Native functionality |
| --- | --- | --- |
| CPU | C++17 standard library | float32 HNSW KNN, KODAMA KNN/PLS-LDA, SIMPLS/LDA, randomized PCA, graph, UMAP/openTSNE, random-walk clustering |
| CPU + OpenMP | OpenMP compiler runtime | Optional parallel CPU loops |
| Apple Metal | Foundation, Metal, MetalPerformanceShaders | exact and IVF-Flat KNN, k-means, KODAMA KNN/PLS-LDA, label-aware SIMPLS/LDA, randomized PCA |
| CUDA | CUDA toolkit | package-owned exact and recall-tuned IVF-Flat KNN, k-means, float32 label-aware SIMPLS/LDA, KODAMA KNN/PLS-LDA, randomized PCA, UMAP/openTSNE |

The CPU, Metal, and CUDA numerical backends do not link FAISS, cuVS, RAFT,
RMM, fastEmbedR, fastPLS, R, or Python. Apple Metal uses only system
frameworks; CUDA is opt-in and uses CUDA Toolkit libraries. Backend names are
strict: an unavailable accelerator raises an error rather than silently running
CPU code under a GPU label.

## Public Scope

- `KNNCV`, `KNNCV_CPU`, `KNNCV_CUDA`, `KNNCV_METAL`
- `PLSLDACV`, `PLSLDACV_CPU`, `PLSLDACV_CUDA`, `PLSLDACV_METAL`
- `CoreKNN_*` and `CorePLSLDA_*`
- `KODAMAMatrix_*`, including matrix+graph and graph-input variants
- `PCA`, `PCA_CPU`, `PCA_CUDA`, `PCA_METAL`
- KNN graph construction, UMAP/openTSNE, and CPU random-walk clustering

KODAMA optimization exposes only the KNN and PLS-LDA classifiers. Numerical
data matrices, PLS/LDA workspaces, and accelerator buffers use float32;
public timing and summary statistics use double precision.

PLS-cKNN is not part of the library API. The CUDA SIMPLS adapter is a native
float32 implementation and does not require Armadillo or a legacy double
fallback.

Metal currently covers nearest-neighbor search, k-means, cross-validation,
and the matrix-based KODAMA KNN/PLS-LDA optimization. Graph-only spectral
feature construction is performed on the CPU before Metal optimization.
UMAP/openTSNE remain explicit CPU/CUDA operations. Random-walk clustering is
CPU-only; callers may construct its input graph with CPU, CUDA, or Metal before
calling the clustering kernel explicitly.

## PCA and visualization

`PCA` is a standalone float32 randomized PCA implementation with CPU, CUDA,
and Metal entry points. It returns scores, loadings, singular values, explained
variance, preprocessing vectors, backend metadata, and runtime. Automatic
oversampling and power-iteration policies match the current fastEmbedR/fastPLS
strategy; callers may set both values explicitly for reproducibility.

The CPU and CUDA UMAP/openTSNE implementations track fastEmbedR commit
`c98b9ecd124d442f20f849ce1be7f5bd4c13d0db`. UMAP defaults to the current
fastEmbedR symmetric binary neighbor graph. Binary and fuzzy graph modes use
its direct float32 CSR construction, smooth-kNN bandwidth rule, and epoch
schedule without per-row graph objects. Set
`UMAPOptions::graph_mode = UMAPGraphMode::Fuzzy` to retain fuzzy UMAP edge
weights. The openTSNE default perplexity is 30. Neither embedding path links
fastEmbedR or an R runtime.

## Build

Dependency-free CPU build:

```sh
cmake -S . -B build \
  -DKODAMA_ENABLE_CUDA=OFF \
  -DKODAMA_ENABLE_METAL=OFF \
  -DKODAMA_ENABLE_OPENMP=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Apple Metal build:

```sh
cmake -S . -B build-metal \
  -DKODAMA_ENABLE_METAL=ON \
  -DKODAMA_ENABLE_CUDA=OFF
cmake --build build-metal -j
ctest --test-dir build-metal --output-on-failure
```

CUDA build, using an installed CUDA toolkit:

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

## CUDA KNN

`CudaExact` and `CudaIVFFlat` are implemented in
`src/native_cuda_backend.cu`. The IVF path uses a package-owned signed-hash
projection, float32 GPU k-means, resident inverted lists, and an exact pilot
to verify recall. Automatic probing starts at
`2 * ceil(sqrt(nlist))` and increases only when the pilot target is not met.
Callers can still set `ivf_nlist` and `ivf_nprobe` explicitly.

No FAISS, cuVS, RAFT, or RMM headers, symbols, or binaries are required by
this path.

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
`ef4aa0e4ea663a097fb3c10f6a4ce9f2884a278f`. The package-owned CUDA search
is informed by the algorithmic organization of MIT-licensed FAISS and
Apache-2.0-licensed cuVS, without copying or linking their source. See
[`PROVENANCE.md`](PROVENANCE.md),
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and `licenses/`. The audit
maps the historical Git identity `tkcaccia <tkcaccia@gmail.com>` to the full
name Stefano Cacciatore; the publication contact is
`stefano.cacciatore@icgeb.org`.

## License

Original kodama-cpp code is MIT licensed. Identified compatible third-party
portions retain their original terms; in particular, `src/metal_backend.mm`
is marked `MIT AND Apache-2.0`. See `LICENSE`, `PROVENANCE.md`,
`THIRD_PARTY_NOTICES.md`, and `licenses/`.

Run the machine-checkable source and retained-license audit with:

```sh
bash tools/check_license_headers.sh
```

Contributor and release procedures are documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md). The implementation delta from
the historical KODAMA R package is recorded in
[`docs/previous-version-delta.md`](docs/previous-version-delta.md).
