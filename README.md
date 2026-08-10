# kodama-cpp

[![CI](https://github.com/tkcaccia/kodama-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/tkcaccia/kodama-cpp/actions/workflows/ci.yml)
[![CPU coverage](https://github.com/tkcaccia/kodama-cpp/actions/workflows/coverage.yml/badge.svg)](https://github.com/tkcaccia/kodama-cpp/actions/workflows/coverage.yml)
[![API documentation](https://github.com/tkcaccia/kodama-cpp/actions/workflows/docs.yml/badge.svg)](https://github.com/tkcaccia/kodama-cpp/actions/workflows/docs.yml)

**Standalone float32 KODAMA for CPU, NVIDIA CUDA, and Apple Metal.**

`kodama-cpp` discovers informative sample structure by evolving label vectors
that maximize held-out classification accuracy. Independent solutions are then
combined into a KODAMA graph that can be analyzed with UMAP or openTSNE.

The numerical core is a C++17 library. It does not require R or Python, and it
does not link FAISS, cuVS, RAFT, fastEmbedR, or fastPLS. Thin R and Python
wrappers are included in this repository.

> **Release status:** `0.1.0` is a source release candidate. It is not yet on
> CRAN, Bioconductor, or PyPI. Install it from this repository as described
> below.

## Start here

| I want to... | Start with... |
| --- | --- |
| Try the C++ library | [Five-minute CPU build](#five-minute-cpu-build) |
| Use KODAMA from R | [Install the R wrapper](#install-the-r-wrapper) |
| Use KODAMA from Python | [`split-repos/kodama-python/README.md`](split-repos/kodama-python/README.md) |
| Build for NVIDIA CUDA or Apple Metal | [Accelerator builds](#accelerator-builds) |
| Inspect the stable public surface | [`API_STABILITY.md`](API_STABILITY.md) |
| Browse the generated API and tutorials | [kodama-cpp documentation](https://tkcaccia.github.io/kodama-cpp/) |

The R and Python wrappers currently live under `split-repos/`. They are being
prepared as independent repositories, but the embedded copies are the
authoritative wrappers for this release candidate.

## What is included

KODAMA optimization supports exactly two classifiers:

- **KNN**, using package-owned nearest-neighbor implementations.
- **PLS-LDA**, using SIMPLS followed by LDA in the latent component space.

The public library also provides:

- cross-validation kernels: `KNNCV` and `PLSLDACV`;
- label optimization: `CoreKNN` and `CorePLSLDA`;
- complete KODAMA matrix construction from data or a supplied KNN graph;
- explicit resident CUDA/Metal IVF indexes for repeated nearest-neighbor calls;
- persistent CUDA/Metal KODAMA workspaces for fold matrices, classifier state,
  projected labels, and the full-data graph across proposal cycles and `M` runs;
- randomized PCA with float32 data and CPU, CUDA, and Metal entry points;
- KODAMA-compatible float32 normalization and scaling with CPU, CUDA, and
  Metal entry points;
- KNN graph construction, UMAP, openTSNE, and random-walk clustering;
- thin R and Python bindings to the same C++ implementation.

Numerical matrices, PLS/LDA workspaces, and accelerator buffers use float32.
Timing and summary statistics use double precision. Requested feasible PLS
component counts are evaluated directly; the implementation does not search
for an internally selected "best" component count.

The package-owned CPU HNSW path uses `n_threads` for both graph construction
and graph querying. It stores vectors and adjacency arrays contiguously,
inserts nodes concurrently with per-node locking, evaluates expansion
candidates in reusable batches, and schedules query rows in contiguous
batches. The maintained quality test requires at least 0.99 recall against
exact neighbors. See [backend validation](docs/backend-validation.md) for
timing and race-detection evidence.

## Five-minute CPU build

Requirements:

- CMake 3.18 or newer;
- a C++17 compiler;
- Git.

```sh
git clone https://github.com/tkcaccia/kodama-cpp.git
cd kodama-cpp

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DKODAMA_ENABLE_CUDA=OFF \
  -DKODAMA_ENABLE_METAL=OFF \
  -DKODAMA_ENABLE_OPENMP=OFF

cmake --build build -j
./build/kodama_cv_example
ctest --test-dir build --output-on-failure
```

The example generates a small three-class dataset and prints the held-out
accuracy and runtime for `KNNCV` and `PLSLDACV`.

OpenMP is optional. It defaults to off on macOS and on elsewhere; set
`-DKODAMA_ENABLE_OPENMP=ON` explicitly when you want the multicore CPU loops.

## Install the R wrapper

The R package is named `kodamaR`. It compiles a small Rcpp bridge and links it
to the `kodama-cpp` library built above.

Install the R dependencies:

```sh
Rscript -e 'install.packages(c("Rcpp", "testthat"), repos = "https://cloud.r-project.org")'
```

On macOS, install the build tools and OpenMP runtime first when needed:

```sh
brew install cmake libomp
```

From the `kodama-cpp` repository root, build the CPU core and install the
embedded wrapper:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DKODAMA_ENABLE_CUDA=OFF \
  -DKODAMA_ENABLE_METAL=OFF
cmake --build build -j

KODAMA_CPP_ROOT="$PWD" \
KODAMA_CPP_BUILD_DIR="$PWD/build" \
R CMD INSTALL split-repos/kodama-r
```

Verify the linked package:

```r
library(kodamaR)
KODAMA.diagnostics()
```

### First KODAMA analysis in R

This example intentionally uses small `M` and `Tcycle` values so it finishes
quickly as an installation test:

```r
library(kodamaR)

set.seed(1)
x <- do.call(rbind, lapply(c(-2.5, 0, 2.5), function(mu) {
  matrix(rnorm(40 * 8, mean = mu, sd = 0.6), nrow = 40)
}))

fit <- KODAMA.matrix(
  x,
  classifier = "knn",
  backend = "cpu",
  M = 4,
  Tcycle = 4,
  landmarks = 90,
  splitting = 12,
  graph.neighbors = 30,
  knn.k = 30,
  progress = TRUE
)

print(fit)
KODAMA.timing(fit)
fit$landmark_seconds
head(fit$best_labels)

embedding <- KODAMA.visualization(
  fit,
  method = "UMAP",
  k = 15,
  backend = "cpu"
)

plot(
  embedding,
  col = fit$best_labels + 1L,
  pch = 19,
  xlab = "UMAP 1",
  ylab = "UMAP 2"
)
```

The graph preparation step can also be called and reused explicitly:

```r
prepared <- KODAMA.graph(
  x, k = 30, backend = "cpu", storage = "handle"
)
fit_graph <- KODAMA.matrix(
  graph = prepared,
  classifier = "knn",
  M = 4,
  Tcycle = 4
)
fit_graph_x <- KODAMA.matrix(
  data = x,
  graph = prepared,
  classifier = "pls_lda",
  M = 4,
  Tcycle = 4
)
```

`KODAMA.graph()` returns one KNN graph, backend-matched PCA starts for UMAP and
openTSNE, metadata, and timings. It deliberately does not retain `x`.
`storage = "handle"` keeps the native float32 graph behind an external pointer
that is consumed directly by matrix optimization and visualization. The
backward-compatible `storage = "matrix"` materializes R matrices using a
cache-blocked conversion; `KODAMA.graph.materialize()` performs that conversion
explicitly when needed.
`KODAMA.matrix()` reserves `data` for raw features and `graph` for a prepared
object or bare `indices`/`distances` graph. Either argument can be supplied
alone, or both can be supplied together. For graph-only PLS-LDA, self-tuning
Laplacian features provide the rectangular input; supplying `data` retains the
ordinary data-input PLS-LDA path.

With raw matrix input, `KODAMA.matrix()` implicitly performs the same graph
preparation step. It constructs its full-data neighbor graph once, before the
independent `M` searches, and returns that graph for direct reuse by
`KODAMA.visualization()`. By default the same native matrix call also performs
one backend-native float32 PCA and stores the separately scaled UMAP and
openTSNE starts. Visualization therefore performs neither neighbor search nor
PCA again when its backend matches the stored initialization. An explicit
`init` takes precedence, followed by a backend-matched stored start. Supplying
`raw.data = x` recomputes the start only when no compatible stored start is
available. Graph-only spectral UMAP or random openTSNE initialization is used
only when no matching raw-data start is available. The result fields
`graph_builds` and `timing$visual_init_seconds` make this lifecycle observable.
The C++ result owns only one graph buffer: final row compaction and KODAMA
distance correction occur in place, then the graph is moved into `knn`.
`knn_is_kodama_corrected` records whether correction was requested, and
`graph_storage_bytes` reports retained C++ capacity. The former duplicate
`base_knn` payload is no longer returned.

For CUDA and Metal, graph matrices are not returned by default. The graph stays
resident through correction and no wrapper-side matrix materialization occurs.
Retain a reusable graph without R matrices with `return.graph = "handle"`.
Request matrices explicitly with `return.graph = TRUE` in R or
`return_graph=True` in Python when visualization or direct graph access is
required.

CUDA and Metal upload the full-data graph once and keep it resident through
all independent `M` runs. Each worker lane owns persistent landmark/fold,
label, projection, PLS, LDA, and voting buffers. A selected landmark matrix
and its fold layouts are refreshed once when a lane starts a new `M` run, then
reused throughout its `Tcycle` proposals; label-dependent cross-products,
weights, and predictions overwrite existing device buffers rather than
allocating new ones. Only compact labels, predictions, accuracies, and the
final corrected graph cross the device boundary in the inner optimization
lifecycle.

For a full analysis, use enough independent searches and proposal cycles to
assess convergence. The release-validation configuration uses `M = 100` and
`Tcycle = 100`; these values are much more expensive than the smoke test.

Switch to PLS-LDA with:

```r
fit_pls <- KODAMA.matrix(
  x,
  classifier = "pls_lda",
  backend = "cpu",
  M = 100,
  Tcycle = 100,
  ncomp = min(50L, ncol(x))
)
```

### R API map

| Task | R function |
| --- | --- |
| Cross-validated KNN | `KNNCV()` |
| Cross-validated SIMPLS plus LDA | `PLSLDACV()` |
| Optimize an initial label vector | `CoreKNN()`, `CorePLSLDA()` |
| Run complete KODAMA | `KODAMA.matrix()` |
| Run KODAMA from KNN indices and distances | `KODAMA.matrix.graph()` |
| Compute float32 PCA | `KODAMA.pca()` |
| Build a reusable graph plus PCA starts | `KODAMA.graph()`, `makeSNNGraph()` |
| Compute UMAP or openTSNE | `KODAMA.visualization()` |
| Cluster a graph or embedding | `KODAMA.clustering()` |
| Inspect runtime by step | `KODAMA.timing()` |

Inside R, run `help(package = "kodamaR")` or open the help for a specific
function, for example `?KODAMA.matrix` or `?KODAMA.visualization`. The complete
source-install guide is in
[`split-repos/kodama-r/README.md`](split-repos/kodama-r/README.md).

## Accelerator builds

Backend names are strict. Requesting an unavailable accelerator raises an
error; the library never reports CUDA or Metal while silently executing the
CPU backend.

### Apple Metal

Metal uses the system `Foundation`, `Metal`, and `MetalPerformanceShaders`
frameworks:

```sh
cmake -S . -B build-metal \
  -DCMAKE_BUILD_TYPE=Release \
  -DKODAMA_ENABLE_METAL=ON \
  -DKODAMA_ENABLE_CUDA=OFF
cmake --build build-metal -j
ctest --test-dir build-metal --output-on-failure
```

Metal provides native nearest-neighbor search, k-means, KODAMA KNN/PLS-LDA,
SIMPLS/LDA, PCA, UMAP, and FFT-grid openTSNE. The visualization code mirrors
fastEmbedR's backend-specific execution: CPU uses the CSR UMAP epoch schedule,
CUDA uses the atomic COO/CSR epoch schedule, and Metal uses the clean row
sampler. openTSNE uses exact/FFT CPU paths and native CUDA/Metal FFT-grid
optimizers. Device discovery
falls back to enumerated Metal devices when the macOS default selector is
unavailable. Independent `M` workers retain separate command queues and float32
PLS workspaces, analogous to CUDA stream lanes.

### NVIDIA CUDA

Set `CUDACXX` to the `nvcc` belonging to the CUDA environment you intend to
use at runtime:

```sh
export CUDACXX=/path/to/nvcc

cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DKODAMA_ENABLE_CUDA=ON \
  -DKODAMA_ENABLE_METAL=OFF \
  -DCMAKE_CUDA_COMPILER="$CUDACXX"
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

Install the R wrapper against this build by changing only the build directory:

```sh
KODAMA_CPP_ROOT="$PWD" \
KODAMA_CPP_BUILD_DIR="$PWD/build-cuda" \
R CMD INSTALL split-repos/kodama-r
```

The R session must see the same CUDA runtime libraries used during the core
build. After installation, select CUDA with `backend = "cuda"`.

### Reuse an accelerator IVF index

The C++ API exposes a move-only owner for repeated searches against one
training matrix:

```cpp
kodama::KNNOptions options;
options.backend = kodama::Backend::CUDA; // or Backend::Metal
options.metric = kodama::DistanceMetric::Euclidean;
options.ivf_nlist = 256;
options.ivf_nprobe = 32;

auto index = kodama::BuildResidentIVFIndex(train, options);
auto graph = kodama::SearchResidentIVFIndexSelf(index, 30);
auto other = kodama::SearchResidentIVFIndex(index, query, 30);
```

The handle retains the float32 training matrix, projected matrix, centroids,
list offsets, and list identifiers on the selected device. Construction uses
device-side assignment accumulation, empty-cluster repair, prefix offsets,
and ID scatter. Self-search reuses the resident training matrix; an external
search uploads only its query matrix. The lifetime is explicit and there is
no process-global cache.

## Parameters that matter most

| Parameter | Meaning |
| --- | --- |
| `M` | Number of independent KODAMA label searches contributing to the final graph |
| `Tcycle` | Number of proposal and cross-validation cycles inside each independent search |
| `landmarks` | Exact effective sample count optimized directly; when `n <= landmarks`, KODAMA uses `ceiling(0.75 * n)` |
| `splitting` | Initial label classes and coarse landmark strata for matrix-only input |
| `knn.k` | K used by the KNN classifier, not the visualization graph |
| `ncomp` | Requested feasible SIMPLS components for PLS-LDA |
| `graph.neighbors` | Neighbors retained in the returned KODAMA graph |
| `backend` | Strict execution backend: CPU, CUDA, or an API-supported Metal path |

The KODAMA KNN classifier defaults to `knn.k = 30`. This does not change the
standalone KNNCV default, full-data graph width, or visualization neighborhood
size. The R wrapper
defaults `splitting` to 100 below 40,000 samples and 300
otherwise. UMAP uses `k = 30` by default; openTSNE uses perplexity 30.

Landmarks use exact population-proportional quotas. Matrix-only runs first
partition the rows into `splitting` coarse classes; 2D/3D coordinate input
instead uses an automatically sized regular grid. Integer quotas are sampled
without replacement and residual slots use randomized systematic rounding, so
the selected count is exact and no occupied grid cell is artificially
guaranteed a landmark. Initial labels are still fitted in expression space on
the selected rows.

## Use from another CMake project

Install the library:

```sh
cmake --install build --prefix /path/to/kodama-cpp-install
```

Then link the exported target:

```cmake
find_package(kodama-cpp CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE kodama::kodama_cpp)
```

Shared libraries can be built with `-DBUILD_SHARED_LIBS=ON`; they are usually
the most convenient artifacts for binary R and Python wrappers.

## Repository layout

```text
include/kodama/          Public C++ API
src/                     CPU, CUDA, and Metal implementations
examples/                Small runnable C++ programs
tests/                   Backend, API, and float32 validation
split-repos/kodama-r/    R wrapper package
split-repos/kodama-python/ Python wrapper package
docs/                    Backend and release-validation notes
benchmarks/              Reproducible benchmark drivers
manuscript/              JMLR manuscript sources and artifacts
```

## Implementation notes

The CPU, CUDA, and Metal UMAP/openTSNE implementations track fastEmbedR commit
`814350a5ca69b0c26e6df40377636f109055f84b`. UMAP uses fuzzy graph weighting
by default and retains symmetric binary graph weighting as an explicit
compatibility mode. The openTSNE default perplexity is 30. A controlled CPU
openTSNE comparison with identical neighbors, raw-data PCA initialization, and
settings is numerically identical. UMAP uses the same fuzzy graph, optimizer
equations, schedule, and initialization; later CPU coordinates need not be
bitwise identical because kodama-cpp keeps the iterative state in float32,
whereas the compared fastEmbedR R path stores its embedding in a double
matrix. Native Metal openTSNE uses the fastEmbedR FFT-grid optimizer with a
portable float-bit compare-and-swap implementation for grid accumulation.
See [`docs/visualization-parity.md`](docs/visualization-parity.md).

CUDA and Metal nearest-neighbor search is package-owned and provides exact
and recall-tuned IVF-Flat paths. Full-data graph preparation automatically
keeps exact search when `n <= 5000` or `n * n * p <= 2e8`; larger accelerator
problems use resident IVF-Flat tuned to a fixed 0.99 exact-pilot recall target.
Automatic `nlist` is `ceil(sqrt(n))`, bounded only by the backend list limit;
it is independent of the smaller `nprobe` kernel limit.
The graph result reports the selected index, `nlist`, `nprobe`, and pilot
recall. Both IVF builders keep assignments, centroid
accumulation, empty-cluster repair, and inverted-list construction on the
accelerator. No FAISS, cuVS, RAFT, or RMM headers or binaries are required.

IVF self-search writes directly into the KODAMA resident graph buffers, so the
full graph is not downloaded and uploaded between graph construction and the
`M` runs. The KODAMA CUDA/Metal lifecycle also retains its full graph and per-worker
classifier allocations across `M` runs. CUDA label-aware SIMPLS uses a
dedicated persistent cuBLAS workspace per stream. Its float32 class
cross-product uses fixed-order accumulation followed by separate column-sum
and centering kernels, avoiding a read/write race while preserving the
accepted SIMPLS formula.

See [`docs/backend-validation.md`](docs/backend-validation.md) for backend
acceptance results and [`docs/release-validation.md`](docs/release-validation.md)
for the release benchmark protocol.
The repository-wide correctness, memory, backend, and wrapper audit is in
[`docs/full-code-audit-2026-08-07.md`](docs/full-code-audit-2026-08-07.md).

## License and provenance

Original `kodama-cpp` code is MIT licensed. Identified compatible third-party
portions retain their original terms. See [`PROVENANCE.md`](PROVENANCE.md),
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and [`licenses/`](licenses/).

Run the machine-checkable license audit with:

```sh
bash tools/check_license_headers.sh
```

Release history and contribution guidance are in [`CHANGELOG.md`](CHANGELOG.md),
[`CONTRIBUTING.md`](CONTRIBUTING.md), and
[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).
