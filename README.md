# kodama-cpp

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
- randomized PCA with float32 data and CPU, CUDA, and Metal entry points;
- KNN graph construction, UMAP, openTSNE, and random-walk clustering;
- thin R and Python bindings to the same C++ implementation.

Numerical matrices, PLS/LDA workspaces, and accelerator buffers use float32.
Timing and summary statistics use double precision. Requested feasible PLS
component counts are evaluated directly; the implementation does not search
for an internally selected "best" component count.

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
  knn.k = 10,
  progress = TRUE
)

print(fit)
KODAMA.timing(fit)
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
| Build a KNN/SNN graph | `KODAMA.graph()`, `makeSNNGraph()` |
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
SIMPLS/LDA, and PCA. UMAP and openTSNE currently have explicit CPU and CUDA
entry points.

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

## Parameters that matter most

| Parameter | Meaning |
| --- | --- |
| `M` | Number of independent KODAMA label searches contributing to the final graph |
| `Tcycle` | Number of proposal and cross-validation cycles inside each independent search |
| `landmarks` | Maximum samples optimized directly; when `n <= landmarks`, KODAMA uses `ceiling(0.75 * n)` |
| `splitting` | Initial number of label classes in each independent search |
| `knn.k` | K used by the KNN classifier, not the visualization graph |
| `ncomp` | Requested feasible SIMPLS components for PLS-LDA |
| `graph.neighbors` | Neighbors retained in the returned KODAMA graph |
| `backend` | Strict execution backend: CPU, CUDA, or an API-supported Metal path |

The R wrapper defaults `splitting` to 100 below 40,000 samples and 300
otherwise. UMAP uses `k = 30` by default; openTSNE uses perplexity 30.

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

The CPU and CUDA UMAP/openTSNE implementations track fastEmbedR commit
`c98b9ecd124d442f20f849ce1be7f5bd4c13d0db`. UMAP supports the current
fastEmbedR symmetric binary graph and an explicit fuzzy graph mode. The
openTSNE default perplexity is 30.

CUDA nearest-neighbor search is package-owned and provides exact and
recall-tuned IVF-Flat paths. Metal provides exact search and an explicit
IVF-Flat alternative. No FAISS, cuVS, RAFT, or RMM headers or binaries are
required.

See [`docs/backend-validation.md`](docs/backend-validation.md) for backend
acceptance results and [`docs/release-validation.md`](docs/release-validation.md)
for the release benchmark protocol.

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
