# Getting Started

## Build and test the C++ core

```sh
git clone https://github.com/tkcaccia/kodama-cpp.git
cd kodama-cpp
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DKODAMA_ENABLE_CUDA=OFF \
  -DKODAMA_ENABLE_METAL=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

An installed consumer can use `find_package(kodama-cpp CONFIG REQUIRED)` and
link `kodama::kodama_cpp`.

## Minimal C++ workflow

```cpp
#include <kodama/kodama.hpp>

#include <vector>

int main() {
  const std::size_t n = 120;
  const std::size_t p = 8;
  std::vector<float> x(n * p, 0.0f); // row-major input

  kodama::KODAMAMatrixOptions options;
  options.runs = 10;
  options.cycles = 20;
  options.landmarks = 90;
  options.splitting = 10;
  options.classifier = kodama::CoreClassifier::KNN;
  options.backend = kodama::Backend::CPU;
  options.knn.k = 10;

  const auto result = kodama::KODAMAMatrix(
    kodama::MatrixView{x.data(), n, p}, {}, {}, {}, options
  );
  return result.res.size() == options.runs * n ? 0 : 1;
}
```

`result.res` stores one row-major label vector per independent run;
`result.acc` stores its raw cross-validated accuracy. Select a run from those
internal scores, not from external reference labels.

## Retain or lazily correct one graph

`KODAMAMatrixResult` owns one `NeighborGraph`. With the default
`apply_kodama_dissimilarity = true`, the base distances are corrected in place
and the final graph is moved into `result.knn`. To retain the base graph and
defer correction:

```cpp
options.apply_kodama_dissimilarity = false;
auto result = kodama::KODAMAMatrix(
  kodama::MatrixView{x.data(), n, p}, {}, {}, {}, options
);

kodama::KODAMADissimilarityInPlace(
  result.knn,
  result.res,
  result.runs,
  result.samples,
  options.backend,
  options.n_threads,
  options.knn.gpu_device
);
```

The in-place function accepts the public one-based graph convention and does
not allocate a second graph. `result.knn_is_kodama_corrected` describes the
state produced by `KODAMAMatrix`; `result.graph_storage_bytes` reports the
retained index-distance capacity.

## Normalize and scale float32 matrices

The standalone preprocessing API accepts row-major training data and an
optional test matrix. Test data always reuse the training PQN reference or
the training centering/scaling statistics.

```cpp
kodama::NormalizationOptions normalization_options;
normalization_options.method = kodama::NormalizationMethod::PQN;
normalization_options.backend = kodama::Backend::CUDA; // CPU or Metal also supported
normalization_options.n_threads = 4;

auto normalized = kodama::Normalization(
  kodama::MatrixView{x_train.data(), train_rows, variables},
  kodama::MatrixView{x_test.data(), test_rows, variables},
  normalization_options
);

kodama::ScalingOptions scaling_options;
scaling_options.method = kodama::ScalingMethod::Autoscaling;
scaling_options.backend = kodama::Backend::CPU;
scaling_options.n_threads = 4;

auto scaled = kodama::Scaling(
  kodama::MatrixView{x_train.data(), train_rows, variables},
  kodama::MatrixView{x_test.data(), test_rows, variables},
  scaling_options
);
```

Normalization methods are `PQN`, `Sum`, `Median`, `Sqrt`, and `None`.
Scaling methods are `None`, `Centering`, `Autoscaling`, `RangeScaling`, and
`ParetoScaling`. The implementation deliberately preserves the historical
KODAMA train/test conventions, including signed training sums and absolute
test sums for sum normalization. Division by zero and missing-value behavior
follow the original formulas; the library does not insert an epsilon or
silently impute values.

`Normalization_CPU` and `Scaling_CPU` parallelize independent rows and
columns. Their CUDA and Metal counterparts compute statistics, PQN medians,
and transformations on device and throw when their requested backend is not
compiled or available; they never fall back to CPU.

## Reuse a CUDA or Metal IVF index

Use an explicit resident handle when several searches share one training
matrix:

```cpp
kodama::KNNOptions ivf;
ivf.backend = kodama::Backend::CUDA; // or Backend::Metal
ivf.metric = kodama::DistanceMetric::Euclidean;
ivf.ivf_nlist = 256;
ivf.ivf_nprobe = 32;

auto index = kodama::BuildResidentIVFIndex(
  kodama::MatrixView{x.data(), n, p},
  ivf
);

kodama::ResidentIVFSearchStats stats;
auto self_graph = kodama::SearchResidentIVFIndexSelf(
  index, 30, true, &stats
);
auto query_graph = kodama::SearchResidentIVFIndex(
  index,
  kodama::MatrixView{query.data(), query_n, p},
  30,
  &stats
);
```

The move-only handle owns the device training matrix, projection, centroids,
and inverted lists. Self-search does not upload the training matrix again;
external queries upload only their query rows. Neighbor identifiers returned
by both calls are one-based.

## Parallel CPU HNSW graphs

`GraphClusterOptions::n_threads`, `KNNOptions::n_threads`, and the wrapper
argument `n.cores` control both native HNSW construction and querying. The
implementation uses lock-protected concurrent insertion and batched parallel
queries over contiguous float32 storage:

```cpp
kodama::GraphClusterOptions graph_options;
graph_options.k = 30;
graph_options.n_threads = 4;
graph_options.metric = kodama::DistanceMetric::Euclidean;

auto graph = kodama::KODAMAKNNGraph_CPU(
  kodama::MatrixView{x.data(), n, p},
  graph_options
);
```

Parallel HNSW insertion is approximate and its graph can vary slightly with
thread scheduling. The maintained regression test requires at least 0.99
recall against exact neighbors rather than bitwise identity with a serial
graph.

## R wrapper

Build the core first, then install the thin wrapper from the checkout:

```sh
KODAMA_CPP_ROOT="$PWD" \
KODAMA_CPP_BUILD_DIR="$PWD/build" \
R CMD INSTALL split-repos/kodama-r
```

```r
library(kodamaR)

set.seed(1)
x <- matrix(rnorm(120 * 8), 120, 8)
fit <- KODAMA.matrix(
  x,
  classifier = "knn",
  backend = "cpu",
  M = 10,
  Tcycle = 20,
  landmarks = 90,
  splitting = 10,
  progress = FALSE
)

fit$best_labels
KODAMA.timing(fit)
fit$landmark_seconds
embedding <- KODAMA.visualization(fit, "UMAP", k = 30, backend = "cpu")
```

The matrix call builds the full-data graph once before all `M` searches. It
also computes one PCA with the selected matrix backend, derives both UMAP and
openTSNE starts from those scores, and stores the graph and starts in `fit`.
The visualization call reuses a stored start only when its backend matches the
requested embedding backend. Check `fit$graph_builds` (normally `1`) and
`fit$timing$visual_init_seconds` when profiling. Landmark selection is retained
separately for every independent run in `fit$landmark_seconds`; its sum, mean,
and median appear in `fit$timing`. Do not attribute this stage to the KNN or
PLS-LDA classifier core. To visualize on another
backend, pass the raw matrix so initialization is recomputed there. Native
UMAP and openTSNE are available on CPU, CUDA, and Metal; no visualization
backend is silently substituted for another:

```r
embedding_cuda <- KODAMA.visualization(
  fit,
  "opentsne",
  raw.data = x,
  backend = "cuda",
  perplexity = 30
)
```

An explicit `init` always wins. Without explicit, raw, or backend-matched
stored initialization, UMAP uses its graph-spectral start and openTSNE uses
its deterministic random start.

The complete installation and `R CMD check` procedure is in
`split-repos/kodama-r/README.md`.

## Python wrapper

```sh
python -m pip install -v "./split-repos/kodama-python[test]" \
  --config-settings=cmake.define.KODAMA_CPP_ROOT="$PWD" \
  --config-settings=cmake.define.KODAMA_CPP_BUILD_DIR="$PWD/build"
```

```python
import numpy as np
import kodama

rng = np.random.default_rng(1)
x = rng.normal(size=(120, 8)).astype(np.float32)
fit = kodama.matrix(
    x,
    classifier="knn",
    backend="cpu",
    M=10,
    Tcycle=20,
    landmarks=90,
    splitting=10,
    progress=False,
)

labels = fit.best_labels
embedding = kodama.visualization(fit, "UMAP", k=30, backend="cpu")
```

`kodama.matrix()` stores raw-data PCA starts by default. Pass
`raw_data=x` to `kodama.visualization()` when changing the CPU/CUDA/Metal
visualization backend, or pass `init=` to provide coordinates explicitly.

## CUDA and Metal

CUDA is enabled with `-DKODAMA_ENABLE_CUDA=ON`; build and run against the same
CUDA Toolkit runtime. Metal is enabled with `-DKODAMA_ENABLE_METAL=ON` on
macOS. Call `KODAMA.diagnostics()` in R or `kodama.diagnostics()` in Python to
inspect the linked wrapper environment. A requested accelerator that is absent
or cannot initialize raises an error instead of changing the backend.
