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
embedding <- KODAMA.visualization(fit, "UMAP", k = 30, backend = "cpu")
```

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

## CUDA and Metal

CUDA is enabled with `-DKODAMA_ENABLE_CUDA=ON`; build and run against the same
CUDA Toolkit runtime. Metal is enabled with `-DKODAMA_ENABLE_METAL=ON` on
macOS. Call `KODAMA.diagnostics()` in R or `kodama.diagnostics()` in Python to
inspect the linked wrapper environment. A requested accelerator that is absent
or cannot initialize raises an error instead of changing the backend.
