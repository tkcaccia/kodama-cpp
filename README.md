# kodama-cpp

`kodama-cpp` is a standalone C++ library for high-performance KODAMA-related
algorithms. The first implementation phase intentionally focuses only on two
cross-validation kernels:

- `KNNCV`
- `PLSCV`

The full KODAMA algorithm is not implemented in this repository yet. The goal is
to keep the core library independent from R and Python while making later
wrappers straightforward.

## Current Status

This repository is an initial C++ core. It provides:

- constrained cross-validation folds through a `constrain` vector;
- CPU `KNNCV` using exact cosine / inner-product KNN;
- backend metadata for future FAISS/cuVS acceleration;
- CPU `PLSCV` using a SIMPLS-style PLS2 component extraction strategy;
- PLS-DA and PLS-LDA classification modes;
- fold-level accuracy, global accuracy, confusion matrix, runtime, and memory reporting;
- a clean CMake build with tests and an example.

The CUDA FAISS/cuVS path is exposed in the design but is not linked by default.
The next backend phase will implement FAISS GPU IVF-Flat / cuVS search behind
the same `KNNCV` API.

## Design

Inputs use a simple row-major `MatrixView`:

```cpp
kodama::MatrixView x{data.data(), n_samples, n_features};
```

Labels are integer class labels. `constrain` is optional; when provided, samples
with the same constraint value are kept in the same CV fold. This mirrors the
fold-control behavior used by KODAMA's `PLSDACV_simpls` workflow.

## KNNCV

`KNNCV` performs:

1. constrained / optionally stratified fold construction;
2. training/validation split per fold;
3. KNN classification of validation samples from the training samples;
4. majority vote among `k` neighbours;
5. fold-level and global accuracy reporting.

Default settings:

```cpp
kodama::KNNOptions opt;
opt.cv.folds = 10;
opt.cv.stratified = true;
opt.cv.seed = 1;
opt.k = 10;
opt.metric = kodama::DistanceMetric::Cosine;
opt.backend = kodama::Backend::Auto;
```

Requested future acceleration:

- CPU: exact KNN and optional FAISS CPU fallback.
- CUDA: FAISS GPU IVF-Flat / cuVS nearest-neighbour search.

## PLSCV

`PLSCV` performs:

1. constrained / optionally stratified fold construction;
2. training-set centering/scaling;
3. SIMPLS-style latent component extraction;
4. validation projection into the PLS latent space;
5. PLS-DA or PLS-LDA classification;
6. accuracy evaluation for `1:max_components`;
7. component selection by cross-validated accuracy.

Default settings:

```cpp
kodama::PLSOptions opt;
opt.cv.folds = 10;
opt.cv.stratified = true;
opt.cv.seed = 1;
opt.max_components = 10;
opt.mode = kodama::PLSMode::PLS_DA;
opt.center = true;
opt.scale = true;
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the example:

```bash
./build/kodama_cv_example
```

## Optional Backends

The CMake options are present for backend-specific builds:

```bash
cmake -S . -B build-faiss -DKODAMA_ENABLE_FAISS=ON
cmake -S . -B build-cuvs  -DKODAMA_ENABLE_CUVS=ON
```

The first release keeps the CPU implementation as the portable reference. FAISS
and cuVS will be added as implementation files behind the existing API so R and
Python wrappers do not need to change.

## Benchmark Plan

The speed and classification performance of `KNNCV` and `PLSCV` should be
tested on the datasets already prepared for the fastEmbedR benchmarks:

- COIL20
- USPS
- FashionMNIST
- FlowRepository_FR-FCM-ZYRM_files
- flow18
- MNIST
- imagenet
- MetRef
- mass41

Metrics to report:

- global CV accuracy;
- fold-level accuracy;
- confusion matrix;
- runtime;
- peak memory;
- backend used;
- FAISS/cuVS parameters, when enabled.

## Acknowledgements

This library is inspired by:

- KODAMA, especially the constrained CV logic around `PLSDACV_simpls`;
- fastPLS, especially the SIMPLS / fast PLS implementation strategy;
- FAISS and cuVS for the planned GPU/ANN backend design.

The C++ code in this repository is a standalone implementation and does not use
Rcpp, R, or Armadillo APIs.

