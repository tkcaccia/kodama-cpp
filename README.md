# kodama-cpp

`kodama-cpp` is a standalone C++ library for high-performance
KODAMA-related cross-validation kernels. It is independent from R and Python,
but the API is designed so wrappers can be added later.

The first implementation phase focuses only on:

- `KNNCV`
- `PLSDACV`
- `PLSLDACV`

The full KODAMA algorithm is intentionally not implemented yet.

## Current Contract

- FAISS is mandatory.
- `KNNCV` uses FAISS IVF-Flat, not a hand-written exact fallback.
- Cosine similarity is implemented by L2-normalizing rows and using inner product.
- CPU and CUDA entry points are exposed.
- CUDA builds require FAISS GPU, cuVS headers, CUDA headers, cuBLAS, and cudart.
- `KNNCV_CUDA` is implemented with FAISS GPU IVF-Flat.
- `PLSDACV_CUDA` and `PLSLDACV_CUDA` are public reserved entry points and fail explicitly until the CUDA SIMPLS backend is added. They never silently run CPU code and report it as CUDA.
- PLS-DA and PLS-LDA are separate functions.
- `constrain` controls CV splitting: samples with the same constraint value are kept in the same fold.

## Functions

### KNNCV

```cpp
auto result = kodama::KNNCV(x, labels, constrain, options);
auto cpu_result = kodama::KNNCV_CPU(x, labels, constrain, options);
auto cuda_result = kodama::KNNCV_CUDA(x, labels, constrain, options);
```

CPU:

- FAISS `IndexIVFFlat`
- inner product / cosine

CUDA:

- FAISS GPU IVF-Flat
- cuVS is required by the CUDA build gate for future native cuVS kernels
- `ivf_nlist` and `ivf_nprobe` are applied through FAISS IVF search parameters

Default options:

```cpp
kodama::KNNOptions opt;
opt.cv.folds = 10;
opt.cv.stratified = true;
opt.cv.seed = 1;
opt.k = 10;
opt.metric = kodama::DistanceMetric::Cosine;
opt.backend = kodama::Backend::CPU;
opt.index_type = kodama::KNNIndexType::FaissIVFFlat;
opt.ivf_nlist = 0;   // automatic
opt.ivf_nprobe = 0;  // automatic
```

### PLSDACV

```cpp
auto result = kodama::PLSDACV(x, labels, constrain, options);
auto cpu_result = kodama::PLSDACV_CPU(x, labels, constrain, options);
auto cuda_result = kodama::PLSDACV_CUDA(x, labels, constrain, options);
```

PLS-DA uses SIMPLS-style latent components followed by argmax/nearest-centroid
classification in the latent space. The CPU implementation is active. The CUDA
entry point exists for wrapper stability, but currently throws a clear
not-implemented error rather than falling back to CPU.

### PLSLDACV

```cpp
auto result = kodama::PLSLDACV(x, labels, constrain, options);
auto cpu_result = kodama::PLSLDACV_CPU(x, labels, constrain, options);
auto cuda_result = kodama::PLSLDACV_CUDA(x, labels, constrain, options);
```

PLS-LDA uses the same latent components followed by covariance-weighted LDA-style
classification in the latent space. The CPU implementation is active. The CUDA
entry point exists for wrapper stability, but currently throws a clear
not-implemented error rather than falling back to CPU.

Default PLS options:

```cpp
kodama::PLSOptions opt;
opt.cv.folds = 10;
opt.cv.stratified = true;
opt.cv.seed = 1;
opt.max_components = 10;
opt.center = true;
opt.scale = true;
opt.backend = kodama::Backend::CPU;
```

## Build

FAISS must be installed and discoverable.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the example:

```bash
./build/kodama_cv_example
```

CUDA build:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DKODAMA_ENABLE_CUDA=ON
cmake --build build-cuda -j
```

On conda or micromamba FAISS/cuVS installations, use the matching compiler and
runtime from the environment so `libstdc++`, FAISS, and cuVS agree:

```bash
export CONDA_PREFIX=/path/to/faiss-cuvs-env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DKODAMA_ENABLE_CUDA=ON \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

`KODAMA_ENABLE_CUDA=ON` requires:

- FAISS
- FAISS GPU library
- cuVS headers
- CUDA toolchain

## Outputs

The result structs include:

- predicted labels
- true labels
- fold assignments
- accuracy per fold
- global accuracy
- confusion matrix
- runtime
- peak memory where measurable
- backend used
- FAISS/cuVS parameters used

## Benchmark Plan

The first benchmark target is the curated fastEmbedR data collection:

- COIL20
- USPS
- FashionMNIST
- FlowRepository_FR-FCM-ZYRM_files
- flow18
- MNIST
- imagenet
- MetRef
- mass41

For `KNNCV`, compare CPU FAISS IVF-Flat and CUDA FAISS/cuVS IVF-Flat with
cosine / inner-product and `k = 10`.

For `PLSDACV` and `PLSLDACV`, compare CPU and CUDA versions across component
counts and report selected components once the CUDA SIMPLS backend is
implemented. In the current revision, CPU PLS-DA and CPU PLS-LDA are the active
implementations.

## Acknowledgements

This library is inspired by:

- KODAMA, especially the constrained CV logic around `PLSDACV_simpls`;
- fastPLS, especially the SIMPLS / fast PLS implementation strategy;
- FAISS and cuVS for high-performance nearest-neighbour search.

The code here is implemented as a standalone C++ library and does not depend on
Rcpp, R, Armadillo, or Python.
