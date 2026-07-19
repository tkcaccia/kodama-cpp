# kodama-cpp API {#mainpage}

`kodama-cpp` is a standalone C++17 library for float32 KODAMA optimization,
cross-validated KNN and PLS-LDA, graph construction, PCA, UMAP, and openTSNE.
The same typed core is called by the R and Python wrappers.

Start with the [C++/R/Python walkthrough](getting-started.md), then use the
namespaced declarations in `kodama/kodama.hpp` as the authoritative C++ API.
The [R API map](r-api.md) links wrapper names to their C++ counterparts, and
the [CPU coverage report](coverage.md) states the measured scope and gaps.

## Backend contract

- `Backend::CPU` runs package-owned CPU code.
- `Backend::CUDA` requires a CUDA-enabled build and reports an error if CUDA is
  unavailable.
- `Backend::Metal` requires a Metal-enabled macOS build and reports an error if
  Metal is unavailable.

Accelerator requests never silently fall back to CPU. Analysis matrices and
intermediate numerical buffers are float32 inside the core. Reference labels
are not inputs to KODAMA optimization; they belong only in downstream
diagnostics.

## Main entry points

- `KNNCV` and `PLSLDACV`: classifier-level cross-validation.
- `CoreKNN` and `CorePLSLDA`: one label-vector optimization.
- `KODAMAMatrix`: complete independent-run ensemble and graph correction.
- `KODAMAMatrixFromGraph`: alternative input when only a KNN graph is available.
- `KODAMAKNNGraph`, `PCA`, `KODAMAUMAP_CPU/CUDA`, and
  `KODAMAOpenTSNE_CPU/CUDA`: graph and visualization primitives.
