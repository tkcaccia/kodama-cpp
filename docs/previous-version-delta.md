# Improvements over KODAMA R 2.4.1/2.4

The KODAMA mathematical principle was introduced in 2014 and its R package was
described in 2017. KODAMA R 2.4.1/2.4 is the sole previous-version performance
comparator in this work. This document records why kodama-cpp is a distinct
software submission while preserving that prior work. Later KODAMA releases,
including 3.3, are excluded from timing and quality comparisons.

| Area | KODAMA R 2.4.1/2.4 | kodama-cpp 0.1.0 |
| --- | --- | --- |
| Core architecture | R-package workflow with R-facing compiled helpers | Standalone C++17 library with no R or Python runtime dependency |
| Search dynamics | Historical stochastic label evolution driven by cross-validated prediction | Grouped adaptive proposals, transition-driven coarsening, and a disclosed label-only degeneracy guard; raw CV accuracy remains separately reported |
| Classifiers in KODAMA | KNN and PLS-DA | KNN and SIMPLS + latent-space PLS-LDA |
| Numerical storage | Host-language and dependency-dependent representations | Float32 analysis matrices and workspaces on every backend |
| Hardware | CPU-oriented R execution | Explicit CPU, NVIDIA CUDA, and Apple Metal entry points |
| Neighbor search | Repeated package-level neighbor-search calls | Package-owned CPU HNSW, CUDA exact/IVF-Flat, and Metal exact/IVF-Flat implementations with reusable graph state |
| PLS execution | Package-level fitting inside repeated cycles | Label-aware SIMPLS cross-products, fixed requested component counts, reusable fold buffers, and native accelerator workspaces |
| Backend behavior | Backend identity not part of a common typed result | Strict backend selection; unavailable accelerators raise errors and results record the backend that executed |
| Language bindings | R is the implementation boundary | Thin R and Python wrappers call the same C++ API |
| Input forms | Data-matrix workflow | Data matrix, matrix plus supplied graph, or graph-only input; graph-only PLS-LDA uses documented self-tuning Laplacian features |
| Output | R objects oriented to the original workflow | Typed predictions, folds, accuracies, label ensembles, graphs, timings, memory, and backend/search metadata |
| Validation | Package examples and method-paper experiments | CTest backend tests, public API snapshot test, wrapper tests, provenance audit, and reproducible benchmark drivers |

The implementation does not claim a replacement learning principle: held-out
label predictability remains the central classifier signal. The current standard
does include explicit search extensions relative to the historical R core. Their
acceptance score, raw CV accuracy, and external diagnostics are kept separate so
the extension can be audited rather than described as trajectory parity. The
software contribution is the typed, heterogeneous, reusable execution of this
procedure and its consistent exposure to multiple host languages.

Runtime comparisons are interpreted by scope. Kernel, core-optimizer, and
end-to-end `KODAMA.matrix` timings are reported separately. The manuscript
quantifies both historical MetRef routes. KNN is a classifier-matched predecessor
comparison. Historical PLS-DA is retained as a contextual predecessor baseline,
but it is not used to claim a classifier-matched speedup because current KODAMA
introduces label-aware SIMPLS followed by LDA in the requested-component latent
space. That PLS-LDA route is a novelty of kodama-cpp rather than an assertion of
numerical equivalence to the 2.4.1 PLS-DA decoder.
