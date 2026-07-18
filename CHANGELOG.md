# Changelog

All notable changes to kodama-cpp are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- Replaced the CPU/CUDA UMAP and openTSNE implementation with the current
  fastEmbedR kernels, including direct float32 CSR graph construction, the
  current smooth-kNN bandwidth rule, pooled CUDA workspaces, and the symmetric
  binary UMAP graph default; fuzzy UMAP weighting remains selectable.
- Added a standalone float32 randomized PCA API with CPU, CUDA, and Metal
  entry points plus thin R and Python bindings.
- Complete the tagged release archive, checksum, and archival DOI.
- Complete the final historical-R, backend, and visualization validation on
  the release hardware.

## [0.1.0] - 2026-07-16

Initial standalone release candidate.

- Added float32 KNN and SIMPLS + PLS-LDA cross-validation kernels.
- Added independent KODAMA KNN and PLS-LDA label-optimization paths.
- Added matrix, matrix-plus-graph, and graph-only KODAMA entry points.
- Added package-owned CPU HNSW, CUDA exact/IVF-Flat, and Metal exact/IVF-Flat
  nearest-neighbor backends.
- Added CPU/CUDA UMAP and openTSNE visualization primitives.
- Added thin R and Python wrappers around the same typed C++ interface.
- Added strict backend reporting, public API compile-link tests, provenance
  records, SPDX checks, and retained third-party license texts.

[Unreleased]: https://github.com/tkcaccia/kodama-cpp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tkcaccia/kodama-cpp/releases/tag/v0.1.0
