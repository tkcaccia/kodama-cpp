# kodama-cpp public API stability

## Version 0.1.0

The public C++ interface is declared in `include/kodama/kodama.hpp`; version
macros and constants are declared in `include/kodama/version.hpp`. The R and
Python wrappers are thin bindings to this interface and carry the same
`0.1.0` release version.

The `0.1.0` source-compatibility surface comprises:

- cross-validation: `KNNCV` and `PLSLDACV`, with explicit CPU, CUDA, and Metal
  entry points where implemented;
- optimization: `CoreKNN` and `CorePLSLDA`, with explicit backend entry points;
- matrix construction: `KODAMAMatrix`, `KODAMAMatrixFromGraph`, and
  `KODAMAMatrixFromGraphData`;
- graph construction: `KODAMAKNNGraph`;
- principal components: `PCA`, `PCA_CPU`, `PCA_CUDA`, and `PCA_METAL`;
- visualization: `KODAMAUMAP_CPU`, `KODAMAUMAP_CUDA`,
  `KODAMAOpenTSNE_CPU`, and `KODAMAOpenTSNE_CUDA`;
- graph clustering: `KODAMAGraphCluster` and `KODAMAEmbeddingCluster`;
- the option, result, graph, matrix-view, and backend types used by these
  functions.

`tests/test_public_api_0_1.cpp` compile-links the release surface. This test is
part of every configured CTest build, so an accidental removal or signature
change fails release validation.

## Compatibility policy

The project follows semantic versioning. Patch releases in the `0.1.x` line
will preserve this source interface. Before version 1.0, an intentional
incompatible API change requires a new minor version, a changelog entry, and a
documented wrapper migration. Backend entry points are strict: an unavailable
CUDA or Metal backend reports an error and does not silently execute CPU code.

Numerical equivalence is tested separately from binary ABI stability. The
library guarantees float32 analysis semantics and the documented output
contract; it does not claim a stable C++ ABI across compilers or standard
library implementations. Consumers should link against the CMake package built
for their toolchain.

## Release identity

The reviewed release must be identified by all of the following before journal
submission:

1. Git tag `v0.1.0` and its full commit hash;
2. a source archive generated from that tag;
3. the archive SHA-256 digest;
4. an archival DOI, recorded in `CITATION.cff` and the manuscript only after the
   archive has been deposited.

No DOI or commit placeholder is presented as a completed archive.
