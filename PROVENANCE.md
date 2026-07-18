# Licensing and Provenance Audit

Audit date: 2026-07-16

Repository baseline inspected: `0eb2261ec10c62082950acb6f7b1c9f98c4617a4`

This is an engineering provenance audit, not a legal opinion. It records the
source lineage visible in the repositories, the maintainer's declarations, the
licenses retained in this distribution, and the checks required before a
release.

## Maintainer identity and declaration

Stefano Cacciatore confirmed the following for this audit:

- his name is **Stefano Cacciatore**;
- `tkcaccia <tkcaccia@gmail.com>` is his historical Git identity;
- `stefano.cacciatore@icgeb.org` is his publication and corresponding-author
  address;
- he developed KODAMA and fastPLS; and
- kodama-cpp is intended to distribute his original and adapted contributions
  under the MIT License.

The inspected kodama-cpp history contains 28 commits, all by
`tkcaccia <tkcaccia@gmail.com>`. The inspected KODAMA code history contains
731 commits by that identity and one documentation-only commit by another
author. The inspected fastPLS history contains 153 commits, all by that
identity. The inspected fastEmbedR history contains 86 commits by that
identity. The inspected faissR history contains 652 commits under `tkcaccia`
or the historical display name `Stefano Caccia`, all using
`tkcaccia@gmail.com`.

This declaration authorizes MIT relicensing only for material whose copyright
Stefano Cacciatore owns. Package-paper and metadata authors remain credited:
KODAMA names Leonardo Tenori as a coauthor; fastPLS names Dupe Ojo, Leonardo
Tenori, and Alessia Vignoli as coauthors. Before a public release, the
maintainer should confirm that no adapted code was supplied by those coauthors
outside the inspected Git history, or retain written permission if it was.

## Distribution license result

The original kodama-cpp code is MIT licensed. The distribution also contains
one mixed-license source file:

- `src/metal_backend.mm`: `MIT AND Apache-2.0`, because the adapted
  fastEmbedR Metal implementation explicitly retains Apache-2.0 terms for the
  Faiss-mlx fused list-scan/top-k organization.

Therefore, the accurate release statement is **MIT-licensed project code with
identified compatible third-party portions retained under their original
terms**, not “every byte is MIT.” The complete license texts are distributed in
`LICENSE` and `licenses/`.

## Upstream snapshots inspected

| Source | Snapshot | Declared license | Audit observation |
| --- | --- | --- | --- |
| [KODAMA](https://github.com/tkcaccia/KODAMA) | `d2dbe30ee66509b82c084616d6961c5b292cd059` | GPL (>= 2) | Public origin of the KODAMA objective, label evolution, matrix construction, and R-facing behavior. Code history is attributed to Stefano Cacciatore; the local adaptations are separately relicensed under MIT by him. |
| [fastPLS](https://github.com/tkcaccia/fastPLS) | `ef4aa0e4ea663a097fb3c10f6a4ce9f2884a278f` | GPL-3 | Source strategy for SIMPLS, label-aware cross-products, latent LDA, randomized PCA, and accelerator organization. Code history is attributed to Stefano Cacciatore; local adaptations are separately relicensed under MIT by him. |
| [fastEmbedR](https://github.com/tkcaccia/fastEmbedR) | `c98b9ecd124d442f20f849ce1be7f5bd4c13d0db` | MIT | Direct source lineage for native CPU HNSW, Metal KNN, randomized-PCA backend policy, UMAP/openTSNE, and accelerator embedding code. |
| [faissR](https://github.com/tkcaccia/faissR) | `b317a9715dd33ad3a49cf7989a83b4f7f9f7b389` | MIT | Direct source lineage for 2D/3D exact grid KNN and its selection rules. |
| [FAISS](https://github.com/facebookresearch/faiss) | `0ca9df4792b173d573044ee14ca0704780176e82` (1.14.3) | MIT | Direct structural source for compact HNSW through fastEmbedR; algorithmic reference for native exact/IVF CUDA and Metal paths. Meta notice retained. |
| [Faiss-mlx](https://github.com/MLXPorts/faiss-mlx) | `d092af559375144fc719cd88a10e414f92c625fa` | Apache-2.0 | Fused Metal IVF list-scan/top-k organization adapted through fastEmbedR. Apache terms and upstream notice retained. |
| [RAPIDS cuVS](https://github.com/rapidsai/cuvs) | `ad9e2d2a617c8d51e3eebc920e5a60ad8dc59bcd` | Apache-2.0 | Algorithmic organization/reference only in the package-owned CUDA search path; no cuVS source or binary is distributed or linked. |

The official cuVS license file at the pinned snapshot has SHA-256
`756005f963846334943e8bfc08ef98cd254257d8467ac7a7ffd42a1be262f442`.
The audit corrected an earlier accidental duplicate of the Faiss-mlx license
at `licenses/CUVS-LICENSE`.

## Component provenance matrix

| Local component | Local files | Classification | License and required action |
| --- | --- | --- | --- |
| Public C++ API and common utilities | `include/kodama/kodama.hpp`, `src/common.*` | Original kodama-cpp work | MIT; retain Stefano Cacciatore SPDX notice. |
| KODAMA label optimization | `src/core.cpp`, `src/kodama_matrix.cpp`, `src/kodama_matrix_cuda.*` | Port/adaptation of KODAMA behavior plus new backend implementation | MIT under Stefano Cacciatore's relicensing declaration; cite KODAMA and preserve publication credit. |
| CPU KNN CV and voting | `src/knncv.cpp`, relevant portions of `src/core.cpp` | Original/adapted kodama-cpp work | MIT. |
| CPU HNSW | `src/native_knn.cpp`, `src/native_knn.hpp` | Directly adapted from fastEmbedR's FAISS-distilled HNSW | MIT; retain both Meta and Stefano Cacciatore notices and `licenses/FAISS-LICENSE`. |
| Exact spatial grid KNN | `src/spatial_grid_knn.hpp` and callers | Direct adaptation from faissR | MIT; pin faissR snapshot in this record. |
| CPU PLS-LDA | `src/plscv.cpp` and PLS portions of `src/core.cpp` | Adapted from fastPLS SIMPLS/LDA | MIT under Stefano Cacciatore's relicensing declaration; cite fastPLS. |
| CUDA PLS-LDA | `src/cuda_simpls_float.cpp`, `src/pls_lda_cuda.cu` | Float32 standalone adaptation of fastPLS CUDA SIMPLS/LDA | MIT under Stefano Cacciatore's relicensing declaration; cite fastPLS. |
| Native CUDA KNN/k-means | `src/native_cuda_backend.*` | Package-owned KNN informed by FAISS, cuVS, fastEmbedR, and faissR; k-means initialization, seed/Lloyd semantics, and empty-cluster repair adapted from FAISS 1.14.3 `faiss/Clustering.cpp` and random utilities | MIT; retain Meta Platforms and Stefano Cacciatore notices plus `licenses/FAISS-LICENSE`. No FAISS/cuVS binary is distributed or linked, and no cuVS source is included. |
| Metal KNN and PLS-LDA | `src/metal_backend.mm` | Adapted from fastEmbedR and fastPLS; contains Faiss-mlx-derived organization | `MIT AND Apache-2.0`; retain Meta, Sydney Bach/The Solace Project, and Stefano Cacciatore notices, the modified-file statement, and both full licenses. |
| Metal stubs/interfaces | `src/metal_backend.hpp`, `src/metal_backend_stub.cpp` | Original kodama-cpp work | MIT. |
| UMAP/openTSNE CPU/CUDA | `src/visualization.cpp`, `src/embedding_cuda_kernels.cu` | Direct adaptation of fastEmbedR's MIT implementation, including direct binary/fuzzy CSR construction, smooth-kNN bandwidths, epoch scheduling, float32 openTSNE, and CUDA workspaces | MIT; retain the algorithmic-reference record below. |
| Randomized PCA CPU/CUDA/Metal | `src/pca.cpp`, `src/pca_cuda.cu`, `src/pca_cuda_backend.hpp`, public wrappers | Standalone float32 adaptation of the current fastEmbedR/fastPLS randomized-PCA strategy; uses package-owned QR/eigensolver code and existing native accelerator matrix multiplication | MIT under Stefano Cacciatore's fastPLS relicensing declaration and fastEmbedR MIT terms; cite both projects. No Armadillo, Eigen, RAFT, fastPLS, or fastEmbedR link is present. |
| Graph and random-walk utilities | `src/graph_cluster.cpp` | Original kodama-cpp implementation | MIT. Louvain, Leiden, and cuGraph code are absent from the audited baseline. |
| Build, examples, tests, benchmarks | `CMakeLists.txt`, `cmake/`, `examples/`, `tests/`, `benchmarks/`, tracked `tools/` scripts | Repository-authored support code | MIT. |
| R and Python wrappers in this monorepo snapshot | `split-repos/kodama-r/`, `split-repos/kodama-python/`, `wrappers/R/` | Repository-authored bindings | MIT; generated Rcpp files carry generated-file and SPDX notices. |
| Manuscript generator | `manuscript/build_kodama_manuscript.py` | Repository-authored publication tooling | MIT. Generated manuscript documents are scholarly outputs, not linked program code. |

## Visualization algorithm references

The local visualization implementation is adapted from the MIT-licensed
fastEmbedR snapshot above. Its upstream provenance record identifies the
following as mathematical, behavioral, or architectural references rather
than vendored source:

- UMAP reference implementation (BSD-3-Clause), umappp (BSD-2-Clause), and
  ensmallen (BSD-3-Clause);
- openTSNE (BSD-3-Clause), t-SNE-CUDA (BSD-3-Clause), Rtsne behavior, and
  opt-SNE/Multicore-opt-SNE (BSD-3-Clause);
- Schraudolph's published exponential approximation and permissive HXA7241
  prior art;
- AppleSiliconFFT (MIT), mlx-vis (Apache-2.0), and annembed
  (MIT OR Apache-2.0) as accelerator-architecture references.

No source from GPL-licensed `uwot` is permitted in the core. It may be used
only as an external benchmark/reference package. If future changes copy or
closely adapt any referenced implementation, its copyright notice, exact
snapshot, and license must be added here and to the affected file.

## Per-file policy

Every tracked C, C++, CUDA, Objective-C++, CMake, R, Python, and shell source
file must contain:

1. at least one `SPDX-FileCopyrightText` line;
2. an `SPDX-License-Identifier` expression; and
3. any additional upstream copyright and modification notice required for
   derivative files.

`tools/check_license_headers.sh` enforces this policy and the special notices
for `src/native_knn.cpp` and `src/metal_backend.mm`. The check also rejects
the known surname truncation “Stefano Caccia” in tracked project files.

## External system libraries

CUDA Toolkit libraries, OpenMP runtimes, and Apple system frameworks are build
or runtime dependencies and are not distributed in this repository. Binary
package distributors must satisfy the licenses and redistribution rules of the
specific binaries they ship.

## Publication assets outside the runtime audit

Generated DOCX, TeX, PDF, figures, benchmark data, and plots are not linked
program code and are outside the tracked-source header check. The generated
`grfext.sty` copy states LPPL-1.3c-or-later terms in its own header. The
inspected JMLR `jmlr2e.sty` upstream repository provides the journal style but
does not publish a standalone license file. Keep that style file outside the
kodama-cpp software release archive unless JMLR confirms redistribution, or
obtain it from JMLR during manuscript build/submission.

## Release findings

| Finding | Status |
| --- | --- |
| Misspelled local copyright name `Stefano Caccia` | Corrected to Stefano Cacciatore. |
| Missing per-file license identifiers | Corrected for all tracked source-like files in the audit scope. |
| Wrong cuVS license text | Corrected and checksum-verified against the pinned official source. |
| Missing Apache designation on Metal derivative | Corrected to `MIT AND Apache-2.0` with upstream notices. |
| GPL-origin KODAMA/fastPLS adaptations under MIT | Documented maintainer authorization and Git-history evidence; final release remains conditional on confirming there were no unrecorded third-party code contributions. |
| Upstream fastEmbedR uses the truncated display name `Stefano Caccia` | Local notices corrected. The upstream repository should be corrected separately for consistency. |
| Faiss-mlx author-name evidence differs | fastEmbedR records “Sydney Bach, The Solace Project”; the inspected Faiss-mlx Git history records “Sydney Renee.” The retained fastEmbedR notice is preserved pending upstream clarification. |
| JMLR style-file redistribution | Not part of the runtime library; upstream redistribution terms should be confirmed before bundling `jmlr2e.sty` in a software release. |

Run before every release:

```sh
bash tools/check_license_headers.sh
cmake --build build -j
ctest --test-dir build --output-on-failure
```
