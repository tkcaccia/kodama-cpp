# Third-Party and Source Acknowledgements

The original kodama-cpp code is released under the MIT License. Identified
third-party portions retain their compatible upstream terms. The formal
component and snapshot audit is in [`PROVENANCE.md`](PROVENANCE.md).

## Maintainer-authored source lineage

Stefano Cacciatore confirmed that `tkcaccia <tkcaccia@gmail.com>` is his
historical Git identity, that he developed KODAMA and fastPLS, and that his
adapted contributions in kodama-cpp are released under MIT. His publication
contact is `stefano.cacciatore@icgeb.org`.

The public KODAMA package is GPL (>= 2), and the public fastPLS package is
GPL-3. Local ports of the KODAMA optimization procedure and fastPLS
SIMPLS/LDA implementation are separately relicensed under MIT by Stefano
Cacciatore for this repository. This permission covers only material whose
copyright he owns; package and publication coauthors remain credited.

The native visualization, CPU HNSW, and Metal implementation were adapted
from the MIT-licensed `fastEmbedR` snapshot
`c98b9ecd124d442f20f849ce1be7f5bd4c13d0db`. The exact 2D/3D grid search was
adapted from the MIT-licensed `faissR` snapshot
`b317a9715dd33ad3a49cf7989a83b4f7f9f7b389`. Both repositories were developed
and committed by Stefano Cacciatore under the Git identities documented above.

## FAISS HNSW

The package-owned CPU HNSW implementation in `src/native_knn.cpp` was
distilled from the HNSW organization in FAISS 1.14.3, commit
`0ca9df4792b173d573044ee14ca0704780176e82` (MIT). The file retains the Meta
Platforms and Stefano Cacciatore notices. The complete FAISS license is in
`licenses/FAISS-LICENSE`.

## Metal KNN

`src/metal_backend.mm` is adapted from fastEmbedR and fastPLS. Its KNN
organization was informed by FAISS 1.14.3 and contains portions inherited from
the fused IVF list-scan/top-k organization of MLXPorts/Faiss-mlx commit
`d092af559375144fc719cd88a10e414f92c625fa`. Those portions remain
Apache-2.0; the FAISS-derived and kodama-cpp portions remain MIT. The file is
therefore marked `MIT AND Apache-2.0` and carries Meta Platforms, Sydney Bach / The
Solace Project, and Stefano Cacciatore notices. The complete licenses are in
`licenses/FAISS-LICENSE` and `licenses/FAISS-MLX-LICENSE`.

The inspected Faiss-mlx Git history uses the author identity Sydney Renee,
whereas the fastEmbedR source notice records Sydney Bach, The Solace Project.
The inherited fastEmbedR notice is retained pending clarification upstream.

## Native CUDA search

The package-owned CUDA exact search, IVF-Flat search, and k-means code in
`src/native_cuda_backend.cu` is informed by the algorithmic organization of
FAISS 1.14.3 (MIT), RAPIDS cuVS commit
`ad9e2d2a617c8d51e3eebc920e5a60ad8dc59bcd` (Apache-2.0), and the native Metal
work in fastEmbedR. Its k-means initialization, seed convention, Lloyd
iteration semantics, and empty-cluster repair are adapted from FAISS 1.14.3
`faiss/Clustering.cpp` and random utilities. The source file therefore retains
the Meta Platforms notice and MIT terms. No FAISS, cuVS, RAFT, or RMM binary is
vendored or linked, and no cuVS source is included. The upstream license texts
are retained in `licenses/FAISS-LICENSE` and `licenses/CUVS-LICENSE`.

## UMAP and openTSNE

`src/visualization.cpp` and `src/embedding_cuda_kernels.cu` are adapted from
fastEmbedR's MIT implementation. Its documented mathematical and architectural
references include the BSD-licensed UMAP reference implementation, openTSNE,
t-SNE-CUDA, Rtsne behavior, opt-SNE/Multicore-opt-SNE, umappp, and ensmallen;
AppleSiliconFFT (MIT); and mlx-vis (Apache-2.0). No source from the GPL-licensed
`uwot` package is included in the core. See `PROVENANCE.md` for the complete
classification and the rule governing future adaptations. The current port
includes fastEmbedR's direct binary/fuzzy CSR construction, smooth-kNN
bandwidth calculation, epoch scheduler, CPU float32 openTSNE optimizer, and
CUDA embedding kernels.

## PCA

`src/pca.cpp`, `src/pca_cuda.cu`, and their public wrappers implement a
standalone float32 randomized PCA using the current fastEmbedR/fastPLS
strategy for Gaussian sketches, power iterations, backend-specific automatic
settings, score/loadings orientation, and explained-variance reporting. The
implementation does not link either package, Armadillo, Eigen, or RAFT. The
fastPLS-derived strategy is included under Stefano Cacciatore's MIT
relicensing declaration above; the complete fastPLS publication and package
credit remain required.

## Runtime and build dependencies

CPU, Metal, and CUDA numerical builds do not link FAISS, cuVS, fastEmbedR,
fastPLS, or an R/Python runtime. Metal builds use Apple system frameworks
(Foundation, Metal, and Metal Performance Shaders). CUDA builds use CUDA
Toolkit libraries. OpenMP is optional.

No external FAISS or cuVS binary is distributed with this repository. Binary
package distributors must include notices required by the particular CUDA,
OpenMP, or platform binaries they redistribute.
