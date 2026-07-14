# Third-Party and Source Acknowledgements

`kodama-cpp` is released under the MIT License.

Portions of the PLS/SIMPLS, visualization, graph search, and KODAMA optimization code are
adapted from projects developed by Stefano Cacciatore, including `fastPLS`,
`fastEmbedR`, `faissR`, and `KODAMA`. These portions are relicensed here under MIT by the
copyright holder.

The package-owned CPU HNSW implementation in `src/native_knn.cpp` was distilled from
the HNSW organization in FAISS 1.14.3, commit
`0ca9df4792b173d573044ee14ca0704780176e82` (MIT). The complete FAISS license is in
`licenses/FAISS-LICENSE`.

The native Objective-C++/Metal nearest-neighbor implementation in
`src/metal_backend.mm` is adapted from `fastEmbedR` and informed by FAISS 1.14.3 and
MLXPorts/Faiss-mlx commit `d092af559375144fc719cd88a10e414f92c625fa`.
The FAISS MIT license and Faiss-mlx Apache-2.0 license are retained in
`licenses/FAISS-LICENSE` and `licenses/FAISS-MLX-LICENSE`. The Metal SIMPLS path is
adapted from `fastPLS` and relicensed here under MIT by its copyright holder.

The package-owned CUDA exact search, IVF-Flat search, and k-means code in
`src/native_cuda_backend.cu` is informed by the algorithmic organization of
FAISS 1.14.3 (MIT), RAPIDS cuVS (Apache-2.0), and the native Metal work in
`fastEmbedR`. No FAISS, cuVS, RAFT, or RMM source is copied or linked. The
complete upstream license texts are retained in `licenses/FAISS-LICENSE` and
`licenses/CUVS-LICENSE`.

CPU, Metal, and CUDA numerical builds do not link FAISS, cuVS, fastEmbedR,
fastPLS, or an R/Python runtime. Metal builds use only Apple system frameworks
(Foundation, Metal, and Metal Performance Shaders). CUDA builds use CUDA
Toolkit libraries.

External runtime/build dependencies keep their own licenses:

- CUDA Toolkit / cuBLAS / cuSOLVER / cuRAND / cuFFT
- optional RAPIDS cuGraph clustering adapter
- OpenMP implementation used by the platform

No external FAISS or cuVS binary is distributed with this repository. Builds
that explicitly enable the optional cuGraph adapter link an installation
supplied by the build environment.

Distributors of binary packages should include the corresponding dependency
license notices required by their packaging channel.
