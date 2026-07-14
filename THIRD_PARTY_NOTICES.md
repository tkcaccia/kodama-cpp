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

CPU and Metal builds do not link FAISS, cuVS, fastEmbedR, fastPLS, or an R/Python
runtime. Metal builds use only Apple system frameworks (Foundation, Metal, and Metal
Performance Shaders).

External runtime/build dependencies keep their own licenses:

- optional FAISS CPU/GPU backends
- CUDA Toolkit / cuBLAS / cuSOLVER / cuRAND / cuFFT
- optional RAPIDS cuVS and cuGraph backends
- OpenMP implementation used by the platform

No FAISS or RAPIDS source or binary is copied into this repository. CUDA builds that
enable those adapters link an installation supplied by the build environment. The cuVS
Apache-2.0 license is retained in `licenses/CUVS-LICENSE` for those distributions.

Distributors of binary packages should include the corresponding dependency
license notices required by their packaging channel.
