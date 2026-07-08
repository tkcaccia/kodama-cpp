# kodama-cpp

Standalone C++/CUDA kernels for high-performance KODAMA-related algorithms.

The C++ core is independent from R and Python. Thin wrappers live in separate
repositories:

- `tkcaccia/kodama-r`
- `tkcaccia/kodama-python`

Current core scope:

- KODAMA matrix optimization with KNN or PLS-LDA classifiers
- KNN cross-validation
- PLS-DA / PLS-LDA cross-validation
- KNN graph, UMAP/openTSNE embedding, and graph clustering helpers

## Build

CPU build:

```sh
cmake -S . -B build -DKODAMA_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA build:

```sh
cmake -S . -B build-cuda -DKODAMA_ENABLE_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

## Install

```sh
cmake --install build --prefix /path/to/kodama-cpp-install
```

The install exports a CMake package:

```cmake
find_package(kodama-cpp CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE kodama::kodama_cpp)
```

## License

MIT. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
