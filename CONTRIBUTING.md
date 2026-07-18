# Contributing to kodama-cpp

kodama-cpp keeps one mathematical contract across CPU, CUDA, and Apple Metal.
Contributions are welcome when they preserve that contract and include enough
evidence to distinguish numerical parity from performance.

## Development setup

Start with the dependency-free CPU build:

```sh
cmake -S . -B build-cpu \
  -DKODAMA_ENABLE_CUDA=OFF \
  -DKODAMA_ENABLE_METAL=OFF \
  -DKODAMA_BUILD_TESTS=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

On macOS, add a separate Metal build with `KODAMA_ENABLE_METAL=ON`. On a CUDA
machine, use a separate build directory with `KODAMA_ENABLE_CUDA=ON`.

## Change requirements

- Keep numerical analysis and accelerator workspaces float32.
- Do not silently fall back from CUDA or Metal to CPU.
- Preserve independent `M` runs and one CV evaluation per proposal cycle.
- Keep KNN and PLS-LDA as separate, clean classifier implementations.
- Add focused tests for behavior, backend identity, and public API changes.
- Report accuracy or parity together with runtime for performance changes.
- Add SPDX headers and update `PROVENANCE.md` for copied or adapted code.
- Update the manuscript when an accepted change affects the algorithm, public
  API, backend coverage, dependency model, or reported performance.

Run before opening a pull request:

```sh
bash tools/check_license_headers.sh
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
git diff --check
```

## Pull requests

Describe the mathematical behavior before and after the change, the datasets
and seeds used for validation, and the exact build commands. Keep benchmark
artifacts out of source commits unless they are small release evidence files.

## Licensing

By contributing, you agree that your contribution is distributed under the
license expression stated in the contributed file. New project-authored files
normally use MIT. Adapted files must retain all upstream notices and compatible
license terms.
