# CPU Source Coverage

Coverage was remeasured on 8 August 2026 from the `kodama-cpp` CPU test binaries
with LLVM 22.1.1 instrumentation (`-fprofile-instr-generate
-fcoverage-mapping`). CUDA and Metal sources are excluded from this report and
are validated separately on accelerator hardware.

| Measure | Covered |
|---|---:|
| Regions | 79.98% |
| Functions | 77.50% |
| Lines | 80.48% |
| Branches | 69.33% |

The measurement executes `kodama_cpp_tests` and
`kodama_public_api_0_1_tests`, uses the substantive numerical-test profile for
source coverage, and reports project sources while excluding tests, examples,
benchmarks, and generated build files. Reproduce it with:

```sh
bash tools/run_cpu_coverage.sh
```

The full text summary, LCOV record, and HTML source report are uploaded by the
`CPU coverage` GitHub Actions workflow. The lowest-covered substantive module
in this snapshot is now the matrix orchestration layer. Cycle 17 added public
contract tests for graph weighting, index bases, pruning, embedding bridging,
target errors, and resident-IVF failure behavior. `graph_cluster.cpp` now has
94.65% line and 73.94% branch coverage. Cycle 19 added public invalid-input,
visualization-option, spatial-metadata, empty-resident-index, unavailable-
backend, and Core state-machine invariant contracts without changing production
code. Accelerator-only Metal stubs and
resident-index code remain in the conservative CPU denominator; CUDA and
physical Metal kernels are validated separately on hardware.

Cycle 20 added fixed-seed replay, monotone best-trace, guarded one-class,
anti-collapse, shake-recovery, PLS-LDA input, and constant-predictor majority-
fallback contracts. These behavioral assertions raise branch coverage to
65.37% and region coverage to 76.33% without changing production code; line
coverage is unchanged because the affected stochastic paths were already
executable.

Cycle 21 added generic raw-data, graph-only, and graph-plus-data matrix
dispatch; CPU cosine graph construction; prepared-graph initialization;
shape-mismatch; malformed dissimilarity; non-identity constraint; and
unavailable CUDA/Metal routing contracts. Total line/branch coverage rose to
77.67%/66.78%, while `kodama_matrix.cpp` rose from 69.98%/57.61% to
74.83%/64.07%. No production numerical source was changed.

Cycle 22 added a public binary-graph UMAP contract covering duplicate and
self-edge compaction, non-default `min_dist` curve fitting, explicit
initialization metadata, finite float32 output, fixed-seed replay, and failure
when no usable non-self edge exists. Total line/branch coverage rose to
80.35%/68.79%, while `visualization.cpp` rose from 70.32%/62.03% to
79.38%/70.57%. Production numerical source again remained unchanged.

Cycle 26 added resident-IVF null-view, non-positive-k, unavailable-backend,
move-construction, and move-assignment contracts. Total line/branch coverage is
now 80.48%/69.33%; `resident_ivf.cpp` is 47.79%/39.58%. The physical Metal
suite separately exercises native resident-index ownership and explicit
failure behavior, which cannot contribute to the portable CPU denominator.

The public-API executable is still run as a compile/link test but is not merged
into source coverage. It contributes no additional project-source lines and
defines a different function named `main`, which otherwise produces an LLVM
profile-hash warning when combined with the numerical test executable.
