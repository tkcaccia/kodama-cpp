# CPU Source Coverage

Coverage was measured on 19 July 2026 from the `kodama-cpp` CPU test binaries
with LLVM 22.1.1 instrumentation (`-fprofile-instr-generate
-fcoverage-mapping`). CUDA and Metal sources are excluded from this report and
are validated separately on accelerator hardware.

| Measure | Covered |
|---|---:|
| Regions | 67.61% |
| Functions | 66.04% |
| Lines | 63.58% |
| Branches | 57.03% |

The measurement executes `kodama_cpp_tests` and
`kodama_public_api_0_1_tests`, merges their raw profiles, and reports project
sources while excluding tests, examples, benchmarks, and generated build
files. Reproduce it with:

```sh
bash tools/run_cpu_coverage.sh
```

The full text summary, LCOV record, and HTML source report are uploaded by the
`CPU coverage` GitHub Actions workflow. The lowest-covered substantive module
in this snapshot is graph clustering; PLS helper functions also contain paths
that are not reached by the current CPU tests. These are test-coverage gaps,
not claims about accelerator behavior.
