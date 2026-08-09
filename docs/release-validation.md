# JMLR release-validation protocol

The release benchmark is intentionally fixed before results are inspected.
Its purpose is to test implementation claims, parameter sensitivity, wrapper
parity, and the boundary between KODAMA's internal objective and external
diagnostics.

## Fixed analysis settings

- `M = 100` independent runs
- `Tcycle = 100` proposal/evaluation cycles
- `landmarks = 100000`, with `ceil(0.75 * n)` when `n <= landmarks`
- `knn.k = 10`
- `ncomp = 50`, limited only by mathematical feasibility
- `splitting = 100` for `n < 40000`, otherwise `300`
- UMAP `k = 30`; openTSNE perplexity `30`
- fixed recorded seeds

## Comparisons

1. Current KNN versus current PLS-LDA.
2. KODAMA graph correction on versus off.
3. Default splitting versus the alternate predeclared value.
4. Single-core CPU, four-core CPU, CUDA, and local Metal with identical
   analysis settings. Cross-machine timings are reported separately rather
   than interpreted as hardware-normalized speedups.
5. Current KNN versus the legacy KNN-capable KODAMA 2.4.1/2.4 release on compatible
   small datasets. These jobs use `splitting = 50`, matching that release's
   internal k-means initialization. This is the designated predecessor baseline;
   different proposal and landmark rules prevent trajectory-level parity claims.
6. Classic and KODAMA-corrected UMAP/openTSNE using the same local embedding
   implementation and initialization policy.
7. Nonspatial `M` and `Tcycle` sensitivity on MetRef and USPS, varying one
   control at a time over 20, 50, and 100 while holding the other at 100.
8. Convergence of the run-wise edge-agreement estimator as `M` increases,
   including empirical RMSE from the `M = 100` graph and the worst-case
   Bernoulli Monte Carlo standard error `0.5 / sqrt(M)`.

## Recorded outcomes

The driver records wall, core, graph, and embedding time separately; raw best
and median CV accuracy; selected and median ARI; active class counts; graph
truth purity; truth-label silhouette; KODAMA-label silhouette; and normalized
per-class compactness. Visualizations are diagnostics, not selection criteria.

## Commands

On the CUDA release machine:

```sh
export ENV_DIR=/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs
export CONDA_PREFIX="$ENV_DIR"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

KODAMA_RELEASE_OUT=/mnt/sata_ssd/kodama-cpp-benchmarks/jmlr-release-validation \
  bash benchmarks/run_jmlr_release_validation.sh
```

The run is resumable: each completed job has a CSV row and an RDS result, and
completed job identifiers are skipped on restart.

The complete release-freeze, CUDA scheduler, visualization, provenance, and
statistical protocol is in `docs/JMLR_CUDA_HPC_EXPERIMENT_PLAN.md`.

## Full-cycle local engineering controls

Before the frozen CUDA experiment, complete shared-graph `M = 100`,
`Tcycle = 100` controls were run locally on MetRef, USPS, ImageSegmentation,
PageBlocks, and PenDigits with CPU4 and
physical Metal. These controls use one graph checksum per dataset and the same
CPU UMAP backend to isolate KODAMA optimization. They show a reproducible
workload crossover rather than universal accelerator gain: Metal accelerates
KNN on MetRef and USPS, while CPU4 is faster on ImageSegmentation and nearly
tied on PageBlocks; CPU4 is faster for complete PLS-LDA on all five datasets.
PageBlocks and PenDigits supply negative quality cases and are used for parity
and scaling evidence rather than a quality claim. The reports and raw CSV files are in
`manuscript/JMLR_CYCLE16_USPS_FULL_CPU_METAL_REPORT_2026-08-08.md` and
`manuscript/JMLR_CYCLE17_IMAGESEGMENTATION_AND_COVERAGE_REPORT_2026-08-08.md`,
and `manuscript/JMLR_CYCLE18_PAGEBLOCKS_RELEASE_REPORT_2026-08-08.md`,
with artifacts under `manuscript/jmlr_cycle16_usps_*_m100_t100_20260808/` and
`manuscript/jmlr_cycle17_imagesegmentation_m100_t100_20260808/` and
`manuscript/jmlr_cycle18_pageblocks_m100_t100_20260808/`. Because the worktree was
dirty, these rows cannot replace the tagged release experiment.

PenDigits (10,992 by 16) extends the adverse control beyond class imbalance.
Classic UMAP truth silhouette was 0.5116, while KODAMA silhouettes ranged from
-0.0832 to -0.1009 despite median CV accuracy of 0.9959--0.9967 for KNN and
0.9720--0.9735 for PLS-LDA. CPU4/Metal optimization required 4.397/4.925
seconds for KNN and 39.311/199.897 seconds for PLS-LDA. The complete record is
`manuscript/JMLR_CYCLE20_STOCHASTIC_CONTRACT_PENDIGITS_REPORT_2026-08-08.md`.

COIL20 (1,440 by 16,384) adds a high-dimensional local KNN stress control.
Metal reduced optimization wall time from 193.377 to 80.753 seconds (2.39x)
and pipeline time from 200.950 to 88.315 seconds, with identical median CV
accuracy of 0.99537. Both KODAMA truth silhouettes (0.4531/0.4278) were below
classic UMAP (0.5266/0.5303), so the result is not a quality claim. Raw CPU4
PLS-LDA exceeded 300 seconds before its first four runs completed and is
retained as a censored local resource-limit cell. See
`manuscript/JMLR_CYCLE21_ORCHESTRATION_TIMING_COIL20_REPORT_2026-08-08.md`.

## Public error-contract coverage

Cycle 19 adds observable error contracts for invalid KODAMA.matrix metadata,
malformed visualization graphs and options, empty resident indices, explicit
unavailable-backend requests, and Core state-machine invariants. The Cycle 19
LLVM snapshot was 76.45% by line, 65.19% by branch, 74.07% by function, and
76.23% by region. Portable,
physical-Metal, and AddressSanitizer tests pass. Apple ASan must use
`detect_leaks=0`; the frozen Linux validation retains LeakSanitizer. The full
record is `manuscript/JMLR_CYCLE19_ERROR_CONTRACT_COVERAGE_REPORT_2026-08-08.md`.

Cycle 20 adds fixed-seed stochastic replay, monotone best traces, guarded
one-class and anti-collapse behavior, shake recovery, and exact degenerate
PLS-LDA majority fallback. Coverage is now 76.45% line, 65.37% branch, 74.07%
function, and 76.33% region. The new assertions pass portable, ASan, and
physical-Metal validation without production changes.

Cycle 21 exercises public raw-data, graph-only, and graph-plus-data dispatch,
cosine graph construction, prepared-initialization behavior, shape errors,
malformed dissimilarity inputs, and unavailable accelerator routing. Coverage
was 77.67% line, 66.78% branch, 75.31% function, and 77.74% region. The R
wrapper also exposes exact per-run landmark times and sum/mean/median timing
aggregates; isolated installation, `testthat`, and `R CMD check` pass.

Cycle 22 adds a deterministic public binary UMAP contract for the
fastEmbedR-derived route. It verifies duplicate/self-edge compaction,
non-default `min_dist`, explicit initialization metadata, finite output,
fixed-seed replay, and self-only graph failure. Current CPU coverage is 80.35%
line, 68.79% branch, 76.73% function, and 79.82% region. ASan, physical Metal,
and installed R-wrapper tests pass with this contract.

## Wrapper relink validation

For the R wrapper, validate core-selection invalidation in addition to a clean
install. Install once, select a core archive at a different path (or rebuild
the archive), and install again without `--preclean`. The second log must
compile both `RcppExports.cpp` and `kodama_r.cpp` and relink `kodamaR`; a third
install against the unchanged archive must be a no-op at the native build
step. The generated `.kodama-core.signature` records the selected archive and
public-header checksums and is removed by the package cleanup script.

For Python, validate the built wheel rather than an editable checkout. Build
with `pip wheel --no-deps --no-build-isolation`, install it into a clean virtual
environment or isolated target directory, print both `kodama.__file__` and
`kodama._core.__file__`, and only then run `pytest`. A developer machine can
contain scikit-build's `ScikitBuildRedirectingFinder`; if either printed path
points to the source tree or global site-packages, the wheel has not been
isolated and the result is invalid. For an installed core, prefer
`CMAKE_ARGS='-Dkodama-cpp_DIR=/path/to/prefix/lib/cmake/kodama-cpp'`; replacing
`CMAKE_PREFIX_PATH` can hide scikit-build's isolated `pybind11` package. The
Cycle-18 installed-core wheel passed all eight tests with physical Metal
access.

For a targeted resumable subset, set `KODAMA_JOB_REGEX` to an R regular
expression matched against the generated job ID. This filters benchmark
orchestration only; it does not change a KODAMA parameter or numerical path.
For example, `KODAMA_JOB_REGEX='historical.*knn|historical_comparison.*knn'`
runs the declared historical/current KNN comparison jobs.

## Preliminary legacy KNN predecessor comparison

The 2026-07-16 MetRef KNN run fixed `M = 100`, `Tcycle = 100`,
`landmarks = 100000` (655 effective under the historical 75% rule),
`splitting = 50`, `knn.k = 10`, graph `k = 100`, and seed 1234.
Wall times were 610.543 s for KODAMA R 2.4.1, 965.348 s for current
single-core CPU, 235.547 s for current four-core CPU, and 2.361 s for
current CUDA. All four rows reached best raw CV accuracy 1.000.

This is a matched-settings KODAMA 2.4.1/2.4 predecessor comparison, not trajectory parity:
the current grouped and guarded proposal dynamics extend the historical
stochastic search. Historical PLS-DA is not used as a parity row for current
SIMPLS plus latent-space LDA.

These are single-run preliminary rows. A later diagnostic used the same
predecessor release but a different worktree/hardware state and produced
different wall times; it is retained for audit but is not substituted into the
manuscript table. The frozen release experiment must run at least three fresh
processes per backend, report within-host medians and dispersion, and preserve
its exact source and wrapper identities.

The sensitivity stage also writes `ensemble_convergence_summary.csv` and
`ensemble_convergence.png`. Set `KODAMA_RUN_ENSEMBLE_CONVERGENCE=0` only when
the convergence diagnostic should be deferred together with its manuscript
evidence.
