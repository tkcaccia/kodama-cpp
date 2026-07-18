# JMLR release-validation protocol

The release benchmark is intentionally fixed before results are inspected.
Its purpose is to test implementation claims, parameter sensitivity, wrapper
parity, and the boundary between KODAMA's internal objective and external
diagnostics.

## Fixed analysis settings

- `M = 100` independent runs
- `Tcycle = 100` proposal/evaluation cycles
- `landmarks = 100000`, with `ceil(0.75 * n)` when `n <= landmarks`
- `knn.k = 30`
- `ncomp = 50`, limited only by mathematical feasibility
- `splitting = 100` for `n < 40000`, otherwise `300`
- UMAP `k = 30`; openTSNE perplexity `30`
- fixed recorded seeds

## Comparisons

1. Current KNN versus current PLS-LDA.
2. KODAMA graph correction on versus off.
3. Default splitting versus the alternate predeclared value.
4. Single-core CPU, four-core CPU, and CUDA with identical analysis settings.
5. Current R wrapper versus current CRAN KODAMA 3.3 as a separately labeled
   end-to-end package comparison. KODAMA 3.3 ignores the deprecated `FUN`
   argument and selects a PLS-DA implementation internally, so this is not
   described as classifier parity with latent-space PLS-LDA.
6. Current KNN versus the legacy KNN-capable KODAMA 2.4.1 release on compatible
   small datasets. These jobs use `splitting = 50`, matching that release's
   internal k-means initialization. This is a KNN predecessor comparison, not a
   claim that 2.4.1 is current on CRAN; different proposal and landmark rules
   also prevent trajectory-level parity claims.
7. Classic and KODAMA-corrected UMAP/openTSNE using the same local embedding
   implementation and initialization policy.
8. Nonspatial `M` and `Tcycle` sensitivity on MetRef and USPS, varying one
   control at a time over 20, 50, and 100 while holding the other at 100.
9. Convergence of the run-wise edge-agreement estimator as `M` increases,
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

For a targeted resumable subset, set `KODAMA_JOB_REGEX` to an R regular
expression matched against the generated job ID. This filters benchmark
orchestration only; it does not change a KODAMA parameter or numerical path.
For example, `KODAMA_JOB_REGEX='historical.*knn|historical_comparison.*knn'`
runs the declared historical/current KNN comparison jobs.

## Completed legacy KNN predecessor comparison

The 2026-07-16 MetRef KNN run fixed `M = 100`, `Tcycle = 100`,
`landmarks = 100000` (655 effective under the historical 75% rule),
`splitting = 50`, `knn.k = 30`, graph `k = 100`, and seed 1234.
Wall times were 610.543 s for KODAMA R 2.4.1, 965.348 s for current
single-core CPU, 235.547 s for current four-core CPU, and 2.361 s for
current CUDA. All four rows reached best raw CV accuracy 1.000.

This is a matched-settings legacy KNN predecessor comparison, not a comparison
with the current CRAN release and not trajectory parity:
the current grouped and guarded proposal dynamics extend the historical
stochastic search. Historical PLS-DA is not used as a parity row for current
SIMPLS plus latent-space LDA.

The sensitivity stage also writes `ensemble_convergence_summary.csv` and
`ensemble_convergence.png`. Set `KODAMA_RUN_ENSEMBLE_CONVERGENCE=0` only when
the convergence diagnostic should be deferred together with its manuscript
evidence.
