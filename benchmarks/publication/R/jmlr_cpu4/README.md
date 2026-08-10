# JMLR MLOSS CPU4 Benchmark Protocol

This is the publication copy of the four-core CPU benchmark suite. It tests
the release implementation at `M=100`, `Tcycle=100`, five folds, and seeds 4,
17, and 42. It covers reusable graph construction, KNN and PLS-LDA KODAMA,
classic fuzzy UMAP and openTSNE, isolated evolution-policy ablations, KNN and
PLS-LDA predictor sensitivity, and the ImageNet raw/PCA50 comparison.

The suite records stage timings, peak memory, external label metrics,
neighborhood preservation, run and cycle diagnostics, graph/data identities,
failures, package versions, source commits, and dataset checksums. Truth labels
are loaded for post-hoc evaluation and are not passed to KODAMA fitting.

## Generate The HPC Run

From a clone of `kodama-cpp` on the HPC:

```bash
ROOT=/scratch/firenze/NN
CODE=$ROOT/kodama-cpp/benchmarks/publication/R/jmlr_cpu4

Rscript "$CODE/generate_slurm.R" \
  --root="$ROOT" \
  --code-dir="$CODE" \
  --release=RELEASE_TAG \
  --image="$ROOT/singularity/fastembedr_cuda.sif" \
  --core-sha=FULL_KODAMA_CPP_SHA \
  --wrapper-sha=FULL_KODAMA_R_SHA \
  --account=immunology \
  --partition=ada
```

Generation does not submit jobs. It creates seven explicit phase launchers.
Run one phase only after the preceding phase has completed successfully:

```bash
RUN=$ROOT/kodama_cpu4_jmlr_RELEASE_TAG

bash "$RUN/submit_01_preflight.sh"
bash "$RUN/submit_02_pca.sh"
bash "$RUN/submit_03_graphs.sh"
bash "$RUN/submit_04_smoke.sh"
bash "$RUN/submit_05_validate.sh"
bash "$RUN/submit_06_full.sh"
bash "$RUN/submit_07_aggregate.sh"
```

Graph preparation submits 12 bundled jobs; smoke and full phases submit 36
bundled jobs each. Every array permits at most four concurrent jobs. The
launchers contain no background controller, dependency chain, polling loop,
or automatic retry.

## Files

- `datasets.csv`: publication dataset registry.
- `preflight.R`: API, dependency, dataset, embedding, and hidden-policy checks.
- `prepare_imagenet_pca.R`: label-blind ImageNet PCA50 preparation.
- `prepare_graph.R`, `prepare_graph_bundle.R`: reusable graph construction.
- `build_cells.R`: complete statistical-cell design.
- `run_cell.R`, `run_cell_bundle.R`: benchmark execution and checkpoints.
- `validate_smoke.R`: diagnostic schema, identity, and CV-count gate.
- `aggregate.R`: paired effects, sensitivity summaries, tests, and figures.
- `common.R`: loading, metrics, hashing, atomic output, and API helpers.
- `generate_slurm.R`, `run_worker.sh`: scheduler and container launch support.
