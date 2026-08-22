# KODAMA JMLR MLOSS CPU4 benchmark suite

This suite prepares the CPU-only release benchmark, isolated evolution-policy
ablations, predictor sensitivity experiments, and ImageNet raw/PCA50 comparison.
It does not submit jobs during generation.

## Fixed protocol

- four CPU workers per process; nested BLAS threading disabled;
- `M=100`, `Tcycle=100`, five folds, seeds 4/17/42;
- `landmarks=100000`, graph `k=100`, KODAMA KNN default `k=30`;
- splitting 100 below 40,000 samples and 300 otherwise;
- classic fuzzy UMAP `k=30` and openTSNE perplexity 30;
- truth labels are loaded only after fitting for external diagnostics;
- graph construction occurs once per dataset/representation/seed;
- ImageNet PCA50 is calculated once without labels;
- native hidden evolution policies are tested, never emulated in R;
- smoke `M=2,Tcycle=2` and schema/identity checks gate the full array;
- the canonical package is `KODAMA >= 0.99.6`; legacy `kodamaR` is rejected;
- native progress reports M, Tcycle, accuracy, active classes, acceptance, and elapsed time;
- `KODAMA.matrix()` retains its corrected graph as an in-process handle, so
  visualization neither rebuilds the graph nor materializes duplicate matrices.

## Install into the synchronized HPC directory

The local mirror is `/Users/stefano/HPC-firenze/NN/kodama_cpu4_jmlr`; the HPC
path is `/scratch/firenze/NN/kodama_cpu4_jmlr`.

Generate scripts on the HPC:

```bash
cd /scratch/firenze/NN
Rscript kodama_cpu4_jmlr/generate_slurm.R \
  --root=/scratch/firenze/NN \
  --code-dir=/scratch/firenze/NN/kodama_cpu4_jmlr \
  --release=RELEASE_TAG \
  --image=/scratch/firenze/NN/singularity/fastembedr_cuda.sif \
  --core-sha=FULL_KODAMA_CPP_SHA \
  --wrapper-sha=FULL_KODAMA_R_SHA
```

When generating from the synchronized Mac folder, set `--code-dir` to the
local folder and add
`--runtime-code-dir=/scratch/firenze/NN/kodama_cpu4_jmlr`. Also set
`--run-root` to the HPC path and `--output-dir` to its local synchronized
counterpart. Generated jobs then contain only HPC runtime paths.

The Firenze defaults are `--account=immunology --partition=ada`, matching the
existing CPU benchmark launchers. They can be overridden explicitly with
`--account=ACCOUNT --partition=PARTITION` for another cluster.

Inspect the generated manifests and scripts. The recommended launcher submits
every phase with explicit `afterok` dependencies, so graph consumers cannot
start before graph production and aggregation cannot start before both full
arrays finish:

```bash
RUN=/scratch/firenze/NN/kodama_cpu4_jmlr_RELEASE_TAG
bash "$RUN/submit_pipeline.sh"
```

Individual phase launchers remain available for diagnosis. When using them,
wait until each phase has completed successfully before submitting the next:

```bash
RUN=/scratch/firenze/NN/kodama_cpu4_jmlr_RELEASE_TAG
bash "$RUN/submit_01_preflight.sh"
bash "$RUN/submit_02_pca.sh"
bash "$RUN/submit_03_graphs_standard.sh"
bash "$RUN/submit_04_graphs_large.sh"
bash "$RUN/submit_05_smoke_standard.sh"
bash "$RUN/submit_06_smoke_large.sh"
bash "$RUN/submit_07_validate.sh"
bash "$RUN/submit_08_full_standard.sh"
bash "$RUN/submit_09_full_large.sh"
bash "$RUN/submit_10_complete.sh"
bash "$RUN/submit_11_aggregate.sh"
```

Use `bash "$RUN/status.sh"` to display only this benchmark's jobs. Each submit
script invokes `sbatch` once and returns immediately. `submit_pipeline.sh`
uses native Slurm dependencies; it has no background controller or polling
loop. A failed phase leaves all dependent phases pending with
`DependencyNeverSatisfied`, preventing invalid partial aggregation.

The scheduler sees balanced bundles rather than one job per statistical cell:
12 graph jobs, 36 smoke jobs, 72 standard full jobs, and 24 high-cost full
jobs. Standard jobs request four CPUs and 32 GB, preventing memory-per-CPU
policy from inflating them to 32 allocated CPUs. FlowRepository and raw
ImageNet use 64 GB in a separate array capped at four concurrent tasks. No
phase runs more than 20 KODAMA jobs concurrently and only 96 full jobs exist.

Dependency order is preflight -> ImageNet PCA50 -> reusable graphs -> smoke ->
schema validation -> standard full array -> large full array -> aggregation.

Each native call writes progress directly to its Slurm log. With `Tcycle=100`,
the core reports at cycle 0 and every five cycles. Bundle status is rewritten
atomically after every cell, so `bundle_status/<phase>/bundle_<id>.csv` shows
completed work while the job is still running. Successful cells are skipped
on restart; failed or interrupted cells are rerun.

## Checkpoints and recovery

The worker saves labels, run diagnostics, cycle diagnostics, summary metrics,
and matrix timing before visualization. Preflight executes the complete
`KODAMA.matrix()` to UMAP/openTSNE handoff, including the retained graph handle.
This prevents a visualization interface error from being discovered only after
an expensive full optimization.

`recover_cell.R` reconstructs the exact KODAMA graph dissimilarity from a saved
`M`-run label matrix and the prepared base graph, then runs only visualization
and post-hoc metrics. `recover_failed.R` applies this operation to every failed
cell with a valid label checkpoint. It never reruns `KODAMA.matrix()`.

For a partially completed run, `generate_recovery_slurm.R` inventories graph
caches and cell outputs, retains successful cells, and creates a dependency-
chained recovery under `RUN/recovery`. The chain performs missing graph builds,
fresh smoke validation, reruns only missing/failed cells, refreshes silhouettes,
validates all expected cells, and finally aggregates:

```bash
CODE=/scratch/firenze/NN/benchmark_code/kodama0996_ecd4f6c_1df098f_clean1
RUN=/scratch/firenze/NN/kodama_cpu4_jmlr_kodama0996_ecd4f6c_1df098f_clean1
IMAGE=/scratch/firenze/NN/singularity/fastembedr_cuda.sif

apptainer exec --cleanenv \
  --bind /scratch/firenze/NN:/scratch/firenze/NN \
  "$IMAGE" /opt/r46/bin/Rscript "$CODE/generate_recovery_slurm.R" \
  --run-root="$RUN" \
  --code-dir="$CODE" \
  --runtime-code-dir="$CODE" \
  --image="$IMAGE" \
  --data-root=/scratch/firenze/NN/Data \
  --account=immunology \
  --partition=ada

cat "$RUN/recovery/recovery_manifest.json"
bash "$RUN/recovery/submit_recovery.sh"
```

`complete_validation.csv` and the `COMPLETE_OK` sentinel are written only when
all 972 expected cells have successful metrics, finite UMAP/openTSNE layouts,
finite truth-label silhouettes, the expected 10,100 CV evaluations, no
single-class collapse, and multiple independent M solutions.

## Outputs

Each cell stores metrics, stage timings, run/cycle diagnostics, labels, UMAP,
openTSNE, identities, warnings, and exit status atomically. The aggregate stage
creates paired full-minus-ablation tables, sensitivity tables, ImageNet
raw/PCA50 tables, statistical tests, failure records, and overview figures.

Classic fastEmbedR fuzzy UMAP and openTSNE are evaluated against both KODAMA
KNN and KODAMA PLS-LDA embeddings. Post-hoc quality metrics include truth-label
silhouette (overall and worst class), Davies-Bouldin, Calinski-Harabasz,
sampled Dunn, within/between dispersion, trustworthiness, continuity,
neighborhood preservation, label-neighbor accuracy, mean neighbor-rank error,
distance correlation and stress, density preservation, centroid-distance
correlation, and rare-class recall. The aggregator reports raw and
direction-adjusted paired KODAMA-minus-classic effects with dataset-level
medians, bootstrap intervals, Wilcoxon/sign tests, and Holm correction.

Silhouette is calculated exactly by a package-independent blockwise
implementation. It matches `cluster::silhouette()` while avoiding both an
optional R dependency and allocation of the complete distance matrix.

The benchmark is intentionally strict. Missing packages, datasets, policies,
diagnostics, graph/data identity, or expected CV counts stop the dependency
chain rather than causing fallback.
