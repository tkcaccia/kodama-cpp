# Consensus-graph validation

This protocol validates PLS-LDA KODAMA on MetRef, MERFISH, and Br8100 with
one production rule across datasets:

- `M = 100`, `Tcycle = 100`, five grouped folds;
- 50 SIMPLS components where mathematically feasible;
- spatial k-means constraint blocks at `spatial.resolution = 0.3`;
- equal weighting of all M label vectors in the corrected graph;
- deterministic exact-K Walktrap on the corrected graph;
- UMAP used only for visualization.

The connected-region alternative remains available through
`spatial.constraint.mode = "graph"`, but is not the production default. In the
controlled August 2026 validation it reduced Br8100 and MERFISH ARI relative
to spatial k-means blocks. Transition evidence is calculated from sample-level
out-of-fold predictions while each spatial block remains an atomic proposal
unit.

The runner checkpoints the corrected graph, all M label vectors, initialization,
and diagnostics immediately after `KODAMA.matrix()`. Spatial results are shown
with `KODAMAextra::plot_slide()`, the fixed publication palette, and ARI for
every slide. Visualization or clustering failures therefore do not erase the
completed KODAMA optimization.

Truth labels provide the requested K and are used only after clustering for
evaluation. No algorithm, resolution, seed, or M solution is selected by ARI.

## Accepted control results (2026-08-22)

All rows use PLS-LDA, `M = 100`, `Tcycle = 100`, five folds, and the fixed
seed. Times below are for `KODAMA.matrix()` only.

| Dataset | Backend | Matrix seconds | ARI | NMI | Silhouette | Median M classes |
|---|---:|---:|---:|---:|---:|---:|
| MetRef | CPU (4 cores) | 1043.50 | 0.9271 | 0.9723 | 0.8417 | 25.5 |
| MetRef | Metal | 294.43 | 0.9258 | 0.9716 | 0.8323 | 25.0 |
| MetRef | CUDA | 62.47 | 0.9237 | 0.9699 | 0.7479 | 25.0 |
| MERFISH | CUDA | 77.72 | 0.5587 | 0.6392 | 0.2034 | 5.0 |
| Br8100 | CUDA | 60.47 | 0.4589 | 0.5794 | 0.3070 | 3.0 |

MERFISH slide ARIs were 0.6095, 0.6174, 0.5850, 0.5194, and 0.4963.
Br8100 slide ARIs were 0.5190, 0.4607, 0.4346, and 0.4350. The fixed-palette
`plot_slide()` figures are stored with the benchmark outputs.

Run a local smoke test:

```sh
KODAMA_BACKEND=cpu KODAMA_M=2 KODAMA_TCYCLE=2 \
KODAMA_DATASETS=MetRef \
Rscript benchmarks/consensus_graph_validation/run_plslda_validation.R
```

Run the release CUDA validation on chiamaka after configuring the documented
CUDA environment:

```sh
KODAMA_BACKEND=cuda KODAMA_M=100 KODAMA_TCYCLE=100 \
KODAMA_DATASETS=MetRef,MERFISH,Br8100 \
KODAMA_SPATIAL_CONSTRAINT_MODE=kmeans \
KODAMA_OUT=/mnt/sata_ssd/kodama-cpp-benchmarks/consensus_graph_validation \
Rscript benchmarks/consensus_graph_validation/run_plslda_validation.R
```
