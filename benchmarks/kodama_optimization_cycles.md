# KODAMA Matrix Optimization Cycles

This log records optimization cycles for the standalone `KODAMA.matrix`
implementation. Each cycle keeps the KODAMA mathematics and visualization
initialization unchanged unless explicitly noted. Full validation uses M=100,
step timings, M-label diagnostics, final UMAP silhouette, and per-truth-label
compactness on the final plot.

## Baseline Contract

- Benchmark output root: `/mnt/sata_ssd/kodama-cpp-benchmarks`
- Current primary benchmark script: `tools/run_plslda_pca_m100_timed_benchmark.R`
- Required diagnostics:
  - total KODAMA runtime and step timings;
  - UMAP runtime;
  - median CV accuracy;
  - median ARI of each KODAMA label vector against truth;
  - number of KODAMA classes across M runs;
  - pairwise ARI among M label vectors;
  - final-plot truth-label silhouette;
  - per-truth-label compactness on the final plot.
- Acceptance rule: a large quality improvement can outweigh a moderate speed
  regression. When quality is unchanged, speed decides. Quality is judged from
  median CV accuracy, median ARI against truth, final-plot silhouette, and
  per-truth-label compactness.

## Cycle 1: PLS-LDA PCA20 Input

Thought: PLS-LDA was unstable and slow on high-variable data. A PCA score input
with 20 components keeps the downstream KODAMA and PLS-LDA CV mathematics
unchanged while reducing the dimension used by every CV fit.

Decision: Accepted as a strong candidate for MERFISH, MNIST, and a speed/ARI
candidate for Br8100. It is not forced as a universal default.

Evidence:

| dataset | variant | KODAMA sec | total sec | median ARI | UMAP silhouette | compactness min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MERFISH | raw | 643.739 | 653.202 | 0.078815 | 0.185378 | 0.610204 |
| MERFISH | pca20 | 279.282 | 288.403 | 0.207211 | 0.259673 | 0.628530 |
| Br8100 | raw | 556.268 | 561.076 | 0.287796 | 0.302283 | 0.730744 |
| Br8100 | pca20 | 259.031 | 263.851 | 0.305622 | 0.281394 | 0.721051 |
| MNIST | pca20 | 538.495 | 566.832 | 0.066169 | 0.435103 | 0.712651 |

## Cycle 2: Direct Fixed-Component PLS-LDA Prediction Storage

Thought: fixed-component PLS-LDA evaluates one component count, but the CV code
still allocated prediction storage for all possible component prefixes. Avoiding
that storage looked like a clean speed improvement without changing predicted
labels.

Decision: Rejected. It did not improve speed and slightly worsened final plot
compactness in the M=100 validation.

Evidence:

| dataset | variant | baseline KODAMA sec | candidate KODAMA sec | quality result |
| --- | --- | ---: | ---: | --- |
| MERFISH | pca20 | 279.282 | 279.957 | same median ARI, worse compactness min |
| Br8100 | pca20 | 259.031 | 259.309 | same median ARI, worse silhouette and compactness min |

## Cycle 3: Two Concurrent CUDA M Workers

Thought: M runs are independent. Running two CUDA-backed M cycles concurrently
can reduce wall time without changing fold construction, proposals, PLS-LDA
math, spatial constraints, KODAMA dissimilarity, or visualization
initialization. This should be treated as a scheduling option, not as an
algorithmic change.

Decision: Accepted as a tested optional CUDA scheduling setting for MERFISH,
Br8100, and MNIST PCA20. It is not promoted as a hidden universal default:
callers should still choose the requested number of M workers explicitly.
Br8100 and MNIST trade a small amount of compactness for the speed gain.

Evidence:

| dataset | variant | workers | KODAMA sec | speedup | median ARI | classes | UMAP silhouette | compactness min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MERFISH | pca20 | 1 | 279.282 | 1.000 | 0.207211 | 52 | 0.259673 | 0.628530 |
| MERFISH | pca20 | 2 | 242.551 | 1.151 | 0.207211 | 52 | 0.262248 | 0.638624 |
| Br8100 | pca20 | 1 | 259.031 | 1.000 | 0.305622 | 37 | 0.281394 | 0.721051 |
| Br8100 | pca20 | 2 | 226.457 | 1.144 | 0.305622 | 37 | 0.275143 | 0.709127 |
| MNIST | pca20 | 1 | 538.495 | 1.000 | 0.066169 | 298 | 0.435103 | 0.712651 |
| MNIST | pca20 | 2 | 415.366 | 1.296 | 0.066169 | 298 | 0.439905 | 0.709147 |

## Cycle 4: Direct Column-Major CUDA Response Matrix

Thought: the CUDA PLS-LDA path used the same centered one-hot response as the
CPU path, then converted it to column-major storage before launching SIMPLS.
The CUDA backend already consumes column-major matrices, so constructing the
centered one-hot response directly in that layout removes a full matrix copy
without changing the labels, centering, SIMPLS math, LDA decoder, spatial
constraints, KODAMA dissimilarity, or visualization initialization.

Decision: Accepted. It is a clean implementation improvement for CUDA PLS-LDA.
Median ARI is unchanged or slightly higher. KODAMA wall time improves on all
three M=100 validation datasets, with the largest gain on MNIST, where the
class response matrix is widest. MERFISH has a small worst-label compactness
tradeoff, but unchanged labels and better silhouette; MNIST has a small
silhouette tradeoff but unchanged median ARI and better worst-label
compactness.

Evidence versus the accepted Cycle 3 two-worker CUDA baseline:

| dataset | variant | baseline KODAMA sec | candidate KODAMA sec | speedup sec | baseline median ARI | candidate median ARI | baseline silhouette | candidate silhouette | baseline compactness min | candidate compactness min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MERFISH | pca20 | 242.551 | 237.666 | 4.885 | 0.207211 | 0.207211 | 0.262248 | 0.265871 | 0.638624 | 0.629097 |
| Br8100 | pca20 | 226.457 | 222.350 | 4.107 | 0.305622 | 0.305625 | 0.275143 | 0.279176 | 0.709127 | 0.725800 |
| MNIST | pca20 | 415.366 | 352.204 | 63.162 | 0.066169 | 0.066169 | 0.439905 | 0.438118 | 0.709147 | 0.709224 |

Step timing highlights:

| dataset | KODAMA optimization wall sec | KODAMA graph sec | KODAMA dissimilarity sec | UMAP sec | compactness sec | total sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MERFISH | 237.202 | 0.197 | 0.016 | 0.444 | 0.018 | 250.808 |
| Br8100 | 222.157 | 0.095 | 0.009 | 0.177 | 0.005 | 227.326 |
| MNIST | 351.753 | 0.322 | 0.063 | 0.864 | 0.024 | 380.475 |

Worst truth-label compactness on the final UMAP:

| dataset | worst truth label | n | compactness |
| --- | --- | ---: | ---: |
| MERFISH | PVH | 3173 | 0.629097 |
| Br8100 | Layer6 | 2430 | 0.725800 |
| MNIST | 1 | 7877 | 0.709224 |
