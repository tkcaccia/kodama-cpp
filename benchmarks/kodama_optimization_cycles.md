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

## Cycle 5: CUDA Auto M Workers and KNN Vote Scratch Reuse

Thought: KODAMA M runs are independent, but the right number of concurrent
CUDA M workers depends on the GPU, dataset size, graph width, classifier, and
free memory. Manual `n.cores=2` helped in Cycle 3, but it should not be a
hard-coded benchmark setting. The library should have an opt-in automatic mode
that selects a conservative worker count from GPU capacity. Separately, KNN
core prediction repeatedly votes over the same precomputed CV neighbor lists;
the vote scratch buffers can be reused across T cycles without changing the
neighbor search, labels, vote rule, or objective.

Implementation:

- `n.cores=0` with CUDA enables automatic M-worker selection.
- Explicit `n.cores > 0` remains manual and unchanged.
- The selector reads CUDA free/total memory and SM count, estimates per-worker
  memory from `n`, `p`, landmarks, graph neighbors, classifier, components, and
  KNN `k`, then chooses a conservative worker count.
- The selected worker count and GPU diagnostics are returned in the R wrapper
  and written by the benchmark script.
- `CoreKNN_CPU` and `CoreKNN_CUDA` now reuse label-code and vote buffers across
  repeated predictions; KNN CV neighbors are still computed once before the T
  cycle and the vote mathematics is unchanged.

Decision: Accepted. This is a clean CUDA scheduling and KNN implementation
improvement. On chiamaka the auto selector chose 2 workers for the 36-SM,
~16 GB GPU. Compared with manual one-worker KNN, auto improved KODAMA wall time
while preserving median accuracy and median ARI. MERFISH labels were identical
between manual and auto runs. Br8100 had a few changed M solutions, but median
row agreement was 1 and median ARI was slightly higher with auto.

Evidence, KNN CUDA, M=100, PCA20 input, `knn.k=30`:

| dataset | workers | selected workers | KODAMA sec | speedup vs 1 worker | median acc | median ARI | UMAP silhouette | compactness min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MERFISH | 1 | 1 | 38.815 | 1.000 | 0.472050 | 0.221672 | 0.273261 | 0.605142 |
| MERFISH | 0 auto | 2 | 32.442 | 1.196 | 0.472050 | 0.221672 | 0.266715 | 0.588225 |
| Br8100 | 1 | 1 | 25.397 | 1.000 | 0.818300 | 0.337290 | 0.241789 | 0.678282 |
| Br8100 | 0 auto | 2 | 20.201 | 1.257 | 0.818300 | 0.338477 | 0.239645 | 0.679925 |

Auto-worker diagnostics:

| dataset | SM count | free memory MiB | total memory MiB | worker estimate MiB |
| --- | ---: | ---: | ---: | ---: |
| MERFISH | 36 | 15180.81 | 15841.31 | 527.51 |
| Br8100 | 36 | 14794.81 | 15841.31 | 523.12 |

Step timing highlights for auto:

| dataset | KODAMA optimization wall sec | KODAMA graph sec | KODAMA dissimilarity sec | UMAP sec | compactness sec | total sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MERFISH | 32.033 | 0.190 | 0.016 | 0.445 | 0.018 | 45.569 |
| Br8100 | 20.001 | 0.098 | 0.009 | 0.187 | 0.005 | 25.073 |

Next GPU traffic target: the full input matrix is still supplied to FAISS
k-means from host memory in each M run. FAISS GPU indices can consume device
pointers, but FAISS `Clustering` still performs centroid updates at the host
algorithm level, so true matrix residency for landmark k-means should be done
as a dedicated cuVS/custom GPU k-means path rather than by passing a device
pointer into the current FAISS clustering call.
