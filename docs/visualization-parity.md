# Visualization parity

kodama-cpp tracks the UMAP and openTSNE implementation at fastEmbedR commit
`814350a5ca69b0c26e6df40377636f109055f84b`. The audited upstream source
hashes are recorded in `PROVENANCE.md`.

On 8 August 2026, fastEmbedR `main` advanced to
`5248ee02376c01bd6e1be788280db4e978623eed`. A follow-up source audit found no
changed native UMAP optimizer file. The complete exported float32
`knn_tsne_opentsne_float_cpp` function had the identical SHA-256
`7fc28f97762cc9a0163a0702a2c03159244a5f0cb1899238d7e0e1a22d688887` in both
revisions. The newer commit reorganizes R-level t-SNE controls, adds native PCA
orchestration, removes the unused double t-SNE path, and adds graph-clustering
backends. None changes the KODAMA UMAP/openTSNE kernels validated below, so the
parity snapshot remains pinned rather than importing unrelated code.

## Initialization contract

The C++ core exposes `KODAMAVisualizationPCAInit()`. It calculates one
backend-native float32 PCA and derives both starts:

- openTSNE centers the scores and scales the largest component standard
  deviation to `1e-4`, matching fastEmbedR;
- UMAP centers the scores, scales the largest absolute coordinate to 10, adds
  deterministic float32 jitter with standard deviation `1e-4`, and recenters.

R and Python use explicit coordinates first, then a stored initialization when
its backend matches the requested CPU, CUDA, or Metal visualization backend,
then explicit raw data. Graph-only initialization is the final fallback.

## Implementation comparison

The port includes fuzzy UMAP graph construction and stochastic optimization,
and openTSNE sparse affinities, exact/FFT repulsion, persistent CPU workers,
cached FFT plans, CUDA graph chunking, and native Metal FFT-grid optimization.

In controlled CPU openTSNE tests with identical KNN input, raw-data PCA
initialization, seed, perplexity, iteration count, and optimizer settings, the
fastEmbedR and kodama-cpp coordinates had zero maximum and mean absolute
difference on MetRef, USPS, and TabulaMuris.

Native Metal openTSNE uses fastEmbedR's FFT-grid optimizer, layout-statistic
reductions, sparse attractive forces, momentum/gain schedule, and centering.
For compatibility with Metal compilers that do not expose `atomic_float`, its
grid accumulators use `atomic_uint` storage plus float-bit compare-and-swap
addition. This preserves float32 values and equations, but parallel
accumulation order remains nondeterministic. On MetRef, USPS, and full
TabulaMuris (873, 11,000, and 100,102 samples), pair-distance correlation with
fastEmbedR Metal was 0.9863, 0.9981, and 0.9928. Correlation with the exact
CPU trajectory was 1.0000, 0.9997, and 0.9990. Warm KODAMA Metal optimizer
times were 0.054, 0.201, and 0.556 seconds; corresponding fastEmbedR Metal
times were 0.047, 0.224, and 0.606 seconds.

UMAP parity is backend-specific, as it is in the audited fastEmbedR source.
CPU uses the CSR epoch-schedule optimizer, CUDA uses its atomic COO/CSR
epoch-schedule optimizer, and Metal uses the clean row sampler with Bernoulli
edge acceptance and negative sampling. KODAMA copies those three execution
strategies separately rather than forcing all devices through one sampler.
The Metal implementation uses fixed-point atomic coordinate updates because
the supported Metal compiler does not expose portable `atomic_float` addition.

With one identical fuzzy graph and CPU spectral initialization, the current
source comparison had exactly matching graph edge counts and maximum weights
on MetRef, USPS, and TabulaMuris: 42,410, 361,242, and 4,390,614 directed
entries, each with maximum weight 1. Pair-distance correlations between KODAMA
and fastEmbedR were 0.9574, 0.8580, and 0.9535 for CPU and 0.9842, 0.7943, and
0.9572 for Metal. All layouts were finite. Stochastic trajectories are not
expected to be bitwise identical because storage layout, float32 reduction
order, and parallel atomic update order differ; graph identity and explicit
optimizer metadata distinguish those numerical effects from graph mismatch.

In a second fixed-200-epoch CPU/Metal screen using the same graph and start,
Metal optimizer speedups were 1.08x, 6.10x, and 6.52x on MetRef, USPS, and
TabulaMuris. Truth-label edge agreement@15 was 0.6225/0.6309,
0.7077/0.7084, and 0.8828/0.8828 for CPU/Metal. These are exploratory local
worktree measurements, not frozen release benchmarks.

R and Python expose the same diagnostics on returned layouts: selected backend,
optimizer, initialization source and backend, runtime, and UMAP fuzzy-graph
edge count and maximum weight. The graph fields are zero for openTSNE because
its sparse probability matrix is not the UMAP fuzzy graph.

The reproducible drivers, CSV files, and plots are under
`benchmarks/run_visualization_fastembedr_parity.R`,
`benchmarks/run_umap_fastembedr_parity.R`,
`benchmarks/visualization_fastembedr_parity_20260807/`, and
`benchmarks/umap_fastembedr_source_parity_20260808_final/`.
