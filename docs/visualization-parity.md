# Visualization parity

kodama-cpp tracks the UMAP and openTSNE implementation at fastEmbedR commit
`ef064f2a13db0b28f257bcb25bd06fd031da6da6`.

## Initialization contract

The C++ core exposes `KODAMAVisualizationPCAInit()`. It calculates one
backend-native float32 PCA and derives both starts:

- openTSNE centers the scores and scales the largest component standard
  deviation to `1e-4`, matching fastEmbedR;
- UMAP centers the scores, scales the largest absolute coordinate to 10, adds
  deterministic float32 jitter with standard deviation `1e-4`, and recenters.

R and Python use explicit coordinates first, then a stored initialization when
its backend matches the requested CPU or CUDA visualization backend, then
explicit raw data. Graph-only initialization is the final fallback.

## Implementation comparison

The port includes the current fuzzy UMAP graph construction and stochastic
optimizer, and the current openTSNE sparse affinities, exact/FFT repulsion,
persistent CPU workers, cached FFT plans, and CUDA graph chunking.

In a controlled CPU openTSNE test with identical KNN input, raw-data PCA
initialization, seed, perplexity, iteration count, and optimizer settings, the
fastEmbedR and kodama-cpp coordinates had zero maximum and mean absolute
difference.

For UMAP, graph construction, initialization, optimizer equations, sampling
schedule, and the first epoch agree. Later CPU coordinates are not expected to
be bitwise identical: kodama-cpp retains all iterative state in float32, while
the compared fastEmbedR R entry point holds its embedding in an R double
matrix. Stochastic updates amplify these rounding differences. This is a
precision-contract distinction, not a different UMAP objective. Comparisons
should therefore use fixed settings plus neighborhood, trustworthiness, or
Procrustes diagnostics rather than raw coordinate equality.
