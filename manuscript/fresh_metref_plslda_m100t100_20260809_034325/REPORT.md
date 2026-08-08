# Fresh MetRef PLS-LDA benchmark

## Protocol

- Dataset: MetRef float32, 873 samples, 375 variables, 22 withheld truth classes
- Dataset MD5: `500e69ce3c5bebc2f57687c93426ed58`
- Classifier: SIMPLS plus latent-space LDA
- Independent runs: `M=100`
- Evolution cycles: `Tcycle=100`
- Requested PLS components: 50
- Requested/effective landmarks: 100,000 / 655
- Splitting: 100
- Graph neighbors: 100
- KNN predictor setting: 30
- CV: five folds
- Seed: 4
- Visualization: fuzzy UMAP, `k=30`, 200 epochs, backend-native PCA initialization

Each backend ran from a fresh R process and built its graph from the input data.
CPU and Metal ran locally on the Apple system; CUDA ran on chiamaka's NVIDIA
GeForce RTX 5060 Ti. The CUDA core and wrapper were rebuilt in an isolated
directory from the local worktree snapshot. Functional CUDA C++ tests passed;
the staged-source license audit was not applicable because auxiliary license
files and unrelated historical scripts were deliberately not copied.

## Timing

| Backend | Workers | Graph (s) | KODAMA.matrix (s) | UMAP (s) | Pipeline (s) | Speedup vs CPU |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 4 | 0.155 | 252.155 | 0.034 | 252.344 | 1.00x |
| Metal | 26 auto | 0.136 | 263.580 | 0.041 | 263.757 | 0.96x |
| CUDA | 4 auto | 0.296 | 146.717 | 0.067 | 147.080 | 1.72x |

Graph construction and visualization are negligible for MetRef. PLS-LDA label
evolution accounts for more than 99.8% of the measured pipeline time.

## Quality

| Backend | Best CV | Median CV | Best-run ARI | Median run ARI | Best classes | Median classes | Truth silhouette | Classic silhouette |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | 1.000 | 0.993893 | 0.843135 | 0.726923 | 23 | 26 | 0.812164 | 0.098705 |
| Metal | 1.000 | 0.993893 | 0.860185 | 0.721507 | 32 | 25 | 0.822186 | 0.103778 |
| CUDA | 1.000 | 0.993893 | 0.772332 | 0.726662 | 25 | 25 | 0.785033 | 0.125757 |

Every backend had zero collapsed runs with two or fewer classes. Median CV,
median ARI, and median class count are closely aligned. The CV-selected best run
can differ across backends because float32 reduction order changes the stochastic
trajectory; withheld truth labels are not used for selection.

## Decision

The implementation passes the fresh quality check on every backend. CUDA is the
fastest backend. Metal remains slower than four-core CPU on this small dataset,
so further Metal optimization should target the repeated PLS-LDA evolution and
not graph construction, landmark selection, or UMAP.
