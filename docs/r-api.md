# R API Map

The R package in `split-repos/kodama-r` contains only conversion, validation,
and S3 convenience code. Numerical work remains in `libkodama_cpp`.

| R function | C++ entry point | Purpose |
|---|---|---|
| `KNNCV()` | `kodama::KNNCV` | Cross-validated KNN classification |
| `PLSLDACV()` | `kodama::PLSLDACV` | SIMPLS plus latent-space LDA CV |
| `CoreKNN()` | `kodama::CoreKNN` | Optimize one label vector with KNN |
| `CorePLSLDA()` | `kodama::CorePLSLDA` | Optimize one label vector with PLS-LDA |
| `KODAMA.matrix()` | `kodama::KODAMAMatrix` | Complete independent-run ensemble |
| `KODAMA.matrix.graph()` | `kodama::KODAMAMatrixFromGraph` | KODAMA from supplied indices/distances |
| `KODAMA.graph()` | `kodama::KODAMAKNNGraph` | Build a neighbor graph |
| `KODAMA.pca()` | `kodama::PCA` | Float32 randomized PCA |
| `KODAMA.visualization()` | `kodama::KODAMAUMAP_*` or `kodama::KODAMAOpenTSNE_*` | Embed a graph |
| `KODAMA.clustering()` | `kodama::KODAMAGraphCluster` | Random-walk graph clustering |

Use `help(package = "kodamaR")` for argument-level R documentation. The
wrapper preserves raw result fields such as `acc`, `res`, `knn`, and `timing`,
and adds `best_labels`, `best_run`, `class_counts`, and `parameters`.

For reproducible evaluation, report at least the classifier, backend, `M`,
`Tcycle`, landmark count, splitting, component count or KNN `k`, graph-neighbor
count, seed, thread count, and whether KODAMA dissimilarity correction was
applied. External labels must not be used to choose the best run.
