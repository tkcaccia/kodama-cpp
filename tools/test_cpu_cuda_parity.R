#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

root <- Sys.getenv("KODAMA_CPP_ROOT", getwd())
build <- Sys.getenv("KODAMA_CPP_BUILD_DIR", file.path(root, "build"))
Sys.setenv(KODAMA_CPP_ROOT = root, KODAMA_CPP_BUILD_DIR = build)

source(file.path(root, "wrappers", "R", "kodama_matrix_temp.R"))

adjusted_rand_index <- function(x, y) {
  x <- as.integer(as.factor(x))
  y <- as.integer(as.factor(y))
  tab <- table(x, y)
  choose2 <- function(v) sum(v * (v - 1) / 2)
  n <- sum(tab)
  a <- rowSums(tab)
  b <- colSums(tab)
  sum_ij <- choose2(as.vector(tab))
  sum_a <- choose2(a)
  sum_b <- choose2(b)
  expected <- sum_a * sum_b / (n * (n - 1) / 2)
  max_index <- 0.5 * (sum_a + sum_b)
  denom <- max_index - expected
  if (!is.finite(denom) || abs(denom) < .Machine$double.eps) return(1)
  (sum_ij - expected) / denom
}

embedding_dist_cor <- function(a, b) {
  a <- as.matrix(a)
  b <- as.matrix(b)
  if (!all(dim(a) == dim(b))) return(NA_real_)
  da <- as.vector(stats::dist(a))
  db <- as.vector(stats::dist(b))
  if (stats::sd(da) == 0 || stats::sd(db) == 0) return(NA_real_)
  suppressWarnings(stats::cor(da, db, method = "spearman"))
}

make_toy <- function(seed = 1001L, n_per_class = 40L, p = 8L) {
  set.seed(seed)
  classes <- 3L
  n <- classes * n_per_class
  x <- matrix(rnorm(n * p, sd = 0.25), n, p)
  y <- rep(seq_len(classes), each = n_per_class)
  for (cl in seq_len(classes)) {
    rows <- which(y == cl)
    x[rows, cl] <- x[rows, cl] + 2.5
    x[rows, cl + 3L] <- x[rows, cl + 3L] + 1.5
  }
  list(data = x, labels = y)
}

run_backend <- function(backend, data, labels) {
  cat("\n== backend:", backend, "==\n")
  .kodama_cpp_temp_load(backend)

  base_graph <- KODAMA.knn.graph.cpp(
    data,
    k = 20L,
    backend = backend,
    n.cores = 2L,
    exclude.self = TRUE
  )
  stopifnot(nrow(base_graph$indices) == nrow(data), ncol(base_graph$indices) == 20L)

  common <- list(
    data = data,
    M = 3L,
    Tcycle = 4L,
    ncomp = 4L,
    landmarks = 80L,
    splitting = 6L,
    n.cores = 2L,
    graph.neighbors = 20L,
    knn.k = 7L,
    backend = backend,
    seed = 77L,
    progress = FALSE,
    visual.init = TRUE
  )
  knn <- do.call(KODAMA.matrix.cpp, c(common, list(classifier = "knn")))
  pls <- do.call(KODAMA.matrix.cpp, c(common, list(classifier = "pls_lda")))

  umap_knn <- KODAMA.umap.knn.cpp(
    knn,
    n.neighbors = 5L,
    backend = backend,
    n.threads = 2L,
    n.epochs = 10L,
    seed = 77L
  )
  tsne_knn <- KODAMA.opentsne.knn.cpp(
    knn,
    perplexity = 3,
    backend = backend,
    n.threads = 2L,
    early.exaggeration.iter = 5L,
    n.iter = 5L,
    seed = 77L
  )
  umap_pls <- KODAMA.umap.knn.cpp(
    pls,
    n.neighbors = 5L,
    backend = backend,
    n.threads = 2L,
    n.epochs = 10L,
    seed = 78L
  )
  tsne_pls <- KODAMA.opentsne.knn.cpp(
    pls,
    perplexity = 3,
    backend = backend,
    n.threads = 2L,
    early.exaggeration.iter = 5L,
    n.iter = 5L,
    seed = 78L
  )

  clu <- KODAMA.graph.cluster.cpp(
    umap_knn,
    method = "louvain",
    backend = backend,
    graph.backend = backend,
    n.clusters = length(unique(labels)),
    k = 10L,
    n.cores = 2L,
    seed = 77L
  )

  matrices <- list(umap_knn = umap_knn, tsne_knn = tsne_knn, umap_pls = umap_pls, tsne_pls = tsne_pls)
  for (name in names(matrices)) {
    stopifnot(all(dim(matrices[[name]]) == c(nrow(data), 2L)))
    stopifnot(all(is.finite(matrices[[name]])))
  }
  stopifnot(length(knn$acc) == 3L, length(pls$acc) == 3L)
  stopifnot(length(clu$membership) == nrow(data))

  best_labels <- function(result) {
    best <- which.max(result$acc)
    result$res[, best]
  }
  out <- list(
    graph = base_graph,
    knn = knn,
    pls = pls,
    umap_knn = umap_knn,
    tsne_knn = tsne_knn,
    umap_pls = umap_pls,
    tsne_pls = tsne_pls,
    clu = clu,
    knn_best = best_labels(knn),
    pls_best = best_labels(pls)
  )

  cat("KNN acc:", paste(round(knn$acc, 4), collapse = ", "), "\n")
  cat("PLS-LDA acc:", paste(round(pls$acc, 4), collapse = ", "), "\n")
  cat("KNN classes:", paste(apply(knn$res, 2, function(z) length(unique(z))), collapse = ", "), "\n")
  cat("PLS-LDA classes:", paste(apply(pls$res, 2, function(z) length(unique(z))), collapse = ", "), "\n")
  cat("cluster classes:", length(unique(clu$membership)), "\n")
  out
}

toy <- make_toy()
cpu <- run_backend("cpu", toy$data, toy$labels)
cuda <- run_backend("cuda", toy$data, toy$labels)

cat("\n== cpu/cuda mirror summary ==\n")
cat("KNN label ARI:", round(adjusted_rand_index(cpu$knn_best, cuda$knn_best), 4), "\n")
cat("PLS-LDA label ARI:", round(adjusted_rand_index(cpu$pls_best, cuda$pls_best), 4), "\n")
cat("UMAP KNN distance correlation:", round(embedding_dist_cor(cpu$umap_knn, cuda$umap_knn), 4), "\n")
cat("openTSNE KNN distance correlation:", round(embedding_dist_cor(cpu$tsne_knn, cuda$tsne_knn), 4), "\n")
cat("UMAP PLS-LDA distance correlation:", round(embedding_dist_cor(cpu$umap_pls, cuda$umap_pls), 4), "\n")
cat("openTSNE PLS-LDA distance correlation:", round(embedding_dist_cor(cpu$tsne_pls, cuda$tsne_pls), 4), "\n")
cat("CPU graph cluster classes:", length(unique(cpu$clu$membership)), "\n")
cat("CUDA graph cluster classes:", length(unique(cuda$clu$membership)), "\n")
cat("cpu/cuda parity smoke OK\n")
