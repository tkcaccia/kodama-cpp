#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

options(stringsAsFactors = FALSE)

value_or <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}

r_lib <- value_or("KODAMA_R_LIB", "")
if (nzchar(r_lib)) .libPaths(c(r_lib, .libPaths()))
suppressPackageStartupMessages({
  library(cluster)
  library(kodamaR)
})

data_root <- normalizePath(
  value_or("KODAMA_DATA_ROOT", "/mnt/sata_ssd/fastEmbedR/Data"),
  mustWork = TRUE
)
out_dir <- value_or(
  "KODAMA_K_SWEEP_OUT",
  "/mnt/sata_ssd/kodama-cpp-benchmarks/knn-k-sweep-fixed-graph"
)
datasets <- strsplit(
  value_or("KODAMA_K_SWEEP_DATASETS", "MNIST,TabulaMuris,FashionMNIST"),
  ",",
  fixed = TRUE
)[[1L]]
k_values <- as.integer(strsplit(
  value_or("KODAMA_K_SWEEP_VALUES", "5,10,15,20,30,50"),
  ",",
  fixed = TRUE
)[[1L]])
M <- as.integer(value_or("KODAMA_K_SWEEP_M", "100"))
Tcycle <- as.integer(value_or("KODAMA_K_SWEEP_TCYCLE", "100"))
landmarks <- as.integer(value_or("KODAMA_K_SWEEP_LANDMARKS", "100000"))
seed <- as.integer(value_or("KODAMA_K_SWEEP_SEED", "1234"))

manifest <- c(
  MNIST = file.path(data_root, "MNIST", "MNIST_float32.RData"),
  TabulaMuris = file.path(data_root, "TabulaMuris", "TabulaMuris_float32.RData"),
  FashionMNIST = file.path(data_root, "FashionMNIST", "FashionMNIST_float32.RData")
)
unknown <- setdiff(datasets, names(manifest))
if (length(unknown)) stop("Unknown datasets: ", paste(unknown, collapse = ", "))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

as_numeric_matrix <- function(x) {
  if (inherits(x, "float32") || inherits(x, "float")) {
    if (!requireNamespace("float", quietly = TRUE)) stop("The float package is required")
    x <- float::dbl(x)
  }
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

load_dataset <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  objects <- mget(ls(env, all.names = TRUE), envir = env)
  object <- if ("dataset" %in% names(objects)) objects$dataset else {
    candidates <- Filter(
      function(x) is.list(x) && !is.null(x$data) && !is.null(x$labels),
      objects
    )
    if (length(candidates) != 1L) stop("Cannot identify dataset in ", path)
    candidates[[1L]]
  }
  list(data = as_numeric_matrix(object$data), labels = droplevels(as.factor(object$labels)))
}

choose2 <- function(x) x * (x - 1) / 2

adjusted_rand <- function(a, b) {
  tab <- table(a, b)
  nij <- sum(choose2(tab))
  ai <- sum(choose2(rowSums(tab)))
  bj <- sum(choose2(colSums(tab)))
  total <- choose2(sum(tab))
  expected <- if (total > 0) ai * bj / total else 0
  maximum <- (ai + bj) / 2
  denominator <- maximum - expected
  if (!is.finite(denominator) || denominator == 0) 1 else (nij - expected) / denominator
}

sampled_silhouette <- function(embedding, labels, max_samples = 5000L) {
  keep <- which(!is.na(labels) & rowSums(is.finite(embedding)) == ncol(embedding))
  if (length(keep) > max_samples) {
    set.seed(seed + 91L)
    keep <- sort(sample(keep, max_samples))
  }
  labels <- as.integer(as.factor(labels))[keep]
  if (length(keep) < 3L || length(unique(labels)) < 2L) return(NA_real_)
  mean(cluster::silhouette(labels, stats::dist(embedding[keep, , drop = FALSE]))[, 3L])
}

for (dataset_name in datasets) {
  message("== ", dataset_name, " ==")
  ds <- load_dataset(unname(manifest[[dataset_name]]))
  x <- ds$data
  truth <- ds$labels
  splitting <- if (nrow(x) < 40000L) 100L else 300L

  graph_start <- proc.time()[["elapsed"]]
  base_graph <- KODAMA.graph(
    x,
    k = 100L,
    backend = "cpu",
    n.cores = 4L,
    seed = seed,
    storage = "matrix"
  )
  graph_seconds <- proc.time()[["elapsed"]] - graph_start

  rows <- list()
  embeddings <- list()
  for (k in k_values) {
    result_path <- file.path(out_dir, sprintf("%s__k%d.rds", dataset_name, k))
    if (file.exists(result_path)) {
      record <- readRDS(result_path)
    } else {
      matrix_start <- proc.time()[["elapsed"]]
      fit <- KODAMA.matrix(
        data = x,
        graph = base_graph,
        classifier = "knn",
        backend = "cuda",
        M = M,
        Tcycle = Tcycle,
        landmarks = landmarks,
        splitting = splitting,
        graph.neighbors = 100L,
        knn.k = k,
        n.cores = 4L,
        seed = seed,
        progress = FALSE,
        visual.init = TRUE,
        apply.kodama.dissimilarity = TRUE,
        return.graph = TRUE
      )
      matrix_seconds <- proc.time()[["elapsed"]] - matrix_start
      embedding_start <- proc.time()[["elapsed"]]
      embedding <- KODAMA.visualization(
        fit,
        method = "UMAP",
        k = 30L,
        backend = "cuda",
        n.cores = 4L,
        seed = seed
      )
      embedding_seconds <- proc.time()[["elapsed"]] - embedding_start
      aris <- apply(as.matrix(fit$res), 1L, adjusted_rand, b = truth)
      classes <- apply(as.matrix(fit$res), 1L, function(z) length(unique(z)))
      record <- list(
        fit = fit,
        embedding = embedding,
        metrics = data.frame(
          dataset = dataset_name,
          k = k,
          samples = nrow(x),
          variables = ncol(x),
          graph_backend = "cpu",
          matrix_backend = "cuda",
          graph_seconds = graph_seconds,
          matrix_seconds = matrix_seconds,
          embedding_seconds = embedding_seconds,
          best_cv_accuracy = max(fit$acc, na.rm = TRUE),
          median_cv_accuracy = median(fit$acc, na.rm = TRUE),
          selected_ari = aris[fit$best_run],
          median_run_ari = median(aris, na.rm = TRUE),
          max_diagnostic_ari = max(aris, na.rm = TRUE),
          selected_classes = classes[fit$best_run],
          median_classes = median(classes),
          truth_silhouette = sampled_silhouette(embedding, truth)
        )
      )
      saveRDS(record, result_path, compress = FALSE)
    }
    rows[[as.character(k)]] <- record$metrics
    embeddings[[as.character(k)]] <- record$embedding
  }

  metrics <- do.call(rbind, rows)
  write.csv(metrics, file.path(out_dir, paste0(dataset_name, "__summary.csv")), row.names = FALSE)
  png(
    file.path(out_dir, paste0(dataset_name, "__knn_k_sweep.png")),
    width = 2400,
    height = 1500,
    res = 180
  )
  old <- par(mfrow = c(2, 3), mar = c(0.8, 0.8, 2.7, 0.4))
  colors <- hcl.colors(nlevels(truth), "Dynamic")[as.integer(truth)]
  for (k in k_values) {
    embedding <- embeddings[[as.character(k)]]
    metric <- metrics[metrics$k == k, ]
    plot(
      embedding,
      pch = 16,
      cex = if (nrow(x) > 50000L) 0.2 else 0.35,
      col = colors,
      axes = FALSE,
      xlab = "",
      ylab = "",
      main = sprintf("k=%d | sil=%.3f | ARI=%.3f", k, metric$truth_silhouette, metric$selected_ari)
    )
    box(col = "grey80")
  }
  par(old)
  dev.off()
  rm(x, ds, base_graph, embeddings)
  gc(verbose = FALSE)
}
