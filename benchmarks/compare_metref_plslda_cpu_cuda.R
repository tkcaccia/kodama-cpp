#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

options(stringsAsFactors = FALSE)

value_or <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}

if (!requireNamespace("kodamaR", quietly = TRUE)) {
  stop("The installed kodamaR package is required for this parity figure.")
}

cpu_path <- value_or(
  "KODAMA_CPU_RESULT",
  "/mnt/sata_ssd/KODAMAopt/metref_cpu_parity_20260720/final_cpu_M100_T100.rds"
)
cuda_path <- value_or(
  "KODAMA_CUDA_RESULT",
  "/mnt/sata_ssd/KODAMAopt/metref_cpu_parity_20260720/final_cuda_curand_skip_M100_T100.rds"
)
data_path <- value_or(
  "KODAMA_DATA_PATH",
  "/mnt/sata_ssd/fastEmbedR/Data/MetRef/MetRef_float32.RData"
)
out_dir <- value_or(
  "KODAMA_COMPARE_OUT",
  "/mnt/sata_ssd/KODAMAopt/metref_cpu_parity_20260720/comparison"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

load_dataset <- function(path) {
  environment <- new.env(parent = emptyenv())
  load(path, envir = environment)
  objects <- mget(ls(environment, all.names = TRUE), envir = environment)
  dataset <- if ("dataset" %in% names(objects)) {
    objects$dataset
  } else {
    candidates <- Filter(
      function(x) is.list(x) && !is.null(x$data) && !is.null(x$labels),
      objects
    )
    if (length(candidates) != 1L) stop("Could not identify dataset in ", path)
    candidates[[1L]]
  }
  labels <- droplevels(as.factor(dataset$labels))
  list(data = dataset$data, labels = labels)
}

choose2 <- function(x) x * (x - 1) / 2

adjusted_rand <- function(a, b) {
  table_ab <- table(a, b)
  nij <- sum(choose2(table_ab))
  ai <- sum(choose2(rowSums(table_ab)))
  bj <- sum(choose2(colSums(table_ab)))
  total <- choose2(sum(table_ab))
  expected <- if (total > 0) ai * bj / total else 0
  maximum <- (ai + bj) / 2
  denominator <- maximum - expected
  if (!is.finite(denominator) || denominator == 0) 1 else (nij - expected) / denominator
}

truth_silhouette <- function(embedding, truth) {
  mean(cluster::silhouette(as.integer(truth), stats::dist(embedding))[, 3L])
}

selected_run <- function(result, truth) {
  labels_by_run <- as.matrix(result$res)
  best <- if (!is.null(result$best_run) && length(result$best_run)) {
    as.integer(result$best_run)
  } else {
    which.max(as.numeric(result$acc))
  }
  labels <- as.integer(labels_by_run[best, ])
  list(
    run = best,
    labels = labels,
    ari = adjusted_rand(labels, truth),
    classes = length(unique(labels))
  )
}

ensemble_metrics <- function(result, truth) {
  labels_by_run <- as.matrix(result$res)
  ari <- apply(labels_by_run, 1L, adjusted_rand, b = truth)
  classes <- apply(labels_by_run, 1L, function(x) length(unique(x)))
  list(
    maximum_ari = max(ari),
    median_ari = stats::median(ari),
    median_classes = stats::median(classes),
    minimum_classes = min(classes),
    maximum_classes = max(classes)
  )
}

dataset <- load_dataset(data_path)
truth <- dataset$labels
if (inherits(dataset$data, "float32") || inherits(dataset$data, "float")) {
  if (!requireNamespace("float", quietly = TRUE)) stop("The float package is required.")
  dataset$data <- float::dbl(dataset$data)
}
cpu <- readRDS(cpu_path)
cuda <- readRDS(cuda_path)

as_visualization_input <- function(result) {
  if (is.null(result$knn)) result$knn <- result$knn_Rnanoflann
  result
}

cpu <- as_visualization_input(cpu)
cuda <- as_visualization_input(cuda)
init <- cpu$visual_init$umap
if (is.null(init)) stop("The CPU result does not contain the original-data UMAP initialization.")

classic_graph <- kodamaR::KODAMA.graph(
  dataset$data,
  k = 30L,
  backend = "cpu",
  n.cores = 4L
)
embed <- function(x) {
  start <- proc.time()[["elapsed"]]
  embedding <- kodamaR::KODAMA.visualization(
    x,
    method = "UMAP",
    init = init,
    k = 30L,
    backend = "cpu",
    n.cores = 4L,
    n.epochs = 200L,
    seed = 4L
  )
  list(embedding = embedding, seconds = proc.time()[["elapsed"]] - start)
}

classic_record <- embed(classic_graph)
cpu_record <- embed(cpu)
cuda_record <- embed(cuda)
classic_umap <- classic_record$embedding
cpu_umap <- cpu_record$embedding
cuda_umap <- cuda_record$embedding

cpu_selected <- selected_run(cpu, truth)
cuda_selected <- selected_run(cuda, truth)
cpu_ensemble <- ensemble_metrics(cpu, truth)
cuda_ensemble <- ensemble_metrics(cuda, truth)
metrics <- data.frame(
  backend = c("cpu", "cuda"),
  kodama_seconds = c(cpu$runtime_seconds, cuda$runtime_seconds),
  umap_seconds = c(cpu_record$seconds, cuda_record$seconds),
  median_cv_accuracy = c(stats::median(cpu$acc), stats::median(cuda$acc)),
  maximum_cv_accuracy = c(max(cpu$acc), max(cuda$acc)),
  selected_run = c(cpu_selected$run, cuda_selected$run),
  selected_ari = c(cpu_selected$ari, cuda_selected$ari),
  selected_classes = c(cpu_selected$classes, cuda_selected$classes),
  maximum_ari = c(cpu_ensemble$maximum_ari, cuda_ensemble$maximum_ari),
  median_ari = c(cpu_ensemble$median_ari, cuda_ensemble$median_ari),
  median_classes = c(cpu_ensemble$median_classes, cuda_ensemble$median_classes),
  minimum_classes = c(cpu_ensemble$minimum_classes, cuda_ensemble$minimum_classes),
  maximum_classes = c(cpu_ensemble$maximum_classes, cuda_ensemble$maximum_classes),
  truth_silhouette = c(
    truth_silhouette(cpu_umap, truth),
    truth_silhouette(cuda_umap, truth)
  )
)
utils::write.csv(metrics, file.path(out_dir, "metrics.csv"), row.names = FALSE)
saveRDS(
  cpu_record,
  file.path(out_dir, "MetRef__pls_lda__cpu__umap.rds"),
  compress = FALSE
)
saveRDS(
  cuda_record,
  file.path(out_dir, "MetRef__pls_lda__cuda_optimized__umap.rds"),
  compress = FALSE
)
saveRDS(
  classic_record,
  file.path(out_dir, "MetRef__classic__umap_exact_k30.rds"),
  compress = FALSE
)

colors <- grDevices::hcl.colors(nlevels(truth), "Dynamic")[as.integer(truth)]
plot_path <- file.path(out_dir, "MetRef__pls_lda__cpu_cuda.png")
grDevices::png(plot_path, width = 2400, height = 820, res = 160)
old <- graphics::par(mfrow = c(1, 3), mar = c(1, 1, 3.1, 0.5), oma = c(0, 0, 2.2, 0))
for (panel in list(
  list(title = "Classic UMAP", embedding = classic_umap),
  list(title = "Optimized CPU KODAMA PLS-LDA", embedding = cpu_umap),
  list(title = "Optimized CUDA KODAMA PLS-LDA", embedding = cuda_umap)
)) {
  silhouette <- truth_silhouette(panel$embedding, truth)
  graphics::plot(
    panel$embedding,
    pch = 16,
    cex = 1.0,
    col = colors,
    axes = FALSE,
    xlab = "",
    ylab = "",
    main = sprintf("%s\ntruth silhouette=%.3f", panel$title, silhouette)
  )
  graphics::box(col = "grey80")
}
graphics::mtext("MetRef, colored by truth labels", outer = TRUE, cex = 1.2, font = 2)
graphics::par(old)
grDevices::dev.off()

print(metrics, row.names = FALSE)
cat("plot", plot_path, "\n")
