#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore and kodama-cpp contributors
# SPDX-License-Identifier: MIT

options(stringsAsFactors = FALSE)
candidate.library <- Sys.getenv("KODAMA_R_LIB", unset = "")
if (nzchar(candidate.library)) .libPaths(c(candidate.library, .libPaths()))
suppressPackageStartupMessages(library(KODAMA))
message("KODAMA ", as.character(utils::packageVersion("KODAMA")),
        " loaded from ", find.package("KODAMA"))
`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x
script.argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script.path <- if (length(script.argument)) sub("^--file=", "", script.argument[[1L]]) else
  "benchmarks/consensus_graph_validation/run_plslda_validation.R"
source(file.path(dirname(normalizePath(script.path)), "metrics.R"))
source(file.path(dirname(dirname(normalizePath(script.path))), "spatial_benchmark_common.R"))

env <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}
as_numeric_matrix <- function(x) {
  if (inherits(x, "float32") || inherits(x, "float")) x <- float::dbl(x)
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}
load_objects <- function(path) {
  e <- new.env(parent = emptyenv())
  load(path, envir = e)
  as.list(e)
}
load_dataset <- function(name, path) {
  x <- load_objects(path)
  if (!is.null(x$dataset)) {
    data <- as_numeric_matrix(x$dataset$data)
    spatial <- if (is.null(x$dataset$spatial)) NULL else as_numeric_matrix(x$dataset$spatial)
    samples <- NULL
    if (name == "MERFISH" && !is.null(spatial) && ncol(spatial) >= 3L) {
      samples <- factor(spatial[, 3L])
    } else if (name == "Br8100" && !is.null(rownames(data))) {
      samples <- factor(sub("^.*-", "", rownames(data)))
    }
    return(list(data = data, labels = factor(x$dataset$labels),
                spatial = spatial, samples = samples))
  }
  if (name == "MetRef") {
    return(list(data = as_numeric_matrix(x$dataset$data), labels = factor(x$dataset$labels),
                spatial = NULL, samples = NULL))
  }
  if (name == "MERFISH") {
    return(list(data = as_numeric_matrix(x$pca.PM), labels = factor(x$tissue_segments),
                spatial = as_numeric_matrix(x$xyz), samples = factor(x$xyz[, 3L])))
  }
  if (name == "Br8100") {
    return(list(data = as_numeric_matrix(x$pca_Br8100), labels = factor(x$labels_Br8100),
                spatial = as_numeric_matrix(x$xy_Br8100), samples = factor(x$samples_Br8100)))
  }
  stop("Unknown dataset: ", name)
}

backend <- env("KODAMA_BACKEND", "cpu")
n.cores <- as.integer(env("KODAMA_N_CORES", "4"))
M <- as.integer(env("KODAMA_M", "100"))
Tcycle <- as.integer(env("KODAMA_TCYCLE", "100"))
constraint.mode <- env("KODAMA_SPATIAL_CONSTRAINT_MODE", "kmeans")
dataset.names <- strsplit(env("KODAMA_DATASETS", "MetRef,MERFISH,Br8100"), ",", fixed = TRUE)[[1L]]
root <- env("KODAMA_DATA_ROOT", if (backend == "cuda") "/mnt/sata_ssd" else "/Users/stefano/Documents")
paths <- c(
  MetRef = file.path(root, "fastEmbedR/Data/MetRef/MetRef_float32.RData"),
  MERFISH = file.path(root, "GitHub/KODAMA-Analysis/data/MERFISH-input.RData"),
  Br8100 = file.path(root, "GitHub/KODAMA-Analysis/data/DLFPC-Br8100-input.RData")
)
if (backend == "cuda") {
  paths["MERFISH"] <- "/mnt/sata_ssd/KODAMAopt/spatial/MERFISH.RData"
  paths["Br8100"] <- "/mnt/sata_ssd/KODAMAopt/spatial/Br8100.RData"
}
out.dir <- env("KODAMA_OUT", file.path(tempdir(), "kodama-consensus-validation"))
dir.create(out.dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out.dir, "plots"), showWarnings = FALSE)
dir.create(file.path(out.dir, "objects"), showWarnings = FALSE)

for (dataset.name in dataset.names) {
  dataset.name <- trimws(dataset.name)
  ds <- load_dataset(dataset.name, paths[[dataset.name]])
  target.k <- nlevels(ds$labels)
  message("[", dataset.name, "] PLS-LDA M=", M, " Tcycle=", Tcycle,
          " backend=", backend, " target K=", target.k)
  graph.start <- proc.time()[["elapsed"]]
  graph <- KODAMA.graph(
    ds$data, spatial = ds$spatial, samples = ds$samples, k = 100L,
    backend = backend, n.cores = n.cores, seed = 1234L, storage = "handle"
  )
  graph.seconds <- proc.time()[["elapsed"]] - graph.start
  matrix.start <- proc.time()[["elapsed"]]
  fit <- KODAMA.matrix(
    data = ds$data, graph = graph, spatial = ds$spatial, samples = ds$samples,
    classifier = "pls_lda", backend = backend, n.cores = n.cores,
    M = M, Tcycle = Tcycle, ncomp = min(50L, ncol(ds$data)),
    landmarks = 100000L, splitting = if (nrow(ds$data) < 40000L) 100L else 300L,
    graph.neighbors = 100L, knn.k = 30L, spatial.resolution = 0.3,
    spatial.constraint.mode = constraint.mode, folds = 5L, seed = 1234L,
    visual.init = TRUE, progress = TRUE, return.graph = "handle"
  )
  matrix.seconds <- proc.time()[["elapsed"]] - matrix.start
  graph.matrix <- KODAMA.graph.materialize(fit$knn)
  saveRDS(
    list(
      res = fit$res, knn = graph.matrix, visual_init = fit$visual_init,
      run_diagnostics = fit$run_diagnostics,
      cycle_diagnostics = fit$cycle_diagnostics,
      truth = ds$labels, samples = ds$samples, spatial = ds$spatial,
      matrix_seconds = matrix.seconds
    ),
    file.path(out.dir, "objects", paste0(dataset.name, "_matrix_checkpoint.rds")),
    compress = FALSE
  )
  visual.start <- proc.time()[["elapsed"]]
  layout <- KODAMA.visualization(
    fit, method = "UMAP", k = 30L, backend = backend, n.cores = n.cores,
    graph.mode = "fuzzy", n.epochs = 200L, seed = 4L
  )
  visual.seconds <- proc.time()[["elapsed"]] - visual.start
  clustering.start <- proc.time()[["elapsed"]]
  clustering <- KODAMA.clustering(
    fit, method = "walktrap", n.clusters = target.k, k = 30L,
    n.cores = n.cores, seed = 1L
  )
  clustering.seconds <- proc.time()[["elapsed"]] - clustering.start
  final.labels <- factor(clustering$membership)

  edges <- graph_edge_table(graph.matrix)
  partition <- partition_metrics(ds$labels, final.labels)
  per.sample <- per_sample_metrics(ds$labels, final.labels, ds$samples)
  conductance <- graph_conductance(edges, final.labels)
  boundary <- boundary_f1(edges, ds$labels, final.labels)
  agreement <- sample_agreement(edges, fit$res)
  classes <- apply(fit$res, 1L, function(labels) length(unique(labels)))
  rare.fraction <- apply(fit$res, 1L, function(labels) min(table(labels)) / length(labels))
  diagnostic <- data.frame(
    dataset = dataset.name, samples = nrow(ds$data), variables = ncol(ds$data),
    backend = backend, M = M, Tcycle = Tcycle,
    spatial_constraint_mode = constraint.mode, target_clusters = target.k,
    graph_seconds = graph.seconds, matrix_seconds = matrix.seconds,
    clustering_seconds = clustering.seconds, visualization_seconds = visual.seconds,
    ari = partition$summary$ari, nmi = partition$summary$nmi,
    homogeneity = partition$summary$homogeneity,
    completeness = partition$summary$completeness,
    silhouette = sampled_silhouette(layout, ds$labels),
    conductance_mean = conductance[["mean"]], conductance_max = conductance[["maximum"]],
    boundary_f1 = boundary[["f1"]], final_clusters = nlevels(final.labels),
    median_M_classes = median(classes), minimum_M_classes = min(classes),
    maximum_M_classes = max(classes), median_rarest_class_fraction = median(rare.fraction),
    corrected_components = fit$corrected_graph_components,
    zero_agreement_edges = fit$corrected_zero_agreement_edges,
    finite_corrected_edges = fit$corrected_finite_edges,
    agreement_q0 = agreement[1L], agreement_q10 = agreement[2L],
    agreement_q25 = agreement[3L], agreement_q50 = agreement[4L],
    agreement_q75 = agreement[5L], agreement_q90 = agreement[6L], agreement_q100 = agreement[7L]
  )
  utils::write.csv(diagnostic, file.path(out.dir, paste0(dataset.name, "_summary.csv")), row.names = FALSE)
  utils::write.csv(per.sample, file.path(out.dir, paste0(dataset.name, "_per_sample.csv")), row.names = FALSE)
  utils::write.csv(partition$per_class, file.path(out.dir, paste0(dataset.name, "_per_class.csv")), row.names = FALSE)
  utils::write.csv(fit$run_diagnostics, file.path(out.dir, paste0(dataset.name, "_run_diagnostics.csv")), row.names = FALSE)
  utils::write.csv(fit$cycle_diagnostics, file.path(out.dir, paste0(dataset.name, "_cycle_diagnostics.csv")), row.names = FALSE)

  grDevices::png(file.path(out.dir, "plots", paste0(dataset.name, "_PLSLDA.png")),
                 width = 2400, height = if (is.null(ds$spatial)) 800 else 1200, res = 150)
  old <- graphics::par(mfrow = if (is.null(ds$spatial)) c(1, 3) else c(2, 3),
                       mar = c(1, 1, 3, 1), oma = c(0, 0, 2.5, 0))
  truth.colors <- grDevices::hcl.colors(nlevels(ds$labels), "Dynamic")[as.integer(ds$labels)]
  cluster.colors <- grDevices::hcl.colors(nlevels(final.labels), "Dark 3")[as.integer(final.labels)]
  point.cex <- if (nrow(ds$data) > 20000L) 0.25 else if (nrow(ds$data) > 5000L) 0.4 else 0.75
  graphics::plot(fit$visual_init$umap, col = truth.colors, pch = 16, cex = point.cex,
                 axes = FALSE, xlab = "", ylab = "", main = "Raw PCA initialization by truth")
  graphics::plot(layout, col = truth.colors, pch = 16, cex = point.cex,
                 axes = FALSE, xlab = "", ylab = "", main = "KODAMA UMAP by truth")
  graphics::plot(layout, col = cluster.colors, pch = 16, cex = point.cex,
                 axes = FALSE, xlab = "", ylab = "", main = "Corrected-graph Walktrap")
  if (!is.null(ds$spatial)) {
    graphics::plot(ds$spatial[, 1:2], col = truth.colors, pch = 16, cex = point.cex,
                   asp = 1, axes = FALSE, xlab = "", ylab = "", main = "Spatial truth")
    graphics::plot(ds$spatial[, 1:2], col = cluster.colors, pch = 16, cex = point.cex,
                   asp = 1, axes = FALSE, xlab = "", ylab = "", main = "Spatial clustering")
    graphics::plot(classes, type = "h", xlab = "M run", ylab = "classes",
                   main = "Classes across all M")
  }
  graphics::mtext(sprintf("%s PLS-LDA | ARI %.3f | NMI %.3f | M=%d T=%d",
                          dataset.name, diagnostic$ari, diagnostic$nmi, M, Tcycle),
                  outer = TRUE, font = 2)
  graphics::par(old)
  grDevices::dev.off()
  if (!is.null(ds$spatial)) {
    for (package in c("clue", "ggplot2", "gridExtra", "KODAMAextra")) {
      if (!requireNamespace(package, quietly = TRUE)) {
        stop("Spatial benchmark plotting requires package: ", package)
      }
    }
    suppressPackageStartupMessages({
      library(ggplot2)
      library(gridExtra)
      library(KODAMAextra)
    })
    aligned.labels <- kodama_align_labels(final.labels, ds$labels)
    plot.spatial <- ds$spatial[, seq_len(2L), drop = FALSE]
    ari.text <- paste(
      sprintf("%s %.3f", per.sample$sample, per.sample$ari), collapse = " | "
    )
    truth.slide <- KODAMAextra::plot_slide(
      plot.spatial, ds$samples, ds$labels, col = kodama_spatial_palette,
      nrow = 1L, size.dot = 0.3
    ) + ggplot2::ggtitle("Truth")
    cluster.slide <- KODAMAextra::plot_slide(
      plot.spatial, ds$samples, aligned.labels, col = kodama_spatial_palette,
      nrow = 1L, size.dot = 0.3
    ) + ggplot2::ggtitle("Corrected-graph Walktrap")
    slide.panel <- gridExtra::arrangeGrob(
      truth.slide, cluster.slide, nrow = 2L,
      top = sprintf("%s | per-slide ARI: %s", dataset.name, ari.text)
    )
    ggplot2::ggsave(
      file.path(out.dir, "plots", paste0(dataset.name, "_PLSLDA_slides.png")),
      slide.panel, width = 18, height = 7.5, dpi = 180, bg = "white"
    )
  }
  saveRDS(list(fit = fit, clustering = clustering, layout = layout, truth = ds$labels,
               samples = ds$samples, metrics = diagnostic),
          file.path(out.dir, "objects", paste0(dataset.name, "_PLSLDA.rds")), compress = FALSE)
  print(diagnostic)
  rm(graph, fit, graph.matrix, edges, layout)
  gc()
}
