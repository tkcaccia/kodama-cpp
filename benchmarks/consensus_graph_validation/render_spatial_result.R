#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore and kodama-cpp contributors
# SPDX-License-Identifier: MIT

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop("Usage: render_spatial_result.R DATASET DATA_RDATA RESULT_RDS METRICS_CSV OUTPUT_PNG")
}
dataset.name <- args[[1L]]
data.path <- args[[2L]]
result.path <- args[[3L]]
metrics.path <- args[[4L]]
output.path <- args[[5L]]

for (package in c("clue", "ggplot2", "gridExtra", "KODAMAextra")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing package: ", package)
}
suppressPackageStartupMessages({
  library(ggplot2)
  library(gridExtra)
  library(KODAMAextra)
})
script.argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script.path <- sub("^--file=", "", script.argument[[1L]])
source(file.path(dirname(dirname(normalizePath(script.path))), "spatial_benchmark_common.R"))

input <- new.env(parent = emptyenv())
load(data.path, envir = input)
objects <- as.list(input)
if (!is.null(objects$dataset)) {
  spatial <- as.matrix(objects$dataset$spatial)
} else if (dataset.name == "MERFISH") {
  spatial <- as.matrix(objects$xyz)
} else if (dataset.name == "Br8100") {
  spatial <- as.matrix(objects$xy_Br8100)
} else {
  stop("Unsupported spatial dataset: ", dataset.name)
}

result <- readRDS(result.path)
truth <- droplevels(as.factor(result$truth))
samples <- droplevels(as.factor(result$samples))
clusters <- droplevels(as.factor(result$clustering$membership))
aligned <- kodama_align_labels(clusters, truth)
metrics <- utils::read.csv(metrics.path, check.names = FALSE)
ari.text <- paste(sprintf("%s %.3f", metrics$sample, metrics$ari), collapse = " | ")
plot.spatial <- spatial[, seq_len(2L), drop = FALSE]

truth.plot <- KODAMAextra::plot_slide(
  plot.spatial, samples, truth, col = kodama_spatial_palette,
  nrow = 1L, size.dot = 0.3
) + ggplot2::ggtitle("Truth")
cluster.plot <- KODAMAextra::plot_slide(
  plot.spatial, samples, aligned, col = kodama_spatial_palette,
  nrow = 1L, size.dot = 0.3
) + ggplot2::ggtitle("Corrected-graph Walktrap")
panel <- gridExtra::arrangeGrob(
  truth.plot, cluster.plot, nrow = 2L,
  top = sprintf("%s | per-slide ARI: %s", dataset.name, ari.text)
)
ggplot2::ggsave(output.path, panel, width = 18, height = 7.5, dpi = 180, bg = "white")
cat(normalizePath(output.path), "\n")
