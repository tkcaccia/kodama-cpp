#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: summarize_fresh_metref_backends.R OUTPUT_DIR DATASET_RDATA")
}
out <- normalizePath(args[[1L]])
data_path <- normalizePath(args[[2L]])

local_metrics <- utils::read.csv(file.path(out, "local_cpu_metal_summary.csv"))
cuda_metrics <- utils::read.csv(file.path(out, "cuda", "local_cpu_metal_summary.csv"))
metrics <- rbind(local_metrics, cuda_metrics)
metrics <- metrics[match(c("cpu", "metal", "cuda"), metrics$backend_requested), ]
stopifnot(identical(metrics$data_md5, rep(metrics$data_md5[[1L]], 3L)))

run_path <- function(backend) {
  root <- if (backend == "cuda") file.path(out, "cuda") else out
  file.path(
    root, "runs",
    sprintf("MetRef__native__%s__4__100__100__pls_lda.rds", backend)
  )
}
runs <- setNames(lapply(c("cpu", "metal", "cuda"), function(x) readRDS(run_path(x))),
                 c("cpu", "metal", "cuda"))

e <- new.env(parent = emptyenv())
load(data_path, envir = e)
dataset <- if (exists("dataset", e, inherits = FALSE)) {
  e$dataset
} else {
  candidates <- Filter(
    function(x) is.list(x) && !is.null(x$data) && !is.null(x$labels),
    mget(ls(e), e)
  )
  if (length(candidates) != 1L) stop("Cannot identify dataset in ", data_path)
  candidates[[1L]]
}
truth <- droplevels(as.factor(dataset$labels))
truth_colors <- grDevices::hcl.colors(nlevels(truth), "Dynamic")[as.integer(truth)]

summary <- data.frame(
  backend = toupper(metrics$backend_requested),
  graph_seconds = metrics$graph_wall_seconds,
  matrix_seconds = metrics$matrix_wall_seconds,
  umap_seconds = metrics$embedding_seconds,
  pipeline_seconds = metrics$pipeline_seconds,
  best_cv_accuracy = metrics$best_cv_accuracy,
  median_cv_accuracy = metrics$median_cv_accuracy,
  best_ari = metrics$best_ari,
  truth_silhouette = metrics$kodama_truth_silhouette,
  classic_truth_silhouette = metrics$classic_truth_silhouette,
  best_classes = metrics$best_classes,
  median_classes = metrics$median_classes,
  collapse_rate_le2 = metrics$collapse_rate_le2,
  workers = metrics$n_cores_selected,
  data_md5 = metrics$data_md5,
  stringsAsFactors = FALSE
)
utils::write.csv(summary, file.path(out, "fresh_backend_summary.csv"), row.names = FALSE)

draw_embedding <- function(x, title, subtitle) {
  graphics::plot(
    x, pch = 16, cex = 0.75, col = truth_colors, axes = FALSE,
    xlab = "", ylab = "", main = title, sub = subtitle
  )
  graphics::box(col = "grey80")
}

plot_path <- file.path(out, "MetRef_fresh_cpu_metal_cuda_summary.png")
grDevices::png(plot_path, width = 2600, height = 1450, res = 170)
old <- graphics::par(
  mfrow = c(2, 4), mar = c(3.2, 3.4, 3.2, 0.8),
  oma = c(0.5, 0.5, 3.0, 0.5), mgp = c(2.0, 0.65, 0), tcl = -0.25
)
draw_embedding(
  runs$cpu$classic_umap, "Classic UMAP",
  sprintf("truth silhouette %.3f", summary$classic_truth_silhouette[[1L]])
)
for (i in seq_along(runs)) {
  draw_embedding(
    runs[[i]]$kodama_umap,
    paste(summary$backend[[i]], "KODAMA PLS-LDA"),
    sprintf("sil %.3f; ARI %.3f", summary$truth_silhouette[[i]], summary$best_ari[[i]])
  )
}

backend_colors <- c("#2A9D8F", "#E9C46A", "#457B9D")
graphics::barplot(
  summary$matrix_seconds, names.arg = summary$backend, col = backend_colors,
  border = NA, ylab = "seconds", main = "KODAMA.matrix time"
)
graphics::text(
  seq_along(summary$matrix_seconds) - 0.5, summary$matrix_seconds,
  labels = sprintf("%.1f", summary$matrix_seconds), pos = 3, cex = 0.85
)

overhead <- rbind(Graph = summary$graph_seconds, UMAP = summary$umap_seconds)
graphics::barplot(
  overhead, beside = TRUE, names.arg = summary$backend,
  col = c("#6D597A", "#F28482"), border = NA,
  ylab = "seconds", main = "Graph and UMAP overhead"
)
graphics::legend("topright", legend = rownames(overhead), fill = c("#6D597A", "#F28482"),
                 bty = "n", cex = 0.8)

quality <- rbind(ARI = summary$best_ari, Silhouette = summary$truth_silhouette)
graphics::barplot(
  quality, beside = TRUE, names.arg = summary$backend,
  col = c("#E76F51", "#264653"), border = NA, ylim = c(0, 1),
  ylab = "value", main = "Label and layout quality"
)
graphics::legend("bottomright", legend = rownames(quality), fill = c("#E76F51", "#264653"),
                 bty = "n", cex = 0.8)

graphics::boxplot(
  lapply(runs, function(x) x$class_counts), names = summary$backend,
  col = backend_colors, border = "grey30", ylab = "classes",
  main = "Classes across 100 M runs"
)
graphics::abline(h = nlevels(truth), lty = 2, col = "grey40")
graphics::mtext(
  "Fresh MetRef benchmark: M=100, Tcycle=100, 50 PLS components, 655 effective landmarks",
  outer = TRUE, font = 2, cex = 1.15, line = 1.1
)
graphics::par(old)
grDevices::dev.off()

print(summary, row.names = FALSE)
cat("PLOT=", normalizePath(plot_path), "\n", sep = "")
