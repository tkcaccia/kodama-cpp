#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1L) args[[1L]] else "manuscript/jmlr_nonspatial_panel_20260718"
input <- file.path(root, "nonspatial_panel_summary.csv")
output <- file.path(root, "nonspatial_visualization_validation.png")

x <- read.csv(input, check.names = FALSE)
stopifnot(
  nrow(x) == 20L,
  all(x$M == 100L),
  all(x$Tcycle == 100L),
  all(x$backend == "cuda")
)

dataset_order <- c("MetRef", "PBMC3K_PCA50", "OptDigits", "USPS", "Macosko2015_retina")
dataset_labels <- c("MetRef", "PBMC3K", "OptDigits", "USPS", "Macosko")
classifier_order <- c("knn", "pls_lda")
classifier_labels <- c("KNN", "PLS-LDA")
colors <- c(classic = "#6B7280", knn = "#009E73", pls_lda = "#0072B2")

lookup <- function(dataset, classifier, method, column) {
  row <- x[
    x$dataset == dataset & x$classifier == classifier & x$method == method,
    ,
    drop = FALSE
  ]
  stopifnot(nrow(row) == 1L)
  row[[column]][[1L]]
}

silhouette_matrix <- function(method) {
  classic <- vapply(
    dataset_order,
    function(dataset) lookup(dataset, "knn", method, "classic_truth_silhouette"),
    numeric(1L)
  )
  knn <- vapply(
    dataset_order,
    function(dataset) lookup(dataset, "knn", method, "kodama_truth_silhouette"),
    numeric(1L)
  )
  pls <- vapply(
    dataset_order,
    function(dataset) lookup(dataset, "pls_lda", method, "kodama_truth_silhouette"),
    numeric(1L)
  )
  rbind(Classic = classic, KNN = knn, `PLS-LDA` = pls)
}

runtime <- vapply(
  dataset_order,
  function(dataset) {
    vapply(
      classifier_order,
      function(classifier) lookup(dataset, classifier, "UMAP", "kodama_wall_seconds"),
      numeric(1L)
    )
  },
  numeric(length(classifier_order))
)

ari <- vapply(
  dataset_order,
  function(dataset) {
    vapply(
      classifier_order,
      function(classifier) lookup(dataset, classifier, "UMAP", "selected_ari"),
      numeric(1L)
    )
  },
  numeric(length(classifier_order))
)

png(output, width = 2400, height = 1450, res = 220, type = "cairo")
par(
  mfrow = c(2, 2),
  mar = c(4.8, 4.8, 3.0, 1.0),
  oma = c(0.5, 0.5, 1.0, 0.5),
  family = "sans",
  las = 1,
  mgp = c(2.8, 0.8, 0),
  tcl = -0.25
)

draw_silhouette <- function(method, panel) {
  values <- silhouette_matrix(method)
  barplot(
    values,
    beside = TRUE,
    names.arg = dataset_labels,
    col = colors[c("classic", "knn", "pls_lda")],
    border = NA,
    ylim = c(-0.2, 0.9),
    ylab = "Truth-label silhouette",
    main = paste0(panel, "  ", if (method == "UMAP") "UMAP" else "openTSNE"),
    cex.names = 0.9
  )
  abline(h = seq(-0.2, 0.8, 0.2), col = "#E5E7EB", lwd = 0.8)
  abline(h = 0, col = "#9CA3AF", lwd = 0.9)
  box(bty = "l")
  legend(
    "topright",
    legend = c("Classic", "KODAMA KNN", "KODAMA PLS-LDA"),
    fill = colors[c("classic", "knn", "pls_lda")],
    border = NA,
    bty = "n",
    cex = 0.78
  )
}

draw_silhouette("UMAP", "A")
draw_silhouette("opentsne", "B")

matplot(
  seq_along(dataset_order),
  t(runtime),
  type = "b",
  log = "y",
  lty = 1,
  lwd = 2.2,
  pch = c(16, 17),
  col = colors[classifier_order],
  xaxt = "n",
  xlab = "",
  ylab = "KODAMA.matrix wall time (s, log scale)",
  main = "C  CUDA runtime",
  ylim = c(2, 1000)
)
axis(1, at = seq_along(dataset_labels), labels = dataset_labels, las = 1)
abline(h = c(3, 10, 30, 100, 300, 1000), col = "#E5E7EB", lwd = 0.8)
box(bty = "l")
legend(
  "topleft",
  legend = classifier_labels,
  col = colors[classifier_order],
  pch = c(16, 17),
  lty = 1,
  lwd = 2.2,
  bty = "n",
  cex = 0.82
)

barplot(
  ari,
  beside = TRUE,
  names.arg = dataset_labels,
  col = colors[classifier_order],
  border = NA,
  ylim = c(0, 1),
  ylab = "Selected-run ARI",
  main = "D  Post hoc label agreement",
  cex.names = 0.9
)
abline(h = seq(0, 1, 0.2), col = "#E5E7EB", lwd = 0.8)
box(bty = "l")
legend(
  "topright",
  legend = classifier_labels,
  fill = colors[classifier_order],
  border = NA,
  bty = "n",
  cex = 0.82
)

mtext(
  "Fixed release-validation protocol: CUDA, M=100, Tcycle=100; truth labels used only after optimization",
  outer = TRUE,
  side = 3,
  line = -0.1,
  cex = 0.9,
  font = 2
)
dev.off()

cat(output, "\n")
