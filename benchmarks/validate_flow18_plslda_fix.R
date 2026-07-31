#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

args <- commandArgs(trailingOnly = TRUE)
dataset_file <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  "/mnt/sata_ssd/fastEmbedR_Data/flow18/flow18.RData"
}
output_dir <- if (length(args) >= 2L) {
  args[[2L]]
} else {
  file.path(tempdir(), "kodama_flow18_plslda_fix")
}
library_dir <- if (length(args) >= 3L) args[[3L]] else NA_character_

if (!is.na(library_dir)) {
  .libPaths(c(library_dir, .libPaths()))
}
suppressPackageStartupMessages(
  library(
    kodamaR,
    lib.loc = if (is.na(library_dir)) NULL else library_dir
  )
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
loaded <- load(dataset_file)
if (!"dataset" %in% loaded || !is.list(dataset)) {
  stop("Expected `dataset` list in ", dataset_file)
}
x <- dataset[["data"]]
if (!is.matrix(x)) {
  stop("Expected `dataset$data` to be a matrix.")
}

configuration <- data.frame(
  dataset = "flow18",
  samples = nrow(x),
  features = ncol(x),
  classifier = "pls_lda",
  backend = "cpu",
  M = 1L,
  Tcycle = 2L,
  landmarks = 1000L,
  ncomp = min(50L, ncol(x)),
  n.cores = 4L,
  seed = 4L,
  stringsAsFactors = FALSE
)
write.csv(
  configuration,
  file.path(output_dir, "flow18_plslda_fix_configuration.csv"),
  row.names = FALSE
)

started <- Sys.time()
fit <- KODAMA.matrix(
  x,
  M = configuration$M,
  Tcycle = configuration$Tcycle,
  ncomp = configuration$ncomp,
  landmarks = configuration$landmarks,
  n.cores = configuration$n.cores,
  classifier = configuration$classifier,
  backend = configuration$backend,
  seed = configuration$seed,
  visual.init = FALSE,
  progress = TRUE,
  apply.kodama.dissimilarity = FALSE
)
elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))

stopifnot(
  inherits(fit, "kodama_matrix"),
  length(fit[["best_labels"]]) == nrow(x),
  all(is.finite(fit[["best_labels"]]))
)

summary <- transform(
  configuration,
  status = "success",
  elapsed_sec = elapsed,
  output_labels = length(fit[["best_labels"]]),
  selected_classes = length(unique(fit[["best_labels"]]))
)
write.csv(
  summary,
  file.path(output_dir, "flow18_plslda_fix_summary.csv"),
  row.names = FALSE
)
saveRDS(
  list(
    summary = summary,
    timing = KODAMA.timing(fit),
    best_labels = fit[["best_labels"]]
  ),
  file.path(output_dir, "flow18_plslda_fix_result.rds"),
  compress = FALSE
)
print(summary)
