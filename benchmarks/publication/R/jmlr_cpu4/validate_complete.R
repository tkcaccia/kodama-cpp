# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)), "common.R"))

args <- parse_cli()
cells <- read.csv(file.path(args$run_root, "cells.csv"), stringsAsFactors = FALSE)
results <- vector("list", nrow(cells))

for (i in seq_len(nrow(cells))) {
  cell <- cells[i, , drop = FALSE]
  setting <- if (!is.na(cell$value) && nzchar(cell$value)) {
    paste0(cell$setting, "_", cell$value)
  } else cell$setting
  out <- file.path(
    args$run_root, "cells", cell$dataset, cell$representation, cell$classifier,
    cell$experiment, setting, paste0("seed_", cell$seed)
  )
  reasons <- character()
  metric_path <- file.path(out, "metrics.csv")
  embedding_path <- file.path(out, "embedding_metrics.csv")
  metrics <- tryCatch(read.csv(metric_path, stringsAsFactors = FALSE, check.names = FALSE), error = function(e) NULL)
  embeddings <- tryCatch(read.csv(embedding_path, stringsAsFactors = FALSE, check.names = FALSE), error = function(e) NULL)
  if (is.null(metrics) || !nrow(metrics) || metrics$status[[1L]] != "success") {
    reasons <- c(reasons, "missing successful metrics")
  }
  if (is.null(embeddings) || !all(c("umap", "opentsne") %in% embeddings$method)) {
    reasons <- c(reasons, "missing embedding metrics")
  } else if (any(!is.finite(embeddings$silhouette)) ||
             any(!is.finite(embeddings$silhouette_class_min))) {
    reasons <- c(reasons, "non-finite truth silhouette")
  }
  for (method in c("umap", "opentsne")) {
    layout <- tryCatch(read.csv(file.path(out, paste0(method, ".csv"))), error = function(e) NULL)
    if (is.null(layout) || !all(c("x", "y") %in% names(layout)) ||
        any(!is.finite(as.matrix(layout[, c("x", "y"), drop = FALSE])))) {
      reasons <- c(reasons, paste0("invalid ", method, " layout"))
    }
  }
  if (cell$classifier != "classic" && !is.null(metrics) && nrow(metrics)) {
    if (!isTRUE(all.equal(as.numeric(metrics$cv_evaluations[[1L]]), 10100))) {
      reasons <- c(reasons, "CV evaluation count is not 10100")
    }
    if (!is.finite(metrics$collapse_one_rate[[1L]]) || metrics$collapse_one_rate[[1L]] > 0) {
      reasons <- c(reasons, "single-class collapse detected")
    }
    if (!is.finite(metrics$distinct_solutions[[1L]]) || metrics$distinct_solutions[[1L]] < 2) {
      reasons <- c(reasons, "insufficient independent M solutions")
    }
  }
  results[[i]] <- data.frame(
    cell_id = cell$cell_id, dataset = cell$dataset, classifier = cell$classifier,
    ok = !length(reasons), reason = paste(unique(reasons), collapse = " | ")
  )
}

report <- do.call(rbind, results)
atomic_write_csv(report, file.path(args$run_root, "complete_validation.csv"))
if (any(!report$ok)) {
  cat("Validation failed for ", sum(!report$ok), " of ", nrow(report), " cells.\n", sep = "")
  quit(save = "no", status = 1L)
}
atomic_write_lines(
  paste("validated", nrow(report), "cells at", format(Sys.time(), tz = "UTC")),
  file.path(args$run_root, "COMPLETE_OK")
)
cat("Validated all ", nrow(report), " benchmark cells.\n", sep = "")
