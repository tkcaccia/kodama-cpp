# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)), "common.R"))

args <- parse_cli()
cells <- read.csv(args$cells %||% stop("--cells required"), stringsAsFactors = FALSE)
bundles <- read.csv(args$bundles %||% stop("--bundles required"), stringsAsFactors = FALSE)
bundle_id <- as_int(args$bundle_id %||% Sys.getenv("SLURM_ARRAY_TASK_ID"))
members <- bundles[bundles$bundle_id == bundle_id, , drop = FALSE]
if (!nrow(members)) stop("Invalid bundle id")
selected <- cells[match(members$cell_id, cells$cell_id), , drop = FALSE]

cell_dir <- function(cell) {
  setting <- if (!is.na(cell$value) && nzchar(cell$value)) {
    paste0(cell$setting, "_", cell$value)
  } else cell$setting
  file.path(
    args$out_root, cell$dataset, cell$representation, cell$classifier,
    cell$experiment, setting, paste0("seed_", cell$seed)
  )
}

records <- list()
failed <- FALSE
for (i in seq_len(nrow(selected))) {
  cell <- selected[i, , drop = FALSE]
  out <- cell_dir(cell)
  started <- Sys.time()
  status <- 0L
  error <- ""
  tryCatch({
    metric_path <- file.path(out, "metrics.csv")
    embedding_path <- file.path(out, "embedding_metrics.csv")
    if (!file.exists(metric_path) || !file.exists(embedding_path)) {
      stop("Successful cell output is incomplete")
    }
    metrics <- read.csv(metric_path, stringsAsFactors = FALSE, check.names = FALSE)
    if (!nrow(metrics) || metrics$status[[1L]] != "success") stop("Cell did not succeed")
    embeddings <- read.csv(embedding_path, stringsAsFactors = FALSE, check.names = FALSE)
    d <- load_dataset(
      cell$dataset, cell$representation, args$data_root,
      args$imagenet_pca %||% NULL
    )
    kodama_labels <- NULL
    if (cell$classifier != "classic") {
      saved <- readRDS(file.path(out, "labels.rds"))
      kodama_labels <- as.integer(flatten_numeric(saved$best_labels))
      if (length(kodama_labels) != nrow(d$x)) stop("Saved label length mismatch")
    }

    for (method in c("umap", "opentsne")) {
      layout_path <- file.path(out, paste0(method, ".csv"))
      if (!file.exists(layout_path)) stop("Missing ", method, " layout")
      layout_csv <- read.csv(layout_path, stringsAsFactors = FALSE)
      layout <- as.matrix(layout_csv[, c("x", "y"), drop = FALSE])
      if (nrow(layout) != nrow(d$x) || any(!is.finite(layout))) {
        stop("Invalid ", method, " layout")
      }
      truth_sil <- silhouette_summary(layout, d$labels, seed = cell$seed)
      label_sil <- if (is.null(kodama_labels)) NA_real_ else {
        silhouette_summary(layout, kodama_labels, seed = cell$seed)[["silhouette"]]
      }
      row <- match(method, embeddings$method)
      if (is.na(row)) stop("Missing embedding metric row for ", method)
      embeddings$silhouette[[row]] <- truth_sil[["silhouette"]]
      embeddings$silhouette_class_min[[row]] <- truth_sil[["silhouette_class_min"]]
      embeddings$kodama_label_silhouette[[row]] <- label_sil
      if (method == "umap") {
        metrics$silhouette[[1L]] <- truth_sil[["silhouette"]]
        metrics$silhouette_class_min[[1L]] <- truth_sil[["silhouette_class_min"]]
        metrics$kodama_label_silhouette[[1L]] <- label_sil
      }
    }
    atomic_write_csv(embeddings, embedding_path)
    atomic_write_csv(metrics, metric_path)
  }, error = function(e) {
    status <<- 1L
    error <<- conditionMessage(e)
    failed <<- TRUE
  })
  records[[i]] <- data.frame(
    bundle_id = bundle_id, cell_id = cell$cell_id, status = status,
    error = error, started = as.character(started), ended = as.character(Sys.time())
  )
  atomic_write_csv(
    do.call(rbind, records),
    file.path(args$status_root, paste0("bundle_", bundle_id, ".csv"))
  )
}
if (failed) quit(save = "no", status = 1L)
