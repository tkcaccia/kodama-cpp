# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)), "common.R"))

correct_graph_from_saved_labels <- function(graph, res) {
  indices <- as.matrix(graph$indices %||% graph$index)
  distances <- as.matrix(graph$distances %||% graph$distance)
  if (!identical(dim(indices), dim(distances))) stop("Prepared graph dimensions differ.")
  res <- as.matrix(res)
  n <- nrow(indices)
  if (ncol(res) != n) stop("Saved labels and prepared graph have different sample counts.")

  corrected_indices <- indices
  corrected_distances <- distances
  for (i in seq_len(n)) {
    neighbors <- as.integer(indices[i, ])
    values <- as.numeric(distances[i, ])
    usable <- neighbors >= 1L & neighbors <= n & is.finite(values)
    values[!usable] <- Inf
    if (any(usable)) {
      lhs <- res[, i]
      rhs <- res[, neighbors[usable], drop = FALSE]
      valid <- rhs != 0L & lhs != 0L
      same <- colSums(valid & rhs == lhs)
      valid_count <- colSums(valid)
      agreement <- same / pmax(valid_count, 1L)
      adjusted <- rep(Inf, length(same))
      keep <- same > 0L & valid_count > 0L
      adjusted[keep] <- (1 + values[usable][keep]) / (agreement[keep]^2)
      values[usable] <- adjusted
    }
    order <- order(values, neighbors, method = "radix", na.last = TRUE)
    corrected_indices[i, ] <- neighbors[order]
    corrected_distances[i, ] <- values[order]
  }
  list(indices = corrected_indices, distances = corrected_distances)
}

args <- parse_cli()
cells <- read.csv(args$cells %||% stop("--cells required"), stringsAsFactors = FALSE)
id <- as_int(args$cell_id %||% stop("--cell-id required"))
if (is.na(id) || !id %in% cells$cell_id) stop("Invalid cell id")
cell <- cells[cells$cell_id == id, , drop = FALSE]
pca_path <- args$imagenet_pca %||% NULL
d <- load_dataset(cell$dataset, cell$representation, args$data_root, pca_path)
graph_path <- file.path(
  args$prepared_root, cell$dataset, cell$representation,
  paste0("seed_", cell$seed), "prepared_graph.rds"
)
prepared <- readRDS(graph_path)
setting_path <- if (!is.na(cell$value) && nzchar(cell$value)) {
  paste0(cell$setting, "_", cell$value)
} else cell$setting
out <- file.path(
  args$out_root, cell$dataset, cell$representation, cell$classifier,
  cell$experiment, setting_path, paste0("seed_", cell$seed)
)
labels_path <- file.path(out, "labels.rds")
if (!file.exists(labels_path)) stop("No saved classifier checkpoint for cell ", id)
if (file.exists(file.path(out, "exit_status.txt")) &&
    identical(readLines(file.path(out, "exit_status.txt")), "0")) {
  quit(save = "no", status = 0L)
}

old_failure <- tryCatch(read.csv(file.path(out, "metrics.csv"), stringsAsFactors = FALSE), error = function(e) NULL)
matrix_seconds <- NA_real_
started <- Sys.time()
if (!is.null(old_failure) && all(c("started", "ended") %in% names(old_failure))) {
  started <- as.POSIXct(old_failure$started[[1L]])
  matrix_seconds <- as.numeric(difftime(as.POSIXct(old_failure$ended[[1L]]), started, units = "secs"))
}

saved <- readRDS(labels_path)
saved$res <- as_run_matrix(saved$res, expected_samples = nrow(d$x))
saved$acc <- flatten_numeric(saved$acc)
saved$best_labels <- as.integer(flatten_numeric(saved$best_labels))
if (length(saved$best_labels) != nrow(d$x)) {
  saved$best_labels <- as.integer(saved$res[which.max(saved$acc), ])
}
saved$run_diagnostics <- readRDS(file.path(out, "run_diagnostics.rds"))
saved$cycle_diagnostics <- readRDS(file.path(out, "cycle_diagnostics.rds"))
saved$peak_memory_mb <- NA_real_
saved$timing <- list(matrix_wall = matrix_seconds)

cat("Recovering corrected graph for cell ", id, " without rerunning KODAMA.matrix\n", sep = "")
corrected <- correct_graph_from_saved_labels(prepared$graph, saved$res)
visual_input <- list(knn = corrected)
u0 <- proc.time()[["elapsed"]]
umap <- extract_layout(KODAMA::KODAMA.visualization(
  visual_input, method = "UMAP", raw.data = d$x, k = 30L,
  graph.mode = "fuzzy", backend = "cpu", n.cores = 4L, seed = cell$seed
))
umap_seconds <- proc.time()[["elapsed"]] - u0
t0 <- proc.time()[["elapsed"]]
opentsne <- extract_layout(KODAMA::KODAMA.visualization(
  visual_input, method = "opentsne", raw.data = d$x, perplexity = 30,
  backend = "cpu", n.cores = 4L, seed = cell$seed
))
opentsne_seconds <- proc.time()[["elapsed"]] - t0
layouts <- list(umap = umap, opentsne = opentsne)
visual_seconds <- c(umap = umap_seconds, opentsne = opentsne_seconds)

for (nm in names(layouts)) {
  atomic_write_csv(
    data.frame(sample = seq_len(nrow(layouts[[nm]])), x = layouts[[nm]][, 1], y = layouts[[nm]][, 2]),
    file.path(out, paste0(nm, ".csv"))
  )
}
fit_stats <- fit_summary(saved)
atomic_write_csv(fit_stats, file.path(out, "run_metrics.csv"))
atomic_write_csv(cycle_deciles(saved), file.path(out, "cycle_deciles.csv"))
atomic_write_csv(agreement_prefix_metrics(saved$res, prepared$graph, cell$seed), file.path(out, "agreement_prefix.csv"))
embedding_rows <- do.call(rbind, lapply(names(layouts), function(nm) {
  e <- external_metrics(d$labels, saved$best_labels, layouts[[nm]], cell$seed)
  q <- embedding_quality(d$x, layouts[[nm]], d$labels, cell$seed)
  data.frame(method = nm, seconds = unname(visual_seconds[[nm]]), t(e), t(q))
}))
atomic_write_csv(embedding_rows, file.path(out, "embedding_metrics.csv"))
ext <- external_metrics(d$labels, saved$best_labels, umap, cell$seed)
metrics <- data.frame(
  cell, status = "success", phase = "full", M = nrow(saved$res), Tcycle = 100L,
  folds = 5L, n = nrow(d$x), p = ncol(d$x), workers = 4L,
  requested_k = if (cell$classifier == "knn") {
    if (cell$experiment == "knn_sensitivity") as.integer(cell$value) else 30L
  } else NA,
  requested_ncomp = if (cell$classifier == "pls_lda") {
    if (cell$experiment == "ncomp_sensitivity") as.integer(cell$value) else 50L
  } else NA,
  matrix_seconds = matrix_seconds, graph_seconds = prepared$graph_seconds,
  pipeline_seconds = matrix_seconds + prepared$graph_seconds + sum(visual_seconds),
  peak_memory_mb = NA_real_, t(ext), nonfinite_layout = sum(!is.finite(umap)),
  warnings = "", fallbacks = "", data_sha256 = d$data_sha256,
  graph_sha256 = sha256(graph_path), started = as.character(started),
  ended = as.character(Sys.time()), recovered_from_checkpoint = TRUE
)
metrics <- cbind(metrics, fit_stats)
atomic_write_csv(metrics, file.path(out, "metrics.csv"))
atomic_write_csv(
  data.frame(
    stage = c("matrix_wall", "visualization_umap", "visualization_opentsne"),
    seconds = c(matrix_seconds, visual_seconds)
  ),
  file.path(out, "timing.csv")
)
atomic_write_csv(
  data.frame(
    cell, status = "success", phase = "full", stage = "recovered",
    M = nrow(saved$res), Tcycle = 100L, started = as.character(started),
    ended = as.character(Sys.time())
  ),
  file.path(out, "progress.csv")
)
atomic_write_lines("0", file.path(out, "exit_status.txt"))
cat("Recovered cell ", id, "\n", sep = "")
