#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(kodamaR))

load_dataset <- function(path) {
  environment <- new.env(parent = emptyenv())
  load(path, envir = environment)
  objects <- mget(ls(environment, all.names = TRUE), envir = environment)
  if ("dataset" %in% names(objects)) return(objects$dataset)
  candidates <- Filter(
    function(x) is.list(x) && !is.null(x$data) && !is.null(x$labels),
    objects
  )
  if (length(candidates) != 1L) {
    stop("Could not identify dataset$data and dataset$labels in ", path)
  }
  candidates[[1L]]
}

as_double_matrix <- function(x) {
  if (inherits(x, "float32") || inherits(x, "float")) {
    if (!requireNamespace("float", quietly = TRUE)) {
      stop("The float package is required to read this dataset.")
    }
    x <- float::dbl(x)
  }
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

label_agreement <- function(a, b) mean(as.integer(a) == as.integer(b))

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) args[[1L]] else file.path(
  tempdir(), "metref_graph_api"
)
data_path <- if (length(args) >= 2L) args[[2L]] else
  "/mnt/sata_ssd/fastEmbedR/Data/MetRef/MetRef_float32.RData"
backend <- if (length(args) >= 3L) args[[3L]] else "cuda"
M <- as.integer(Sys.getenv("KODAMA_M", "10"))
Tcycle <- as.integer(Sys.getenv("KODAMA_TCYCLE", "10"))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dataset <- load_dataset(data_path)
x <- as_double_matrix(dataset$data)
truth <- droplevels(as.factor(dataset$labels))
if (nrow(x) != 873L || ncol(x) != 375L) {
  stop("Expected the 873 x 375 MetRef matrix.")
}

common <- list(
  M = M,
  Tcycle = Tcycle,
  ncomp = 50L,
  landmarks = 10000000L,
  splitting = 100L,
  n.cores = 4L,
  graph.neighbors = 100L,
  knn.k = 50L,
  metric = "euclidean",
  backend = backend,
  seed = 1234L,
  visual.init = TRUE,
  progress = FALSE
)

graph_start <- proc.time()[["elapsed"]]
prepared <- KODAMA.graph(
  x,
  k = 100L,
  metric = "euclidean",
  backend = backend,
  n.cores = 4L,
  seed = 1234L
)
graph_wall <- proc.time()[["elapsed"]] - graph_start
if (!is.null(prepared$data)) stop("KODAMA.graph retained the raw matrix.")

bare <- list(indices = prepared$indices, distances = prepared$distances)
runs <- list()
rows <- list()

run_one <- function(classifier, input_mode) {
  call <- common
  call$classifier <- classifier
  if (input_mode == "raw") {
    call$data <- x
  } else if (input_mode == "prepared") {
    call$data <- prepared
  } else if (input_mode == "prepared_plus_raw") {
    call$data <- prepared
    call$raw.data <- x
  } else if (input_mode == "bare") {
    call$data <- bare
  } else {
    stop("Unknown input mode.")
  }
  started <- proc.time()[["elapsed"]]
  fit <- do.call(KODAMA.matrix, call)
  wall <- proc.time()[["elapsed"]] - started
  key <- paste(classifier, input_mode, sep = "__")
  runs[[key]] <<- fit
  rows[[key]] <<- data.frame(
    classifier = classifier,
    input_mode = input_mode,
    graph_builds = fit$graph_builds,
    graph_seconds = fit$timing$graph_seconds,
    visual_init_seconds = fit$timing$visual_init_seconds,
    optimization_seconds = fit$timing$optimization_wall_seconds,
    runtime_seconds = fit$runtime_seconds,
    wall_seconds = wall,
    best_accuracy = max(fit$acc, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

for (classifier in c("knn", "pls_lda")) {
  for (input_mode in c("raw", "prepared_plus_raw", "prepared", "bare")) {
    run_one(classifier, input_mode)
  }
}

metrics <- do.call(rbind, rows)
metrics$graph_preparation_wall_seconds <- graph_wall
metrics$graph_object_bytes <- as.numeric(object.size(prepared))
metrics$raw_matrix_bytes <- as.numeric(object.size(x))
metrics$contains_raw_data <- !is.null(prepared$data)
metrics$comparison_label_agreement <- NA_real_
for (classifier in c("knn", "pls_lda")) {
  raw <- runs[[paste(classifier, "raw", sep = "__")]]
  prepared_raw <- runs[[paste(classifier, "prepared_plus_raw", sep = "__")]]
  prepared_only <- runs[[paste(classifier, "prepared", sep = "__")]]
  bare_only <- runs[[paste(classifier, "bare", sep = "__")]]
  metrics$comparison_label_agreement[
    metrics$classifier == classifier & metrics$input_mode %in% c("raw", "prepared_plus_raw")
  ] <- label_agreement(raw$res, prepared_raw$res)
  metrics$comparison_label_agreement[
    metrics$classifier == classifier & metrics$input_mode %in% c("prepared", "bare")
  ] <- label_agreement(prepared_only$res, bare_only$res)
}
write.csv(metrics, file.path(output_dir, "metrics.csv"), row.names = FALSE)
saveRDS(prepared, file.path(output_dir, "metref_kodama_graph.rds"), compress = FALSE)
saveRDS(runs, file.path(output_dir, "metref_graph_api_runs.rds"), compress = FALSE)

plot_keys <- c(
  "knn__raw", "knn__prepared_plus_raw",
  "pls_lda__raw", "pls_lda__prepared_plus_raw"
)
layouts <- lapply(plot_keys, function(key) {
  KODAMA.visualization(
    runs[[key]],
    method = "UMAP",
    k = 30L,
    backend = backend,
    n.cores = 4L,
    n.epochs = 200L,
    seed = 4L,
    graph.mode = "fuzzy"
  )
})
names(layouts) <- plot_keys
saveRDS(layouts, file.path(output_dir, "metref_graph_api_umap.rds"), compress = FALSE)

png(
  file.path(output_dir, "metref_graph_api_umap.png"),
  width = 2000,
  height = 1800,
  res = 180
)
par(mfrow = c(2, 2), mar = c(1, 1, 3, 1))
colors <- grDevices::hcl.colors(nlevels(truth), "Dynamic")[as.integer(truth)]
titles <- c(
  "KNN: raw matrix",
  "KNN: KODAMA.graph + raw matrix",
  "PLS-LDA: raw matrix",
  "PLS-LDA: KODAMA.graph + raw matrix"
)
for (index in seq_along(layouts)) {
  plot(
    layouts[[index]],
    col = colors,
    pch = 19,
    cex = 0.45,
    axes = FALSE,
    xlab = "",
    ylab = "",
    main = titles[[index]]
  )
  box()
}
dev.off()

print(metrics)
cat("output_dir=", output_dir, "\n", sep = "")
