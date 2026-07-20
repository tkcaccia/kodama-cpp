#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

options(stringsAsFactors = FALSE)

env_value <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}

load_dataset <- function(path) {
  environment <- new.env(parent = emptyenv())
  load(path, envir = environment)
  objects <- mget(ls(environment, all.names = TRUE), envir = environment)
  if ("dataset" %in% names(objects)) return(objects$dataset)
  candidates <- Filter(
    function(x) is.list(x) && !is.null(x$data) && !is.null(x$labels),
    objects
  )
  if (length(candidates) != 1L) stop("Could not identify dataset in ", path)
  candidates[[1L]]
}

root <- normalizePath(env_value("KODAMA_CPP_ROOT", getwd()), mustWork = TRUE)
source(file.path(root, "wrappers", "R", "kodama_matrix_temp.R"))

data_path <- env_value(
  "KODAMA_DATA_PATH",
  "/mnt/sata_ssd/fastEmbedR/Data/MetRef/MetRef_float32.RData"
)
backend <- env_value("KODAMA_BACKEND", "cuda")
runs <- as.integer(env_value("KODAMA_M", "8"))
cycles <- as.integer(env_value("KODAMA_TCYCLE", "20"))
threads <- as.integer(env_value("KODAMA_THREADS", "4"))
output <- env_value("KODAMA_OUTPUT", "")

dataset <- load_dataset(data_path)
start <- proc.time()[["elapsed"]]
result <- KODAMA.matrix.cpp(
  data = dataset$data,
  M = runs,
  Tcycle = cycles,
  ncomp = 50L,
  landmarks = 100000L,
  splitting = 100L,
  n.cores = threads,
  graph.neighbors = 100L,
  knn.k = 50L,
  classifier = "pls_lda",
  backend = backend,
  seed = 1234L,
  progress = TRUE
)
elapsed <- proc.time()[["elapsed"]] - start

if (nzchar(output)) saveRDS(result, output, compress = FALSE)
cat(
  "backend", backend,
  "M", runs,
  "Tcycle", cycles,
  "elapsed", elapsed,
  "runtime", result$runtime_seconds,
  "median_acc", stats::median(result$acc),
  "max_acc", max(result$acc),
  "median_classes", stats::median(apply(result$res, 1L, function(x) length(unique(x)))),
  "\n"
)
