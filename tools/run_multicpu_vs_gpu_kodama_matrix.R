#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

suppressPackageStartupMessages({
  library(kodamaR)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L) y else x

parse_csv_env <- function(name, default) {
  value <- Sys.getenv(name, default)
  value <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  value[nzchar(value)]
}

as_numeric_matrix <- function(x) {
  if (inherits(x, "float32")) {
    if (!requireNamespace("float", quietly = TRUE)) {
      stop("The float package is required to read float32 RData matrices.")
    }
    x <- float::dbl(x)
  }
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

dataset_paths <- function(root) {
  c(
    COIL20 = file.path(root, "COIL20", "COIL20_float32.RData"),
    FashionMNIST = file.path(root, "FashionMNIST", "FashionMNIST_float32.RData"),
    MNIST = file.path(root, "MNIST", "MNIST_float32.RData"),
    Macosko2015_retina = file.path(root, "Macosko2015_retina", "Macosko2015_retina_float32.RData"),
    MetRef = file.path(root, "MetRef", "MetRef_float32.RData"),
    TabulaMuris = file.path(root, "TabulaMuris", "TabulaMuris_float32.RData"),
    USPS = file.path(root, "USPS", "USPS_float32.RData"),
    flow18 = file.path(root, "flow18", "flow18_float32.RData"),
    mass41 = file.path(root, "mass41", "mass41_float32.RData")
  )
}

load_dataset <- function(path, max_rows = 0L, seed = 1L) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  obj <- if (exists("dataset", env, inherits = FALSE)) env$dataset else mget(ls(env), env)[[1L]]
  x <- as_numeric_matrix(obj$data)
  y <- as.factor(obj$labels)
  if (max_rows > 0L && nrow(x) > max_rows) {
    set.seed(seed)
    keep <- sort(sample.int(nrow(x), max_rows))
    x <- x[keep, , drop = FALSE]
    y <- droplevels(y[keep])
  }
  list(data = x, labels = y)
}

choose2 <- function(x) x * (x - 1) / 2

adjusted_rand_index <- function(a, b) {
  a <- as.factor(a)
  b <- as.factor(b)
  tab <- table(a, b)
  n <- sum(tab)
  if (n < 2L) return(NA_real_)
  sum_nij <- sum(choose2(as.numeric(tab)))
  sum_ai <- sum(choose2(rowSums(tab)))
  sum_bj <- sum(choose2(colSums(tab)))
  expected <- sum_ai * sum_bj / choose2(n)
  max_index <- 0.5 * (sum_ai + sum_bj)
  denom <- max_index - expected
  if (!is.finite(denom) || abs(denom) < .Machine$double.eps) return(NA_real_)
  (sum_nij - expected) / denom
}

best_labels <- function(result) {
  if (!is.null(result$best_labels)) return(result$best_labels)
  if (!is.null(result$res) && !is.null(result$acc) && length(result$acc) > 0L) {
    return(as.integer(result$res[which.max(result$acc), ]))
  }
  integer()
}

timing_value <- function(result, name) {
  if (!is.null(result$timing) && !is.null(result$timing[[name]])) {
    return(as.numeric(result$timing[[name]]))
  }
  NA_real_
}

main <- function() {
  root <- Sys.getenv("KODAMA_DATA_ROOT", "/mnt/sata_ssd/fastEmbedR/Data")
  out_dir <- Sys.getenv("KODAMA_MATRIX_BACKEND_OUT", "/mnt/sata_ssd/KODAMAopt/multicpu_vs_gpu_kodama_matrix")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  datasets <- parse_csv_env("KODAMA_MATRIX_DATASETS", "MetRef,USPS,COIL20")
  classifiers <- parse_csv_env("KODAMA_MATRIX_CLASSIFIERS", "knn,pls_lda")
  backends <- parse_csv_env("KODAMA_MATRIX_BACKENDS", "cpu,cuda")
  paths <- dataset_paths(root)

  M <- as.integer(Sys.getenv("KODAMA_MATRIX_M", "100"))
  Tcycle <- as.integer(Sys.getenv("KODAMA_MATRIX_TCYCLE", "100"))
  ncomp <- as.integer(Sys.getenv("KODAMA_MATRIX_NCOMP", "50"))
  landmarks <- as.integer(Sys.getenv("KODAMA_MATRIX_LANDMARKS", "100000"))
  graph_neighbors <- as.integer(Sys.getenv("KODAMA_MATRIX_GRAPH_NEIGHBORS", "100"))
  knn_k <- as.integer(Sys.getenv("KODAMA_MATRIX_KNN_K", "30"))
  cpu_cores <- as.integer(Sys.getenv("KODAMA_MATRIX_CPU_CORES", "4"))
  gpu_cores <- as.integer(Sys.getenv("KODAMA_MATRIX_GPU_CORES", "0"))
  seed <- as.integer(Sys.getenv("KODAMA_MATRIX_SEED", "1234"))
  max_rows <- as.integer(Sys.getenv("KODAMA_MATRIX_MAX_ROWS", "0"))

  rows <- list()
  summary_path <- file.path(out_dir, "multicpu_vs_gpu_kodama_matrix_summary.csv")

  for (dataset in datasets) {
    if (!dataset %in% names(paths)) stop("Unknown dataset: ", dataset)
    message("== ", dataset, " ==")
    ds <- load_dataset(paths[[dataset]], max_rows = max_rows, seed = seed)
    x <- ds$data
    truth <- ds$labels
    n <- nrow(x)
    p <- ncol(x)
    splitting <- if (n < 40000L) 100L else 300L

    for (classifier in classifiers) {
      for (backend in backends) {
        n_cores <- if (backend == "cuda") gpu_cores else cpu_cores
        message("  ", classifier, " / ", backend, " n.cores=", n_cores)
        started <- proc.time()[["elapsed"]]
        result <- tryCatch(
          kodama_matrix(
            x,
            M = M,
            Tcycle = Tcycle,
            ncomp = min(ncomp, p, n - 1L),
            landmarks = landmarks,
            splitting = splitting,
            n.cores = n_cores,
            graph.neighbors = graph_neighbors,
            knn.k = knn_k,
            classifier = classifier,
            backend = backend,
            seed = seed,
            progress = TRUE,
            apply.kodama.dissimilarity = TRUE
          ),
          error = function(e) structure(list(error = conditionMessage(e)), class = "kodama_error")
        )
        elapsed <- proc.time()[["elapsed"]] - started
        if (inherits(result, "kodama_error")) {
          rows[[length(rows) + 1L]] <- data.frame(
            dataset = dataset, classifier = classifier, backend = backend,
            n = n, p = p, truth_classes = length(levels(truth)),
            M = M, Tcycle = Tcycle, n_cores_requested = n_cores,
            n_cores_used = NA_integer_, elapsed_sec = elapsed,
            runtime_sec = NA_real_, optimization_wall_sec = NA_real_,
            graph_sec = NA_real_, dissimilarity_sec = NA_real_,
            best_acc = NA_real_, median_acc = NA_real_, best_ari = NA_real_,
            median_classes = NA_real_, min_classes = NA_integer_, max_classes = NA_integer_,
            status = result$error, stringsAsFactors = FALSE
          )
        } else {
          labels <- best_labels(result)
          rows[[length(rows) + 1L]] <- data.frame(
            dataset = dataset, classifier = classifier, backend = backend,
            n = n, p = p, truth_classes = length(levels(truth)),
            M = M, Tcycle = Tcycle, n_cores_requested = n_cores,
            n_cores_used = as.integer(result$n.cores %||% result$n_threads %||% n_cores),
            elapsed_sec = elapsed,
            runtime_sec = as.numeric(result$runtime_seconds %||% timing_value(result, "runtime_seconds")),
            optimization_wall_sec = timing_value(result, "optimization_wall_seconds"),
            graph_sec = timing_value(result, "graph_seconds"),
            dissimilarity_sec = timing_value(result, "dissimilarity_seconds"),
            best_acc = if (length(result$acc)) max(result$acc, na.rm = TRUE) else NA_real_,
            median_acc = if (length(result$acc)) median(result$acc, na.rm = TRUE) else NA_real_,
            best_ari = if (length(labels)) adjusted_rand_index(labels, truth) else NA_real_,
            median_classes = if (!is.null(result$class_counts)) median(result$class_counts) else NA_real_,
            min_classes = if (!is.null(result$class_counts)) min(result$class_counts) else NA_integer_,
            max_classes = if (!is.null(result$class_counts)) max(result$class_counts) else NA_integer_,
            status = "ok",
            stringsAsFactors = FALSE
          )
          saveRDS(result, file.path(out_dir, sprintf("%s_%s_%s_M%d_T%d.rds", dataset, classifier, backend, M, Tcycle)))
        }
        utils::write.csv(do.call(rbind, rows), summary_path, row.names = FALSE)
      }
    }
  }
  message("wrote ", summary_path)
}

main()
