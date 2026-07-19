#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

wrapper_library <- Sys.getenv("KODAMA_R_LIB", "")
if (nzchar(wrapper_library)) {
  .libPaths(c(wrapper_library, .libPaths()))
}

suppressPackageStartupMessages({
  library(KODAMA)
  library(kodamaR)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L) y else x

csv_env <- function(name, default) {
  value <- trimws(strsplit(Sys.getenv(name, default), ",", fixed = TRUE)[[1L]])
  value[nzchar(value)]
}

integer_env <- function(name, default) {
  value <- suppressWarnings(as.integer(Sys.getenv(name, as.character(default))))
  if (!is.finite(value)) stop("Invalid integer in ", name)
  value
}

as_numeric_matrix <- function(x) {
  if (inherits(x, "float32") || inherits(x, "float")) {
    if (!requireNamespace("float", quietly = TRUE)) {
      stop("The float package is required to read this RData matrix.")
    }
    x <- float::dbl(x)
  }
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

dataset_paths <- function(root) {
  c(
    MetRef = file.path(root, "MetRef", "MetRef_float32.RData"),
    USPS = file.path(root, "USPS", "USPS_float32.RData")
  )
}

load_dataset <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  object <- if (exists("dataset", env, inherits = FALSE)) {
    env$dataset
  } else {
    candidates <- mget(ls(env), env, inherits = FALSE)
    candidates[[which(vapply(candidates, function(x) {
      is.list(x) && all(c("data", "labels") %in% names(x))
    }, logical(1L)))[1L]]]
  }
  if (is.null(object$data) || is.null(object$labels)) {
    stop("Expected a list with data and labels in ", path)
  }
  list(data = as_numeric_matrix(object$data), labels = droplevels(as.factor(object$labels)))
}

choose2 <- function(x) x * (x - 1) / 2

adjusted_rand_index <- function(a, b) {
  tab <- table(as.factor(a), as.factor(b))
  n <- sum(tab)
  if (n < 2L) return(NA_real_)
  sum_nij <- sum(choose2(as.numeric(tab)))
  sum_ai <- sum(choose2(rowSums(tab)))
  sum_bj <- sum(choose2(colSums(tab)))
  expected <- sum_ai * sum_bj / choose2(n)
  maximum <- 0.5 * (sum_ai + sum_bj)
  denominator <- maximum - expected
  if (!is.finite(denominator) || abs(denominator) < .Machine$double.eps) return(NA_real_)
  (sum_nij - expected) / denominator
}

run_matrix <- function(result, n_samples) {
  labels <- as.matrix(result$res)
  if (ncol(labels) == n_samples) return(labels)
  if (nrow(labels) == n_samples) return(t(labels))
  stop("Cannot align result$res with ", n_samples, " samples.")
}

summarize_result <- function(result, truth) {
  labels <- run_matrix(result, length(truth))
  accuracy <- as.numeric(result$acc)
  if (length(accuracy) != nrow(labels)) {
    stop("Result accuracy and label-run counts differ.")
  }
  valid <- which(is.finite(accuracy) & apply(labels, 1L, function(x) all(is.finite(x))))
  if (!length(valid)) stop("No valid KODAMA runs were returned.")
  selected <- valid[which.max(accuracy[valid])]
  aris <- apply(labels, 1L, adjusted_rand_index, b = truth)
  classes <- apply(labels, 1L, function(x) length(unique(x)))
  list(
    selected_run = selected,
    selected_accuracy = accuracy[selected],
    median_accuracy = stats::median(accuracy[valid]),
    selected_ari = aris[selected],
    median_ari = stats::median(aris[valid], na.rm = TRUE),
    selected_classes = classes[selected],
    median_classes = stats::median(classes[valid]),
    min_classes = min(classes[valid]),
    max_classes = max(classes[valid])
  )
}

atomic_csv <- function(value, path) {
  temporary <- tempfile(pattern = paste0(basename(path), "."), tmpdir = dirname(path))
  utils::write.csv(value, temporary, row.names = FALSE)
  if (!file.rename(temporary, path)) stop("Could not replace ", path)
}

record_path <- function(out_dir, dataset, implementation) {
  file.path(out_dir, "records", sprintf("%s__%s.rds", dataset, implementation))
}

collect_records <- function(out_dir) {
  paths <- list.files(file.path(out_dir, "records"), pattern = "\\.rds$", full.names = TRUE)
  if (!length(paths)) return(data.frame())
  rows <- lapply(paths, function(path) readRDS(path)$summary)
  output <- do.call(rbind, rows)
  output[order(output$dataset, output$implementation), , drop = FALSE]
}

main <- function() {
  data_root <- Sys.getenv("KODAMA_DATA_ROOT", "/mnt/sata_ssd/fastEmbedR/Data")
  out_dir <- Sys.getenv(
    "KODAMA_COMPARE_OUT",
    "/mnt/sata_ssd/kodama-jmlr-current-cran-comparison"
  )
  dir.create(file.path(out_dir, "records"), recursive = TRUE, showWarnings = FALSE)

  datasets <- csv_env("KODAMA_COMPARE_DATASETS", "MetRef")
  implementations <- csv_env("KODAMA_COMPARE_IMPLEMENTATIONS", "cran3.3,cpu4")
  paths <- dataset_paths(data_root)
  M <- integer_env("KODAMA_COMPARE_M", 100L)
  Tcycle <- integer_env("KODAMA_COMPARE_TCYCLE", 100L)
  landmarks <- integer_env("KODAMA_COMPARE_LANDMARKS", 100000L)
  splitting <- integer_env("KODAMA_COMPARE_SPLITTING", 100L)
  ncomp_requested <- integer_env("KODAMA_COMPARE_NCOMP", 50L)
  knn_k <- integer_env("KODAMA_COMPARE_KNN_K", 30L)
  seed <- integer_env("KODAMA_COMPARE_SEED", 1234L)
  cran_cores <- integer_env("KODAMA_COMPARE_CRAN_CORES", 4L)
  cpu_cores <- integer_env("KODAMA_COMPARE_CPU_CORES", 4L)

  metadata <- c(
    sprintf("timestamp_utc=%s", format(Sys.time(), tz = "UTC", usetz = TRUE)),
    sprintf("KODAMA_version=%s", as.character(packageVersion("KODAMA"))),
    sprintf("kodamaR_version=%s", as.character(packageVersion("kodamaR"))),
    sprintf("M=%d", M), sprintf("Tcycle=%d", Tcycle),
    sprintf("landmarks_requested=%d", landmarks), sprintf("splitting=%d", splitting),
    sprintf("ncomp_requested=%d", ncomp_requested), sprintf("seed=%d", seed),
    capture.output(sessionInfo())
  )
  writeLines(metadata, file.path(out_dir, "session-info.txt"))

  for (dataset in datasets) {
    if (!dataset %in% names(paths)) stop("Unknown dataset: ", dataset)
    if (!file.exists(paths[[dataset]])) stop("Missing dataset: ", paths[[dataset]])
    ds <- load_dataset(paths[[dataset]])
    x <- ds$data
    truth <- ds$labels
    n <- nrow(x)
    p <- ncol(x)
    ncomp <- min(ncomp_requested, p, n - 1L)
    graph_neighbors <- max(1L, floor(min(landmarks, n * 0.75 - 1, 500)))
    if ("cuda" %in% implementations && graph_neighbors > 256L) {
      stop(
        "The matched current-CRAN graph requests k=", graph_neighbors,
        ", but the native CUDA graph builder supports k <= 256. ",
        "Do not change landmarks or graph k silently; omit cuda or run a separately labeled design."
      )
    }

    for (implementation in implementations) {
      path <- record_path(out_dir, dataset, implementation)
      if (file.exists(path)) {
        message("Skipping completed ", dataset, " / ", implementation)
        next
      }
      message("Running ", dataset, " / ", implementation, " at ", Sys.time())
      started <- proc.time()[["elapsed"]]
      result <- switch(
        implementation,
        "cran3.3" = KODAMA::KODAMA.matrix(
          x,
          M = M,
          Tcycle = Tcycle,
          ncomp = ncomp,
          landmarks = landmarks,
          splitting = splitting,
          n.cores = cran_cores,
          seed = seed
        ),
        "cpu4" = kodamaR::kodama_matrix(
          x,
          M = M,
          Tcycle = Tcycle,
          ncomp = ncomp,
          landmarks = landmarks,
          splitting = splitting,
          n.cores = cpu_cores,
          graph.neighbors = graph_neighbors,
          knn.k = knn_k,
          classifier = "pls_lda",
          backend = "cpu",
          seed = seed,
          visual.init = FALSE,
          progress = TRUE,
          apply.kodama.dissimilarity = TRUE
        ),
        "cuda" = kodamaR::kodama_matrix(
          x,
          M = M,
          Tcycle = Tcycle,
          ncomp = ncomp,
          landmarks = landmarks,
          splitting = splitting,
          n.cores = 0L,
          graph.neighbors = graph_neighbors,
          knn.k = knn_k,
          classifier = "pls_lda",
          backend = "cuda",
          seed = seed,
          visual.init = FALSE,
          progress = TRUE,
          apply.kodama.dissimilarity = TRUE
        ),
        stop("Unknown implementation: ", implementation)
      )
      elapsed <- proc.time()[["elapsed"]] - started
      metrics <- summarize_result(result, truth)
      scope <- if (implementation == "cran3.3") {
        "current CRAN KODAMA 3.3 automatic PLS route"
      } else {
        "kodama-cpp SIMPLS plus latent-space LDA"
      }
      cores <- if (implementation == "cran3.3") cran_cores else if (implementation == "cpu4") cpu_cores else 0L
      summary <- data.frame(
        dataset = dataset,
        implementation = implementation,
        scope = scope,
        n = n,
        p = p,
        truth_classes = nlevels(truth),
        M = M,
        Tcycle = Tcycle,
        landmarks_requested = landmarks,
        graph_neighbors = graph_neighbors,
        splitting = splitting,
        ncomp = ncomp,
        seed = seed,
        cores_requested = cores,
        wall_seconds = elapsed,
        selected_run = metrics$selected_run,
        selected_accuracy = metrics$selected_accuracy,
        median_accuracy = metrics$median_accuracy,
        selected_ari = metrics$selected_ari,
        median_ari = metrics$median_ari,
        selected_classes = metrics$selected_classes,
        median_classes = metrics$median_classes,
        min_classes = metrics$min_classes,
        max_classes = metrics$max_classes,
        stringsAsFactors = FALSE
      )
      saveRDS(list(summary = summary, result = result), path)
      atomic_csv(collect_records(out_dir), file.path(out_dir, "current_cran_comparison.csv"))
      message(sprintf(
        "Completed %s / %s in %.3f s; selected A=%.4f, ARI=%.4f, classes=%d",
        dataset, implementation, elapsed, metrics$selected_accuracy,
        metrics$selected_ari, metrics$selected_classes
      ))
      rm(result)
      invisible(gc())
    }
  }

  atomic_csv(collect_records(out_dir), file.path(out_dir, "current_cran_comparison.csv"))
}

main()
