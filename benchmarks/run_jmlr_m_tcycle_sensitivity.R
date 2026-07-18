#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

options(stringsAsFactors = FALSE)

value_or <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}

csv_int <- function(name, default) {
  as.integer(trimws(strsplit(value_or(name, default), ",", fixed = TRUE)[[1L]]))
}

csv_chr <- function(name, default) {
  trimws(strsplit(value_or(name, default), ",", fixed = TRUE)[[1L]])
}

as_numeric_matrix <- function(x) {
  if (inherits(x, "float32")) {
    if (!requireNamespace("float", quietly = TRUE)) {
      stop("Loading a float32 RData matrix requires the float package")
    }
    x <- float::dbl(x)
  } else {
    x <- as.matrix(x)
  }
  storage.mode(x) <- "double"
  x
}

out_dir <- normalizePath(
  value_or("KODAMA_SENS_OUT", "/mnt/sata_ssd/kodama-cpp-benchmarks/jmlr-m-tcycle-sensitivity"),
  mustWork = FALSE
)
data_root <- normalizePath(
  value_or("KODAMA_DATA_ROOT", "/mnt/sata_ssd/fastEmbedR/Data"),
  mustWork = FALSE
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "runs"), recursive = TRUE, showWarnings = FALSE)

current_lib <- value_or("KODAMA_CURRENT_R_LIB", "")
if (nzchar(current_lib) && dir.exists(current_lib)) {
  .libPaths(unique(c(current_lib, .libPaths())))
}
suppressPackageStartupMessages(library(kodamaR))

datasets <- csv_chr("KODAMA_SENS_DATASETS", "MetRef,USPS")
classifiers <- csv_chr("KODAMA_SENS_CLASSIFIERS", "knn,pls_lda")
M_values <- csv_int("KODAMA_SENS_M_VALUES", "20,50,100")
T_values <- csv_int("KODAMA_SENS_T_VALUES", "20,50,100")
M_anchor <- as.integer(value_or("KODAMA_SENS_M_ANCHOR", "100"))
T_anchor <- as.integer(value_or("KODAMA_SENS_T_ANCHOR", "100"))
landmarks <- as.integer(value_or("KODAMA_BENCH_LANDMARKS", "100000"))
knn_k <- as.integer(value_or("KODAMA_BENCH_KNN_K", "30"))
ncomp_requested <- as.integer(value_or("KODAMA_BENCH_NCOMP", "50"))
graph_k <- as.integer(value_or("KODAMA_BENCH_GRAPH_K", "100"))
seed <- as.integer(value_or("KODAMA_BENCH_SEED", "1234"))
backend <- value_or("KODAMA_SENS_BACKEND", "cuda")
n_cores <- as.integer(value_or("KODAMA_SENS_CORES", "4"))

dataset_paths <- list(
  MetRef = c(
    file.path(data_root, "MetRef", "MetRef_float32.RData"),
    file.path(data_root, "MetRef.RData")
  ),
  USPS = c(
    file.path(data_root, "USPS", "USPS_float32.RData"),
    file.path(data_root, "USPS.RData")
  ),
  MNIST = c(
    file.path(data_root, "MNIST", "MNIST_float32.RData"),
    file.path(data_root, "MNIST.RData")
  )
)

load_dataset <- function(name) {
  candidates <- dataset_paths[[name]]
  if (is.null(candidates)) stop("Unknown dataset: ", name)
  path <- candidates[file.exists(candidates)][1L]
  if (is.na(path)) stop("Missing ", name, "; checked ", paste(candidates, collapse = ", "))
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  object <- if (exists("dataset", env, inherits = FALSE)) {
    env$dataset
  } else {
    objects <- mget(ls(env), envir = env)
    candidates <- Filter(function(x) is.list(x) && !is.null(x$data) && !is.null(x$labels), objects)
    if (length(candidates) != 1L) stop("Cannot identify dataset list in ", path)
    candidates[[1L]]
  }
  x <- as_numeric_matrix(object$data)
  list(data = x, labels = as.factor(object$labels), path = path)
}

choose2 <- function(x) x * (x - 1) / 2

adjusted_rand <- function(a, b) {
  tab <- table(a, b)
  nij <- sum(choose2(tab))
  ai <- sum(choose2(rowSums(tab)))
  bj <- sum(choose2(colSums(tab)))
  total <- choose2(sum(tab))
  expected <- if (total > 0) ai * bj / total else 0
  maximum <- (ai + bj) / 2
  if (maximum == expected) 1 else (nij - expected) / (maximum - expected)
}

run_metrics <- function(result, truth) {
  labels <- as.matrix(result$res)
  accuracy <- as.numeric(result$acc)
  aris <- apply(labels, 1L, adjusted_rand, b = truth)
  classes <- apply(labels, 1L, function(x) length(unique(x)))
  selected <- result$best_run
  if (is.null(selected) || is.na(selected) || selected < 1L) selected <- which.max(accuracy)
  data.frame(
    best_cv_accuracy = max(accuracy, na.rm = TRUE),
    median_cv_accuracy = median(accuracy, na.rm = TRUE),
    selected_ari = aris[[selected]],
    median_run_ari = median(aris, na.rm = TRUE),
    selected_classes = classes[[selected]],
    median_classes = median(classes, na.rm = TRUE),
    min_classes = min(classes, na.rm = TRUE),
    max_classes = max(classes, na.rm = TRUE),
    best_run = selected
  )
}

splitting_for <- function(n) if (n < 40000L) 100L else 300L

grid <- rbind(
  data.frame(sweep = "Tcycle", M = M_anchor, Tcycle = T_values),
  data.frame(sweep = "M", M = M_values, Tcycle = T_anchor)
)

summary_path <- file.path(out_dir, "m_tcycle_sensitivity_summary.csv")
summary_rows <- list()

for (dataset in datasets) {
  ds <- load_dataset(dataset)
  ncomp <- max(1L, min(ncomp_requested, ncol(ds$data), nrow(ds$data) - 1L))
  splitting <- splitting_for(nrow(ds$data))
  for (classifier in classifiers) {
    for (i in seq_len(nrow(grid))) {
      M <- grid$M[[i]]
      Tcycle <- grid$Tcycle[[i]]
      key <- sprintf("%s__%s__M%d__T%d", dataset, classifier, M, Tcycle)
      rds_path <- file.path(out_dir, "runs", paste0(key, ".rds"))
      timing_path <- file.path(out_dir, "runs", paste0(key, "_timing.rds"))
      message(sprintf("[%s/%s] %s", i, nrow(grid), key))
      if (file.exists(rds_path) && file.exists(timing_path)) {
        result <- readRDS(rds_path)
        timing <- readRDS(timing_path)
      } else {
        start <- proc.time()[["elapsed"]]
        result <- kodamaR::kodama_matrix(
          ds$data,
          M = M,
          Tcycle = Tcycle,
          ncomp = ncomp,
          landmarks = landmarks,
          splitting = splitting,
          n.cores = n_cores,
          graph.neighbors = graph_k,
          knn.k = knn_k,
          classifier = classifier,
          backend = backend,
          seed = seed,
          visual.init = TRUE,
          progress = TRUE,
          apply.kodama.dissimilarity = TRUE
        )
        timing <- list(wall_seconds = proc.time()[["elapsed"]] - start)
        saveRDS(result, rds_path, compress = FALSE)
        saveRDS(timing, timing_path)
      }
      metrics <- run_metrics(result, ds$labels)
      summary_rows[[length(summary_rows) + 1L]] <- cbind(
        data.frame(
          dataset = dataset,
          classifier = classifier,
          sweep = grid$sweep[[i]],
          samples = nrow(ds$data),
          variables = ncol(ds$data),
          M = M,
          Tcycle = Tcycle,
          landmarks_requested = landmarks,
          effective_landmarks = if (nrow(ds$data) <= landmarks) ceiling(0.75 * nrow(ds$data)) else landmarks,
          splitting = splitting,
          knn_k = knn_k,
          ncomp = ncomp,
          graph_neighbors = graph_k,
          backend = backend,
          n_cores = n_cores,
          seed = seed,
          wall_seconds = timing$wall_seconds,
          reported_core_seconds = as.numeric(result$runtime_seconds)
        ),
        metrics
      )
      utils::write.csv(do.call(rbind, summary_rows), summary_path, row.names = FALSE)
    }
  }
}

summary <- do.call(rbind, summary_rows)
utils::write.csv(summary, summary_path, row.names = FALSE)

plot_metric <- function(sweep, x_name, y_name, y_label, title) {
  block <- summary[summary$sweep == sweep, , drop = FALSE]
  groups <- unique(interaction(block$dataset, block$classifier, drop = TRUE))
  colors <- setNames(grDevices::hcl.colors(length(groups), "Dark 3"), groups)
  graphics::plot(range(block[[x_name]]), range(block[[y_name]], finite = TRUE),
                 type = "n", xlab = x_name, ylab = y_label, main = title)
  for (group in groups) {
    selected <- interaction(block$dataset, block$classifier, drop = TRUE) == group
    values <- block[selected, , drop = FALSE]
    values <- values[order(values[[x_name]]), , drop = FALSE]
    graphics::lines(values[[x_name]], values[[y_name]], type = "b", pch = 16,
                    lwd = 2, col = colors[[as.character(group)]])
  }
  graphics::legend("topright", legend = levels(groups), col = colors, lwd = 2,
                   pch = 16, cex = 0.72, bg = "white")
}

grDevices::png(file.path(out_dir, "m_tcycle_sensitivity.png"), width = 2200, height = 1400, res = 170)
old <- graphics::par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot_metric("Tcycle", "Tcycle", "best_cv_accuracy", "best CV accuracy", "Tcycle quality, M=100")
plot_metric("Tcycle", "Tcycle", "median_classes", "median active classes", "Tcycle fragmentation")
plot_metric("M", "M", "selected_ari", "selected-run ARI", "M external diagnostic")
plot_metric("M", "M", "wall_seconds", "wall seconds", "M cost")
graphics::par(old)
grDevices::dev.off()

message("Sensitivity summary: ", summary_path)
