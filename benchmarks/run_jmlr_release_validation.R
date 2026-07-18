#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

options(stringsAsFactors = FALSE)

env_value <- function(name, default = "") {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}

csv_values <- function(name, default) {
  value <- env_value(name, default)
  trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
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
  env_value("KODAMA_RELEASE_OUT", "jmlr-release-validation"),
  mustWork = FALSE
)
data_root <- normalizePath(
  env_value("KODAMA_DATA_ROOT", "/mnt/sata_ssd/fastEmbedR/Data"),
  mustWork = FALSE
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "runs"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "plots"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "embeddings"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "compactness"), recursive = TRUE, showWarnings = FALSE)

current_lib <- env_value("KODAMA_CURRENT_R_LIB")
historical_lib <- env_value("KODAMA_HISTORICAL_R_LIB")
extra_libs <- c(current_lib, historical_lib)
extra_libs <- extra_libs[nzchar(extra_libs) & dir.exists(extra_libs)]
.libPaths(unique(c(extra_libs, .libPaths())))

suppressPackageStartupMessages(library(kodamaR))

M <- as.integer(env_value("KODAMA_BENCH_M", "100"))
Tcycle <- as.integer(env_value("KODAMA_BENCH_TCYCLE", "100"))
landmarks_requested <- as.integer(env_value("KODAMA_BENCH_LANDMARKS", "100000"))
knn_k <- as.integer(env_value("KODAMA_BENCH_KNN_K", "30"))
ncomp <- as.integer(env_value("KODAMA_BENCH_NCOMP", "50"))
graph_neighbors <- as.integer(env_value("KODAMA_BENCH_GRAPH_K", "100"))
seed <- as.integer(env_value("KODAMA_BENCH_SEED", "1234"))
profile <- env_value("KODAMA_VALIDATION_PROFILE", "full")

dataset_candidates <- function(name) {
  switch(
    name,
    MetRef = c(
      file.path(data_root, "MetRef", "MetRef_float32.RData"),
      file.path(data_root, "MetRef.RData")
    ),
    COIL20 = c(
      file.path(data_root, "COIL20", "COIL20_float32.RData"),
      file.path(data_root, "COIL20.RData")
    ),
    USPS = c(
      file.path(data_root, "USPS", "USPS_float32.RData"),
      file.path(data_root, "USPS.RData")
    ),
    MNIST = c(
      file.path(data_root, "MNIST", "MNIST_float32.RData"),
      file.path(data_root, "MNIST.RData")
    ),
    stop("Unknown release-validation dataset: ", name)
  )
}

load_dataset <- function(name) {
  candidates <- dataset_candidates(name)
  path <- candidates[file.exists(candidates)][1L]
  if (is.na(path)) stop("Missing ", name, "; checked: ", paste(candidates, collapse = ", "))
  e <- new.env(parent = emptyenv())
  load(path, envir = e)
  if (exists("dataset", envir = e, inherits = FALSE)) {
    object <- get("dataset", envir = e, inherits = FALSE)
    x <- object$data
    labels <- object$labels
  } else {
    objects <- mget(ls(e), envir = e)
    lists <- Filter(function(x) is.list(x) && !is.null(x$data) && !is.null(x$labels), objects)
    if (length(lists) != 1L) stop("Cannot identify dataset$data and dataset$labels in ", path)
    x <- lists[[1L]]$data
    labels <- lists[[1L]]$labels
  }
  x <- as_numeric_matrix(x)
  list(name = name, path = path, data = x, labels = as.factor(labels))
}

adjusted_rand <- function(a, b) {
  tab <- table(a, b)
  choose2 <- function(x) x * (x - 1) / 2
  nij <- sum(choose2(tab))
  ai <- sum(choose2(rowSums(tab)))
  bj <- sum(choose2(colSums(tab)))
  total <- choose2(sum(tab))
  expected <- if (total > 0) ai * bj / total else 0
  maximum <- (ai + bj) / 2
  denominator <- maximum - expected
  if (denominator == 0) 1 else (nij - expected) / denominator
}

normalize_indices <- function(indices, n) {
  indices <- as.matrix(indices)
  if (length(indices) && min(indices, na.rm = TRUE) == 0L) indices <- indices + 1L
  indices[indices < 1L | indices > n] <- NA_integer_
  indices
}

graph_truth_purity <- function(graph, truth) {
  if (is.null(graph) || is.null(graph$indices)) return(NA_real_)
  indices <- normalize_indices(graph$indices, length(truth))
  same <- matrix(FALSE, nrow(indices), ncol(indices))
  valid <- !is.na(indices)
  same[valid] <- truth[row(indices)[valid]] == truth[indices[valid]]
  mean(same[valid])
}

sampled_silhouette <- function(embedding, labels, max_samples = 5000L, sample_seed = 909L) {
  labels <- as.integer(as.factor(labels))
  keep <- which(!is.na(labels) & rowSums(is.finite(embedding)) == ncol(embedding))
  if (length(unique(labels[keep])) < 2L || length(keep) < 3L) return(NA_real_)
  if (length(keep) > max_samples) {
    set.seed(sample_seed)
    keep <- sort(sample(keep, max_samples))
  }
  mean(cluster::silhouette(labels[keep], stats::dist(embedding[keep, , drop = FALSE]))[, 3L])
}

cluster_compactness <- function(embedding, labels) {
  labels <- as.factor(labels)
  centered <- sweep(embedding, 2L, colMeans(embedding), check.margin = FALSE)
  overall <- sqrt(mean(rowSums(centered^2)))
  levels_out <- levels(labels)
  rows <- lapply(levels_out, function(level) {
    selected <- which(labels == level)
    block <- embedding[selected, , drop = FALSE]
    centroid <- colMeans(block)
    rms <- sqrt(mean(rowSums(sweep(block, 2L, centroid, check.margin = FALSE)^2)))
    data.frame(label = level, samples = length(selected), rms = rms,
               normalized_rms = if (overall > 0) rms / overall else NA_real_)
  })
  do.call(rbind, rows)
}

point_size <- function(n) {
  if (n >= 50000L) 0.25 else if (n >= 10000L) 0.45 else if (n >= 2000L) 0.7 else 1.2
}

plot_embedding <- function(path, classic, kodama, truth, selected, title, stats_line) {
  grDevices::png(path, width = 2400, height = 800, res = 140)
  on.exit(grDevices::dev.off(), add = TRUE)
  old <- graphics::par(mfrow = c(1, 3), mar = c(1.5, 1.5, 3.0, 0.5), oma = c(0, 0, 2.5, 0))
  on.exit(graphics::par(old), add = TRUE)
  truth_colors <- grDevices::hcl.colors(length(levels(truth)), "Dynamic")[as.integer(truth)]
  selected_factor <- as.factor(selected)
  selected_colors <- grDevices::hcl.colors(length(levels(selected_factor)), "Dark 3")[as.integer(selected_factor)]
  cex <- point_size(nrow(kodama))
  graphics::plot(classic, pch = 16, cex = cex, col = truth_colors, axes = FALSE,
                 xlab = "", ylab = "", main = "Classic UMAP by truth")
  graphics::plot(kodama, pch = 16, cex = cex, col = truth_colors, axes = FALSE,
                 xlab = "", ylab = "", main = "KODAMA UMAP by truth")
  graphics::plot(kodama, pch = 16, cex = cex, col = selected_colors, axes = FALSE,
                 xlab = "", ylab = "", main = "Best CV-selected labels")
  graphics::mtext(paste(title, stats_line, sep = " | "), outer = TRUE, side = 3, line = 0.6, font = 2)
}

final_trace_accuracy <- function(v) {
  apply(v, 1L, function(row) {
    finite <- row[is.finite(row)]
    if (length(finite)) finite[length(finite)] else NA_real_
  })
}

current_result <- function(ds, job) {
  kodamaR::kodama_matrix(
    ds$data,
    M = M,
    Tcycle = Tcycle,
    ncomp = ncomp,
    landmarks = landmarks_requested,
    splitting = job$splitting,
    n.cores = job$n_cores,
    graph.neighbors = graph_neighbors,
    knn.k = knn_k,
    classifier = job$classifier,
    backend = job$backend,
    seed = seed,
    visual.init = TRUE,
    progress = TRUE,
    apply.kodama.dissimilarity = job$graph_correction
  )
}

historical_result <- function(ds, job) {
  if (!requireNamespace("KODAMA", quietly = TRUE)) {
    stop("Historical KODAMA 2.4.1 is not installed in KODAMA_HISTORICAL_R_LIB")
  }
  effective_landmarks <- if (nrow(ds$data) <= landmarks_requested) {
    ceiling(nrow(ds$data) * 0.75)
  } else {
    landmarks_requested
  }
  set.seed(seed)
  KODAMA::KODAMA.matrix(
    ds$data,
    M = M,
    Tcycle = Tcycle,
    FUN = if (job$classifier == "knn") "KNN" else "PLS-DA",
    f.par = if (job$classifier == "knn") knn_k else min(ncomp, ncol(ds$data)),
    landmarks = effective_landmarks,
    neighbors = min(graph_neighbors, effective_landmarks - 1L)
  )
}

extract_result <- function(result, implementation, truth) {
  if (implementation == "historical-r-2.4.1") {
    acc <- final_trace_accuracy(result$v)
    labels_by_run <- result$res
    best_run <- which.max(acc)
    graph <- list(
      indices = result$knn_Armadillo$nn_index,
      distances = result$knn_Armadillo$distances
    )
    runtime_reported <- NA_real_
    visual_init <- NULL
  } else {
    acc <- as.numeric(result$acc)
    labels_by_run <- as.matrix(result$res)
    best_run <- result$best_run
    graph <- result$knn
    runtime_reported <- result$runtime_seconds
    visual_init <- result$visual_init
  }
  if (is.na(best_run) || best_run < 1L) best_run <- which.max(acc)
  best_labels <- as.integer(labels_by_run[best_run, ])
  ari_by_run <- apply(labels_by_run, 1L, adjusted_rand, b = truth)
  classes_by_run <- apply(labels_by_run, 1L, function(x) length(unique(x)))
  list(
    acc = acc,
    labels_by_run = labels_by_run,
    best_run = best_run,
    best_labels = best_labels,
    ari_by_run = ari_by_run,
    classes_by_run = classes_by_run,
    graph = graph,
    runtime_reported = runtime_reported,
    visual_init = visual_init
  )
}

classic_cache_path <- function(dataset) file.path(out_dir, "embeddings", paste0(dataset, "_classic_umap.rds"))

build_embeddings <- function(ds, extracted) {
  init <- if (is.list(extracted$visual_init)) extracted$visual_init$umap else NULL
  cache <- classic_cache_path(ds$name)
  if (file.exists(cache)) {
    classic_record <- readRDS(cache)
  } else {
    graph_start <- proc.time()[["elapsed"]]
    classic_graph <- kodamaR::KODAMA.graph(
      ds$data, k = 30L, metric = "euclidean", backend = "cuda", n.cores = 4L
    )
    classic <- kodamaR::KODAMA.visualization(
      classic_graph, method = "UMAP", init = init, k = 30L, backend = "cuda",
      n.cores = 4L, n.epochs = 200L, seed = 4L
    )
    classic_record <- list(
      embedding = classic,
      seconds = proc.time()[["elapsed"]] - graph_start,
      graph = classic_graph
    )
    saveRDS(classic_record, cache)
  }
  kodama_start <- proc.time()[["elapsed"]]
  kodama <- kodamaR::KODAMA.visualization(
    extracted$graph, method = "UMAP", init = init, k = 30L, backend = "cuda",
    n.cores = 4L, n.epochs = 200L, seed = 4L
  )
  list(
    classic = classic_record$embedding,
    classic_seconds = classic_record$seconds,
    kodama = kodama,
    kodama_seconds = proc.time()[["elapsed"]] - kodama_start
  )
}

default_splitting <- function(n) if (n < 40000L) 100L else 300L

make_plan <- function() {
  rows <- list()
  add <- function(dataset, implementation, classifier, backend, n_cores,
                  variant, splitting, graph_correction = TRUE) {
    rows[[length(rows) + 1L]] <<- data.frame(
      dataset = dataset,
      implementation = implementation,
      classifier = classifier,
      backend = backend,
      n_cores = as.integer(n_cores),
      variant = variant,
      splitting = as.integer(splitting),
      graph_correction = isTRUE(graph_correction)
    )
  }

  current_datasets <- csv_values("KODAMA_ABLATION_DATASETS", "MetRef,USPS,MNIST")
  for (dataset in current_datasets) {
    n_hint <- c(MetRef = 873L, USPS = 11000L, MNIST = 70000L)[[dataset]]
    split_default <- default_splitting(n_hint)
    for (classifier in c("knn", "pls_lda")) {
      add(dataset, "kodama-cpp", classifier, "cuda", 4L,
          "reference", split_default, TRUE)
      add(dataset, "kodama-cpp", classifier, "cuda", 4L,
          "no_graph_correction", split_default, FALSE)
      for (split_value in c(100L, 300L)) {
        if (split_value != split_default) {
          add(dataset, "kodama-cpp", classifier, "cuda", 4L,
              paste0("splitting_", split_value), split_value, TRUE)
        }
      }
    }
  }

  for (dataset in csv_values("KODAMA_BACKEND_DATASETS", "MetRef,USPS")) {
    n_hint <- c(MetRef = 873L, USPS = 11000L)[[dataset]]
    for (classifier in c("knn", "pls_lda")) {
      add(dataset, "kodama-cpp", classifier, "cpu", 1L,
          "reference_cpu1", default_splitting(n_hint), TRUE)
      add(dataset, "kodama-cpp", classifier, "cpu", 4L,
          "reference_cpu4", default_splitting(n_hint), TRUE)
    }
  }

  historical_datasets <- csv_values("KODAMA_HISTORICAL_DATASETS", "MetRef,COIL20")
  for (dataset in historical_datasets) {
    classifiers <- if (dataset == "COIL20") "knn" else c("knn", "pls_da")
    n_hint <- c(MetRef = 873L, COIL20 = 1440L)[[dataset]]
    for (classifier in classifiers) {
      add(dataset, "historical-r-2.4.1", classifier, "cpu", 1L,
          "historical_reference", 50L, TRUE)
      current_classifier <- if (classifier == "pls_da") "pls_lda" else classifier
      add(dataset, "kodama-cpp", current_classifier, "cpu", 1L,
          "historical_comparison_cpu1", 50L, TRUE)
      add(dataset, "kodama-cpp", current_classifier, "cpu", 4L,
          "historical_comparison_cpu4", 50L, TRUE)
      add(dataset, "kodama-cpp", current_classifier, "cuda", 4L,
          "historical_comparison_cuda", 50L, TRUE)
    }
  }

  plan <- unique(do.call(rbind, rows))
  plan$job_id <- sprintf(
    "%s__%s__%s__%s__c%d__%s",
    plan$dataset, plan$implementation, plan$classifier, plan$backend,
    plan$n_cores, plan$variant
  )
  if (profile == "pilot") plan <- plan[seq_len(min(4L, nrow(plan))), , drop = FALSE]
  plan
}

summary_path <- file.path(out_dir, "release_validation_summary.csv")
plan_path <- file.path(out_dir, "release_validation_plan.csv")
plan <- make_plan()
job_regex <- env_value("KODAMA_JOB_REGEX")
if (nzchar(job_regex)) {
  plan <- plan[grepl(job_regex, plan$job_id), , drop = FALSE]
  if (!nrow(plan)) stop("KODAMA_JOB_REGEX matched no release-validation jobs: ", job_regex)
}
utils::write.csv(plan, plan_path, row.names = FALSE)
completed <- if (file.exists(summary_path)) utils::read.csv(summary_path) else data.frame()

append_summary <- function(row) {
  old <- if (file.exists(summary_path)) utils::read.csv(summary_path) else data.frame()
  combined <- if (nrow(old)) rbind(old, row) else row
  utils::write.csv(combined, summary_path, row.names = FALSE)
}

for (i in seq_len(nrow(plan))) {
  job <- plan[i, , drop = FALSE]
  if (nrow(completed) && job$job_id %in% completed$job_id) {
    message("[skip] ", job$job_id)
    next
  }
  message(sprintf("[%d/%d] %s", i, nrow(plan), job$job_id))
  row <- tryCatch({
    ds <- load_dataset(job$dataset)
    start <- proc.time()[["elapsed"]]
    result <- if (job$implementation == "historical-r-2.4.1") {
      historical_result(ds, job)
    } else {
      current_result(ds, job)
    }
    wall_seconds <- proc.time()[["elapsed"]] - start
    extracted <- extract_result(result, job$implementation, ds$labels)
    embedding_start <- proc.time()[["elapsed"]]
    embeddings <- build_embeddings(ds, extracted)
    embedding_total <- proc.time()[["elapsed"]] - embedding_start
    classic_sil <- sampled_silhouette(embeddings$classic, ds$labels)
    kodama_sil <- sampled_silhouette(embeddings$kodama, ds$labels)
    label_sil <- sampled_silhouette(embeddings$kodama, extracted$best_labels)
    compactness <- cluster_compactness(embeddings$kodama, ds$labels)
    utils::write.csv(
      compactness,
      file.path(out_dir, "compactness", paste0(job$job_id, ".csv")),
      row.names = FALSE
    )
    saveRDS(
      list(classic = embeddings$classic, kodama = embeddings$kodama,
           truth = ds$labels, best_labels = extracted$best_labels),
      file.path(out_dir, "embeddings", paste0(job$job_id, ".rds")),
      compress = FALSE
    )
    stats_line <- sprintf(
      "ARI=%.3f; truth sil=%.3f; classes=%d; %.1fs",
      adjusted_rand(extracted$best_labels, ds$labels), kodama_sil,
      length(unique(extracted$best_labels)), wall_seconds
    )
    plot_embedding(
      file.path(out_dir, "plots", paste0(job$job_id, ".png")),
      embeddings$classic, embeddings$kodama, ds$labels,
      extracted$best_labels, job$job_id, stats_line
    )
    saveRDS(result, file.path(out_dir, "runs", paste0(job$job_id, ".rds")), compress = FALSE)
    data.frame(
      job_id = job$job_id,
      dataset = job$dataset,
      implementation = job$implementation,
      classifier = job$classifier,
      backend = job$backend,
      n_cores = job$n_cores,
      variant = job$variant,
      samples = nrow(ds$data),
      variables = ncol(ds$data),
      truth_classes = nlevels(ds$labels),
      M = M,
      Tcycle = Tcycle,
      landmarks_requested = landmarks_requested,
      effective_landmarks = if (nrow(ds$data) <= landmarks_requested) ceiling(nrow(ds$data) * 0.75) else landmarks_requested,
      splitting = job$splitting,
      knn_k = knn_k,
      ncomp = ncomp,
      graph_neighbors = graph_neighbors,
      graph_correction = job$graph_correction,
      seed = seed,
      wall_seconds = wall_seconds,
      reported_core_seconds = extracted$runtime_reported,
      embedding_seconds = embedding_total,
      classic_umap_seconds = embeddings$classic_seconds,
      kodama_umap_seconds = embeddings$kodama_seconds,
      best_cv_accuracy = max(extracted$acc, na.rm = TRUE),
      median_cv_accuracy = stats::median(extracted$acc, na.rm = TRUE),
      best_run = extracted$best_run,
      selected_ari = adjusted_rand(extracted$best_labels, ds$labels),
      median_run_ari = stats::median(extracted$ari_by_run, na.rm = TRUE),
      selected_classes = length(unique(extracted$best_labels)),
      median_classes = stats::median(extracted$classes_by_run, na.rm = TRUE),
      graph_truth_purity = graph_truth_purity(extracted$graph, ds$labels),
      classic_truth_silhouette = classic_sil,
      kodama_truth_silhouette = kodama_sil,
      kodama_label_silhouette = label_sil,
      median_truth_cluster_compactness = stats::median(compactness$normalized_rms, na.rm = TRUE),
      status = "complete",
      error = ""
    )
  }, error = function(e) {
    data.frame(
      job_id = job$job_id,
      dataset = job$dataset,
      implementation = job$implementation,
      classifier = job$classifier,
      backend = job$backend,
      n_cores = job$n_cores,
      variant = job$variant,
      samples = NA_integer_, variables = NA_integer_, truth_classes = NA_integer_,
      M = M, Tcycle = Tcycle, landmarks_requested = landmarks_requested,
      effective_landmarks = NA_integer_, splitting = job$splitting,
      knn_k = knn_k, ncomp = ncomp, graph_neighbors = graph_neighbors,
      graph_correction = job$graph_correction, seed = seed,
      wall_seconds = NA_real_, reported_core_seconds = NA_real_,
      embedding_seconds = NA_real_, classic_umap_seconds = NA_real_,
      kodama_umap_seconds = NA_real_, best_cv_accuracy = NA_real_,
      median_cv_accuracy = NA_real_, best_run = NA_integer_,
      selected_ari = NA_real_, median_run_ari = NA_real_,
      selected_classes = NA_integer_, median_classes = NA_real_,
      graph_truth_purity = NA_real_, classic_truth_silhouette = NA_real_,
      kodama_truth_silhouette = NA_real_, kodama_label_silhouette = NA_real_,
      median_truth_cluster_compactness = NA_real_, status = "error",
      error = conditionMessage(e)
    )
  })
  append_summary(row)
  completed <- if (nrow(completed)) rbind(completed, row) else row
  message("[", row$status, "] ", job$job_id,
          if (row$status == "error") paste0(": ", row$error) else "")
}

message("Release validation summary: ", summary_path)
