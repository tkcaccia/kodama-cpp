#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

options(stringsAsFactors = FALSE)
kodama_r_lib <- Sys.getenv("KODAMA_R_LIB", unset = "")
if (nzchar(kodama_r_lib)) .libPaths(c(kodama_r_lib, .libPaths()))
suppressPackageStartupMessages({
  library(kodamaR)
  library(cluster)
})

required_visualization_formals <- c("raw.data", "initialize.from.raw", "graph.mode")
installed_visualization_formals <- names(formals(kodamaR::KODAMA.visualization))
missing_visualization_formals <- setdiff(
  required_visualization_formals,
  installed_visualization_formals
)
if (length(missing_visualization_formals)) {
  stop(
    "The installed kodamaR wrapper is stale; KODAMA.visualization() is missing: ",
    paste(missing_visualization_formals, collapse = ", "),
    ". Reinstall split-repos/kodama-r against the current core before benchmarking."
  )
}
installed_backends <- eval(formals(kodamaR::KODAMA.visualization)$backend)
if (!"metal" %in% installed_backends) {
  stop(
    "The installed kodamaR wrapper does not expose backend='metal'. ",
    "Reinstall split-repos/kodama-r against the current core before benchmarking."
  )
}

env_value <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}

csv_values <- function(name, default) {
  trimws(strsplit(env_value(name, default), ",", fixed = TRUE)[[1L]])
}

dataset_path <- function(root, name, local_root) {
  candidates <- c(
    file.path(root, name, paste0(name, "_float32.RData")),
    file.path(root, name, paste0(name, ".RData")),
    file.path(local_root, paste0(name, ".RData")),
    file.path(local_root, paste0(name, "_float32.RData"))
  )
  existing <- candidates[file.exists(candidates)]
  if (!length(existing)) {
    stop("Missing dataset ", name, "; checked: ", paste(candidates, collapse = ", "))
  }
  normalizePath(existing[[1L]])
}

as_numeric_matrix <- function(x) {
  if (inherits(x, "float32")) {
    if (!requireNamespace("float", quietly = TRUE)) stop("The float package is required")
    x <- float::dbl(x)
  }
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

load_dataset <- function(path) {
  e <- new.env(parent = emptyenv())
  load(path, envir = e)
  object <- if (exists("dataset", e, inherits = FALSE)) {
    e$dataset
  } else {
    candidates <- Filter(
      function(x) is.list(x) && !is.null(x$data) && !is.null(x$labels),
      mget(ls(e), e)
    )
    if (length(candidates) != 1L) stop("Cannot identify dataset list in ", path)
    candidates[[1L]]
  }
  list(data = as_numeric_matrix(object$data), labels = as.factor(object$labels))
}

choose2 <- function(x) x * (x - 1) / 2

adjusted_rand <- function(a, b) {
  tab <- table(as.factor(a), as.factor(b))
  n <- sum(tab)
  if (n < 2L) return(NA_real_)
  nij <- sum(choose2(as.numeric(tab)))
  ai <- sum(choose2(rowSums(tab)))
  bj <- sum(choose2(colSums(tab)))
  expected <- ai * bj / choose2(n)
  maximum <- 0.5 * (ai + bj)
  if (abs(maximum - expected) < .Machine$double.eps) return(NA_real_)
  (nij - expected) / (maximum - expected)
}

median_pairwise_ari <- function(labels_by_run, seed = 7331L, max_pairs = 200L) {
  runs <- nrow(labels_by_run)
  if (runs < 2L) return(NA_real_)
  pairs <- utils::combn(runs, 2L)
  if (ncol(pairs) > max_pairs) {
    set.seed(seed)
    pairs <- pairs[, sort(sample.int(ncol(pairs), max_pairs)), drop = FALSE]
  }
  values <- vapply(seq_len(ncol(pairs)), function(i) {
    adjusted_rand(labels_by_run[pairs[1L, i], ], labels_by_run[pairs[2L, i], ])
  }, numeric(1L))
  stats::median(values, na.rm = TRUE)
}

sampled_silhouette <- function(x, labels, max_n = 5000L, seed = 991L) {
  labels <- as.factor(labels)
  keep <- seq_len(nrow(x))
  if (length(keep) > max_n) {
    set.seed(seed)
    keep <- sort(sample(keep, max_n))
  }
  labels <- droplevels(labels[keep])
  valid_levels <- names(which(table(labels) >= 2L))
  valid <- labels %in% valid_levels
  labels <- droplevels(labels[valid])
  x <- x[keep[valid], , drop = FALSE]
  if (nrow(x) < 3L || nlevels(labels) < 2L) return(NA_real_)
  mean(cluster::silhouette(as.integer(labels), stats::dist(x))[, "sil_width"])
}

git_value <- function(args, default = "NA") {
  value <- tryCatch(system2("git", args, stdout = TRUE, stderr = FALSE), error = function(e) character())
  if (length(value)) paste(value, collapse = " ") else default
}

bind_rows_fill <- function(left, right) {
  columns <- union(names(left), names(right))
  for (name in setdiff(columns, names(left))) left[[name]] <- NA
  for (name in setdiff(columns, names(right))) right[[name]] <- NA
  rbind(left[, columns, drop = FALSE], right[, columns, drop = FALSE])
}

append_row <- function(row, path) {
  old <- if (file.exists(path)) utils::read.csv(path, check.names = FALSE) else data.frame()
  out <- if (nrow(old)) bind_rows_fill(old, row) else row
  utils::write.csv(out, path, row.names = FALSE)
}

timing_value <- function(x, name) {
  value <- x$timing[[name]]
  if (is.null(value) || !length(value)) NA_real_ else as.numeric(value[[1L]])
}

plot_comparison <- function(path, classic, kodama, truth, inferred, title, subtitle) {
  grDevices::png(path, width = 2400, height = 820, res = 180)
  old <- graphics::par(no.readonly = TRUE)
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  })
  graphics::par(mfrow = c(1, 3), oma = c(0, 0, 3, 0), mar = c(1, 1, 3, 1))
  palette_truth <- grDevices::hcl.colors(nlevels(truth), "Dynamic")
  palette_inferred <- grDevices::hcl.colors(length(unique(inferred)), "Dynamic")
  draw <- function(x, colors, heading) {
    graphics::plot(
      x[, 1L], x[, 2L], pch = 16, cex = if (nrow(x) < 2000L) 0.45 else 0.18,
      col = colors, axes = FALSE, xlab = "", ylab = "", main = heading
    )
    graphics::box(col = "grey80")
  }
  draw(classic, palette_truth[as.integer(truth)], "Classic UMAP by truth")
  draw(kodama, palette_truth[as.integer(truth)], "KODAMA UMAP by truth")
  draw(kodama, palette_inferred[as.integer(as.factor(inferred))], "Best KODAMA labels")
  graphics::mtext(paste(title, subtitle, sep = " | "), outer = TRUE, font = 2, line = 0.8)
}

root <- normalizePath(env_value("KODAMA_DATA_ROOT", "/Users/stefano/Documents/fastEmbedR/Data"))
local_data_root <- normalizePath(
  env_value("KODAMA_LOCAL_DATA_ROOT", "benchmark_datasets/RData"),
  mustWork = FALSE
)
out_dir <- env_value(
  "KODAMA_LOCAL_OUT",
  file.path("manuscript", "jmlr_local_cpu_metal_20260807")
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "runs"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "plots"), recursive = TRUE, showWarnings = FALSE)

datasets <- csv_values("KODAMA_LOCAL_DATASETS", "MetRef,COIL20,USPS")
backends <- csv_values("KODAMA_LOCAL_BACKENDS", "cpu,metal")
classifiers <- csv_values("KODAMA_LOCAL_CLASSIFIERS", "knn,pls_lda")
seeds <- as.integer(csv_values("KODAMA_LOCAL_SEEDS", "4,17,42"))
M <- as.integer(env_value("KODAMA_LOCAL_M", "20"))
Tcycle <- as.integer(env_value("KODAMA_LOCAL_TCYCLE", "20"))
landmarks <- as.integer(env_value("KODAMA_LOCAL_LANDMARKS", "100000"))
graph_k <- as.integer(env_value("KODAMA_LOCAL_GRAPH_K", "100"))
knn_k <- as.integer(env_value("KODAMA_LOCAL_KNN_K", "30"))
ncomp <- as.integer(env_value("KODAMA_LOCAL_NCOMP", "50"))
n_cores <- as.integer(env_value("KODAMA_LOCAL_CORES", "4"))
accelerator_cores <- as.integer(env_value("KODAMA_LOCAL_ACCELERATOR_CORES", "0"))
host_threads <- max(1L, n_cores)
splitting_override <- as.integer(env_value("KODAMA_LOCAL_SPLITTING", "0"))
graph_mode <- env_value("KODAMA_LOCAL_GRAPH_MODE", "native")
if (!graph_mode %in% c("native", "shared_cpu")) {
  stop("KODAMA_LOCAL_GRAPH_MODE must be 'native' or 'shared_cpu'")
}

summary_path <- file.path(out_dir, "local_cpu_metal_summary.csv")
core_commit <- git_value(c("rev-parse", "HEAD"))
dirty_lines <- tryCatch(
  system2("git", c("status", "--porcelain"), stdout = TRUE, stderr = FALSE),
  error = function(e) character()
)
core_dirty <- if (length(dirty_lines)) "TRUE" else "FALSE"
machine <- paste(
  Sys.info()[c("sysname", "release", "machine")],
  collapse = " | "
)

manifest <- c(
  paste("created_utc", format(Sys.time(), tz = "UTC", usetz = TRUE), sep = "="),
  paste("core_commit", core_commit, sep = "="),
  paste("core_dirty", core_dirty, sep = "="),
  paste("kodamaR_version", as.character(utils::packageVersion("kodamaR")), sep = "="),
  paste("R_version", R.version.string, sep = "="),
  paste("machine", machine, sep = "="),
  paste("datasets", paste(datasets, collapse = ","), sep = "="),
  paste("backends", paste(backends, collapse = ","), sep = "="),
  paste("classifiers", paste(classifiers, collapse = ","), sep = "="),
  paste("seeds", paste(seeds, collapse = ","), sep = "="),
  paste("M", M, sep = "="),
  paste("Tcycle", Tcycle, sep = "="),
  paste("landmarks", landmarks, sep = "="),
  paste("graph_k", graph_k, sep = "="),
  paste("knn_k", knn_k, sep = "="),
  paste("ncomp", ncomp, sep = "="),
  paste("n_cores", n_cores, sep = "="),
  paste("accelerator_cores", accelerator_cores, sep = "="),
  paste("graph_mode", graph_mode, sep = "=")
)
manifest_dir <- file.path(out_dir, "manifests")
dir.create(manifest_dir, recursive = TRUE, showWarnings = FALSE)
manifest_stamp <- format(Sys.time(), "%Y%m%dT%H%M%SZ", tz = "UTC")
manifest_path <- file.path(
  manifest_dir,
  sprintf("invocation_%s_pid%d.txt", manifest_stamp, Sys.getpid())
)
writeLines(manifest, manifest_path)
study_manifest <- file.path(out_dir, "manifest.txt")
if (!file.exists(study_manifest)) {
  writeLines(c(
    "format=kodama-local-cpu-metal-study-v2",
    "invocation_manifests=manifests/invocation_*.txt",
    "results=local_cpu_metal_summary.csv",
    "censored_results=censored_cells.csv"
  ), study_manifest)
}

completed <- if (file.exists(summary_path)) utils::read.csv(summary_path) else data.frame()

for (dataset_name in datasets) {
  dataset_file <- dataset_path(root, dataset_name, local_data_root)
  ds <- load_dataset(dataset_file)
  x <- ds$data
  truth <- ds$labels
  splitting <- if (splitting_override > 0L) {
    splitting_override
  } else if (nrow(x) < 40000L) {
    100L
  } else {
    300L
  }
  data_md5 <- unname(tools::md5sum(dataset_file))

  for (seed in seeds) {
    classic <- KODAMA.visualization(
      x, method = "UMAP", raw.data = x, initialize.from.raw = TRUE,
      k = 30L, backend = "cpu", n.cores = host_threads, n.epochs = 200L, seed = seed
    )
    classic_sil <- sampled_silhouette(classic, truth, seed = seed)

    shared_prepared <- NULL
    shared_graph_wall <- NA_real_
    if (identical(graph_mode, "shared_cpu")) {
      graph_start <- proc.time()[["elapsed"]]
      shared_prepared <- KODAMA.graph(
        x, k = graph_k, metric = "euclidean", backend = "cpu",
        n.cores = host_threads, seed = seed, storage = "handle"
      )
      shared_graph_wall <- proc.time()[["elapsed"]] - graph_start
      if (!identical(shared_prepared$backend, "cpu")) {
        stop("Shared graph identity failure: expected cpu, received ", shared_prepared$backend)
      }
    }

    for (backend in backends) {
      graph_id <- paste(dataset_name, graph_mode, backend, seed, M, Tcycle, sep = "__")
      if (identical(graph_mode, "shared_cpu")) {
        prepared <- shared_prepared
        graph_wall <- shared_graph_wall
      } else {
        graph_start <- proc.time()[["elapsed"]]
        prepared <- KODAMA.graph(
          x, k = graph_k, metric = "euclidean", backend = backend,
          n.cores = host_threads, seed = seed, storage = "handle"
        )
        graph_wall <- proc.time()[["elapsed"]] - graph_start
        if (!identical(prepared$backend, backend)) {
          stop("Backend identity failure: requested ", backend, ", received ", prepared$backend)
        }
      }

      for (classifier in classifiers) {
        job_id <- paste(graph_id, classifier, sep = "__")
        if (nrow(completed) && job_id %in% completed$job_id) {
          message("[skip] ", job_id)
          next
        }
        message("[run] ", job_id)
        started <- format(Sys.time(), tz = "UTC", usetz = TRUE)
        wall_start <- proc.time()[["elapsed"]]
        requested_cores <- if (identical(backend, "cpu")) n_cores else accelerator_cores
        result <- KODAMA.matrix(
          data = x,
          graph = prepared,
          M = M,
          Tcycle = Tcycle,
          ncomp = min(ncomp, ncol(x), nrow(x) - 1L),
          landmarks = landmarks,
          splitting = splitting,
          n.cores = requested_cores,
          graph.neighbors = graph_k,
          knn.k = knn_k,
          classifier = classifier,
          backend = backend,
          seed = seed,
          # Shared-graph mode isolates optimization and later uses one CPU
          # raw-data PCA start for every final UMAP. Native mode retains the
          # backend-matched initialization in the matrix result.
          visual.init = identical(graph_mode, "native"),
          progress = TRUE,
          apply.kodama.dissimilarity = TRUE,
          return.graph = TRUE
        )
        matrix_wall <- proc.time()[["elapsed"]] - wall_start
        if (!identical(result$backend, backend) ||
            !identical(result$optimization_backend, backend)) {
          stop("Matrix backend identity failure for ", job_id)
        }

        embedding_start <- proc.time()[["elapsed"]]
        embedding <- KODAMA.visualization(
          result, method = "UMAP", raw.data = x, initialize.from.raw = TRUE,
          k = 30L, backend = "cpu", n.cores = host_threads, n.epochs = 200L, seed = seed
        )
        embedding_seconds <- proc.time()[["elapsed"]] - embedding_start
        best_labels <- as.integer(result$best_labels)
        best_ari <- adjusted_rand(best_labels, truth)
        kodama_sil <- sampled_silhouette(embedding, truth, seed = seed)
        label_sil <- sampled_silhouette(embedding, best_labels, seed = seed)
        acc <- as.numeric(result$acc)
        labels_by_run <- as.matrix(result$res)
        run_aris <- apply(labels_by_run, 1L, adjusted_rand, b = truth)
        class_counts <- as.integer(result$class_counts)

        compact <- list(
          job_id = job_id,
          acc = acc,
          class_counts = as.integer(result$class_counts),
          best_run = as.integer(result$best_run),
          best_labels = best_labels,
          landmark_seconds = as.numeric(result$landmark_seconds),
          classic_umap = classic,
          kodama_umap = embedding
        )
        saveRDS(compact, file.path(out_dir, "runs", paste0(job_id, ".rds")), compress = FALSE)
        plot_comparison(
          file.path(out_dir, "plots", paste0(job_id, ".png")),
          classic, embedding, truth, best_labels,
          paste(dataset_name, classifier, toupper(backend)),
          sprintf("M=%d T=%d sec=%.2f ARI=%.3f sil=%.3f", M, Tcycle, graph_wall + matrix_wall, best_ari, kodama_sil)
        )

        row <- data.frame(
          job_id = job_id,
          started_utc = started,
          dataset = dataset_name,
          data_md5 = data_md5,
          samples = nrow(x),
          variables = ncol(x),
          truth_classes = nlevels(truth),
          classifier = classifier,
          graph_mode = graph_mode,
          graph_build_key = if (identical(graph_mode, "shared_cpu")) {
            paste(dataset_name, "shared_cpu", seed, sep = "__")
          } else {
            paste(dataset_name, backend, seed, sep = "__")
          },
          backend_requested = backend,
          graph_backend = as.character(result$graph_backend),
          optimization_backend = as.character(result$optimization_backend),
          dissimilarity_backend = as.character(result$dissimilarity_backend),
          n_cores_requested = requested_cores,
          n_cores_selected = as.integer(result$n.cores),
          gpu_auto_workers = isTRUE(result$gpu_auto_workers),
          gpu_scheduler_lanes = as.integer(result$gpu_scheduler_lanes),
          gpu_worker_memory_estimate_mb = as.numeric(result$gpu_worker_memory_estimate_mb),
          seed = seed,
          M = M,
          Tcycle = Tcycle,
          landmarks_requested = landmarks,
          effective_landmarks = if (nrow(x) <= landmarks) ceiling(0.75 * nrow(x)) else landmarks,
          splitting = splitting,
          graph_k = graph_k,
          knn_k = knn_k,
          ncomp = min(ncomp, ncol(x), nrow(x) - 1L),
          graph_index_type = as.character(prepared$index_type),
          graph_ivf_nlist = as.integer(prepared$ivf_nlist),
          graph_ivf_nprobe = as.integer(prepared$ivf_nprobe),
          graph_ivf_pilot_recall = as.numeric(prepared$ivf_pilot_recall),
          graph_wall_seconds = graph_wall,
          graph_reported_seconds = as.numeric(prepared$runtime_seconds),
          input_copy_seconds = timing_value(result, "input_copy_seconds"),
          visual_init_seconds = as.numeric(prepared$timing$visual_init_seconds),
          landmark_sum_seconds = timing_value(result, "landmark_sum_seconds"),
          landmark_mean_seconds = timing_value(result, "landmark_mean_seconds"),
          landmark_median_seconds = timing_value(result, "landmark_median_seconds"),
          optimization_wall_seconds = timing_value(result, "optimization_wall_seconds"),
          optimization_sum_seconds = timing_value(result, "optimization_sum_seconds"),
          dissimilarity_seconds = timing_value(result, "dissimilarity_seconds"),
          graph_conversion_seconds = timing_value(result, "r_graph_conversion_seconds"),
          matrix_wall_seconds = matrix_wall,
          matrix_reported_seconds = as.numeric(result$runtime_seconds),
          embedding_seconds = embedding_seconds,
          pipeline_seconds = graph_wall + matrix_wall + embedding_seconds,
          peak_memory_mb = as.numeric(result$peak_memory_mb),
          best_cv_accuracy = max(acc, na.rm = TRUE),
          median_cv_accuracy = stats::median(acc, na.rm = TRUE),
          iqr_cv_accuracy = stats::IQR(acc, na.rm = TRUE),
          best_ari = best_ari,
          median_run_ari = stats::median(run_aris, na.rm = TRUE),
          max_diagnostic_ari = max(run_aris, na.rm = TRUE),
          best_classes = length(unique(best_labels)),
          median_classes = stats::median(class_counts),
          iqr_classes = stats::IQR(class_counts),
          min_classes = min(class_counts),
          max_classes = max(class_counts),
          collapse_rate_le2 = mean(class_counts <= 2L),
          median_pairwise_m_ari = median_pairwise_ari(labels_by_run, seed = seed),
          classic_truth_silhouette = classic_sil,
          kodama_truth_silhouette = kodama_sil,
          kodama_label_silhouette = label_sil,
          core_commit = core_commit,
          core_dirty = core_dirty,
          stringsAsFactors = FALSE
        )
        append_row(row, summary_path)
        completed <- if (nrow(completed)) bind_rows_fill(completed, row) else row
      }
    }
  }
}

message("Wrote ", normalizePath(summary_path))
