#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

options(stringsAsFactors = FALSE)

value_or <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}

csv_values <- function(name, default) {
  values <- trimws(strsplit(value_or(name, default), ",", fixed = TRUE)[[1L]])
  values[nzchar(values)]
}

r_lib <- value_or("KODAMA_R_LIB", "")
if (nzchar(r_lib)) .libPaths(c(r_lib, .libPaths()))
suppressPackageStartupMessages({
  library(cluster)
  library(kodamaR)
})

data_root <- normalizePath(
  value_or("KODAMA_DATA_ROOT", "/mnt/sata_ssd/fastEmbedR/Data"),
  mustWork = TRUE
)
out_dir <- value_or(
  "KODAMA_PANEL_OUT",
  "/mnt/sata_ssd/kodama-cpp-benchmarks/jmlr-nonspatial-panel-20260718"
)
datasets <- csv_values(
  "KODAMA_PANEL_DATASETS",
  paste(
    c(
      "MetRef", "COIL20", "PBMC3K_PCA50", "OptDigits", "USPS",
      "HumanActivityRecognition", "Macosko2015_retina", "MNIST",
      "TabulaMuris", "FashionMNIST"
    ),
    collapse = ","
  )
)
classifiers <- csv_values("KODAMA_PANEL_CLASSIFIERS", "knn,pls_lda")
methods <- csv_values("KODAMA_PANEL_METHODS", "UMAP,opentsne")

M <- as.integer(value_or("KODAMA_PANEL_M", "100"))
Tcycle <- as.integer(value_or("KODAMA_PANEL_TCYCLE", "100"))
landmarks <- as.integer(value_or("KODAMA_PANEL_LANDMARKS", "100000"))
ncomp <- as.integer(value_or("KODAMA_PANEL_NCOMP", "50"))
knn_k <- as.integer(value_or("KODAMA_PANEL_KNN_K", "30"))
graph_k <- as.integer(value_or("KODAMA_PANEL_GRAPH_K", "100"))
embedding_k <- as.integer(value_or("KODAMA_PANEL_EMBED_K", "30"))
perplexity <- as.numeric(value_or("KODAMA_PANEL_PERPLEXITY", "30"))
umap_epochs <- as.integer(value_or("KODAMA_PANEL_UMAP_EPOCHS", "200"))
tsne_iterations <- as.integer(value_or("KODAMA_PANEL_TSNE_ITER", "500"))
backend <- value_or("KODAMA_PANEL_BACKEND", "cuda")
n_cores <- as.integer(value_or("KODAMA_PANEL_CORES", "4"))
gpu_device <- as.integer(value_or("KODAMA_PANEL_GPU", "0"))
seed <- as.integer(value_or("KODAMA_PANEL_SEED", "1234"))

if (!all(classifiers %in% c("knn", "pls_lda"))) {
  stop("KODAMA_PANEL_CLASSIFIERS must contain only knn and/or pls_lda")
}
if (!all(methods %in% c("UMAP", "opentsne"))) {
  stop("KODAMA_PANEL_METHODS must contain only UMAP and/or opentsne")
}

subdirs <- c("results", "embeddings", "graphs", "metrics", "plots", "logs")
for (subdir in subdirs) {
  dir.create(file.path(out_dir, subdir), recursive = TRUE, showWarnings = FALSE)
}

dataset_manifest <- c(
  MetRef = file.path(data_root, "MetRef", "MetRef_float32.RData"),
  COIL20 = file.path(data_root, "COIL20", "COIL20_float32.RData"),
  FashionMNIST = file.path(data_root, "FashionMNIST", "FashionMNIST_float32.RData"),
  MNIST = file.path(data_root, "MNIST", "MNIST_float32.RData"),
  Macosko2015_retina = file.path(data_root, "Macosko2015_retina", "Macosko2015_retina_float32.RData"),
  TabulaMuris = file.path(data_root, "TabulaMuris", "TabulaMuris_float32.RData"),
  USPS = file.path(data_root, "USPS", "USPS_float32.RData"),
  flow18 = file.path(data_root, "flow18", "flow18_float32.RData"),
  mass41 = file.path(data_root, "mass41", "mass41_float32.RData"),
  FlowRepository = file.path(
    data_root,
    "FlowRepository_FR-FCM-ZYRM_files",
    "van_unen_FR-FCM-ZYRM_float32.RData"
  ),
  BreastCancerDiagnostic = file.path(data_root, "kodama_extra_benchmarks", "BreastCancerDiagnostic.RData"),
  HumanActivityRecognition = file.path(data_root, "kodama_extra_benchmarks", "HumanActivityRecognition.RData"),
  ImageSegmentation = file.path(data_root, "kodama_extra_benchmarks", "ImageSegmentation.RData"),
  LetterRecognition = file.path(data_root, "kodama_extra_benchmarks", "LetterRecognition.RData"),
  OptDigits = file.path(data_root, "kodama_extra_benchmarks", "OptDigits.RData"),
  PageBlocks = file.path(data_root, "kodama_extra_benchmarks", "PageBlocks.RData"),
  PBMC3K_PCA50 = file.path(data_root, "kodama_extra_benchmarks", "PBMC3K_PCA50.RData"),
  PenDigits = file.path(data_root, "kodama_extra_benchmarks", "PenDigits.RData"),
  SatImage = file.path(data_root, "kodama_extra_benchmarks", "SatImage.RData"),
  SklearnDigits8x8 = file.path(data_root, "kodama_extra_benchmarks", "SklearnDigits8x8.RData"),
  Spambase = file.path(data_root, "kodama_extra_benchmarks", "Spambase.RData"),
  Vehicle = file.path(data_root, "kodama_extra_benchmarks", "Vehicle.RData"),
  Yeast = file.path(data_root, "kodama_extra_benchmarks", "Yeast.RData")
)

unknown <- setdiff(datasets, names(dataset_manifest))
if (length(unknown)) stop("Unknown datasets: ", paste(unknown, collapse = ", "))

as_numeric_matrix <- function(x) {
  if (inherits(x, "float32") || inherits(x, "float")) {
    if (!requireNamespace("float", quietly = TRUE)) {
      stop("The float package is required to load float32 RData matrices")
    }
    x <- float::dbl(x)
  }
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

extract_dataset <- function(path) {
  e <- new.env(parent = emptyenv())
  load(path, envir = e)
  objects <- mget(ls(e, all.names = TRUE), envir = e)
  object <- if ("dataset" %in% names(objects)) {
    objects$dataset
  } else {
    candidates <- Filter(
      function(x) is.list(x) && !is.null(x$data) && !is.null(x$labels),
      objects
    )
    if (length(candidates) != 1L) {
      stop("Expected exactly one list with $data and $labels in ", path)
    }
    candidates[[1L]]
  }
  if (!is.list(object) || is.null(object$data) || is.null(object$labels)) {
    stop("Dataset object does not contain $data and $labels: ", path)
  }
  x <- as_numeric_matrix(object$data)
  labels <- droplevels(as.factor(object$labels))
  if (nrow(x) != length(labels)) {
    stop("Label count does not match matrix rows in ", path)
  }
  if (any(!is.finite(x))) stop("Non-finite values in ", path)
  list(data = x, labels = labels, path = path)
}

atomic_save_rds <- function(object, path) {
  temporary <- paste0(path, ".tmp-", Sys.getpid())
  saveRDS(object, temporary, compress = FALSE)
  if (!file.rename(temporary, path)) {
    unlink(temporary)
    stop("Could not atomically replace ", path)
  }
}

atomic_write_csv <- function(object, path) {
  temporary <- paste0(path, ".tmp-", Sys.getpid())
  utils::write.csv(object, temporary, row.names = FALSE)
  if (!file.rename(temporary, path)) {
    unlink(temporary)
    stop("Could not atomically replace ", path)
  }
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
  denominator <- maximum - expected
  if (!is.finite(denominator) || denominator == 0) 1 else (nij - expected) / denominator
}

sampled_silhouette <- function(embedding, labels, max_samples = 5000L, sample_seed = seed) {
  labels <- as.integer(as.factor(labels))
  keep <- which(!is.na(labels) & rowSums(is.finite(embedding)) == ncol(embedding))
  if (length(keep) < 3L || length(unique(labels[keep])) < 2L) return(NA_real_)
  if (length(keep) > max_samples) {
    set.seed(sample_seed)
    keep <- sort(sample(keep, max_samples))
  }
  mean(cluster::silhouette(labels[keep], stats::dist(embedding[keep, , drop = FALSE]))[, 3L])
}

sampled_neighbor_purity <- function(embedding, labels, k = 15L, max_samples = 3000L) {
  labels <- as.factor(labels)
  keep <- which(!is.na(labels) & rowSums(is.finite(embedding)) == ncol(embedding))
  if (length(keep) < k + 2L) return(NA_real_)
  if (length(keep) > max_samples) {
    set.seed(seed + 13L)
    keep <- sort(sample(keep, max_samples))
  }
  coordinates <- embedding[keep, , drop = FALSE]
  selected_labels <- labels[keep]
  purity <- numeric(nrow(coordinates))
  for (row in seq_len(nrow(coordinates))) {
    distances <- rowSums((coordinates - matrix(coordinates[row, ], nrow(coordinates), 2L, byrow = TRUE))^2)
    neighbors <- order(distances)[seq_len(min(k + 1L, length(distances)))]
    neighbors <- head(neighbors[neighbors != row], k)
    purity[row] <- mean(selected_labels[neighbors] == selected_labels[row])
  }
  mean(purity)
}

median_truth_compactness <- function(embedding, labels) {
  labels <- as.factor(labels)
  centered <- sweep(embedding, 2L, colMeans(embedding), check.margin = FALSE)
  overall <- sqrt(mean(rowSums(centered^2)))
  values <- vapply(levels(labels), function(level) {
    block <- embedding[labels == level, , drop = FALSE]
    center <- colMeans(block)
    sqrt(mean(rowSums(sweep(block, 2L, center, check.margin = FALSE)^2))) / overall
  }, numeric(1L))
  stats::median(values)
}

graph_truth_purity <- function(graph, labels) {
  if (is.null(graph) || is.null(graph$indices)) return(NA_real_)
  indices <- as.matrix(graph$indices)
  if (length(indices) && min(indices, na.rm = TRUE) == 0L) indices <- indices + 1L
  valid <- indices >= 1L & indices <= length(labels)
  if (!any(valid)) return(NA_real_)
  query <- row(indices)
  mean(labels[query[valid]] == labels[indices[valid]])
}

selected_run_data <- function(result, truth) {
  accuracies <- as.numeric(result$acc)
  labels_by_run <- as.matrix(result$res)
  best_run <- if (!is.null(result$best_run) && length(result$best_run) && is.finite(result$best_run)) {
    as.integer(result$best_run)
  } else {
    which.max(accuracies)
  }
  if (best_run < 1L || best_run > nrow(labels_by_run)) best_run <- which.max(accuracies)
  aris <- apply(labels_by_run, 1L, adjusted_rand, b = truth)
  classes <- apply(labels_by_run, 1L, function(x) length(unique(x)))
  list(
    labels = as.integer(labels_by_run[best_run, ]),
    best_run = best_run,
    accuracies = accuracies,
    aris = aris,
    classes = classes
  )
}

method_init <- function(result, method) {
  if (is.null(result$visual_init)) return(NULL)
  key <- if (method == "UMAP") "umap" else "opentsne"
  init <- result$visual_init
  if (is.matrix(init)) init else init[[key]]
}

run_embedding <- function(input, method, init = NULL) {
  start <- proc.time()[["elapsed"]]
  embedding <- KODAMA.visualization(
    input,
    method = method,
    init = init,
    k = embedding_k,
    backend = backend,
    n.cores = n_cores,
    gpu.device = gpu_device,
    n.epochs = umap_epochs,
    n.iter = tsne_iterations,
    perplexity = perplexity,
    seed = seed
  )
  list(embedding = embedding, seconds = proc.time()[["elapsed"]] - start)
}

point_size <- function(n) {
  if (n >= 50000L) 0.24 else if (n >= 10000L) 0.42 else if (n >= 2000L) 0.65 else 1.0
}

embedding_key <- function(dataset, source, method) {
  paste(dataset, source, tolower(method), sep = "__")
}

embedding_path <- function(dataset, source, method) {
  file.path(out_dir, "embeddings", paste0(embedding_key(dataset, source, method), ".rds"))
}

read_embedding <- function(dataset, source, method) {
  path <- embedding_path(dataset, source, method)
  if (file.exists(path)) readRDS(path) else NULL
}

write_panel <- function(dataset, truth) {
  path <- file.path(out_dir, "plots", paste0(dataset, "__six_panel.png"))
  sources <- c("classic", "knn", "pls_lda")
  display <- c(classic = "Classic", knn = "KODAMA KNN", pls_lda = "KODAMA PLS-LDA")
  colors <- grDevices::hcl.colors(nlevels(truth), "Dynamic")[as.integer(truth)]
  grDevices::png(path, width = 2700, height = 1750, res = 160)
  old <- graphics::par(mfrow = c(2, 3), mar = c(1.1, 1.1, 3.0, 0.5), oma = c(0, 0, 3.0, 0))
  on.exit({
    graphics::par(old)
    grDevices::dev.off()
  }, add = TRUE)
  for (method in c("UMAP", "opentsne")) {
    for (source in sources) {
      record <- read_embedding(dataset, source, method)
      if (is.null(record)) {
        graphics::plot.new()
        graphics::title(main = paste(display[[source]], if (method == "UMAP") "UMAP" else "openTSNE"))
        graphics::text(0.5, 0.5, "pending", col = "grey55")
      } else {
        silhouette <- sampled_silhouette(record$embedding, truth)
        graphics::plot(
          record$embedding,
          pch = 16,
          cex = point_size(nrow(record$embedding)),
          col = colors,
          axes = FALSE,
          xlab = "",
          ylab = "",
          main = sprintf(
            "%s %s\ntruth silhouette=%.3f",
            display[[source]],
            if (method == "UMAP") "UMAP" else "openTSNE",
            silhouette
          )
        )
        graphics::box(col = "grey80")
      }
    }
  }
  graphics::mtext(
    sprintf(
      "%s | CUDA release validation | M=%d Tcycle=%d landmarks=%d k=%d graph-k=%d",
      dataset, M, Tcycle, landmarks, knn_k, graph_k
    ),
    outer = TRUE,
    side = 3,
    line = 0.8,
    font = 2
  )
  invisible(path)
}

collect_metrics <- function() {
  paths <- list.files(file.path(out_dir, "metrics"), pattern = "\\.csv$", full.names = TRUE)
  if (!length(paths)) return(invisible(NULL))
  rows <- lapply(sort(paths), utils::read.csv)
  all_metrics <- do.call(rbind, rows)
  atomic_write_csv(all_metrics, file.path(out_dir, "nonspatial_panel_summary.csv"))
  invisible(all_metrics)
}

run_dataset <- function(dataset) {
  path <- unname(dataset_manifest[[dataset]])
  if (!file.exists(path)) stop("Missing dataset: ", path)
  message("== ", dataset, " ==")
  load_start <- proc.time()[["elapsed"]]
  ds <- extract_dataset(path)
  load_seconds <- proc.time()[["elapsed"]] - load_start
  x <- ds$data
  truth <- ds$labels
  n <- nrow(x)
  p <- ncol(x)
  splitting <- if (n < 40000L) 100L else 300L
  effective_landmarks <- if (n <= landmarks) ceiling(0.75 * n) else landmarks
  message(sprintf("n=%d p=%d classes=%d load=%.2fs", n, p, nlevels(truth), load_seconds))

  classic_graph_file <- file.path(out_dir, "graphs", paste0(dataset, "__classic_graph.rds"))
  classic_graph_record <- if (file.exists(classic_graph_file)) readRDS(classic_graph_file) else NULL

  for (classifier in classifiers) {
    job <- paste(dataset, classifier, backend, paste0("M", M), paste0("T", Tcycle), sep = "__")
    result_file <- file.path(out_dir, "results", paste0(job, ".rds"))
    timing_file <- file.path(out_dir, "results", paste0(job, "__timing.rds"))
    if (file.exists(result_file) && file.exists(timing_file)) {
      message("[cache] ", job)
      result <- readRDS(result_file)
      matrix_timing <- readRDS(timing_file)
    } else {
      message("[run] ", job)
      start <- proc.time()[["elapsed"]]
      result <- kodama_matrix(
        x,
        M = M,
        Tcycle = Tcycle,
        ncomp = min(ncomp, p, n - 1L),
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
      matrix_timing <- list(
        wall_seconds = proc.time()[["elapsed"]] - start,
        reported_seconds = if (is.null(result$runtime_seconds)) NA_real_ else result$runtime_seconds
      )
      atomic_save_rds(result, result_file)
      atomic_save_rds(matrix_timing, timing_file)
    }

    if (is.null(classic_graph_record)) {
      message("[run] ", dataset, " classic graph")
      start <- proc.time()[["elapsed"]]
      classic_graph <- KODAMA.graph(
        x,
        k = embedding_k,
        backend = backend,
        n.cores = n_cores,
        gpu.device = gpu_device
      )
      classic_graph_record <- list(
        graph = classic_graph,
        seconds = proc.time()[["elapsed"]] - start
      )
      atomic_save_rds(classic_graph_record, classic_graph_file)
    }

    selected <- selected_run_data(result, truth)
    metric_rows <- list()
    for (method in methods) {
      classic_file <- embedding_path(dataset, "classic", method)
      if (file.exists(classic_file)) {
        classic_record <- readRDS(classic_file)
      } else {
        message("[embed] ", dataset, " classic ", method)
        classic_record <- run_embedding(
          classic_graph_record$graph,
          method,
          init = method_init(result, method)
        )
        atomic_save_rds(classic_record, classic_file)
      }

      kodama_file <- embedding_path(dataset, classifier, method)
      if (file.exists(kodama_file)) {
        kodama_record <- readRDS(kodama_file)
      } else {
        message("[embed] ", dataset, " ", classifier, " ", method)
        kodama_record <- run_embedding(result, method)
        atomic_save_rds(kodama_record, kodama_file)
      }

      metric_rows[[method]] <- data.frame(
        dataset = dataset,
        classifier = classifier,
        method = method,
        backend = backend,
        samples = n,
        variables = p,
        truth_classes = nlevels(truth),
        M = M,
        Tcycle = Tcycle,
        landmarks_requested = landmarks,
        landmarks_effective = effective_landmarks,
        splitting = splitting,
        ncomp = min(ncomp, p, n - 1L),
        knn_k = knn_k,
        graph_k = graph_k,
        embedding_k = embedding_k,
        perplexity = perplexity,
        seed = seed,
        workers = n_cores,
        load_seconds = load_seconds,
        classic_graph_seconds = classic_graph_record$seconds,
        kodama_wall_seconds = matrix_timing$wall_seconds,
        kodama_reported_seconds = matrix_timing$reported_seconds,
        classic_embedding_seconds = classic_record$seconds,
        kodama_embedding_seconds = kodama_record$seconds,
        best_cv_accuracy = max(selected$accuracies, na.rm = TRUE),
        median_cv_accuracy = stats::median(selected$accuracies, na.rm = TRUE),
        selected_run = selected$best_run,
        selected_ari = selected$aris[selected$best_run],
        median_run_ari = stats::median(selected$aris, na.rm = TRUE),
        max_diagnostic_ari = max(selected$aris, na.rm = TRUE),
        selected_classes = selected$classes[selected$best_run],
        median_classes = stats::median(selected$classes, na.rm = TRUE),
        min_classes = min(selected$classes, na.rm = TRUE),
        max_classes = max(selected$classes, na.rm = TRUE),
        base_graph_truth_purity = graph_truth_purity(classic_graph_record$graph, truth),
        kodama_graph_truth_purity = graph_truth_purity(result$knn, truth),
        classic_truth_silhouette = sampled_silhouette(classic_record$embedding, truth),
        kodama_truth_silhouette = sampled_silhouette(kodama_record$embedding, truth),
        classic_truth_neighbor_purity = sampled_neighbor_purity(classic_record$embedding, truth),
        kodama_truth_neighbor_purity = sampled_neighbor_purity(kodama_record$embedding, truth),
        classic_truth_compactness = median_truth_compactness(classic_record$embedding, truth),
        kodama_truth_compactness = median_truth_compactness(kodama_record$embedding, truth),
        kodama_label_silhouette = sampled_silhouette(kodama_record$embedding, selected$labels)
      )
    }
    job_metrics <- do.call(rbind, metric_rows)
    atomic_write_csv(job_metrics, file.path(out_dir, "metrics", paste0(job, ".csv")))
    panel <- write_panel(dataset, truth)
    collect_metrics()
    message("[complete] ", job, " plot=", panel)
    rm(result)
    gc(verbose = FALSE)
  }
  rm(x, ds)
  gc(verbose = FALSE)
}

metadata <- data.frame(
  key = c(
    "datasets", "classifiers", "methods", "M", "Tcycle", "landmarks",
    "ncomp", "knn_k", "graph_k", "embedding_k", "perplexity", "backend",
    "workers", "gpu_device", "seed"
  ),
  value = c(
    paste(datasets, collapse = ","), paste(classifiers, collapse = ","),
    paste(methods, collapse = ","), M, Tcycle, landmarks, ncomp, knn_k,
    graph_k, embedding_k, perplexity, backend, n_cores, gpu_device, seed
  )
)
atomic_write_csv(metadata, file.path(out_dir, "benchmark_configuration.csv"))

for (dataset in datasets) {
  tryCatch(
    run_dataset(dataset),
    error = function(e) {
      error_path <- file.path(out_dir, "logs", paste0(dataset, "__error.txt"))
      writeLines(conditionMessage(e), error_path)
      message("[error] ", dataset, ": ", conditionMessage(e))
    }
  )
}

collect_metrics()
message("Summary: ", file.path(out_dir, "nonspatial_panel_summary.csv"))
