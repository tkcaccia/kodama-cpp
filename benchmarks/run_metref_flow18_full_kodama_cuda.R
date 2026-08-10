#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

suppressPackageStartupMessages(library(kodamaR))

env_int <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (!nzchar(value)) default else as.integer(value)
}

dataset_specs <- list(
  MetRef = list(
    paths = c(
      "/mnt/sata_ssd/fastEmbedR/Data/MetRef/MetRef_float32.RData",
      "/mnt/sata_ssd/KODAMAopt/nonspatial/MetRef.RData",
      "/Users/stefano/Documents/fastEmbedR/Data/MetRef/MetRef_float32.RData"
    ),
    landmarks = 10000000L,
    ncomp = 50L
  ),
  flow18 = list(
    paths = c(
      "/mnt/sata_ssd/fastEmbedR/Data/flow18/flow18_float32.RData",
      "/mnt/sata_ssd/KODAMAopt/nonspatial/flow18.RData",
      "/Users/stefano/Documents/fastEmbedR/Data/flow18/flow18_float32.RData"
    ),
    landmarks = 1000L,
    ncomp = 10L
  )
)

M <- env_int("KODAMA_M", 100L)
Tcycle <- env_int("KODAMA_TCYCLE", 100L)
knn_k <- env_int("KODAMA_KNN_K", 15L)
graph_k <- env_int("KODAMA_GRAPH_K", 100L)
embedding_k <- env_int("KODAMA_EMBEDDING_K", 30L)
perplexity <- env_int("KODAMA_PERPLEXITY", 30L)
seed <- env_int("KODAMA_SEED", 4L)
workers <- env_int("KODAMA_WORKERS", 0L)
backend <- Sys.getenv("KODAMA_BACKEND", "cuda")
out_dir <- Sys.getenv(
  "KODAMA_OUTPUT",
  "/mnt/sata_ssd/kodama-cpp-benchmarks/metref_flow18_full_k15"
)
datasets <- strsplit(Sys.getenv("KODAMA_DATASETS", "MetRef,flow18"), ",", fixed = TRUE)[[1L]]
datasets <- trimws(datasets)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "objects"), showWarnings = FALSE)
dir.create(file.path(out_dir, "plots"), showWarnings = FALSE)
dir.create(file.path(out_dir, "metrics"), showWarnings = FALSE)

elapsed <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, seconds = proc.time()[["elapsed"]] - start)
}

load_dataset <- function(paths) {
  path <- paths[file.exists(paths)][1L]
  if (is.na(path)) stop("Dataset not found. Checked: ", paste(paths, collapse = ", "))
  environment <- new.env(parent = emptyenv())
  load(path, envir = environment)
  objects <- as.list(environment)
  dataset <- if (!is.null(objects$dataset)) objects$dataset else objects[[1L]]
  if (!is.list(dataset) || is.null(dataset$data) || is.null(dataset$labels)) {
    stop("Expected dataset=list(data=..., labels=...) in ", path)
  }
  x <- dataset$data
  if (inherits(x, "float32") || inherits(x, "float")) {
    x <- float::dbl(x)
  } else {
    x <- as.matrix(x)
    storage.mode(x) <- "double"
  }
  list(path = path, data = x, truth = as.factor(dataset$labels))
}

class_counts <- function(result) {
  apply(result$res, 1L, function(labels) length(unique(labels[labels != 0L])))
}

selected_labels <- function(result) {
  selected <- which.max(result$acc)
  as.integer(result$res[selected, ])
}

point_size <- function(n) {
  if (n >= 500000L) 0.09 else if (n >= 50000L) 0.2 else if (n >= 10000L) 0.35 else 0.7
}

plot_four <- function(dataset, records, truth, metrics) {
  colors <- grDevices::hcl.colors(nlevels(truth), "Dynamic")[as.integer(truth)]
  path <- file.path(out_dir, "plots", paste0(dataset, "__KODAMA_KNN15_PLSLDA__UMAP_openTSNE.png"))
  grDevices::png(path, width = 2600, height = 1900, res = 180)
  old <- graphics::par(mfrow = c(2, 2), mar = c(0.7, 0.7, 3.2, 0.5), oma = c(0, 0, 3.1, 0))
  on.exit({ graphics::par(old); grDevices::dev.off() }, add = TRUE)
  for (classifier in c("knn", "pls_lda")) {
    for (method in c("UMAP", "opentsne")) {
      key <- paste(classifier, method, sep = "__")
      record <- records[[key]]
      label <- if (classifier == "knn") "KODAMA KNN (k=15)" else "KODAMA PLS-LDA"
      method_label <- if (method == "UMAP") "UMAP" else "openTSNE"
      graphics::plot(
        record$embedding,
        pch = 16,
        cex = point_size(nrow(record$embedding)),
        col = colors,
        axes = FALSE,
        xlab = "",
        ylab = "",
        main = sprintf("%s + %s\nmatrix %.1fs | visualization %.1fs", label, method_label,
                       metrics[[classifier]]$matrix_seconds, record$seconds)
      )
      graphics::box(col = "grey80")
    }
  }
  graphics::mtext(
    sprintf("%s | full KODAMA | %s | M=%d Tcycle=%d | shared graph %.1fs",
            dataset, toupper(backend), M, Tcycle, metrics$graph_seconds),
    side = 3, outer = TRUE, line = 0.7, font = 2
  )
  path
}

all_rows <- list()
for (dataset_name in datasets) {
  spec <- dataset_specs[[dataset_name]]
  if (is.null(spec)) stop("Unknown dataset: ", dataset_name)
  message("== ", dataset_name, " ==")
  loaded <- load_dataset(spec$paths)
  x <- loaded$data
  truth <- loaded$truth
  n <- nrow(x)
  p <- ncol(x)
  splitting <- if (n < 40000L) 100L else 300L
  effective_landmarks <- if (n <= spec$landmarks) ceiling(n * 0.75) else spec$landmarks
  ncomp <- min(spec$ncomp, p, n - 1L)

  graph_timed <- elapsed(KODAMA.graph(
    x, k = graph_k, backend = backend, n.cores = workers,
    gpu.device = 0L, seed = seed, storage = "handle"
  ))
  graph <- graph_timed$value
  metrics <- list(graph_seconds = graph_timed$seconds)
  records <- list()

  for (classifier in c("knn", "pls_lda")) {
    message("[matrix] ", dataset_name, " ", classifier)
    fit_timed <- elapsed(KODAMA.matrix(
      data = x,
      graph = graph,
      classifier = classifier,
      backend = backend,
      M = M,
      Tcycle = Tcycle,
      ncomp = ncomp,
      landmarks = spec$landmarks,
      splitting = splitting,
      n.cores = workers,
      graph.neighbors = graph_k,
      knn.k = knn_k,
      folds = 5L,
      seed = seed,
      visual.init = TRUE,
      progress = TRUE,
      apply.kodama.dissimilarity = TRUE,
      return.graph = "handle"
    ))
    fit <- fit_timed$value
    counts <- class_counts(fit)
    labels <- selected_labels(fit)
    metrics[[classifier]] <- list(
      matrix_seconds = fit_timed$seconds,
      best_accuracy = max(fit$acc, na.rm = TRUE),
      median_accuracy = stats::median(fit$acc, na.rm = TRUE),
      mean_classes = mean(counts),
      median_classes = stats::median(counts),
      min_classes = min(counts),
      max_classes = max(counts)
    )

    saveRDS(
      list(labels = labels, accuracies = fit$acc, class_counts = counts,
           parameters = fit$parameters, timing = fit$timing),
      file.path(out_dir, "objects", paste0(dataset_name, "__", classifier, "__summary.rds")),
      compress = FALSE
    )

    for (method in c("UMAP", "opentsne")) {
      message("[visualization] ", dataset_name, " ", classifier, " ", method)
      visual_timed <- elapsed(KODAMA.visualization(
        fit,
        method = method,
        k = embedding_k,
        backend = backend,
        n.cores = workers,
        gpu.device = 0L,
        n.epochs = 200L,
        n.iter = 500L,
        perplexity = perplexity,
        graph.mode = "fuzzy",
        seed = seed
      ))
      records[[paste(classifier, method, sep = "__")]] <- list(
        embedding = visual_timed$value,
        seconds = visual_timed$seconds
      )
      saveRDS(
        records[[paste(classifier, method, sep = "__")]],
        file.path(out_dir, "objects", paste0(dataset_name, "__", classifier, "__", tolower(method), ".rds")),
        compress = FALSE
      )
    }

    all_rows[[length(all_rows) + 1L]] <- data.frame(
      dataset = dataset_name,
      classifier = classifier,
      samples = n,
      variables = p,
      truth_classes = nlevels(truth),
      backend = backend,
      M = M,
      Tcycle = Tcycle,
      folds = 5L,
      graph_k = graph_k,
      knn_k = knn_k,
      embedding_k = embedding_k,
      perplexity = perplexity,
      landmarks_requested = spec$landmarks,
      landmarks_effective = effective_landmarks,
      splitting = splitting,
      ncomp = ncomp,
      workers = workers,
      graph_seconds = graph_timed$seconds,
      matrix_seconds = fit_timed$seconds,
      umap_seconds = records[[paste(classifier, "UMAP", sep = "__")]]$seconds,
      opentsne_seconds = records[[paste(classifier, "opentsne", sep = "__")]]$seconds,
      best_cv_accuracy = metrics[[classifier]]$best_accuracy,
      median_cv_accuracy = metrics[[classifier]]$median_accuracy,
      mean_classes = metrics[[classifier]]$mean_classes,
      median_classes = metrics[[classifier]]$median_classes,
      min_classes = metrics[[classifier]]$min_classes,
      max_classes = metrics[[classifier]]$max_classes
    )
    utils::write.csv(do.call(rbind, all_rows), file.path(out_dir, "metrics", "summary.csv"), row.names = FALSE)
    rm(fit)
    invisible(gc())
  }

  plot_path <- plot_four(dataset_name, records, truth, metrics)
  message("[plot] ", plot_path)
  rm(x, graph, records)
  invisible(gc())
}
