#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

suppressPackageStartupMessages({
  library(kodamaR)
  library(cluster)
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
    FlowRepository = file.path(root, "FlowRepository_FR-FCM-ZYRM_files", "van_unen_FR-FCM-ZYRM_float32.RData"),
    MNIST = file.path(root, "MNIST", "MNIST_float32.RData"),
    Macosko2015_retina = file.path(root, "Macosko2015_retina", "Macosko2015_retina_float32.RData"),
    MetRef = file.path(root, "MetRef", "MetRef_float32.RData"),
    TabulaMuris = file.path(root, "TabulaMuris", "TabulaMuris_float32.RData"),
    USPS = file.path(root, "USPS", "USPS_float32.RData"),
    flow18 = file.path(root, "flow18", "flow18_float32.RData"),
    imagenet = file.path(root, "imagenet", "imagenet_float32.RData"),
    mass41 = file.path(root, "mass41", "mass41_float32.RData")
  )
}

load_dataset <- function(path, max_rows = 0L, seed = 1L) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  obj <- if (exists("dataset", env, inherits = FALSE)) env$dataset else mget(ls(env), env)[[1L]]
  data <- as_numeric_matrix(obj$data)
  labels <- as.factor(obj$labels)
  if (nrow(data) != length(labels)) {
    stop("Dataset has ", nrow(data), " rows but ", length(labels), " labels: ", path)
  }
  if (max_rows > 0L && nrow(data) > max_rows) {
    set.seed(seed)
    keep <- sort(sample.int(nrow(data), max_rows))
    data <- data[keep, , drop = FALSE]
    labels <- droplevels(labels[keep])
  }
  list(data = data, labels = labels)
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

silhouette_sample <- function(embedding, labels, max_n = 5000L, seed = 1L) {
  if (is.null(embedding)) return(NA_real_)
  labels <- as.factor(labels)
  valid <- !is.na(labels)
  if (sum(valid) < 3L || length(unique(labels[valid])) < 2L) return(NA_real_)
  idx <- which(valid)
  if (length(idx) > max_n) {
    set.seed(seed)
    idx <- sort(sample(idx, max_n))
  }
  lab <- droplevels(labels[idx])
  if (any(tabulate(as.integer(lab)) < 2L)) {
    keep_levels <- names(which(table(lab) >= 2L))
    keep <- lab %in% keep_levels
    idx <- idx[keep]
    lab <- droplevels(lab[keep])
  }
  if (length(idx) < 3L || length(unique(lab)) < 2L) return(NA_real_)
  out <- tryCatch({
    sil <- cluster::silhouette(as.integer(lab), dist(embedding[idx, , drop = FALSE]))
    mean(sil[, "sil_width"])
  }, error = function(e) NA_real_)
  as.numeric(out)
}

neighbor_purity_sample <- function(embedding, labels, k = 15L, max_n = 3000L, seed = 1L) {
  if (is.null(embedding)) return(NA_real_)
  labels <- as.factor(labels)
  valid <- !is.na(labels)
  idx <- which(valid)
  if (length(idx) < k + 2L) return(NA_real_)
  if (length(idx) > max_n) {
    set.seed(seed)
    idx <- sort(sample(idx, max_n))
  }
  x <- embedding[idx, , drop = FALSE]
  lab <- labels[idx]
  purity <- numeric(nrow(x))
  for (i in seq_len(nrow(x))) {
    d <- rowSums((t(t(x) - x[i, ]))^2)
    nn <- order(d)[seq_len(min(k + 1L, length(d)))]
    nn <- nn[nn != i]
    nn <- nn[seq_len(min(k, length(nn)))]
    purity[i] <- mean(lab[nn] == lab[i])
  }
  mean(purity)
}

plot_embedding <- function(embedding, labels, main, sub = "", cex = 0.18) {
  if (is.null(embedding) || any(!is.finite(embedding))) {
    plot.new()
    title(main = main)
    text(0.5, 0.5, "not available")
    return(invisible(NULL))
  }
  labels <- as.factor(labels)
  colors <- grDevices::rainbow(length(levels(labels)), s = 0.75, v = 0.85)[as.integer(labels)]
  plot(embedding[, 1L], embedding[, 2L],
       pch = 16, cex = cex, col = colors,
       xlab = "", ylab = "", axes = FALSE, main = main, sub = sub,
       cex.main = 0.9, cex.sub = 0.75)
  box(col = "grey75")
}

run_embedding <- function(x, method, backend, k, n_epochs, n_iter, perplexity, n_cores, gpu_device, seed) {
  start <- proc.time()[["elapsed"]]
  emb <- KODAMA.visualization(
    x,
    method = method,
    k = k,
    backend = backend,
    n.cores = n_cores,
    gpu.device = gpu_device,
    n.epochs = n_epochs,
    n.iter = n_iter,
    perplexity = perplexity,
    seed = seed
  )
  list(embedding = emb, seconds = proc.time()[["elapsed"]] - start)
}

kodama_runtime <- function(result, fallback_seconds) {
  if (!is.null(result$runtime_seconds)) return(as.numeric(result$runtime_seconds))
  if (!is.null(result$timing) && !is.null(result$timing$runtime_seconds)) {
    return(as.numeric(result$timing$runtime_seconds))
  }
  fallback_seconds
}

record_row <- function(dataset, n, p, classes, source, method, backend, M, Tcycle,
                       classifier, kodama_seconds, embedding_seconds, embedding,
                       truth, kodama_result, status = "ok") {
  best_labels <- if (!is.null(kodama_result)) kodama_result$best_labels else NULL
  data.frame(
    dataset = dataset,
    n = n,
    p = p,
    truth_classes = classes,
    source = source,
    method = method,
    classifier = classifier %||% "",
    backend = backend,
    M = M,
    Tcycle = Tcycle,
    kodama_seconds = kodama_seconds,
    embedding_seconds = embedding_seconds,
    total_seconds = kodama_seconds + embedding_seconds,
    truth_silhouette = silhouette_sample(embedding, truth, seed = 11L),
    truth_neighbor_purity = neighbor_purity_sample(embedding, truth, seed = 13L),
    best_label_ari = if (is.null(best_labels)) NA_real_ else adjusted_rand_index(best_labels, truth),
    best_cv_accuracy = if (is.null(kodama_result) || length(kodama_result$acc) == 0L) NA_real_ else max(kodama_result$acc, na.rm = TRUE),
    median_cv_accuracy = if (is.null(kodama_result) || length(kodama_result$acc) == 0L) NA_real_ else median(kodama_result$acc, na.rm = TRUE),
    best_label_classes = if (is.null(best_labels)) NA_integer_ else length(unique(best_labels)),
    status = status,
    stringsAsFactors = FALSE
  )
}

main <- function() {
  root <- Sys.getenv("KODAMA_DATA_ROOT", "/Users/stefano/Documents/fastEmbedR/Data")
  out_dir <- Sys.getenv("KODAMA_VIS_BENCH_OUT", file.path(getwd(), "manuscript", "visualization_benchmark_20260707"))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  datasets <- parse_csv_env("KODAMA_VIS_DATASETS", "MetRef,USPS,COIL20")
  classifiers <- parse_csv_env("KODAMA_VIS_CLASSIFIERS", "knn,pls_lda")
  methods <- parse_csv_env("KODAMA_VIS_METHODS", "UMAP,opentsne")
  paths <- dataset_paths(root)

  backend <- Sys.getenv("KODAMA_VIS_BACKEND", "cpu")
  M <- as.integer(Sys.getenv("KODAMA_VIS_M", "4"))
  Tcycle <- as.integer(Sys.getenv("KODAMA_VIS_TCYCLE", "20"))
  ncomp <- as.integer(Sys.getenv("KODAMA_VIS_NCOMP", "50"))
  landmarks <- as.integer(Sys.getenv("KODAMA_VIS_LANDMARKS", "100000"))
  graph_neighbors <- as.integer(Sys.getenv("KODAMA_VIS_GRAPH_NEIGHBORS", "100"))
  knn_k <- as.integer(Sys.getenv("KODAMA_VIS_KNN_K", "30"))
  embed_k <- as.integer(Sys.getenv("KODAMA_VIS_EMBED_K", "30"))
  perplexity <- as.numeric(Sys.getenv("KODAMA_VIS_PERPLEXITY", "30"))
  n_epochs <- as.integer(Sys.getenv("KODAMA_VIS_UMAP_EPOCHS", "200"))
  n_iter <- as.integer(Sys.getenv("KODAMA_VIS_TSNE_ITER", "500"))
  n_cores <- as.integer(Sys.getenv("KODAMA_VIS_CORES", "4"))
  gpu_device <- as.integer(Sys.getenv("KODAMA_VIS_GPU", "0"))
  seed <- as.integer(Sys.getenv("KODAMA_VIS_SEED", "1234"))
  max_rows <- as.integer(Sys.getenv("KODAMA_VIS_MAX_ROWS", "0"))

  all_rows <- list()

  for (dataset in datasets) {
    if (!dataset %in% names(paths)) stop("Unknown dataset: ", dataset)
    if (!file.exists(paths[[dataset]])) stop("Missing dataset file: ", paths[[dataset]])
    message("== ", dataset, " ==")
    ds <- load_dataset(paths[[dataset]], max_rows = max_rows, seed = seed)
    x <- ds$data
    truth <- ds$labels
    n <- nrow(x)
    p <- ncol(x)
    classes <- length(levels(truth))
    splitting <- if (n < 40000L) 100L else 300L
    point_cex <- if (n <= 2000L) 0.55 else if (n <= 12000L) 0.28 else 0.12

    embeddings <- list()
    kodama_results <- list()

    for (method in methods) {
      key <- paste("classic", method, sep = "_")
      message("  classic ", method)
      res <- tryCatch(
        run_embedding(x, method, backend, embed_k, n_epochs, n_iter, perplexity, n_cores, gpu_device, seed),
        error = function(e) list(embedding = NULL, seconds = NA_real_, error = conditionMessage(e))
      )
      embeddings[[key]] <- res$embedding
      all_rows[[length(all_rows) + 1L]] <- record_row(
        dataset, n, p, classes, "classic", method, backend, M, Tcycle,
        NA_character_, 0, res$seconds %||% NA_real_, res$embedding, truth, NULL,
        status = if (is.null(res$error)) "ok" else res$error
      )
    }

    for (classifier in classifiers) {
      message("  KODAMA ", classifier)
      start <- proc.time()[["elapsed"]]
      kk <- tryCatch(
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
      elapsed <- proc.time()[["elapsed"]] - start
      if (inherits(kk, "kodama_error")) {
        warning("KODAMA ", classifier, " failed on ", dataset, ": ", kk$error)
        for (method in methods) {
          all_rows[[length(all_rows) + 1L]] <- record_row(
            dataset, n, p, classes, paste0("kodama_", classifier), method, backend, M, Tcycle,
            classifier, elapsed, NA_real_, NULL, truth, NULL, status = kk$error
          )
        }
        next
      }
      kodama_results[[classifier]] <- kk
      ksec <- kodama_runtime(kk, elapsed)
      for (method in methods) {
        key <- paste(classifier, method, sep = "_")
        message("    ", method)
        res <- tryCatch(
          run_embedding(kk, method, backend, embed_k, n_epochs, n_iter, perplexity, n_cores, gpu_device, seed),
          error = function(e) list(embedding = NULL, seconds = NA_real_, error = conditionMessage(e))
        )
        embeddings[[key]] <- res$embedding
        all_rows[[length(all_rows) + 1L]] <- record_row(
          dataset, n, p, classes, paste0("kodama_", classifier), method, backend, M, Tcycle,
          classifier, ksec, res$seconds %||% NA_real_, res$embedding, truth, kk,
          status = if (is.null(res$error)) "ok" else res$error
        )
      }
    }

    columns <- c("classic", classifiers)
    n_cols <- length(columns)
    png_file <- file.path(out_dir, paste0(dataset, "_classic_vs_kodama.png"))
    grDevices::png(png_file, width = max(1500, 850 * n_cols), height = 1600, res = 170)
    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par), add = TRUE)
    par(mfrow = c(length(methods), n_cols), oma = c(0, 0, 4, 0), mar = c(1.5, 1.5, 3.5, 1))
    for (method in methods) {
      for (column in columns) {
        key <- if (column == "classic") paste("classic", method, sep = "_") else paste(column, method, sep = "_")
        label <- if (column == "classic") {
          paste("Classic", if (method == "UMAP") "UMAP" else "openTSNE")
        } else {
          paste("KODAMA", toupper(gsub("_", "-", column)), if (method == "UMAP") "UMAP" else "openTSNE")
        }
        sil <- silhouette_sample(embeddings[[key]], truth)
        plot_embedding(embeddings[[key]], truth, label,
                       sprintf("sil=%.3f", sil), cex = point_cex)
      }
    }
    mtext(sprintf("%s | backend=%s M=%d Tcycle=%d embed-k=%d perplexity=%g",
                  dataset, backend, M, Tcycle, embed_k, perplexity),
          outer = TRUE, line = 1.2, font = 2, cex = 0.95)
    par(old_par)
    grDevices::dev.off()
    message("  wrote ", png_file)

    summary_file <- file.path(out_dir, "visualization_comparison_summary.csv")
    utils::write.csv(do.call(rbind, all_rows), summary_file, row.names = FALSE)
    message("  updated ", summary_file)
  }
}

main()
