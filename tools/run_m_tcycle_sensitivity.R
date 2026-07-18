# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

suppressPackageStartupMessages({
  library(cluster)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

split_env_int <- function(name, default) {
  raw <- Sys.getenv(name, default)
  out <- as.integer(trimws(strsplit(raw, ",", fixed = TRUE)[[1]]))
  out[!is.na(out)]
}

split_env_chr <- function(name, default) {
  raw <- Sys.getenv(name, default)
  out <- trimws(strsplit(raw, ",", fixed = TRUE)[[1]])
  out[nzchar(out)]
}

elapsed <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, seconds = proc.time()[["elapsed"]] - t0)
}

ari <- function(x, y) {
  x <- as.integer(as.factor(x))
  y <- as.integer(as.factor(y))
  tab <- table(x, y)
  c2 <- function(z) z * (z - 1) / 2
  sij <- sum(c2(tab))
  si <- sum(c2(rowSums(tab)))
  sj <- sum(c2(colSums(tab)))
  expected <- si * sj / c2(length(x))
  maximum <- 0.5 * (si + sj)
  if (maximum == expected) 0 else (sij - expected) / (maximum - expected)
}

classes_by_run <- function(res) {
  apply(res, 1L, function(z) length(unique(z[z != 0L])))
}

largest_by_run <- function(res) {
  apply(res, 1L, function(z) max(tabulate(as.integer(as.factor(z)))) / length(z))
}

ari_by_run <- function(res, labels) {
  apply(res, 1L, function(z) ari(z, labels))
}

best_index <- function(x) {
  which.max(as.numeric(x))
}

timing_value <- function(kk, name) {
  if (!is.null(kk$timing) && !is.null(kk$timing[[name]])) {
    as.numeric(kk$timing[[name]])
  } else {
    NA_real_
  }
}

silhouette_sample <- function(y, labels, n_sample = 5000L, seed = 11L) {
  labels <- as.factor(labels)
  if (nlevels(labels) < 2L) return(NA_real_)
  set.seed(seed)
  idx <- seq_len(nrow(y))
  if (length(idx) > n_sample) idx <- sample(idx, n_sample)
  mean(cluster::silhouette(as.integer(as.factor(labels[idx])),
                           dist(y[idx, , drop = FALSE]))[, 3])
}

load_case <- function(name, cfg) {
  if (!file.exists(cfg$path)) stop("Missing dataset file: ", cfg$path)
  env <- new.env(parent = emptyenv())
  load(cfg$path, envir = env)
  if (exists("dataset", env, inherits = FALSE)) {
    obj <- env$dataset
    x <- obj$data
    labels <- obj$labels
    spatial <- obj$spatial %||% NULL
  } else {
    x <- env$data %||% env$x %||% env$X %||% env$pca.PM %||% env$pca_Br8100
    labels <- env$labels %||% env$label %||% env$class %||% env$tissue_segments %||% env$labels_Br8100
    spatial <- env$spatial %||% env$xy %||% env$xyz %||% env$xy_Br8100 %||% NULL
  }
  if (inherits(x, "float32") || inherits(x, "float")) {
    if (requireNamespace("float", quietly = TRUE)) x <- float::dbl(x)
  }
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (!is.null(spatial)) {
    spatial <- as.matrix(spatial)
    storage.mode(spatial) <- "double"
  }
  list(data = x, labels = as.factor(labels), spatial = spatial)
}

write_if_new <- function(x, path) {
  write.csv(x, path, row.names = FALSE)
  invisible(path)
}

plot_sensitivity <- function(summary, outdir) {
  if (nrow(summary) == 0L) return(invisible(NULL))
  png(file.path(outdir, "m_tcycle_sensitivity_curves.png"), width = 2200, height = 1400, res = 170)
  op <- par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
  on.exit({
    par(op)
    dev.off()
  }, add = TRUE)

  cols <- c("#2B8CBE", "#E34A33", "#31A354", "#756BB1")
  groups <- unique(paste(summary$dataset, summary$classifier, sep = " / "))
  group_col <- setNames(rep(cols, length.out = length(groups)), groups)

  draw_panel <- function(sweep, xvar, yvar, ylab, main) {
    d <- summary[summary$sweep == sweep, , drop = FALSE]
    if (nrow(d) == 0L) {
      plot.new()
      title(main)
      return()
    }
    x <- d[[xvar]]
    y <- d[[yvar]]
    plot(range(x), range(y, finite = TRUE), type = "n", xlab = xvar, ylab = ylab, main = main)
    for (g in groups) {
      dg <- d[paste(d$dataset, d$classifier, sep = " / ") == g, , drop = FALSE]
      if (nrow(dg) == 0L) next
      dg <- dg[order(dg[[xvar]]), , drop = FALSE]
      lines(dg[[xvar]], dg[[yvar]], type = "b", pch = 16, lwd = 2, col = group_col[[g]])
    }
    legend("bottomright", legend = groups, col = group_col, lwd = 2, pch = 16,
           cex = 0.72, bg = "white")
  }

  draw_panel("tcycle", "Tcycle", "best_accuracy", "best CV accuracy", "Tcycle sweep, M = 100")
  draw_panel("tcycle", "Tcycle", "elapsed_sec", "elapsed seconds", "Tcycle cost")
  draw_panel("M", "M", "best_accuracy", "best CV accuracy", "M sweep, Tcycle = 100")
  draw_panel("M", "M", "best_ari", "best-run ARI", "M quality")
  mtext("KODAMA sensitivity to M and Tcycle", outer = TRUE, font = 2)
}

root <- Sys.getenv("KODAMA_CPP_ROOT", "/mnt/sata_ssd/kodama-cpp")
build <- Sys.getenv("KODAMA_CPP_BUILD_DIR", file.path(root, "build-cuda"))
env_dir <- Sys.getenv("KODAMA_ENV_DIR", "/home/chiamaka/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs")
outdir <- Sys.getenv("KODAMA_SENS_OUTDIR", "/mnt/sata_ssd/KODAMAopt/m_tcycle_sensitivity")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

Sys.setenv(
  CONDA_PREFIX = env_dir,
  LD_LIBRARY_PATH = paste(
    file.path(env_dir, "lib"),
    file.path(env_dir, "targets/x86_64-linux/lib"),
    "/usr/local/cuda-13.0/targets/x86_64-linux/lib",
    Sys.getenv("LD_LIBRARY_PATH"),
    sep = ":"
  ),
  KODAMA_CPP_ROOT = root,
  KODAMA_CPP_BUILD_DIR = build,
  KODAMA_RCPP_REBUILD = Sys.getenv("KODAMA_RCPP_REBUILD", "FALSE")
)

source(file.path(root, "wrappers", "R", "kodama_matrix_temp.R"))

datasets <- list(
  MERFISH = list(path = "/mnt/sata_ssd/KODAMAopt/spatial/MERFISH.RData"),
  Br8100 = list(path = "/mnt/sata_ssd/KODAMAopt/spatial/Br8100.RData"),
  MNIST = list(path = "/mnt/sata_ssd/fastEmbedR_Data/MNIST/MNIST_float32.RData")
)
selected_datasets <- split_env_chr("KODAMA_SENS_DATASETS", "MERFISH,Br8100")
datasets <- datasets[names(datasets) %in% selected_datasets]
if (length(datasets) == 0L) stop("No selected datasets are available.")

classifiers <- split_env_chr("KODAMA_SENS_CLASSIFIERS", "knn,pls_lda")
M_values <- split_env_int("KODAMA_SENS_M_VALUES", "20,50,100")
Tcycle_values <- split_env_int("KODAMA_SENS_TCYCLE_VALUES", "20,50,100")
M_anchor <- as.integer(Sys.getenv("KODAMA_SENS_M_ANCHOR", "100"))
Tcycle_anchor <- as.integer(Sys.getenv("KODAMA_SENS_TCYCLE_ANCHOR", "100"))
landmarks <- as.integer(Sys.getenv("KODAMA_SENS_LANDMARKS", "100000"))
graph_neighbors <- as.integer(Sys.getenv("KODAMA_SENS_GRAPH_NEIGHBORS", "100"))
knn_k <- as.integer(Sys.getenv("KODAMA_SENS_KNN_K", "30"))
ncomp_default <- as.integer(Sys.getenv("KODAMA_SENS_NCOMP", "50"))
n_cores <- as.integer(Sys.getenv("KODAMA_SENS_N_CORES", "0"))
seed <- as.integer(Sys.getenv("KODAMA_SENS_SEED", "1234"))
backend <- Sys.getenv("KODAMA_SENS_BACKEND", "cuda")

grid_t <- data.frame(M = M_anchor, Tcycle = Tcycle_values, sweep = "tcycle")
grid_m <- data.frame(M = M_values, Tcycle = Tcycle_anchor, sweep = "M")
grid <- rbind(grid_t, grid_m)
grid <- grid[order(grid$M, grid$Tcycle), , drop = FALSE]

cat(format(Sys.time()), "Sensitivity output:", outdir, "\n")
cat(format(Sys.time()), "Grid:\n")
print(grid)

summaries <- list()

for (dataset in names(datasets)) {
  loaded <- elapsed(load_case(dataset, datasets[[dataset]]))
  d <- loaded$value
  splitting <- ifelse(nrow(d$data) < 40000L, 100L, 300L)
  ncomp <- min(ncomp_default, ncol(d$data), max(1L, nrow(d$data) - 1L))
  for (classifier in classifiers) {
    for (i in seq_len(nrow(grid))) {
      M <- grid$M[[i]]
      Tcycle <- grid$Tcycle[[i]]
      sweep <- grid$sweep[[i]]
      key <- sprintf("%s_%s_M%d_T%d", dataset, classifier, M, Tcycle)
      summary_key <- paste(key, sweep, sep = "_")
      rds_path <- file.path(outdir, paste0(key, ".rds"))
      csv_path <- file.path(outdir, paste0(key, "_summary.csv"))
      if (file.exists(rds_path) && file.exists(csv_path)) {
        cat(format(Sys.time()), "SKIP", key, "\n")
        cached <- read.csv(csv_path, stringsAsFactors = FALSE)
        cached$sweep <- sweep
        summaries[[summary_key]] <- cached
        next
      }

      cat(format(Sys.time()), "RUN", key,
          "n", nrow(d$data), "p", ncol(d$data),
          "landmarks", landmarks, "splitting", splitting,
          "ncomp", ncomp, "knn.k", knn_k, "backend", backend, "\n")
      run <- elapsed({
        KODAMA.matrix.cpp(
          d$data,
          spatial = d$spatial,
          M = M,
          Tcycle = Tcycle,
          ncomp = ncomp,
          landmarks = landmarks,
          splitting = splitting,
          n.cores = n_cores,
          graph.neighbors = graph_neighbors,
          knn.k = knn_k,
          classifier = classifier,
          backend = backend,
          seed = seed,
          progress = TRUE,
          apply.kodama.dissimilarity = TRUE,
          visual.init = TRUE
        )
      })
      kk <- run$value
      saveRDS(kk, rds_path)

      acc <- as.numeric(kk$acc)
      aris <- ari_by_run(kk$res, d$labels)
      classes <- classes_by_run(kk$res)
      largest <- largest_by_run(kk$res)
      best <- best_index(acc)
      row <- data.frame(
        dataset = dataset,
        classifier = classifier,
        sweep = sweep,
        M = M,
        Tcycle = Tcycle,
        n = nrow(d$data),
        p = ncol(d$data),
        landmarks = landmarks,
        result_samples = ncol(kk$res),
        splitting = splitting,
        ncomp = ncomp,
        knn_k = knn_k,
        backend = backend,
        selected_workers = as.integer(kk$n.cores %||% NA_integer_),
        input_copy_sec = timing_value(kk, "input_copy_seconds"),
        graph_sec = timing_value(kk, "graph_seconds"),
        optimization_wall_sec = timing_value(kk, "optimization_wall_seconds"),
        dissimilarity_sec = timing_value(kk, "dissimilarity_seconds"),
        kodama_runtime_sec = as.numeric(kk$runtime_seconds),
        elapsed_sec = run$seconds,
        best_accuracy = max(acc),
        median_accuracy = median(acc),
        min_accuracy = min(acc),
        best_run = best,
        best_ari = aris[[best]],
        median_ari = median(aris),
        best_classes = classes[[best]],
        median_classes = median(classes),
        min_classes = min(classes),
        max_classes = max(classes),
        best_largest_fraction = largest[[best]],
        median_largest_fraction = median(largest),
        stringsAsFactors = FALSE
      )
      write_if_new(row, csv_path)
      print(row)
      summaries[[summary_key]] <- row
      write_if_new(do.call(rbind, summaries), file.path(outdir, "summary_partial.csv"))
      plot_sensitivity(do.call(rbind, summaries), outdir)
    }
  }
}

summary <- do.call(rbind, summaries)
summary <- summary[order(summary$dataset, summary$classifier, summary$M, summary$Tcycle), ]
write_if_new(summary, file.path(outdir, "summary_all.csv"))
plot_sensitivity(summary, outdir)

cat(format(Sys.time()), "DONE\n")
print(summary)
