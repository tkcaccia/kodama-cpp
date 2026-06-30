suppressPackageStartupMessages({
  library(cluster)
  library(fastEmbedR)
  library(fastPLS)
  library(float)
})

outdir <- Sys.getenv(
  "KODAMA_TIMED_OUTDIR",
  "/mnt/sata_ssd/kodama-cpp-benchmarks/plslda_pca_m100_timed"
)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

extract_layout <- function(x) {
  if (is.list(x) && !is.null(x$layout)) return(as.matrix(x$layout))
  if (is.list(x) && !is.null(x$Y)) return(as.matrix(x$Y))
  as.matrix(x)
}

elapsed <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, seconds = proc.time()[["elapsed"]] - start)
}

ari <- function(x, y) {
  x <- as.integer(as.factor(x))
  y <- as.integer(as.factor(y))
  tab <- table(x, y)
  c2 <- function(z) z * (z - 1) / 2
  sij <- sum(c2(tab))
  si <- sum(c2(rowSums(tab)))
  sj <- sum(c2(colSums(tab)))
  n <- length(x)
  total <- c2(n)
  expected <- si * sj / total
  maximum <- 0.5 * (si + sj)
  if (maximum == expected) 0 else (sij - expected) / (maximum - expected)
}

silhouette_sample <- function(y, labels, n.sample = 5000L, seed = 11L) {
  set.seed(seed)
  idx <- seq_len(nrow(y))
  if (length(idx) > n.sample) idx <- sample(idx, n.sample)
  mean(cluster::silhouette(as.integer(as.factor(labels[idx])), dist(y[idx, , drop = FALSE]))[, 3])
}

cluster_compactness <- function(y, labels, dataset, variant) {
  labels <- as.factor(labels)
  global_radius <- sqrt(rowSums((y - matrix(colMeans(y), nrow(y), 2L, byrow = TRUE))^2))
  global_rms <- sqrt(mean(global_radius^2))
  do.call(rbind, lapply(levels(labels), function(level) {
    idx <- which(labels == level)
    center <- colMeans(y[idx, , drop = FALSE])
    radius <- sqrt(rowSums((y[idx, , drop = FALSE] - matrix(center, length(idx), 2L, byrow = TRUE))^2))
    normalized_rms <- sqrt(mean(radius^2)) / global_rms
    data.frame(
      dataset = dataset,
      variant = variant,
      label = level,
      n = length(idx),
      mean_radius = mean(radius),
      median_radius = median(radius),
      q90_radius = as.numeric(quantile(radius, 0.9)),
      rms_radius = sqrt(mean(radius^2)),
      normalized_rms = normalized_rms,
      compactness = 1 / (1 + normalized_rms),
      stringsAsFactors = FALSE
    )
  }))
}

load_case <- function(dataset) {
  if (dataset == "MNIST") {
    load("/mnt/sata_ssd/fastEmbedR_Data/MNIST/MNIST_float32.RData")
    x <- dataset$data
    labels <- dataset$labels
    spatial <- NULL
  } else {
    path <- switch(
      dataset,
      MERFISH = "/mnt/sata_ssd/KODAMAopt/spatial/MERFISH.RData",
      Br8100 = "/mnt/sata_ssd/KODAMAopt/spatial/Br8100.RData",
      stop("Unknown dataset: ", dataset)
    )
    env <- new.env(parent = emptyenv())
    load(path, envir = env)
    x <- env$dataset$data
    labels <- env$dataset$labels
    spatial <- env$dataset$spatial
  }
  if (inherits(x, "float32") || inherits(x, "float")) x <- float::dbl(x)
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (!is.null(spatial)) {
    spatial <- as.matrix(spatial)
    storage.mode(spatial) <- "double"
  }
  list(data = x, labels = as.factor(labels), spatial = spatial)
}

make_input <- function(x, variant) {
  if (variant == "raw") return(list(x = x, pca_sec = 0))
  npc <- as.integer(sub("pca", "", variant))
  npc <- min(npc, ncol(x), nrow(x) - 1L)
  timed <- elapsed({
    fastPLS::pca(x, ncomp = npc, backend = "cuda", method = "rsvd")$scores
  })
  pc <- as.matrix(timed$value)
  storage.mode(pc) <- "double"
  list(x = pc, pca_sec = timed$seconds)
}

plot_one <- function(y, labels, path, title) {
  labels <- as.factor(labels)
  cols <- grDevices::hcl.colors(nlevels(labels), "Dark 3")[as.integer(labels)]
  idx <- seq_len(nrow(y))
  if (length(idx) > 30000L) {
    set.seed(4)
    idx <- sample(idx, 30000L)
  }
  png(path, width = 1200, height = 1000, res = 150)
  par(mar = c(1, 1, 3, 1))
  plot(y[idx, 1], y[idx, 2], pch = 16, cex = 0.28, col = cols[idx],
       axes = FALSE, xlab = "", ylab = "", main = title)
  box(col = "grey70")
  invisible(dev.off())
}

timing_value <- function(kk, name) {
  if (!is.null(kk$timing) && !is.null(kk$timing[[name]])) as.numeric(kk$timing[[name]]) else NA_real_
}

run_case <- function(dataset, variant) {
  key <- paste(dataset, variant, sep = "_")
  rds <- file.path(outdir, paste0(key, ".rds"))
  summary_csv <- file.path(outdir, paste0(key, "_summary.csv"))
  compact_csv <- file.path(outdir, paste0(key, "_compactness.csv"))
  plot_png <- file.path(outdir, paste0(key, "_umap.png"))
  if (file.exists(summary_csv) && file.exists(compact_csv) && file.exists(plot_png)) {
    cat("SKIP existing", key, "\n")
    return(read.csv(summary_csv, stringsAsFactors = FALSE))
  }

  load_timed <- elapsed(load_case(dataset))
  d <- load_timed$value
  input <- make_input(d$data, variant)
  x <- input$x
  workers <- as.integer(Sys.getenv("KODAMA_CUDA_M_WORKERS", "1"))
  cat(format(Sys.time()), "RUN", key, "M=100 n", nrow(x), "p", ncol(x),
      "workers", workers, "pca_sec", round(input$pca_sec, 3), "\n")

  kodama_timed <- elapsed({
    KODAMA.matrix.cpp(
      x,
      spatial = d$spatial,
      M = 100L,
      Tcycle = 20L,
      ncomp = min(50L, ncol(x)),
      landmarks = 10000L,
      splitting = ifelse(nrow(x) < 40000L, 100L, 300L),
      n.cores = workers,
      graph.neighbors = 100L,
      classifier = "pls_lda",
      backend = "cuda",
      seed = 987L,
      progress = TRUE,
      apply.kodama.dissimilarity = TRUE,
      visual.init = TRUE
    )
  })
  kk <- kodama_timed$value

  classes <- apply(kk$res, 1L, function(z) length(unique(z[z != 0L])))
  largest <- apply(kk$res, 1L, function(z) max(tabulate(as.integer(as.factor(z)))) / length(z))
  aris <- apply(kk$res, 1L, function(z) ari(z, d$labels))

  cat(format(Sys.time()), "UMAP", key, "\n")
  umap_timed <- elapsed({
    extract_layout(KODAMA.umap.knn.fastEmbedR(kk, n.neighbors = 30L, backend = "cuda", seed = 4L))
  })
  y <- umap_timed$value

  compact_timed <- elapsed(cluster_compactness(y, d$labels, dataset, variant))
  comp <- compact_timed$value
  write.csv(comp, compact_csv, row.names = FALSE)

  sil_timed <- elapsed(silhouette_sample(y, d$labels))
  plot_timed <- elapsed(plot_one(
    y, d$labels, plot_png,
    paste(dataset, variant, "sil", round(sil_timed$value, 3), "ARI", round(median(aris), 3))
  ))
  save_timed <- elapsed(saveRDS(list(summary = NULL, compactness = comp, kk = kk, layout = y, labels = d$labels), rds))

  summary <- data.frame(
    dataset = dataset,
    variant = variant,
    M = 100L,
    workers = workers,
    n = nrow(kk$res),
    p = ncol(x),
    load_sec = load_timed$seconds,
    pca_sec = input$pca_sec,
    kodama_input_copy_sec = timing_value(kk, "input_copy_seconds"),
    kodama_spatial_precompute_sec = timing_value(kk, "spatial_precompute_seconds"),
    kodama_graph_sec = timing_value(kk, "graph_seconds"),
    kodama_spatial_graph_sec = timing_value(kk, "spatial_graph_seconds"),
    kodama_optimization_wall_sec = timing_value(kk, "optimization_wall_seconds"),
    kodama_optimization_sum_sec = timing_value(kk, "optimization_sum_seconds"),
    kodama_dissimilarity_sec = timing_value(kk, "dissimilarity_seconds"),
    kodama_total_sec = kk$runtime_seconds,
    kodama_r_elapsed_sec = kodama_timed$seconds,
    umap_sec = umap_timed$seconds,
    compactness_sec = compact_timed$seconds,
    silhouette_sec = sil_timed$seconds,
    plot_sec = plot_timed$seconds,
    save_sec = save_timed$seconds,
    total_sec = load_timed$seconds + input$pca_sec + kodama_timed$seconds +
      umap_timed$seconds + compact_timed$seconds + sil_timed$seconds +
      plot_timed$seconds + save_timed$seconds,
    median_acc = median(kk$acc),
    median_classes = median(classes),
    median_ari = median(aris),
    median_largest = median(largest),
    umap_silhouette = sil_timed$value,
    compactness_median = median(comp$compactness),
    compactness_min = min(comp$compactness),
    stringsAsFactors = FALSE
  )
  saveRDS(list(summary = summary, compactness = comp, kk = kk, layout = y, labels = d$labels), rds)
  write.csv(summary, summary_csv, row.names = FALSE)
  print(summary)
  summary
}

Sys.setenv(
  KODAMA_CPP_ROOT = "/mnt/sata_ssd/kodama-cpp",
  KODAMA_CPP_BUILD_DIR = "/mnt/sata_ssd/kodama-cpp/build-cuda",
  KODAMA_RCPP_REBUILD = Sys.getenv("KODAMA_RCPP_REBUILD", "TRUE")
)
source("/mnt/sata_ssd/kodama-cpp/wrappers/R/kodama_matrix_temp.R")

configs <- data.frame(
  dataset = c("MNIST", "MNIST", "MERFISH", "MERFISH", "MERFISH", "Br8100", "Br8100", "Br8100"),
  variant = c("pca50", "pca20", "raw", "pca50", "pca20", "raw", "pca50", "pca20"),
  stringsAsFactors = FALSE
)

filter <- Sys.getenv("KODAMA_CASES", "")
if (nzchar(filter)) {
  keys <- strsplit(filter, ",", fixed = TRUE)[[1]]
  configs <- configs[paste(configs$dataset, configs$variant, sep = "_") %in% keys, , drop = FALSE]
}

all <- data.frame()
summary_all <- file.path(outdir, "summary_all.csv")
for (i in seq_len(nrow(configs))) {
  res <- tryCatch(
    run_case(configs$dataset[i], configs$variant[i]),
    error = function(e) {
      data.frame(
        dataset = configs$dataset[i],
        variant = configs$variant[i],
        error = conditionMessage(e),
        stringsAsFactors = FALSE
      )
    }
  )
  all <- if (nrow(all)) plyr::rbind.fill(all, res) else res
  write.csv(all, summary_all, row.names = FALSE)
}
print(all)
