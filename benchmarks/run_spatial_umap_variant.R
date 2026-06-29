suppressPackageStartupMessages({
  library(uwot)
  library(ggplot2)
  library(cluster)
})

args <- commandArgs(trailingOnly = TRUE)
kv <- strsplit(args, "=", fixed = TRUE)
opt <- setNames(vapply(kv, function(x) if (length(x) > 1L) x[[2L]] else "", ""), vapply(kv, `[[`, "", 1L))

src_root <- opt[["src"]]
out_root <- opt[["out"]]
variant <- if (!is.na(opt[["variant"]])) opt[["variant"]] else "variant"
init_kind <- if (!is.na(opt[["init"]])) opt[["init"]] else "spectral"
n_neighbors <- as.integer(if (!is.na(opt[["n_neighbors"]])) opt[["n_neighbors"]] else "0")
n_epochs <- as.integer(if (!is.na(opt[["n_epochs"]])) opt[["n_epochs"]] else "200")
min_dist <- as.numeric(if (!is.na(opt[["min_dist"]])) opt[["min_dist"]] else "0.1")

if (is.na(src_root) || is.na(out_root)) {
  stop("Required arguments: src=... out=...")
}

dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

scaled2 <- function(x) {
  x <- as.matrix(x[, 1:2, drop = FALSE])
  x <- scale(x)
  x[!is.finite(x)] <- 0
  x
}

sil2d <- function(emb, lab, seed = 543210, n_sample = 5000L) {
  ok <- !is.na(lab)
  emb <- emb[ok, , drop = FALSE]
  lab <- droplevels(factor(lab[ok]))
  if (nlevels(lab) < 2L) return(NA_real_)
  set.seed(seed)
  if (nrow(emb) > n_sample) {
    ii <- sort(sample(seq_len(nrow(emb)), n_sample))
    emb <- emb[ii, , drop = FALSE]
    lab <- lab[ii]
  }
  mean(cluster::silhouette(as.integer(lab), dist(emb))[, "sil_width"])
}

make_init <- function(dataset) {
  if (init_kind == "spectral") return("spectral")
  if (dataset == "MERFISH") {
    load("/mnt/sata_ssd/kodama-cpp-benchmarks/MERFISH-input.RData")
    pca_init <- scaled2(pca.PM)
    spatial_init <- scaled2(xyz)
  } else {
    load("/mnt/sata_ssd/kodama-cpp-benchmarks/DLFPC-Br8100-input.RData")
    pca_init <- scaled2(pca_Br8100)
    spatial_init <- scaled2(xy_Br8100)
  }
  if (init_kind == "pca12") return(pca_init)
  if (init_kind == "spatial12") return(spatial_init)
  if (init_kind == "pca_spatial_50") return(scaled2(0.5 * pca_init + 0.5 * spatial_init))
  if (init_kind == "pca_spatial_25") return(scaled2(0.75 * pca_init + 0.25 * spatial_init))
  if (init_kind == "pca_spatial_75") return(scaled2(0.25 * pca_init + 0.75 * spatial_init))
  stop("Unknown init=", init_kind)
}

run_one <- function(dataset, classifier) {
  in_dir <- file.path(src_root, paste(dataset, classifier, sep = "_"))
  out <- file.path(out_root, paste(dataset, classifier, sep = "_"))
  dir.create(out, recursive = TRUE, showWarnings = FALSE)

  obj <- readRDS(file.path(in_dir, "result.rds"))
  kk <- obj$kk
  labels <- obj$labels
  pred <- obj$best_labels
  idx <- as.matrix(kk$knn_Rnanoflann$indices)
  dst <- as.matrix(kk$knn_Rnanoflann$distances)
  idx[idx < 1L] <- 1L
  dst[!is.finite(dst)] <- max(dst[is.finite(dst)], na.rm = TRUE)

  nn <- ncol(idx)
  if (n_neighbors > 0L && n_neighbors < nn) {
    idx <- idx[, seq_len(n_neighbors), drop = FALSE]
    dst <- dst[, seq_len(n_neighbors), drop = FALSE]
    nn <- n_neighbors
  }

  init <- make_init(dataset)
  set.seed(543210)
  t0 <- proc.time()[["elapsed"]]
  emb <- uwot::umap(
    X = NULL,
    nn_method = list(idx = idx, dist = dst),
    n_neighbors = nn,
    n_epochs = n_epochs,
    min_dist = min_dist,
    n_threads = 4,
    init = init,
    verbose = FALSE
  )
  elapsed <- proc.time()[["elapsed"]] - t0

  old <- read.csv(file.path(in_dir, "metrics.csv"))
  metrics <- old
  metrics$variant <- variant
  metrics$umap_init <- init_kind
  metrics$umap_n_neighbors <- nn
  metrics$umap_n_epochs <- n_epochs
  metrics$umap_min_dist <- min_dist
  metrics$umap_elapsed_sec <- elapsed
  metrics$silhouette_labels_on_kodama_umap <- sil2d(emb, labels)
  write.csv(metrics, file.path(out, "metrics.csv"), row.names = FALSE, quote = TRUE)

  plot_df <- data.frame(u1 = emb[, 1], u2 = emb[, 2], label = labels, kodama = factor(pred))
  base_theme <- theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank(), legend.position = "right")
  title <- paste(dataset, classifier, variant)
  ggsave(file.path(out, "kodama_umap_labels_truth.png"),
    ggplot(plot_df, aes(u1, u2, color = label)) +
      geom_point(size = 0.35, alpha = 0.85) +
      coord_equal() + base_theme +
      labs(title = title, color = "Label"),
    width = 8, height = 6, dpi = 200
  )
  ggsave(file.path(out, "kodama_umap_labels_kodama.png"),
    ggplot(plot_df, aes(u1, u2, color = kodama)) +
      geom_point(size = 0.35, alpha = 0.85, show.legend = FALSE) +
      coord_equal() + base_theme +
      labs(title = paste(title, "KODAMA labels")),
    width = 8, height = 6, dpi = 200
  )
  metrics
}

runs <- list(
  run_one("MERFISH", "knn"),
  run_one("MERFISH", "pls_lda"),
  run_one("Br8100", "knn"),
  run_one("Br8100", "pls_lda")
)
all_cols <- unique(unlist(lapply(runs, names)))
runs <- lapply(runs, function(x) {
  missing <- setdiff(all_cols, names(x))
  for (m in missing) x[[m]] <- NA
  x[, all_cols, drop = FALSE]
})
res <- do.call(rbind, runs)
write.csv(res, file.path(out_root, "summary_metrics.csv"), row.names = FALSE, quote = TRUE)
print(res[, c("dataset", "classifier", "silhouette_labels_on_kodama_umap", "umap_elapsed_sec")])
cat("mean silhouette", mean(res$silhouette_labels_on_kodama_umap, na.rm = TRUE), "\n")
