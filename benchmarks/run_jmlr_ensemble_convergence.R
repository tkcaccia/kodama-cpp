#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

options(stringsAsFactors = FALSE)

value_or <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}

csv_chr <- function(name, default) {
  trimws(strsplit(value_or(name, default), ",", fixed = TRUE)[[1L]])
}

csv_int <- function(name, default) {
  as.integer(csv_chr(name, default))
}

safe_correlation <- function(a, b) {
  if (stats::sd(a) == 0 || stats::sd(b) == 0) return(NA_real_)
  stats::cor(a, b)
}

result_dir <- normalizePath(
  value_or("KODAMA_CONVERGENCE_RESULT_DIR", "jmlr-m-tcycle-sensitivity/runs"),
  mustWork = TRUE
)
out_dir <- normalizePath(
  value_or("KODAMA_CONVERGENCE_OUT", dirname(result_dir)),
  mustWork = FALSE
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- csv_chr("KODAMA_CONVERGENCE_DATASETS", "MetRef")
classifiers <- csv_chr("KODAMA_CONVERGENCE_CLASSIFIERS", "knn,pls_lda")
m_values <- csv_int("KODAMA_CONVERGENCE_M_VALUES", "10,20,50,100")
max_edges <- as.integer(value_or("KODAMA_CONVERGENCE_MAX_EDGES", "100000"))
seed <- as.integer(value_or("KODAMA_CONVERGENCE_SEED", "731"))

rows <- list()

for (dataset in datasets) {
  for (classifier in classifiers) {
    path <- file.path(
      result_dir,
      sprintf("%s__%s__M100__T100.rds", dataset, classifier)
    )
    if (!file.exists(path)) stop("Missing M=100 result: ", path)
    result <- readRDS(path)
    labels <- as.matrix(result$res)
    indices <- as.matrix(result$knn$indices)
    if (length(indices) && min(indices, na.rm = TRUE) == 0L) indices <- indices + 1L

    edge_i <- rep(seq_len(nrow(indices)), times = ncol(indices))
    edge_j <- as.integer(indices)
    valid <- is.finite(edge_j) & edge_j >= 1L & edge_j <= ncol(labels) & edge_i != edge_j
    edge_i <- edge_i[valid]
    edge_j <- edge_j[valid]
    available_edges <- length(edge_i)

    if (available_edges > max_edges) {
      set.seed(seed + sum(utf8ToInt(paste(dataset, classifier))))
      keep <- sort(sample.int(available_edges, max_edges))
      edge_i <- edge_i[keep]
      edge_j <- edge_j[keep]
    }

    total_runs <- nrow(labels)
    same <- matrix(FALSE, nrow = length(edge_i), ncol = total_runs)
    for (run in seq_len(total_runs)) {
      same[, run] <- labels[run, edge_i] == labels[run, edge_j]
    }
    full_agreement <- rowMeans(same)

    for (m in m_values[m_values <= total_runs]) {
      prefix_agreement <- rowMeans(same[, seq_len(m), drop = FALSE])
      prefix_rmse <- sqrt(mean((prefix_agreement - full_agreement)^2))
      prefix_correlation <- safe_correlation(prefix_agreement, full_agreement)

      pair_count <- floor(total_runs / (2L * m))
      pair_rmse <- pair_correlation <- numeric(0L)
      if (pair_count > 0L) {
        for (pair in seq_len(pair_count)) {
          first <- ((pair - 1L) * 2L * m + 1L):((pair - 1L) * 2L * m + m)
          second <- (max(first) + 1L):(max(first) + m)
          first_agreement <- rowMeans(same[, first, drop = FALSE])
          second_agreement <- rowMeans(same[, second, drop = FALSE])
          pair_rmse <- c(pair_rmse, sqrt(mean((first_agreement - second_agreement)^2)))
          pair_correlation <- c(
            pair_correlation,
            safe_correlation(first_agreement, second_agreement)
          )
        }
      }

      rows[[length(rows) + 1L]] <- data.frame(
        dataset = dataset,
        classifier = classifier,
        M = m,
        total_runs = total_runs,
        available_edges = available_edges,
        sampled_edges = length(edge_i),
        prefix_rmse_to_M100 = prefix_rmse,
        prefix_correlation_to_M100 = prefix_correlation,
        independent_pair_count = pair_count,
        independent_pair_rmse = if (length(pair_rmse)) mean(pair_rmse) else NA_real_,
        independent_pair_correlation = if (length(pair_correlation)) {
          mean(pair_correlation, na.rm = TRUE)
        } else {
          NA_real_
        },
        worst_case_agreement_se = 0.5 / sqrt(m),
        worst_case_pairwise_rmse = sqrt(0.5 / m)
      )
    }
  }
}

summary <- do.call(rbind, rows)
summary_path <- file.path(out_dir, "ensemble_convergence_summary.csv")
utils::write.csv(summary, summary_path, row.names = FALSE)

groups <- interaction(summary$dataset, summary$classifier, drop = TRUE)
group_levels <- levels(groups)
colors <- setNames(grDevices::hcl.colors(length(group_levels), "Dark 3"), group_levels)

grDevices::png(
  file.path(out_dir, "ensemble_convergence.png"),
  width = 1800,
  height = 800,
  res = 160
)
old <- graphics::par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

graphics::plot(
  range(summary$M),
  range(summary$prefix_rmse_to_M100, finite = TRUE),
  type = "n",
  xlab = "M",
  ylab = "RMSE from M=100 edge agreement",
  main = "KODAMA graph convergence"
)
for (group in group_levels) {
  block <- summary[groups == group, , drop = FALSE]
  block <- block[order(block$M), , drop = FALSE]
  graphics::lines(
    block$M,
    block$prefix_rmse_to_M100,
    type = "b",
    pch = 16,
    lwd = 2,
    col = colors[[group]]
  )
}
graphics::legend(
  "topright",
  legend = group_levels,
  col = colors,
  lwd = 2,
  pch = 16,
  cex = 0.8,
  bg = "white"
)

graphics::plot(
  range(summary$M),
  c(0, max(summary$worst_case_agreement_se)),
  type = "n",
  xlab = "M",
  ylab = "Maximum standard error",
  main = "Monte Carlo precision of edge agreement"
)
theory <- unique(summary[c("M", "worst_case_agreement_se")])
theory <- theory[order(theory$M), , drop = FALSE]
graphics::lines(
  theory$M,
  theory$worst_case_agreement_se,
  type = "b",
  pch = 16,
  lwd = 2,
  col = "#2C7FB8"
)

graphics::par(old)
grDevices::dev.off()

message("Ensemble-convergence summary: ", summary_path)
