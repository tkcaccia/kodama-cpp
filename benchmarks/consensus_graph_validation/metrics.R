# SPDX-FileCopyrightText: 2026 Stefano Cacciatore and kodama-cpp contributors
# SPDX-License-Identifier: MIT

choose2 <- function(x) x * (x - 1) / 2

partition_metrics <- function(truth, prediction) {
  truth <- droplevels(factor(truth))
  prediction <- droplevels(factor(prediction))
  tab <- table(truth, prediction)
  n <- sum(tab)
  row.mass <- rowSums(tab)
  col.mass <- colSums(tab)
  expected <- outer(row.mass, col.mass) / n
  nz <- tab > 0
  mutual.information <- sum((tab[nz] / n) * log(tab[nz] / expected[nz]))
  entropy <- function(mass) {
    probability <- mass[mass > 0] / sum(mass)
    -sum(probability * log(probability))
  }
  truth.entropy <- entropy(row.mass)
  prediction.entropy <- entropy(col.mass)
  nmi <- if (truth.entropy > 0 && prediction.entropy > 0) {
    mutual.information / sqrt(truth.entropy * prediction.entropy)
  } else 1
  homogeneity <- if (truth.entropy > 0) mutual.information / truth.entropy else 1
  completeness <- if (prediction.entropy > 0) mutual.information / prediction.entropy else 1
  pair.total <- choose2(n)
  pair.index <- sum(choose2(tab))
  pair.rows <- sum(choose2(row.mass))
  pair.cols <- sum(choose2(col.mass))
  pair.expected <- if (pair.total > 0) pair.rows * pair.cols / pair.total else 0
  ari.denominator <- 0.5 * (pair.rows + pair.cols) - pair.expected
  ari <- if (ari.denominator == 0) 1 else (pair.index - pair.expected) / ari.denominator
  class.metrics <- do.call(rbind, lapply(seq_len(nrow(tab)), function(i) {
    j <- which.max(tab[i, ])
    true.positive <- tab[i, j]
    recall <- if (row.mass[i] > 0) true.positive / row.mass[i] else NA_real_
    precision <- if (col.mass[j] > 0) true.positive / col.mass[j] else NA_real_
    dice <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
    data.frame(
      truth_class = rownames(tab)[i], matched_cluster = colnames(tab)[j],
      support = row.mass[i], recall = recall, precision = precision, dice = dice
    )
  }))
  list(
    summary = data.frame(
      ari = ari, nmi = nmi, homogeneity = homogeneity,
      completeness = completeness, clusters = nlevels(prediction)
    ),
    per_class = class.metrics
  )
}

sampled_silhouette <- function(layout, labels, maximum = 5000L, seed = 909L) {
  labels <- as.integer(factor(labels))
  keep <- which(rowSums(is.finite(layout)) == ncol(layout))
  if (length(keep) < 3L || length(unique(labels[keep])) < 2L) return(NA_real_)
  if (length(keep) > maximum) {
    set.seed(seed)
    keep <- sort(sample(keep, maximum))
  }
  mean(cluster::silhouette(labels[keep], stats::dist(layout[keep, , drop = FALSE]))[, 3L])
}

graph_edge_table <- function(graph) {
  graph <- KODAMA::KODAMA.graph.materialize(graph)
  indices <- as.matrix(graph$indices)
  distances <- as.matrix(graph$distances)
  finite.indices <- indices[is.finite(distances) & !is.na(indices)]
  one.based <- length(finite.indices) && min(finite.indices) >= 1L &&
    max(finite.indices) <= nrow(indices)
  from <- rep.int(seq_len(nrow(indices)), ncol(indices))
  to <- as.integer(indices)
  if (!one.based) to <- to + 1L
  distance <- as.numeric(distances)
  keep <- is.finite(distance) & to >= 1L & to <= nrow(indices) & from != to
  data.frame(from = from[keep], to = to[keep], distance = distance[keep])
}

graph_conductance <- function(edges, labels) {
  labels <- as.integer(factor(labels))
  weight <- 1 / (1 + pmax(edges$distance, 0))
  total.volume <- sum(weight)
  values <- vapply(sort(unique(labels)), function(group) {
    source.in <- labels[edges$from] == group
    volume <- sum(weight[source.in])
    cut <- sum(weight[source.in & labels[edges$to] != group])
    denominator <- min(volume, total.volume - volume)
    if (denominator > 0) cut / denominator else 0
  }, numeric(1))
  c(mean = mean(values), maximum = max(values))
}

boundary_f1 <- function(edges, truth, prediction) {
  truth.boundary <- truth[edges$from] != truth[edges$to]
  prediction.boundary <- prediction[edges$from] != prediction[edges$to]
  true.positive <- sum(truth.boundary & prediction.boundary)
  false.positive <- sum(!truth.boundary & prediction.boundary)
  false.negative <- sum(truth.boundary & !prediction.boundary)
  precision <- true.positive / max(1, true.positive + false.positive)
  recall <- true.positive / max(1, true.positive + false.negative)
  f1 <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
  c(precision = precision, recall = recall, f1 = f1)
}

sample_agreement <- function(edges, all.labels, maximum = 100000L, seed = 812L) {
  if (nrow(edges) > maximum) {
    set.seed(seed)
    edges <- edges[sample.int(nrow(edges), maximum), , drop = FALSE]
  }
  agreement <- vapply(seq_len(nrow(edges)), function(i) {
    mean(all.labels[, edges$from[i]] == all.labels[, edges$to[i]])
  }, numeric(1))
  stats::quantile(agreement, c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1), names = FALSE)
}

per_sample_metrics <- function(truth, prediction, samples) {
  samples <- if (is.null(samples)) factor(rep("all", length(truth))) else factor(samples)
  do.call(rbind, lapply(levels(samples), function(sample.name) {
    selected <- samples == sample.name
    out <- partition_metrics(truth[selected], prediction[selected])$summary
    out$sample <- sample.name
    out$n <- sum(selected)
    out[, c("sample", "n", "ari", "nmi", "homogeneity", "completeness", "clusters")]
  }))
}
