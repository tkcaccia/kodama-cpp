# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

kodama_spatial_palette <- c(
  "#0000b6", "#81b29a", "#f2cc8f", "#e07a5f",
  "#cc00b6", "#81ccff", "#33b233", "#f28e2b"
)

kodama_timed <- function(name, expression, timings, log_file = NULL) {
  gc()
  started <- proc.time()[["elapsed"]]
  value <- force(expression)
  elapsed <- proc.time()[["elapsed"]] - started
  timings[[name]] <- elapsed
  line <- sprintf("%s %s completed in %.3f seconds", Sys.time(), name, elapsed)
  cat(line, "\n")
  if (!is.null(log_file)) cat(line, "\n", file = log_file, append = TRUE)
  list(value = value, timings = timings)
}

kodama_align_labels <- function(predicted, reference) {
  predicted <- droplevels(as.factor(predicted))
  reference <- droplevels(as.factor(reference))
  counts <- table(predicted, reference)
  padded <- matrix(0, max(dim(counts)), max(dim(counts)))
  padded[seq_len(nrow(counts)), seq_len(ncol(counts))] <- counts
  assignment <- clue::solve_LSAP(padded, maximum = TRUE)
  mapping <- levels(reference)[as.integer(assignment)[seq_len(nlevels(predicted))]]
  factor(mapping[as.integer(predicted)], levels = levels(reference))
}

fastembedr_exact_community_cut <- function(graph, method, nclusters,
                                           seed = 1L) {
  method <- match.arg(method, c("louvain", "leiden"))
  nclusters <- as.integer(nclusters)
  cache <- new.env(parent = emptyenv())
  evaluate <- function(resolution) {
    key <- format(resolution, digits = 17)
    if (!exists(key, envir = cache, inherits = FALSE)) {
      assign(key, fastEmbedR::graph_cluster(
        graph, method = method, backend = "cpu", resolution = resolution,
        n_iterations = 20L, n_runs = 3L, seed = as.integer(seed)
      ), envir = cache)
    }
    get(key, envir = cache, inherits = FALSE)
  }

  fits <- list(evaluate(1))
  resolutions <- 1
  if (fits[[1L]]$n_communities < nclusters) {
    while (tail(fits, 1L)[[1L]]$n_communities < nclusters &&
           tail(resolutions, 1L) < 1024) {
      resolutions <- c(resolutions, tail(resolutions, 1L) * 2)
      fits <- c(fits, list(evaluate(tail(resolutions, 1L))))
    }
  } else if (fits[[1L]]$n_communities > nclusters) {
    while (tail(fits, 1L)[[1L]]$n_communities > nclusters &&
           tail(resolutions, 1L) > 2^-10) {
      resolutions <- c(resolutions, tail(resolutions, 1L) / 2)
      fits <- c(fits, list(evaluate(tail(resolutions, 1L))))
    }
  }

  counts <- vapply(fits, `[[`, integer(1), "n_communities")
  if (!any(counts == nclusters) && min(counts) < nclusters &&
      max(counts) > nclusters) {
    low <- max(resolutions[counts < nclusters])
    high <- min(resolutions[counts > nclusters])
    for (iteration in seq_len(30L)) {
      middle <- sqrt(low * high)
      fit <- evaluate(middle)
      fits <- c(fits, list(fit))
      resolutions <- c(resolutions, middle)
      counts <- c(counts, fit$n_communities)
      if (fit$n_communities == nclusters) break
      if (fit$n_communities < nclusters) low <- middle else high <- middle
    }
  }

  exact <- which(counts == nclusters)
  if (!length(exact)) {
    stop(
      "fastEmbedR ", method, " did not produce exactly ", nclusters,
      " communities in the deterministic resolution search; observed: ",
      paste(sort(unique(counts)), collapse = ", "), call. = FALSE
    )
  }
  best <- exact[[which.max(vapply(fits[exact], `[[`, numeric(1), "modularity"))]]
  fits[[best]]
}

kodama_cluster_and_refine <- function(embedding, spatial, samples, nclusters,
                                      snn_k, method = c("louvain", "leiden"),
                                      workers = 4L, seed = 1L) {
  method <- match.arg(method)
  graph <- fastEmbedR::knn_graph(
    embedding, k = as.integer(snn_k), backend = "cpu", metric = "euclidean",
    weight = "snn", n.cores = as.integer(workers)
  )
  clustering <- fastembedr_exact_community_cut(
    graph, method = method, nclusters = nclusters, seed = seed
  )
  initial <- as.factor(clustering$membership)
  stopifnot(nlevels(droplevels(initial)) == as.integer(nclusters))
  refined <- fibermargin::refine_spatial_labels(
    spatial, initial, samples = samples, workers = as.integer(workers)
  )
  list(initial = initial, refined = refined, graph = graph,
       clustering = clustering)
}

kodama_slide_metrics <- function(dataset, truth, samples, initial, refined,
                                 historical) {
  do.call(rbind, lapply(levels(as.factor(samples)), function(slide) {
    selected <- samples == slide
    target <- unname(historical[[slide]])
    data.frame(
      dataset = dataset,
      slide = slide,
      n = sum(selected),
      initial_ari = mclust::adjustedRandIndex(truth[selected], initial[selected]),
      refined_ari = mclust::adjustedRandIndex(truth[selected], refined[selected]),
      historical_ari = target,
      historical_gap = mclust::adjustedRandIndex(
        truth[selected], refined[selected]
      ) - target,
      changed_fraction = mean(attr(refined, "changed")[selected])
    )
  }))
}

kodama_embedding_plot <- function(embedding, labels, title, palette = kodama_spatial_palette) {
  labels <- as.factor(labels)
  frame <- data.frame(x = embedding[, 1L], y = embedding[, 2L], label = labels)
  ggplot2::ggplot(frame, ggplot2::aes(x, y, colour = label)) +
    ggplot2::geom_point(size = 0.18, alpha = 0.9) +
    ggplot2::scale_colour_manual(
      values = setNames(palette[seq_len(nlevels(labels))], levels(labels)),
      drop = FALSE
    ) +
    ggplot2::coord_equal() +
    ggplot2::labs(title = title, colour = NULL) +
    ggplot2::theme_bw(base_size = 9) +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      axis.title = ggplot2::element_blank(),
      axis.text = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      legend.position = "none",
      plot.title = ggplot2::element_text(hjust = 0.5)
    )
}

kodama_spatial_panel <- function(dataset, embedding, spatial, samples, truth,
                                 initial, refined, metrics, output_file,
                                 M, Tcycle,
                                 palette = kodama_spatial_palette) {
  initial_aligned <- kodama_align_labels(initial, truth)
  refined_aligned <- kodama_align_labels(refined, truth)
  plots <- list(
    kodama_embedding_plot(embedding, truth, "KODAMA UMAP by truth", palette),
    kodama_embedding_plot(
      embedding, initial_aligned,
      sprintf("Graph clustering | mean ARI %.3f", mean(metrics$initial_ari)), palette
    ),
    kodama_embedding_plot(
      embedding, refined_aligned,
      sprintf("FiberMargin | mean ARI %.3f", mean(metrics$refined_ari)), palette
    ),
    KODAMAextra::plot_slide(
      spatial, samples, truth, col = palette, nrow = 1L, size.dot = 0.3
    ) + ggplot2::ggtitle("Tissue by truth"),
    KODAMAextra::plot_slide(
      spatial, samples, initial_aligned, col = palette, nrow = 1L, size.dot = 0.3
    ) + ggplot2::ggtitle("Tissue: graph clustering"),
    KODAMAextra::plot_slide(
      spatial, samples, refined_aligned, col = palette, nrow = 1L, size.dot = 0.3
    ) + ggplot2::ggtitle("Tissue: FiberMargin")
  )
  panel <- gridExtra::arrangeGrob(
    grobs = plots, nrow = 2L,
    top = sprintf(
      "%s | PCA50 -> KODAMA PLS-LDA50 | M=%d T=%d | min historical gap %+.3f",
      dataset, as.integer(M), as.integer(Tcycle), min(metrics$historical_gap)
    )
  )
  ggplot2::ggsave(
    output_file, panel, width = 18, height = 9, dpi = 180, bg = "white"
  )
  invisible(list(initial_aligned = initial_aligned, refined_aligned = refined_aligned))
}
