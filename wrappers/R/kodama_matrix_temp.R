# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.kodama_cpp_temp_load <- local({
  loaded <- FALSE

  function(backend = "cpu") {
    if (loaded) return(invisible(TRUE))
    if (!requireNamespace("Rcpp", quietly = TRUE)) {
      stop("Rcpp is required for the temporary kodama-cpp R wrapper")
    }
    root <- Sys.getenv("KODAMA_CPP_ROOT", "")
    if (!nzchar(root)) {
      root <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||% getwd()), "..", ".."), mustWork = FALSE)
    }
    if (!file.exists(file.path(root, "include", "kodama", "kodama.hpp"))) {
      root <- normalizePath(getwd(), mustWork = TRUE)
    }
    build <- Sys.getenv("KODAMA_CPP_BUILD_DIR", file.path(root, "build"))
    conda <- Sys.getenv("CONDA_PREFIX", "")
    lib_dirs <- unique(c(
      if (nzchar(conda)) file.path(conda, "lib") else character(),
      if (nzchar(conda)) file.path(conda, "targets", "x86_64-linux", "lib") else character(),
      "/usr/local/cuda-13.0/targets/x86_64-linux/lib",
      "/usr/local/cuda/lib64",
      "/opt/homebrew/opt/faiss/lib",
      "/opt/homebrew/opt/libomp/lib",
      "/usr/local/lib",
      "/usr/lib/x86_64-linux-gnu"
    ))
    lib_dirs <- lib_dirs[file.exists(lib_dirs)]
    lib_flags <- paste(paste0("-L", shQuote(lib_dirs)), collapse = " ")
    omp_dylib <- "/opt/homebrew/opt/libomp/lib/libomp.dylib"
    omp_flag <- if (file.exists(omp_dylib)) shQuote(omp_dylib) else "-lomp"
    conda_stdcxx <- if (nzchar(conda) && file.exists(file.path(conda, "lib", "libstdc++.so.6"))) {
      shQuote(file.path(conda, "lib", "libstdc++.so.6"))
    } else {
      ""
    }
    rpath_dirs <- lib_dirs
    if (dir.exists("/opt/homebrew/opt/llvm/lib")) {
      rpath_dirs <- unique(c(rpath_dirs, "/opt/homebrew/opt/llvm/lib"))
    }
    rpath_flags <- paste(paste0("-Wl,-rpath,", shQuote(rpath_dirs)), collapse = " ")
    accelerator_flags <- if (backend == "cuda") {
      "-lcublas -lcusolver -lcurand -lcufft -lcudart"
    } else if (backend == "metal") {
      "-framework Metal -framework MetalPerformanceShaders -framework Foundation"
    } else {
      ""
    }
    old_pkgs <- Sys.getenv("PKG_CPPFLAGS")
    old_libs <- Sys.getenv("PKG_LIBS")
    on.exit({
      Sys.setenv(PKG_CPPFLAGS = old_pkgs)
      Sys.setenv(PKG_LIBS = old_libs)
    }, add = TRUE)
    Sys.setenv(
      PKG_CPPFLAGS = paste(old_pkgs, paste0("-I", shQuote(file.path(root, "include")))),
      PKG_LIBS = paste(
        old_libs,
        shQuote(file.path(build, "libkodama_cpp.a")),
        lib_flags,
        rpath_flags,
        accelerator_flags,
        omp_flag,
        conda_stdcxx
      )
    )
    Rcpp::sourceCpp(
      file.path(root, "wrappers", "R", "kodama_matrix_wrapper.cpp"),
      rebuild = isTRUE(as.logical(Sys.getenv("KODAMA_RCPP_REBUILD", "FALSE")))
    )
    loaded <<- TRUE
    invisible(TRUE)
  }
})

.kodama_as_numeric_matrix <- function(x) {
  # Temporary sourceCpp ABI: R passes a NumericMatrix; C++ immediately downcasts
  # to float32 before running KODAMA analysis.
  if (inherits(x, "float32") || inherits(x, "float")) {
    if (!requireNamespace("float", quietly = TRUE)) {
      stop("The float package is required to convert a float matrix.")
    }
    x <- float::dbl(x)
  }
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

.kodama_scale_init_sd <- function(scores, target = 1e-4) {
  init <- sweep(as.matrix(scores), 2L, colMeans(scores), check.margin = FALSE)
  scale <- max(apply(init, 2L, stats::sd))
  if (is.finite(scale) && scale > 0) init <- init * (target / scale)
  init
}

.kodama_scale_init_max_abs <- function(scores, target = 10, seed = 4L) {
  init <- sweep(as.matrix(scores), 2L, colMeans(scores), check.margin = FALSE)
  scale <- max(abs(init))
  if (is.finite(scale) && scale > 0) init <- init * (target / scale)
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (had_seed) get(".Random.seed", envir = .GlobalEnv, inherits = FALSE) else NULL
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed))
  init <- init + matrix(stats::rnorm(length(init), sd = 1e-4), nrow = nrow(init), ncol = ncol(init))
  sweep(init, 2L, colMeans(init), check.margin = FALSE)
}

.kodama_visual_pca_scores <- function(data, n.components = 2L, seed = 4L, backend = "cpu") {
  n.components <- min(as.integer(n.components), ncol(data), max(1L, nrow(data) - 1L))
  if (n.components < 1L) stop("Cannot build a visual initialization from empty data.")
  method <- "unknown"
  used_backend <- backend
  scores <- NULL

  if (is.null(scores) && requireNamespace("irlba", quietly = TRUE)) {
    pca <- irlba::prcomp_irlba(data, n = n.components, center = TRUE, scale. = FALSE)
    scores <- as.matrix(pca$x[, seq_len(n.components), drop = FALSE])
    method <- "irlba_prcomp"
    used_backend <- "cpu"
  }

  if (is.null(scores)) {
    pca <- stats::prcomp(data, rank. = n.components, center = TRUE, scale. = FALSE)
    scores <- as.matrix(pca$x[, seq_len(n.components), drop = FALSE])
    method <- "stats_prcomp"
    used_backend <- "cpu"
  }

  if (ncol(scores) < 2L) {
    scores <- cbind(scores, 0)
  }
  scores <- scores[, 1:2, drop = FALSE]
  attr(scores, "method") <- method
  attr(scores, "backend") <- used_backend
  scores
}

.kodama_make_visual_init <- function(data, seed = 4L, backend = "cpu") {
  scores <- .kodama_visual_pca_scores(data, n.components = 2L, seed = seed, backend = backend)
  out <- list(
    opentsne = .kodama_scale_init_sd(scores, target = 1e-4),
    umap = .kodama_scale_init_max_abs(scores, target = 10, seed = seed),
    method = attr(scores, "method"),
    backend = attr(scores, "backend"),
    seed = as.integer(seed)
  )
  attr(out$opentsne, "visual_init") <- "opentsne_pca"
  attr(out$umap, "visual_init") <- "umap_pca"
  out
}

.kodama_extract_knn <- function(x) {
  if (is.list(x) && !is.null(x$knn_Rnanoflann)) return(x$knn_Rnanoflann)
  x
}

.kodama_visual_init <- function(x,
                                method = c("opentsne", "umap"),
                                backend = "cpu",
                                seed = 4L) {
  method <- match.arg(method)
  if (!is.list(x)) return(NULL)
  if (!is.null(x$visual_init)) {
    init <- x$visual_init
    if (is.matrix(init)) return(init)
    if (is.list(init) && !is.null(init[[method]])) return(init[[method]])
  }
  if (!is.null(x$data)) {
    init <- tryCatch(
      .kodama_make_visual_init(.kodama_as_numeric_matrix(x$data), seed = seed, backend = backend),
      error = function(e) NULL
    )
    if (is.list(init) && !is.null(init[[method]])) return(init[[method]])
  }
  NULL
}

KODAMA.matrix.cpp <- local({
  loaded <- FALSE

  function(data,
           spatial = NULL,
           W = NULL,
           constrain = NULL,
           fix = NULL,
           M = 100,
           Tcycle = 20,
           ncomp = min(50L, ncol(data)),
           landmarks = 10000L,
           splitting = ifelse(nrow(data) < 40000, 100L, 300L),
           n.cores = 4L,
           graph.neighbors = 100L,
           knn.k = 30L,
           spatial.resolution = 0.3,
           spatial.graph.mix = FALSE,
           spatial.constraint.mode = c("kmeans", "graph", "auto"),
           metrics = "euclidean",
           classifier = c("pls_lda", "knn"),
           backend = c("cpu", "cuda", "metal"),
           seed = 1234L,
           progress = TRUE,
           apply.kodama.dissimilarity = TRUE,
           visual.init = TRUE) {
    classifier <- match.arg(classifier)
    backend <- match.arg(backend)
    spatial.constraint.mode <- match.arg(spatial.constraint.mode)
    data <- .kodama_as_numeric_matrix(data)
    visual_init <- if (isTRUE(visual.init)) .kodama_make_visual_init(data, seed = seed, backend = backend) else NULL

    if (!loaded) {
      if (!requireNamespace("Rcpp", quietly = TRUE)) {
        stop("Rcpp is required for the temporary kodama-cpp R wrapper")
      }
      root <- Sys.getenv("KODAMA_CPP_ROOT", "")
      if (!nzchar(root)) {
        root <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||% getwd()), "..", ".."), mustWork = FALSE)
      }
      if (!file.exists(file.path(root, "include", "kodama", "kodama.hpp"))) {
        root <- normalizePath(getwd(), mustWork = TRUE)
      }
      build <- Sys.getenv("KODAMA_CPP_BUILD_DIR", file.path(root, "build"))
      conda <- Sys.getenv("CONDA_PREFIX", "")
      lib_dirs <- unique(c(
        if (nzchar(conda)) file.path(conda, "lib") else character(),
        if (nzchar(conda)) file.path(conda, "targets", "x86_64-linux", "lib") else character(),
        "/usr/local/cuda-13.0/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
        "/opt/homebrew/opt/faiss/lib",
        "/opt/homebrew/opt/libomp/lib",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu"
      ))
      lib_dirs <- lib_dirs[file.exists(lib_dirs)]
      lib_flags <- paste(paste0("-L", shQuote(lib_dirs)), collapse = " ")
      omp_dylib <- "/opt/homebrew/opt/libomp/lib/libomp.dylib"
      omp_flag <- if (file.exists(omp_dylib)) shQuote(omp_dylib) else "-lomp"
      conda_stdcxx <- if (nzchar(conda) && file.exists(file.path(conda, "lib", "libstdc++.so.6"))) {
        shQuote(file.path(conda, "lib", "libstdc++.so.6"))
      } else {
        ""
      }
      rpath_dirs <- lib_dirs
      if (dir.exists("/opt/homebrew/opt/llvm/lib")) {
        rpath_dirs <- unique(c(rpath_dirs, "/opt/homebrew/opt/llvm/lib"))
      }
      rpath_flags <- paste(paste0("-Wl,-rpath,", shQuote(rpath_dirs)), collapse = " ")
      accelerator_flags <- if (backend == "cuda") {
        "-lcublas -lcusolver -lcurand -lcufft -lcudart"
      } else if (backend == "metal") {
        "-framework Metal -framework MetalPerformanceShaders -framework Foundation"
      } else {
        ""
      }
      old_pkgs <- Sys.getenv("PKG_CPPFLAGS")
      old_libs <- Sys.getenv("PKG_LIBS")
      on.exit({
        Sys.setenv(PKG_CPPFLAGS = old_pkgs)
        Sys.setenv(PKG_LIBS = old_libs)
      }, add = TRUE)
      Sys.setenv(
        PKG_CPPFLAGS = paste(old_pkgs, paste0("-I", shQuote(file.path(root, "include")))),
        PKG_LIBS = paste(
          old_libs,
          shQuote(file.path(build, "libkodama_cpp.a")),
          lib_flags,
          rpath_flags,
          accelerator_flags,
          omp_flag,
          conda_stdcxx
        )
      )
      Rcpp::sourceCpp(
        file.path(root, "wrappers", "R", "kodama_matrix_wrapper.cpp"),
        rebuild = isTRUE(as.logical(Sys.getenv("KODAMA_RCPP_REBUILD", "FALSE")))
      )
      loaded <<- TRUE
    }

    out <- kodama_matrix_cpp_temp(
      data = data,
      spatial = if (is.null(spatial)) NULL else {
        .kodama_as_numeric_matrix(spatial)
      },
      W = if (is.null(W)) NULL else as.integer(W),
      constrain = if (is.null(constrain)) NULL else as.integer(constrain),
      fix = if (is.null(fix)) NULL else as.integer(fix),
      M = as.integer(M),
      Tcycle = as.integer(Tcycle),
      ncomp = as.integer(ncomp),
      landmarks = as.integer(landmarks),
      splitting = as.integer(splitting),
      n_cores = as.integer(n.cores),
      graph_neighbors = as.integer(graph.neighbors),
      knn_k = as.integer(knn.k),
      spatial_resolution = as.numeric(spatial.resolution),
      spatial_graph_mix = isTRUE(spatial.graph.mix),
      spatial_constraint_mode = if (spatial.constraint.mode == "auto") -1L else if (spatial.constraint.mode == "graph") 1L else 0L,
      metric = metrics,
      classifier = classifier,
      backend = backend,
      seed = as.integer(seed),
      progress = isTRUE(progress),
      apply_kodama_dissimilarity = isTRUE(apply.kodama.dissimilarity)
    )
    if (!is.null(visual_init)) out$visual_init <- visual_init
    out
  }
})

KODAMA.umap.cuda.cpp <- function(knn,
                                 init = NULL,
                                 n.neighbors = 30L,
                                 n.epochs = 200L,
                                 min.dist = 0.01,
                                 spectral.n.iter = 20L,
                                 seed = 1234L) {
  KODAMA.umap.knn.cpp(
    knn,
    n.neighbors = n.neighbors,
    backend = "cuda",
    seed = seed,
    n.threads = 1L,
    n.epochs = n.epochs,
    min.dist = min.dist,
    spectral.n.iter = spectral.n.iter,
    init = init
  )
}

KODAMA.umap.cpp <- function(data,
                            n.neighbors = 30L,
                            backend = c("cuda", "cpu", "metal"),
                            seed = 4L,
                            n.threads = 4L,
                            metric = "euclidean",
                            ...) {
  backend <- match.arg(backend)
  data <- .kodama_as_numeric_matrix(data)
  graph <- KODAMA.knn.graph.cpp(
    data,
    k = as.integer(n.neighbors),
    metric = metric,
    backend = backend,
    n.cores = as.integer(n.threads),
    exclude.self = TRUE
  )
  KODAMA.umap.knn.cpp(graph, n.neighbors = n.neighbors, backend = backend,
                      seed = seed, n.threads = n.threads, ...)
}

KODAMA.umap.knn.cpp <- function(knn,
                                n.neighbors = NULL,
                                backend = c("cuda", "cpu", "metal"),
                                seed = 4L,
                                n.threads = 4L,
                                use.visual.init = TRUE,
                                n.epochs = 200L,
                                min.dist = 0.01,
                                learning.rate = 1.0,
                                repulsion.strength = 1.0,
                                negative.sample.rate = 5L,
                                spectral.n.iter = 20L,
                                init = NULL,
                                ...) {
  backend <- match.arg(backend)
  .kodama_cpp_temp_load(backend)
  source <- knn
  if (is.null(init) && isTRUE(use.visual.init)) {
    init <- .kodama_visual_init(source, "umap", backend = backend, seed = seed)
  }
  knn <- .kodama_extract_knn(source)
  if (is.null(n.neighbors)) n.neighbors <- ncol(knn$indices)
  width <- min(as.integer(round(n.neighbors)), ncol(knn$indices))
  knn <- .kodama_prepare_visual_knn(knn, width, replace.inf = TRUE)
  kodama_umap_temp(
    indices = knn$indices,
    distances = knn$distances,
    init = if (is.null(init)) NULL else {
      init <- as.matrix(init)
      storage.mode(init) <- "double"
      init
    },
    n_neighbors = as.integer(width),
    n_epochs = as.integer(n.epochs),
    learning_rate = as.numeric(learning.rate),
    min_dist = as.numeric(min.dist),
    repulsion_strength = as.numeric(repulsion.strength),
    negative_sample_rate = as.integer(negative.sample.rate),
    spectral_n_iter = as.integer(spectral.n.iter),
    n_threads = as.integer(n.threads),
    seed = as.integer(seed),
    backend = backend
  )
}

KODAMA.opentsne.cuda.cpp <- function(knn,
                                     init = NULL,
                                     n.neighbors = NULL,
                                     perplexity = 15,
                                     n.iter = 500L,
                                     seed = 4L) {
  KODAMA.opentsne.knn.cpp(
    knn,
    perplexity = perplexity,
    n.neighbors = n.neighbors,
    backend = "cuda",
    seed = seed,
    n.threads = 1L,
    Y.init = init,
    n.iter = n.iter
  )
}

KODAMA.opentsne.cpp <- function(data,
                                perplexity = 15,
                                backend = c("cuda", "cpu", "metal"),
                                seed = 4L,
                                n.threads = 4L,
                                metric = "euclidean",
                                ...) {
  backend <- match.arg(backend)
  data <- .kodama_as_numeric_matrix(data)
  graph <- KODAMA.knn.graph.cpp(
    data,
    k = as.integer(ceiling(perplexity) * 3L),
    metric = metric,
    backend = backend,
    n.cores = as.integer(n.threads),
    exclude.self = TRUE
  )
  KODAMA.opentsne.knn.cpp(graph, perplexity = perplexity, backend = backend,
                          seed = seed, n.threads = n.threads, ...)
}

KODAMA.opentsne.knn.cpp <- function(knn,
                                    perplexity = 15,
                                    n.neighbors = NULL,
                                    backend = c("cuda", "cpu", "metal"),
                                    seed = 4L,
                                    n.threads = 4L,
                                    Y.init = NULL,
                                    use.visual.init = TRUE,
                                    n.iter = 500L,
                                    early.exaggeration.iter = 250L,
                                    early.exaggeration = 12,
                                    exaggeration = 1,
                                    learning.rate = 0,
                                    learning.rate.auto = TRUE,
                                    initial.momentum = 0.8,
                                    final.momentum = 0.8,
                                    min.gain = 0.01,
                                    max.step.norm = 5,
                                    theta = 0.5,
                                    ...) {
  backend <- match.arg(backend)
  .kodama_cpp_temp_load(backend)
  source <- knn
  if (is.null(Y.init) && isTRUE(use.visual.init)) Y.init <- .kodama_visual_init(source, "opentsne", backend = backend, seed = seed)
  knn <- .kodama_extract_knn(source)
  if (is.null(n.neighbors)) n.neighbors <- ceiling(perplexity)
  width <- min(as.integer(round(perplexity) * 3L), ncol(knn$indices))
  knn <- .kodama_prepare_visual_knn(knn, width, replace.inf = TRUE)
  kodama_opentsne_temp(
    indices = knn$indices,
    distances = knn$distances,
    init = if (is.null(Y.init)) NULL else {
      Y.init <- as.matrix(Y.init)
      storage.mode(Y.init) <- "double"
      Y.init
    },
    n_neighbors = width,
    perplexity = as.numeric(perplexity),
    theta = as.numeric(theta),
    early_exaggeration_iter = as.integer(early.exaggeration.iter),
    n_iter = as.integer(n.iter),
    early_exaggeration = as.numeric(early.exaggeration),
    exaggeration = as.numeric(exaggeration),
    learning_rate = as.numeric(learning.rate),
    learning_rate_auto = isTRUE(learning.rate.auto),
    initial_momentum = as.numeric(initial.momentum),
    final_momentum = as.numeric(final.momentum),
    min_gain = as.numeric(min.gain),
    max_step_norm = as.numeric(max.step.norm),
    n_threads = as.integer(n.threads),
    seed = as.integer(seed),
    backend = backend
  )
}

.kodama_prepare_visual_knn <- function(knn, width, replace.inf = FALSE) {
  idx <- as.matrix(knn$indices)
  dst <- as.matrix(knn$distances)
  storage.mode(idx) <- "integer"
  storage.mode(dst) <- "double"
  width <- min(as.integer(width), ncol(idx))
  idx <- idx[, seq_len(width), drop = FALSE]
  dst <- dst[, seq_len(width), drop = FALSE]
  if (isTRUE(replace.inf) && any(!is.finite(dst))) {
    finite <- is.finite(dst)
    fill <- if (any(finite)) max(dst[finite]) else 0
    dst[!finite] <- fill
  }
  list(indices = idx, distances = dst, neighbors = as.integer(width))
}

.kodama_strip_self_knn <- function(knn, k) {
  idx <- as.matrix(knn$indices)
  dst <- as.matrix(knn$distances)
  storage.mode(idx) <- "integer"
  storage.mode(dst) <- "double"
  n <- nrow(idx)
  expected0 <- seq_len(n) - 1L
  expected1 <- seq_len(n)
  tol <- max(sqrt(.Machine$double.eps), 1e-12)
  use_one <- all(idx >= 1L & idx <= n, na.rm = TRUE)
  expected <- if (use_one) expected1 else expected0
  out_idx <- matrix(NA_integer_, n, k)
  out_dst <- matrix(NA_real_, n, k)
  for (i in seq_len(n)) {
    keep <- which(!(idx[i, ] == expected[i] & dst[i, ] <= tol))
    if (length(keep) < k) keep <- seq_len(ncol(idx))
    keep <- keep[seq_len(k)]
    out_idx[i, ] <- idx[i, keep]
    out_dst[i, ] <- dst[i, keep]
  }
  list(indices = out_idx, distances = out_dst, neighbors = as.integer(k))
}

KODAMA.knn.graph.cpp <- function(data,
                                 k = 30L,
                                 metric = "euclidean",
                                 backend = c("cpu", "cuda", "metal"),
                                 n.cores = 4L,
                                 gpu.device = 0L,
                                 exclude.self = TRUE) {
  backend <- match.arg(backend)
  .kodama_cpp_temp_load(backend)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  search.k <- as.integer(k) + if (isTRUE(exclude.self)) 1L else 0L
  graph <- kodama_knn_graph_temp(
    data = data,
    k = search.k,
    metric = metric,
    backend = backend,
    n_threads = as.integer(n.cores),
    gpu_device = as.integer(gpu.device)
  )
  if (isTRUE(exclude.self)) graph <- .kodama_strip_self_knn(graph, as.integer(k))
  graph
}

KODAMA.graph.cluster.from.knn.cpp <- function(knn,
                                              weight = c("distance", "adaptive", "binary", "snn"),
                                              n.cores = 4L,
                                              n.iterations = 10L,
                                              random.walk.steps = 4L,
                                              n.clusters = NULL,
                                              prune = 0,
                                              mutual = FALSE) {
  .kodama_cpp_temp_load("cpu")
  weight <- match.arg(weight)
  kodama_graph_cluster_from_knn_temp(
    indices = knn$indices,
    distances = knn$distances,
    weight = weight,
    n_threads = as.integer(n.cores),
    n_iterations = as.integer(n.iterations),
    random_walk_steps = as.integer(random.walk.steps),
    n_clusters = if (is.null(n.clusters)) 0L else as.integer(n.clusters),
    prune = as.numeric(prune),
    mutual = isTRUE(mutual)
  )
}

KODAMA.graph.cluster.cpp <- function(embedding,
                                     graph.backend = c("cpu", "cuda", "metal"),
                                     weight = c("distance", "adaptive", "binary", "snn"),
                                     metric = "euclidean",
                                     k = 30L,
                                     n.cores = 4L,
                                     n.iterations = 10L,
                                     random.walk.steps = 4L,
                                     n.clusters = NULL,
                                     prune = 0,
                                     mutual = FALSE,
                                     gpu.device = 0L) {
  graph.backend <- match.arg(graph.backend)
  .kodama_cpp_temp_load(graph.backend)
  weight <- match.arg(weight)
  embedding <- as.matrix(embedding)
  storage.mode(embedding) <- "double"
  kodama_embedding_cluster_temp(
    embedding = embedding,
    graph_backend = graph.backend,
    weight = weight,
    metric = metric,
    k = as.integer(k),
    n_threads = as.integer(n.cores),
    n_iterations = as.integer(n.iterations),
    random_walk_steps = as.integer(random.walk.steps),
    n_clusters = if (is.null(n.clusters)) 0L else as.integer(n.clusters),
    prune = as.numeric(prune),
    mutual = isTRUE(mutual),
    gpu_device = as.integer(gpu.device)
  )
}

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
