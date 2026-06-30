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
    faiss_gpu_flag <- if (backend == "cuda" && any(file.exists(file.path(lib_dirs, "libfaiss_gpu.so")))) {
      "-lfaiss_gpu"
    } else {
      ""
    }
    cugraph_flag <- if (backend == "cuda" && any(file.exists(file.path(lib_dirs, "libcugraph_c.so")))) {
      "-lcugraph_c"
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
        "-lfaiss",
        if (backend == "cuda") paste(faiss_gpu_flag, "-lcuvs -lcublas -lcusolver -lcurand -lcufft -lcudart") else "",
        cugraph_flag,
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

  if (requireNamespace("fastEmbedR", quietly = TRUE)) {
    pca_fun <- tryCatch(getFromNamespace("fastpls_rsvd_pca_scores", "fastEmbedR"), error = function(e) NULL)
    if (!is.null(pca_fun)) {
      centered <- sweep(data, 2L, colMeans(data), check.margin = FALSE)
      pca <- pca_fun(centered, rank = n.components, seed = as.integer(seed), backend = backend)
      scores <- as.matrix(pca$scores[, seq_len(n.components), drop = FALSE])
      method <- paste0("fastEmbedR_", pca$method %||% "rsvd")
      used_backend <- pca$backend %||% backend
    }
  }

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

.kodama_umap_knn_fastembedr_init <- function(knn,
                                             init,
                                             backend = c("cuda", "cpu"),
                                             seed = 4L,
                                             n.threads = 4L,
                                             graph.mode = "fuzzy",
                                             n.epochs = NULL) {
  if (!requireNamespace("fastEmbedR", quietly = TRUE)) {
    stop("fastEmbedR is required for initialized KODAMA UMAP.")
  }
  backend <- match.arg(backend)
  if (backend != "cuda") stop("Initialized fast KODAMA UMAP currently requires backend='cuda'.")
  ns <- asNamespace("fastEmbedR")
  idx <- as.matrix(knn$indices)
  dst <- as.matrix(knn$distances)
  storage.mode(idx) <- "integer"
  storage.mode(dst) <- "double"
  init <- as.matrix(init)
  storage.mode(init) <- "double"
  if (nrow(init) != nrow(idx) || ncol(init) != 2L) {
    stop("UMAP init must have one row per sample and 2 columns.")
  }

  cfg <- get("fast_knn_umap_config", envir = ns)(n = nrow(idx), k = ncol(idx), backend = backend)
  auto_policy <- tryCatch(
    get("umap_auto_parameters_cpp", envir = ns)(dst, as.integer(ncol(idx)), as.character(cfg$backend)),
    error = function(e) list(error = conditionMessage(e))
  )
  if (is.null(auto_policy$error)) {
    cfg$n_epochs <- as.integer(auto_policy$n_epochs)
    cfg$min_dist <- as.numeric(auto_policy$min_dist)
    cfg$negative_sample_rate <- as.integer(auto_policy$negative_sample_rate)
    cfg$learning_rate <- as.numeric(auto_policy$learning_rate)
    cfg$spectral_n_iter <- as.integer(auto_policy$spectral_n_iter)
    cfg$init_scale <- as.numeric(auto_policy$init_scale)
    cfg$auto_parameter_backend <- "cpp_knn_distance_profile"
    cfg$auto_parameter_rule <- as.character(auto_policy$rule)
  } else {
    cfg$auto_parameter_backend <- "r_size_rule_fallback"
    cfg$auto_parameter_error <- auto_policy$error
  }
  if (!is.null(n.threads)) cfg$n_threads <- as.integer(max(1L, min(4L, n.threads)))
  if (!is.null(n.epochs)) cfg$n_epochs <- as.integer(n.epochs)
  cfg$input_had_self <- FALSE
  cfg$knn_col_start <- 0L
  cfg$knn_n_neighbors <- as.integer(ncol(idx))
  cfg$knn_materialized <- TRUE
  cfg$knn_backend <- "supplied"
  cfg$graph_mode <- graph.mode
  cfg$sgd_loop <- "csr_float32_contiguous_inplace"
  cfg <- get("apply_umap_connectivity_spectral_rule", envir = ns)(
    cfg,
    idx,
    col_start = 0L,
    n_neighbors = ncol(idx)
  )

  graph <- get("umap_build_csr_graph", envir = ns)(
    idx,
    dst,
    0L,
    as.integer(ncol(idx)),
    as.integer(ncol(idx)),
    as.integer(cfg$n_threads),
    graph_mode = graph.mode
  )
  cfg$graph_prep_backend <- "cpu_fuzzy_csr"
  cfg$graph_storage <- "cpu_csr_uploaded_to_cuda"
  cfg$graph_nnz <- as.integer(graph$nnz)
  cfg$graph_max_weight <- as.numeric(graph$max_weight)
  cfg$graph_cuda_like_width <- graph$cuda_like_width
  cfg$graph_builder <- graph$graph_builder
  cfg$init_backend <- "kodama_original_data_pca"
  cfg$gpu_umap_path <- "cuda_fuzzy_graph_atomic_supplied_init"
  cfg$backend <- "cuda"

  layout <- get("umap_cuda_optimize_csr_cpp", envir = ns)(
    graph$offsets,
    graph$neighbors,
    graph$weights,
    graph$epochs_per_sample,
    init,
    as.integer(cfg$n_epochs),
    as.integer(cfg$negative_sample_rate),
    cfg$learning_rate,
    cfg$min_dist,
    cfg$repulsion_strength,
    as.integer(seed),
    0L
  )
  layout <- get("finalize_embedding_layout", envir = ns)(layout, "UMAP", return_float32 = FALSE)
  attr(layout, "fastEmbedR_config") <- cfg
  layout
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
           backend = c("cpu", "cuda"),
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
      faiss_gpu_flag <- if (backend == "cuda" && any(file.exists(file.path(lib_dirs, "libfaiss_gpu.so")))) {
        "-lfaiss_gpu"
      } else {
        ""
      }
      cugraph_flag <- if (backend == "cuda" && any(file.exists(file.path(lib_dirs, "libcugraph_c.so")))) {
        "-lcugraph_c"
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
          "-lfaiss",
          if (backend == "cuda") paste(faiss_gpu_flag, "-lcuvs -lcublas -lcusolver -lcurand -lcufft -lcudart") else "",
          cugraph_flag,
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
  source <- knn
  if (is.null(init)) init <- .kodama_visual_init(source, "umap", backend = "cuda", seed = seed)
  knn <- .kodama_extract_knn(source)
  knn <- .kodama_strip_self_knn(knn, as.integer(n.neighbors))
  kodama_umap_cuda_temp(
    indices = knn$indices,
    distances = knn$distances,
    init = if (is.null(init)) NULL else {
      init <- as.matrix(init)
      storage.mode(init) <- "double"
      init
    },
    n_neighbors = as.integer(n.neighbors),
    n_epochs = as.integer(n.epochs),
    min_dist = as.numeric(min.dist),
    spectral_n_iter = as.integer(spectral.n.iter),
    seed = as.integer(seed)
  )
}

KODAMA.umap.fastEmbedR <- function(data,
                                   n.neighbors = 30L,
                                   backend = c("cuda", "cpu"),
                                   seed = 4L,
                                   n.threads = 4L,
                                   graph.mode = "fuzzy",
                                   ...) {
  if (!requireNamespace("fastEmbedR", quietly = TRUE)) {
    stop("fastEmbedR is required for KODAMA.umap.fastEmbedR().")
  }
  backend <- match.arg(backend)
  data <- .kodama_as_numeric_matrix(data)
  out <- fastEmbedR::umap(
    data,
    n_neighbors = as.integer(n.neighbors),
    backend = backend,
    graph_mode = graph.mode,
    n_threads = as.integer(n.threads),
    seed = as.integer(seed),
    ...
  )
  if (is.list(out) && !is.null(out$layout)) return(as.matrix(out$layout))
  as.matrix(out)
}

KODAMA.umap.knn.fastEmbedR <- function(knn,
                                       n.neighbors = NULL,
                                       backend = c("cuda", "cpu"),
                                       seed = 4L,
                                       n.threads = 4L,
                                       graph.mode = "fuzzy",
                                       use.visual.init = TRUE,
                                       ...) {
  if (!requireNamespace("fastEmbedR", quietly = TRUE)) {
    stop("fastEmbedR is required for KODAMA.umap.knn.fastEmbedR().")
  }
  backend <- match.arg(backend)
  source <- knn
  init <- if (isTRUE(use.visual.init)) .kodama_visual_init(source, "umap", backend = backend, seed = seed) else NULL
  if (!is.null(init) && backend == "cuda") {
    if (is.null(n.neighbors)) n.neighbors <- 30L
    knn_for_width <- .kodama_extract_knn(source)
    width <- min(as.integer(round(n.neighbors) * 3L), ncol(knn_for_width$indices))
    knn_for_width <- .kodama_prepare_visual_knn(knn_for_width, width, replace.inf = TRUE)
    return(tryCatch(
      .kodama_umap_knn_fastembedr_init(
        knn_for_width,
        init = init,
        backend = backend,
        seed = seed,
        n.threads = n.threads,
        graph.mode = graph.mode
      ),
      error = function(e) {
        warning("Fast initialized UMAP path failed; falling back to direct CUDA UMAP: ", conditionMessage(e))
        KODAMA.umap.cuda.cpp(source, init = init, n.neighbors = width, seed = seed)
      }
    ))
  }
  knn <- .kodama_extract_knn(source)
  if (is.null(n.neighbors)) n.neighbors <- ncol(knn$indices)
  # KODAMA.visualization uses 3 * n_neighbors from the precomputed,
  # KODAMA-reordered dissimilarity graph and replaces Inf by the largest
  # finite distance before calling UMAP.
  width <- min(as.integer(round(n.neighbors) * 3L), ncol(knn$indices))
  knn <- .kodama_prepare_visual_knn(knn, width, replace.inf = TRUE)
  out <- fastEmbedR::umap_knn(
    knn$indices,
    knn$distances,
    backend = backend,
    graph_mode = graph.mode,
    n_threads = as.integer(n.threads),
    seed = as.integer(seed),
    ...
  )
  if (is.list(out) && !is.null(out$layout)) return(as.matrix(out$layout))
  as.matrix(out)
}

KODAMA.opentsne.cuda.cpp <- function(knn,
                                     init = NULL,
                                     n.neighbors = NULL,
                                     perplexity = 15,
                                     n.iter = 500L,
                                     seed = 4L) {
  source <- knn
  if (is.null(init)) init <- .kodama_visual_init(source, "opentsne", backend = "cuda", seed = seed)
  knn <- .kodama_extract_knn(source)
  if (is.null(n.neighbors)) n.neighbors <- ceiling(perplexity)
  knn <- .kodama_strip_self_knn(knn, as.integer(n.neighbors))
  if (is.null(init)) {
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
    init <- matrix(stats::rnorm(nrow(knn$indices) * 2L, sd = 1e-4),
                   nrow = nrow(knn$indices), ncol = 2L)
    init <- sweep(init, 2L, colMeans(init), check.margin = FALSE)
  }
  kodama_opentsne_cuda_temp(
    indices = knn$indices,
    distances = knn$distances,
    init = if (is.null(init)) NULL else {
      init <- as.matrix(init)
      storage.mode(init) <- "double"
      init
    },
    n_neighbors = as.integer(n.neighbors),
    perplexity = as.numeric(perplexity),
    n_iter = as.integer(n.iter),
    early_exaggeration_iter = 250L,
    early_exaggeration = 12,
    exaggeration = 1,
    learning_rate = 0,
    learning_rate_auto = TRUE,
    initial_momentum = 0.8,
    final_momentum = 0.8,
    min_gain = 0.01,
    max_step_norm = 5,
    seed = as.integer(seed)
  )
}

KODAMA.opentsne.fastEmbedR <- function(data,
                                       perplexity = 15,
                                       backend = c("cuda", "cpu"),
                                       seed = 4L,
                                       n.threads = 4L,
                                       ...) {
  if (!requireNamespace("fastEmbedR", quietly = TRUE)) {
    stop("fastEmbedR is required for KODAMA.opentsne.fastEmbedR().")
  }
  backend <- match.arg(backend)
  data <- .kodama_as_numeric_matrix(data)
  out <- fastEmbedR::opentsne(
    data,
    perplexity = as.numeric(perplexity),
    backend = backend,
    n_threads = as.integer(n.threads),
    seed = as.integer(seed),
    ...
  )
  if (is.list(out) && !is.null(out$layout)) return(as.matrix(out$layout))
  if (is.list(out) && !is.null(out$Y)) return(as.matrix(out$Y))
  as.matrix(out)
}

KODAMA.opentsne.knn.fastEmbedR <- function(knn,
                                           perplexity = 15,
                                           n.neighbors = NULL,
                                           backend = c("cuda", "cpu"),
                                           seed = 4L,
                                           n.threads = 4L,
                                           Y.init = NULL,
                                           use.visual.init = TRUE,
                                           ...) {
  if (!requireNamespace("fastEmbedR", quietly = TRUE)) {
    stop("fastEmbedR is required for KODAMA.opentsne.knn.fastEmbedR().")
  }
  backend <- match.arg(backend)
  source <- knn
  if (is.null(Y.init) && isTRUE(use.visual.init)) Y.init <- .kodama_visual_init(source, "opentsne", backend = backend, seed = seed)
  knn <- .kodama_extract_knn(source)
  if (is.null(n.neighbors)) n.neighbors <- ceiling(perplexity)
  # KODAMA.visualization uses 3 * perplexity neighbours for t-SNE from the
  # precomputed KODAMA dissimilarity graph. fastEmbedR requires finite KNN
  # distances, so the original UMAP-style Inf replacement is applied at this
  # adapter boundary while preserving KODAMA's row order.
  width <- min(as.integer(round(perplexity) * 3L), ncol(knn$indices))
  knn <- .kodama_prepare_visual_knn(knn, width, replace.inf = TRUE)
  out <- fastEmbedR::opentsne_knn(
    knn$indices,
    knn$distances,
    n_neighbors = width,
    perplexity = as.numeric(perplexity),
    Y_init = Y.init,
    backend = backend,
    n_threads = as.integer(n.threads),
    seed = as.integer(seed),
    ...
  )
  if (is.list(out) && !is.null(out$layout)) return(as.matrix(out$layout))
  if (is.list(out) && !is.null(out$Y)) return(as.matrix(out$Y))
  as.matrix(out)
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
                                 backend = c("cpu", "cuda"),
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
                                              method = c("louvain", "leiden", "random_walking"),
                                              backend = c("cpu", "cuda"),
                                              weight = c("distance", "adaptive", "binary", "snn"),
                                              n.cores = 4L,
                                              n.runs = 1L,
                                              n.iterations = 10L,
                                              random.walk.steps = 4L,
                                              n.clusters = NULL,
                                              resolution = NULL,
                                              resolution.init = 0,
                                              resolution.delta = 0.2,
                                              prune = 0,
                                              mutual = FALSE,
                                              seed = 1L,
                                              gpu.device = 0L) {
  method <- match.arg(method)
  backend <- match.arg(backend)
  .kodama_cpp_temp_load(backend)
  weight <- match.arg(weight)
  .kodama_validate_cluster_target(resolution, n.clusters)
  kodama_graph_cluster_from_knn_temp(
    indices = knn$indices,
    distances = knn$distances,
    method = method,
    backend = backend,
    weight = weight,
    n_threads = as.integer(n.cores),
    n_runs = as.integer(n.runs),
    n_iterations = as.integer(n.iterations),
    random_walk_steps = as.integer(random.walk.steps),
    n_clusters = if (is.null(n.clusters)) 0L else as.integer(n.clusters),
    resolution = if (is.null(resolution)) 1 else as.numeric(resolution),
    resolution_init = as.numeric(resolution.init),
    resolution_delta = as.numeric(resolution.delta),
    prune = as.numeric(prune),
    mutual = isTRUE(mutual),
    seed = as.integer(seed),
    gpu_device = as.integer(gpu.device)
  )
}

KODAMA.graph.cluster.cpp <- function(embedding,
                                     method = c("louvain", "leiden", "random_walking"),
                                     backend = c("cpu", "cuda"),
                                     graph.backend = c("cpu", "cuda"),
                                     weight = c("distance", "adaptive", "binary", "snn"),
                                     metric = "euclidean",
                                     k = 30L,
                                     n.cores = 4L,
                                     n.runs = 1L,
                                     n.iterations = 10L,
                                     random.walk.steps = 4L,
                                     n.clusters = NULL,
                                     resolution = NULL,
                                     resolution.init = 0,
                                     resolution.delta = 0.2,
                                     prune = 0,
                                     mutual = FALSE,
                                     seed = 1L,
                                     gpu.device = 0L) {
  method <- match.arg(method)
  backend <- match.arg(backend)
  graph.backend <- match.arg(graph.backend)
  .kodama_cpp_temp_load(if (backend == "cuda" || graph.backend == "cuda") "cuda" else "cpu")
  weight <- match.arg(weight)
  .kodama_validate_cluster_target(resolution, n.clusters)
  embedding <- as.matrix(embedding)
  storage.mode(embedding) <- "double"
  kodama_embedding_cluster_temp(
    embedding = embedding,
    method = method,
    backend = backend,
    graph_backend = graph.backend,
    weight = weight,
    metric = metric,
    k = as.integer(k),
    n_threads = as.integer(n.cores),
    n_runs = as.integer(n.runs),
    n_iterations = as.integer(n.iterations),
    random_walk_steps = as.integer(random.walk.steps),
    n_clusters = if (is.null(n.clusters)) 0L else as.integer(n.clusters),
    resolution = if (is.null(resolution)) 1 else as.numeric(resolution),
    resolution_init = as.numeric(resolution.init),
    resolution_delta = as.numeric(resolution.delta),
    prune = as.numeric(prune),
    mutual = isTRUE(mutual),
    seed = as.integer(seed),
    gpu_device = as.integer(gpu.device)
  )
}

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

.kodama_validate_cluster_target <- function(resolution, n.clusters) {
  has_resolution <- !is.null(resolution)
  has_clusters <- !is.null(n.clusters)
  if (has_resolution == has_clusters) {
    stop("Provide exactly one of `resolution` or `n.clusters`.", call. = FALSE)
  }
  if (has_clusters) {
    n.clusters <- suppressWarnings(as.integer(n.clusters))
    if (length(n.clusters) != 1L || is.na(n.clusters) || n.clusters < 1L) {
      stop("`n.clusters` must be a positive integer.", call. = FALSE)
    }
  }
  if (has_resolution) {
    resolution <- suppressWarnings(as.numeric(resolution))
    if (length(resolution) != 1L || is.na(resolution) || !is.finite(resolution) || resolution <= 0) {
      stop("`resolution` must be a positive number.", call. = FALSE)
    }
  }
  invisible(TRUE)
}
