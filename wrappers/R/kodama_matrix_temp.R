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
           progress = TRUE) {
    classifier <- match.arg(classifier)
    backend <- match.arg(backend)
    spatial.constraint.mode <- match.arg(spatial.constraint.mode)
    data <- as.matrix(data)
    storage.mode(data) <- "double"

    if (!loaded) {
      if (!requireNamespace("Rcpp", quietly = TRUE)) {
        stop("Rcpp is required for the temporary kodama-cpp R wrapper")
      }
      root <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||% getwd()), "..", ".."), mustWork = FALSE)
      if (!file.exists(file.path(root, "include", "kodama", "kodama.hpp"))) {
        root <- normalizePath(getwd(), mustWork = TRUE)
      }
      build <- Sys.getenv("KODAMA_CPP_BUILD_DIR", file.path(root, "build"))
      conda <- Sys.getenv("CONDA_PREFIX", "")
      lib_dirs <- unique(c(
        if (nzchar(conda)) file.path(conda, "lib") else character(),
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
          omp_flag,
          conda_stdcxx
        )
      )
      Rcpp::sourceCpp(file.path(root, "wrappers", "R", "kodama_matrix_wrapper.cpp"), rebuild = FALSE)
      loaded <<- TRUE
    }

    kodama_matrix_cpp_temp(
      data = data,
      spatial = if (is.null(spatial)) NULL else {
        spatial <- as.matrix(spatial)
        storage.mode(spatial) <- "double"
        spatial
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
      progress = isTRUE(progress)
    )
  }
})

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
