# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x

parse_cli <- function(x = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  for (arg in x) {
    if (!startsWith(arg, "--")) next
    z <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[gsub("-", "_", z[[1L]])]] <- if (length(z) == 1L) TRUE else paste(z[-1L], collapse = "=")
  }
  out
}

script_file <- function() {
  z <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))])
  if (!length(z)) stop("Cannot identify the running script")
  normalizePath(z[[1L]], mustWork = TRUE)
}
suite_dir <- function() dirname(script_file())
as_int <- function(x, default = NA_integer_) {
  z <- suppressWarnings(as.integer(x %||% default)); if (is.na(z)) as.integer(default) else z
}
safe_name <- function(x) gsub("[^A-Za-z0-9_.-]", "_", x)

sha256 <- function(path) {
  if (!file.exists(path)) return(NA_character_)
  cmd <- if (nzchar(Sys.which("sha256sum"))) "sha256sum" else "shasum"
  args <- if (cmd == "shasum") c("-a", "256", path) else path
  out <- system2(cmd, args, stdout = TRUE, stderr = TRUE)
  if (!length(out)) NA_character_ else strsplit(out[[1L]], "[[:space:]]+")[[1L]][[1L]]
}
hash_object <- function(x) {
  path <- tempfile(); on.exit(unlink(path), add = TRUE)
  saveRDS(x, path, version = 3, compress = FALSE); sha256(path)
}

atomic_save_rds <- function(x, path, compress = FALSE) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid()); saveRDS(x, tmp, compress = compress)
  if (!file.rename(tmp, path)) stop("Atomic rename failed: ", path)
}
atomic_write_csv <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid()); write.csv(x, tmp, row.names = FALSE, na = "")
  if (!file.rename(tmp, path)) stop("Atomic rename failed: ", path)
}
atomic_write_lines <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid()); writeLines(x, tmp)
  if (!file.rename(tmp, path)) stop("Atomic rename failed: ", path)
}
atomic_write_json <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid())
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    jsonlite::write_json(x, tmp, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null")
  } else {
    escape_json <- function(z) {
      z <- gsub("\\\\", "\\\\\\\\", z)
      z <- gsub('"', '\\\\"', z)
      z <- gsub("\n", "\\\\n", z, fixed = TRUE)
      z <- gsub("\r", "\\\\r", z, fixed = TRUE)
      z <- gsub("\t", "\\\\t", z, fixed = TRUE)
      paste0('"', z, '"')
    }
    encode <- function(z) {
      if (is.null(z)) return("null")
      if (is.data.frame(z)) z <- lapply(seq_len(nrow(z)), function(i) as.list(z[i, , drop = FALSE]))
      if (is.list(z)) {
        values <- vapply(z, encode, character(1))
        if (!is.null(names(z)) && all(nzchar(names(z)))) {
          return(paste0("{", paste0(escape_json(names(z)), ":", values, collapse = ","), "}"))
        }
        return(paste0("[", paste(values, collapse = ","), "]"))
      }
      if (length(z) != 1L) return(paste0("[", paste(vapply(as.list(z), encode, character(1)), collapse = ","), "]"))
      if (length(z) == 0L || is.na(z)) return("null")
      if (is.logical(z)) return(if (z) "true" else "false")
      if (is.numeric(z)) return(format(z, scientific = FALSE, trim = TRUE, digits = 17))
      escape_json(as.character(z))
    }
    writeLines(encode(x), tmp, useBytes = TRUE)
  }
  if (!file.rename(tmp, path)) stop("Atomic rename failed: ", path)
}

expected_policy_names <- c(
  "full", "no_prediction_guidance", "fixed_proposal_budget",
  "no_transition_proposal", "greedy_acceptance", "raw_cv_score",
  "no_pls_transition_coarsening", "no_pls_fragmentation_penalty"
)
common_policies <- expected_policy_names[1:6]
pls_policies <- expected_policy_names
seeds <- c(4L, 17L, 42L)

build_cell_table <- function(reg = load_registry()) {
  rows <- list()
  add <- function(dataset, representation, classifier, experiment, setting, value, seed) {
    rows[[length(rows)+1L]] <<- data.frame(dataset, representation, classifier, experiment, setting,
      value=as.character(value), seed=as.integer(seed), stringsAsFactors=FALSE)
  }
  for (i in seq_len(nrow(reg))) for (seed in seeds) {
    d <- reg$dataset[[i]]; r <- reg$representation[[i]]
    for (policy in common_policies) add(d,r,"knn","ablation",policy,NA,seed)
    for (policy in pls_policies) add(d,r,"pls_lda","ablation",policy,NA,seed)
    for (k in c(5L,10L,15L,20L,30L,50L,100L)) add(d,r,"knn","knn_sensitivity","k",k,seed)
    for (nc in c(5L,10L,20L,50L)) add(d,r,"pls_lda","ncomp_sensitivity","ncomp",nc,seed)
    for (method in c("umap","opentsne")) add(d,r,"classic","classic",method,NA,seed)
  }
  cells <- do.call(rbind,rows); cells$cell_id <- seq_len(nrow(cells)); cells[,c("cell_id",setdiff(names(cells),"cell_id"))]
}

load_registry <- function(path = file.path(suite_dir(), "datasets.csv")) {
  x <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("dataset", "representation", "relative_path", "enabled")
  if (!all(required %in% names(x))) stop("Invalid dataset registry")
  x[as.logical(x$enabled), , drop = FALSE]
}

load_dataset <- function(dataset, representation, data_root, pca_path = NULL) {
  reg <- load_registry(); row <- reg[reg$dataset == dataset & reg$representation == representation, , drop = FALSE]
  if (nrow(row) != 1L) stop("Dataset representation absent or duplicated: ", dataset, "/", representation)
  if (dataset == "imagenet" && representation == "pca50") {
    if (is.null(pca_path) || !file.exists(pca_path)) stop("Prepared ImageNet PCA50 is missing")
    obj <- readRDS(pca_path); x <- as.matrix(obj$scores)
    raw_path <- file.path(data_root, row$relative_path)
    env <- new.env(parent=emptyenv()); loaded <- load(raw_path,envir=env)
    objects <- mget(loaded,env,inherits=FALSE)
    raw <- if("dataset"%in%names(objects)) objects$dataset else objects[vapply(objects,function(z)is.list(z)&&!is.null(z$data),logical(1))][[1L]]
    labels <- factor(raw$labels %||% raw$label)
    if(length(labels)!=nrow(x)) stop("ImageNet PCA sample-order mismatch")
    return(list(x = x, labels = labels, path = normalizePath(pca_path), file_sha256 = sha256(pca_path),
                data_sha256 = hash_object(x), label_sha256 = hash_object(labels), metadata = row))
  }
  path <- file.path(data_root, row$relative_path)
  if (!file.exists(path)) stop("Dataset file not found: ", path)
  env <- new.env(parent = emptyenv()); loaded <- load(path, envir = env)
  objects <- mget(loaded, env, inherits = FALSE)
  obj <- if ("dataset" %in% names(objects)) objects$dataset else {
    hits <- objects[vapply(objects, function(z) is.list(z) && !is.null(z$data), logical(1))]
    if (length(hits)) hits[[1L]] else NULL
  }
  if (is.null(obj)) stop("No list containing $data in ", path)
  x <- as.matrix(obj$data); storage.mode(x) <- "double"
  labels <- obj$labels %||% obj$label %||% NULL
  if (!nrow(x) || !ncol(x) || any(!is.finite(x))) stop("Invalid or non-finite matrix in ", path)
  if (is.null(labels) || length(labels) != nrow(x)) stop("Missing or mismatched labels in ", path)
  labels <- factor(labels)
  list(x = x, labels = labels, path = normalizePath(path), file_sha256 = sha256(path),
       data_sha256 = hash_object(x), label_sha256 = hash_object(labels), metadata = row)
}

matrix_call <- function(x, graph, classifier, M, Tcycle, folds, ncomp, landmarks,
                        splitting, k, seed, policy, progress = FALSE,
                        progress_file = NULL) {
  do.call(KODAMA::KODAMA.matrix, list(
    data = x, graph = graph, M = as.integer(M), Tcycle = as.integer(Tcycle),
    folds = as.integer(folds), ncomp = as.integer(ncomp), landmarks = as.integer(landmarks),
    splitting = as.integer(splitting), n.cores = 4L, graph.neighbors = 100L,
    knn.k = as.integer(k), classifier = classifier, backend = "cpu", seed = as.integer(seed),
    visual.init = TRUE, progress = progress, progress.file = progress_file,
    return.graph = "handle",
    .evolution.policy = policy
  ))
}

assert_visualizable_fit <- function(fit) {
  if (!is.list(fit) || is.null(fit$knn)) {
    stop(
      "KODAMA.matrix did not retain its corrected graph; ",
      "the benchmark requires return.graph = 'handle'."
    )
  }
  invisible(fit)
}

flatten_numeric <- function(x) {
  if (is.null(x)) return(numeric())
  as.numeric(unlist(x, recursive = TRUE, use.names = FALSE))
}

as_run_matrix <- function(x, expected_runs = NULL, expected_samples = NULL) {
  if (is.null(x)) return(matrix(integer(), 0L, 0L))
  if (is.data.frame(x) || is.matrix(x)) {
    out <- as.matrix(x)
  } else if (is.list(x)) {
    rows <- lapply(x, function(z) unlist(z, recursive = TRUE, use.names = FALSE))
    lengths <- lengths(rows)
    if (length(rows) && !any(lengths == 0L) && length(unique(lengths)) == 1L) {
      out <- do.call(rbind, rows)
    } else {
      values <- unlist(rows, recursive = TRUE, use.names = FALSE)
      out <- matrix(values, nrow = 1L)
    }
  } else {
    out <- matrix(x, nrow = 1L)
  }
  if (!is.null(expected_samples) && ncol(out) != expected_samples) {
    if (nrow(out) == expected_samples) {
      out <- t(out)
    } else if (length(out) %% expected_samples == 0L) {
      out <- matrix(unlist(out, use.names = FALSE), ncol = expected_samples,
                    byrow = TRUE)
    } else {
      stop("KODAMA result labels do not match the sample count")
    }
  }
  if (!is.null(expected_runs) && nrow(out) != expected_runs)
    stop("KODAMA result labels do not match M")
  storage.mode(out) <- "integer"
  out
}

as_diagnostic_frame <- function(x) {
  if (is.null(x)) return(data.frame())
  if (is.data.frame(x)) return(x)
  if (is.matrix(x)) return(as.data.frame(x, stringsAsFactors = FALSE))
  if (is.list(x) && length(x) && all(vapply(x, is.data.frame, logical(1))))
    return(do.call(rbind, x))
  tryCatch(as.data.frame(x, stringsAsFactors = FALSE),
           error = function(e) data.frame())
}

normalize_fit_result <- function(fit, expected_runs, expected_samples) {
  if (!is.list(fit)) stop("KODAMA.matrix returned a non-list result")
  fit$acc <- flatten_numeric(fit$acc)
  if (length(fit$acc) != expected_runs)
    stop("KODAMA accuracies do not match M")
  fit$res <- as_run_matrix(fit$res, expected_runs, expected_samples)
  fit$run_diagnostics <- as_diagnostic_frame(fit$run_diagnostics)
  fit$cycle_diagnostics <- as_diagnostic_frame(fit$cycle_diagnostics)
  fit$best_labels <- as.integer(flatten_numeric(fit$best_labels))
  if (length(fit$best_labels) != expected_samples)
    fit$best_labels <- as.integer(fit$res[which.max(fit$acc), ])
  peak <- flatten_numeric(fit$peak_memory_mb)
  fit$peak_memory_mb <- if (length(peak)) peak[[1L]] else NA_real_
  timing <- unlist(fit$timing %||% fit$timings %||% list(), recursive = TRUE)
  fit$timing <- as.list(as.numeric(timing))
  names(fit$timing) <- names(timing)
  fit
}

check_protocol_api <- function(strict = TRUE) {
  if (!requireNamespace("KODAMA", quietly = TRUE)) {
    stop("The canonical KODAMA package is not installed; legacy kodamaR is not supported by this benchmark")
  }
  f <- names(formals(KODAMA::KODAMA.matrix))
  required <- c("data", "graph", "M", "Tcycle", "folds", "ncomp", "landmarks", "splitting",
                "n.cores", "graph.neighbors", "knn.k", "classifier", "backend", "seed",
                "progress.file")
  hidden_policy <- "..." %in% f
  status <- data.frame(requirement = c(required, "hidden native evolution policy"),
                       available = c(required %in% f, hidden_policy), stringsAsFactors = FALSE)
  if (strict && any(!status$available)) stop("Protocol API incomplete: ", paste(status$requirement[!status$available], collapse = ", "))
  status
}

extract_layout <- function(x) {
  if (is.matrix(x)) return(x[, 1:2, drop = FALSE])
  for (nm in c("layout", "embedding", "Y", "coordinates")) if (!is.null(x[[nm]])) return(as.matrix(x[[nm]])[, 1:2, drop = FALSE])
  stop("No 2D layout found")
}

ari <- function(a, b) {
  tab <- table(a, b); n <- sum(tab); c2 <- function(z) z * (z - 1) / 2
  idx <- sum(c2(tab)); r <- sum(c2(rowSums(tab))); c <- sum(c2(colSums(tab))); total <- c2(n)
  exp <- if (total) r * c / total else 0; lim <- (r + c) / 2
  if (lim == exp) as.numeric(idx == exp) else (idx - exp) / (lim - exp)
}
information_metrics <- function(a, b) {
  tab <- table(a, b); p <- tab / sum(tab); pa <- rowSums(p); pb <- colSums(p)
  nz <- which(p > 0, arr.ind = TRUE)
  mi <- sum(vapply(seq_len(nrow(nz)), function(i) { r <- nz[i,1]; c <- nz[i,2]; p[r,c] * log(p[r,c] / (pa[r]*pb[c])) }, numeric(1)))
  entropy <- function(z) -sum(z[z > 0] * log(z[z > 0])); ha <- entropy(pa); hb <- entropy(pb)
  hom <- if (!ha) 1 else mi/ha; comp <- if (!hb) 1 else mi/hb
  c(nmi = if (!(ha+hb)) 1 else 2*mi/(ha+hb), homogeneity = hom, completeness = comp,
    v_measure = if (!(hom+comp)) 0 else 2*hom*comp/(hom+comp))
}
purity <- function(truth, cluster) sum(apply(table(cluster, truth), 1L, max)) / length(truth)

silhouette_summary <- function(layout, labels, max_n = 5000L, seed = 4L,
                               block_size = 256L) {
  x <- as.matrix(layout)
  keep <- apply(is.finite(x), 1L, all) & !is.na(labels)
  x <- x[keep, , drop = FALSE]
  labels <- labels[keep]
  if (nrow(x) < 2L || length(unique(labels)) < 2L) {
    return(c(silhouette = NA_real_, silhouette_class_min = NA_real_))
  }

  set.seed(seed + 313L)
  rows <- if (nrow(x) <= max_n) seq_len(nrow(x)) else {
    sort(sample.int(nrow(x), max_n))
  }
  x <- x[rows, , drop = FALSE]
  groups <- droplevels(factor(labels[rows]))
  group_id <- as.integer(groups)
  group_size <- tabulate(group_id, nbins = nlevels(groups))
  n <- nrow(x)
  block_size <- max(1L, as.integer(block_size))
  widths <- numeric(n)

  # Exact silhouette values, accumulated in row blocks to avoid materializing
  # the complete n x n distance matrix used by cluster::silhouette().
  for (first in seq.int(1L, n, by = block_size)) {
    last <- min(n, first + block_size - 1L)
    query <- x[first:last, , drop = FALSE]
    query_sq <- rowSums(query * query)
    reference_sq <- rowSums(x * x)
    distance_sq <- outer(query_sq, reference_sq, "+") -
      2 * tcrossprod(query, x)
    distances <- sqrt(pmax(distance_sq, 0))
    block_groups <- group_id[first:last]

    sums <- vapply(seq_len(nlevels(groups)), function(g) {
      rowSums(distances[, group_id == g, drop = FALSE])
    }, numeric(nrow(query)))
    if (is.null(dim(sums))) sums <- matrix(sums, nrow = nrow(query))

    for (i in seq_len(nrow(query))) {
      own <- block_groups[[i]]
      if (group_size[[own]] <= 1L) {
        widths[[first + i - 1L]] <- 0
        next
      }
      means <- sums[i, ] / group_size
      a <- sums[i, own] / (group_size[[own]] - 1L)
      means[[own]] <- Inf
      b <- min(means)
      widths[[first + i - 1L]] <- if (max(a, b) > 0) (b - a) / max(a, b) else 0
    }
  }

  class_means <- tapply(widths, groups, mean)
  c(silhouette = mean(widths), silhouette_class_min = min(class_means))
}

cluster_geometry_metrics <- function(layout, labels, seed=4L, dunn_max_n=2000L) {
  x <- as.matrix(layout); g <- factor(labels); keep <- !is.na(g)&apply(is.finite(x),1L,all); x<-x[keep,,drop=FALSE];g<-droplevels(g[keep])
  n<-nrow(x); k<-nlevels(g); if(n<2L||k<2L||k>=n)return(c(db_index=NA,calinski_harabasz=NA,within_ss=NA,between_ss=NA,separation_ratio=NA,dunn_sampled=NA))
  centers <- do.call(rbind,lapply(levels(g),function(z)colMeans(x[g==z,,drop=FALSE]))); counts<-as.numeric(table(g)); overall<-colMeans(x)
  within_by_class<-vapply(seq_len(k),function(j){z<-x[g==levels(g)[j],,drop=FALSE];sum(rowSums((z-matrix(centers[j,],nrow(z),ncol(x),byrow=TRUE))^2))},numeric(1))
  within<-sum(within_by_class); between<-sum(counts*rowSums((centers-matrix(overall,k,ncol(x),byrow=TRUE))^2))
  scatter<-sqrt(within_by_class/pmax(counts,1)); cd<-as.matrix(dist(centers)); ratios<-outer(scatter,scatter,"+")/pmax(cd,.Machine$double.eps);diag(ratios)<--Inf
  db<-mean(apply(ratios,1L,max)); ch<-(between/(k-1))/(within/(n-k)); sep<-between/pmax(within,.Machine$double.eps)
  set.seed(seed+1777L); rows<-if(n<=dunn_max_n)seq_len(n)else sort(sample.int(n,dunn_max_n)); dx<-as.matrix(dist(x[rows,,drop=FALSE])); gr<-g[rows]
  same<-outer(as.integer(gr),as.integer(gr),"=="); upper<-upper.tri(same); intra_mask<-same&upper;inter_mask<-(!same)&upper
  intra<-if(any(intra_mask))max(dx[intra_mask])else NA_real_;inter<-if(any(inter_mask))min(dx[inter_mask])else NA_real_
  c(db_index=db,calinski_harabasz=ch,within_ss=within,between_ss=between,separation_ratio=sep,dunn_sampled=inter/intra)
}

truth_geometry_metrics <- function(layout, truth, seed) c(silhouette_summary(layout,truth,seed=seed),cluster_geometry_metrics(layout,truth,seed=seed))

row_solution_hashes <- function(res) {
  if (is.null(res)) return(character())
  res <- as.matrix(res)
  vapply(seq_len(nrow(res)), function(i) hash_object(as.integer(res[i, ])), character(1))
}

agreement_prefix_metrics <- function(res, graph, seed, max_edges=1000000L) {
  res <- as.matrix(res); idx <- as.matrix(graph$indices %||% graph$index)
  if(!nrow(res)||is.null(idx)) return(data.frame())
  n <- ncol(res); if(nrow(idx)!=n) stop("Agreement graph/result sample mismatch")
  edge_count <- as.double(n) * ncol(idx)
  set.seed(seed + 9091L)
  edge_position <- if (edge_count > max_edges) {
    sort(sample.int(edge_count, max_edges))
  } else {
    seq_len(edge_count)
  }
  # R matrices are column-major, so a sampled linear position identifies the
  # source row without materializing the complete n-by-k edge list.
  from <- as.integer((edge_position - 1) %% n + 1)
  to <- as.integer(idx[edge_position])
  valid <- is.finite(to) & to >= 1L & to <= n & to != from
  from <- from[valid]; to <- to[valid]
  acc <- numeric(length(from)); prefixes <- unique(pmin(c(10L,20L,50L,100L),nrow(res))); out <- list(); previous<-0L
  for(prefix in prefixes){for(m in seq.int(previous+1L,prefix))acc<-acc+(res[m,from]==res[m,to]);out[[length(out)+1L]]<-data.frame(prefix=prefix,edges=length(from),mean_agreement=mean(acc/prefix),sd_agreement=sd(acc/prefix));previous<-prefix}
  do.call(rbind,out)
}

fit_summary <- function(fit) {
  acc <- as.numeric(fit$acc %||% numeric()); res <- as.matrix(fit$res %||% matrix(integer(), 0, 0))
  classes <- if (nrow(res)) apply(res, 1L, function(z) length(unique(z))) else numeric()
  hashes <- row_solution_hashes(res); rd <- fit$run_diagnostics %||% data.frame(); cd <- fit$cycle_diagnostics %||% data.frame()
  safe <- function(z, f) if (length(z)) f(z, na.rm = TRUE) else NA_real_
  data.frame(
    cv_mean = safe(acc, mean), cv_median = safe(acc, median), cv_sd = safe(acc, sd), cv_min = safe(acc, min), cv_max = safe(acc, max),
    classes_mean = safe(classes, mean), classes_median = safe(classes, median), classes_iqr = safe(classes, IQR),
    collapse_one_rate = safe(classes <= 1, mean), collapse_two_rate = safe(classes <= 2, mean),
    distinct_solutions = length(unique(hashes)), acceptance_rate = safe(cd$accepted, mean),
    improving_acceptance_rate = safe(cd$improving_acceptance, mean), temperature_acceptance_rate = safe(cd$temperature_acceptance, mean),
    proposal_size_mean = safe(cd$proposal_size, mean), cv_evaluations = sum(rd$cv_evaluations %||% 0), stringsAsFactors = FALSE
  )
}

cycle_deciles <- function(fit) {
  x <- fit$cycle_diagnostics %||% data.frame(); if (!nrow(x)) return(x)
  x$decile <- pmin(10L, pmax(1L, ceiling(10 * x$cycle / max(x$cycle))))
  do.call(rbind, lapply(split(x, x$decile), function(z) data.frame(
    decile = z$decile[[1]], cycles = nrow(z), proposal_size = mean(z$proposal_size), active_classes = mean(z$active_classes),
    acceptance_rate = mean(z$accepted), improving_rate = mean(z$improving_acceptance), temperature_rate = mean(z$temperature_acceptance))))
}

external_metrics <- function(truth, labels, layout, seed) c(
  ari = ari(truth, labels), information_metrics(truth, labels), purity = purity(truth, labels),
  truth_geometry_metrics(layout, truth, seed), kodama_label_silhouette = silhouette_summary(layout, labels, seed = seed)[["silhouette"]]
)

embedding_quality <- function(x, layout, truth, seed, max_n=5000L) {
  set.seed(seed+7919L); rows <- if(nrow(x)<=max_n) seq_len(nrow(x)) else sort(sample.int(nrow(x),max_n))
  out <- c(trust15=NA_real_,trust30=NA_real_,continuity15=NA_real_,continuity30=NA_real_,preserve15=NA_real_,preserve30=NA_real_,
    label_knn15=NA_real_,label_knn30=NA_real_,mean_neighbor_rank_error15=NA_real_,mean_neighbor_rank_error30=NA_real_,
    pair_spearman=NA_real_,distance_pearson=NA_real_,stress=NA_real_,density_spearman=NA_real_,density_pearson=NA_real_,
    density_log_radius_rmse=NA_real_,centroid_distance_correlation=NA_real_,rare_class_recall=NA_real_)
  if(requireNamespace("fastEmbedR",quietly=TRUE)) for(k in c(15L,30L)) {
    z <- tryCatch(fastEmbedR::evaluate_embedding(x_high=x[rows,,drop=FALSE],embedding=layout[rows,,drop=FALSE],labels=truth[rows],
      k=k,primary_k=k,seed=seed,backend="cpu",n_threads=1L,use_cache=FALSE,
      sample_size_for_global_metrics=length(rows),sample_size_for_local_metrics=length(rows)),error=function(e)NULL)
    if(!is.null(z)&&nrow(z)) {
      pick <- function(candidates){h<-intersect(candidates,names(z));if(length(h))as.numeric(z[[h[[1L]]]][[1L]])else NA_real_}
      out[[paste0("trust",k)]] <- pick(c("trustworthiness",paste0("trustworthiness_",k)))
      out[[paste0("continuity",k)]] <- pick(c("continuity",paste0("continuity_",k)))
      out[[paste0("preserve",k)]] <- pick(c(paste0("knn_preservation_",k),"knn_preservation","neighborhood_preservation"))
      out[[paste0("label_knn",k)]] <- pick(c("label_knn_accuracy","nn_accuracy",paste0("label_knn_accuracy_",k)))
      out[[paste0("mean_neighbor_rank_error",k)]] <- pick("mean_neighbor_rank_error")
      if(k==30L) for(nm in c("distance_pearson","stress","density_spearman","density_pearson","density_log_radius_rmse","centroid_distance_correlation","rare_class_recall")) out[[nm]]<-pick(nm)
    }
  }
  p <- min(length(rows),2000L); take <- if(length(rows)==p)seq_along(rows)else sort(sample.int(length(rows),p))
  out[["pair_spearman"]] <- suppressWarnings(cor(as.vector(dist(x[rows[take],,drop=FALSE])),as.vector(dist(layout[rows[take],,drop=FALSE])),method="spearman"))
  out
}

release_info <- function(image) {
  package_version <- function(x) if (requireNamespace(x, quietly=TRUE)) as.character(packageVersion(x)) else NA_character_
  loadNamespace("KODAMA")
  so <- tryCatch(getLoadedDLLs()[["KODAMA"]][["path"]], error = function(e) NA_character_)
  list(created = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"), image = normalizePath(image), image_sha256 = sha256(image),
       image_bytes = file.info(image)$size, KODAMA_version = package_version("KODAMA"), fastEmbedR_version = package_version("fastEmbedR"),
       KODAMA_shared_library = so, KODAMA_shared_library_sha256 = if (!is.na(so)) sha256(so) else NA_character_,
       r_version = R.version.string, host = unname(Sys.info()[["nodename"]]), os = paste(Sys.info()[c("sysname","release","machine")], collapse=" "),
       cpu = paste(system2("sh", c("-c", shQuote("lscpu 2>/dev/null | tr '\\n' ';'")), stdout=TRUE), collapse=""),
       threads = as.list(Sys.getenv(c("OMP_NUM_THREADS","RCPP_PARALLEL_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"), unset="")))
}
