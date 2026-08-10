# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)), "common.R"))
args <- parse_cli(); out <- args$out_dir %||% stop("--out-dir required")
image <- args$image %||% stop("--image required"); data_root <- args$data_root %||% stop("--data-root required")
dir.create(out, recursive=TRUE, showWarnings=FALSE)
status <- check_protocol_api(FALSE); atomic_write_csv(status, file.path(out,"api_capabilities.csv"))
manifest <- release_info(image)
manifest$core_commit_expected <- args$core_sha %||% Sys.getenv("KODAMA_CORE_SHA", unset="")
manifest$wrapper_commit_expected <- args$wrapper_sha %||% Sys.getenv("KODAMA_R_SHA", unset="")
manifest$command <- paste(commandArgs(FALSE), collapse=" ")
atomic_write_json(manifest, file.path(out,"release_manifest.json"))
capture.output(sessionInfo(), file=file.path(out,"sessionInfo.txt"))

reg <- load_registry(); ds <- vector("list",nrow(reg))
for (i in seq_len(nrow(reg))) {
  if (reg$dataset[[i]] == "imagenet" && reg$representation[[i]] == "pca50") next
  z <- tryCatch(load_dataset(reg$dataset[[i]], reg$representation[[i]], data_root), error=identity)
  ds[[i]] <- data.frame(dataset=reg$dataset[[i]], representation=reg$representation[[i]],
    status=if(inherits(z,"error")) "failed" else "ready", n=if(inherits(z,"error")) NA else nrow(z$x),
    p=if(inherits(z,"error")) NA else ncol(z$x), classes=if(inherits(z,"error")) NA else nlevels(z$labels),
    file_sha256=if(inherits(z,"error")) NA else z$file_sha256, data_sha256=if(inherits(z,"error")) NA else z$data_sha256,
    label_sha256=if(inherits(z,"error")) NA else z$label_sha256, source=reg$source[[i]], preprocessing=reg$preprocessing[[i]],
    version=reg$version[[i]], license=reg$license[[i]], error=if(inherits(z,"error")) conditionMessage(z) else NA_character_)
}
ds <- do.call(rbind, ds[!vapply(ds,is.null,logical(1))]); atomic_write_csv(ds,file.path(out,"dataset_manifest_pre_pca.csv"))
if (!requireNamespace("fastEmbedR",quietly=TRUE)) stop("fastEmbedR is required for classic fuzzy UMAP/openTSNE benchmarks")
if (any(!status$available) || any(ds$status != "ready")) stop("Preflight failed; see machine-readable reports")

# Exercise every hidden native policy. This proves dispatch without emulating it in R.
set.seed(4); x <- matrix(rnorm(120), 30, 4)
g <- kodamaR::KODAMA.graph(x, k=10L, metric="euclidean", backend="cpu", n.cores=4L, seed=4L, storage="matrix")
classic_checks <- list(
  fuzzy_umap=tryCatch(fastEmbedR::umap(x,n_neighbors=10L,graph_mode="fuzzy",backend="cpu",seed=4L),error=identity),
  opentsne=tryCatch(fastEmbedR::opentsne(x,perplexity=5,backend="cpu",seed=4L,n_iter=20L,early_exaggeration_iter=10L),error=identity))
classic_status <- data.frame(method=names(classic_checks),ok=!vapply(classic_checks,inherits,logical(1),"error"),
  error=vapply(classic_checks,function(z)if(inherits(z,"error"))conditionMessage(z)else NA_character_,character(1)))
atomic_write_csv(classic_status,file.path(out,"classic_embedding_preflight.csv"));if(any(!classic_status$ok))stop("Classic fastEmbedR embedding preflight failed")
checks <- list()
for (classifier in c("knn","pls_lda")) for (policy in if(classifier=="knn") common_policies else pls_policies) {
  z <- tryCatch(matrix_call(x,g,classifier,2L,2L,5L,2L,30L,10L,5L,4L,policy), error=identity)
  ok <- !inherits(z,"error") && all(c("run_diagnostics","cycle_diagnostics") %in% names(z))
  checks[[length(checks)+1L]] <- data.frame(classifier,policy,ok,
    cv_evaluations=if(ok) sum(z$run_diagnostics$cv_evaluations) else NA,
    expected=if(ok) 2L*(2L+1L) else 6L, error=if(inherits(z,"error")) conditionMessage(z) else NA_character_)
}
checks <- do.call(rbind,checks); atomic_write_csv(checks,file.path(out,"policy_preflight.csv"))
if (any(!checks$ok) || any(checks$cv_evaluations != checks$expected)) stop("Native policy/diagnostic preflight failed")
atomic_write_lines("ready",file.path(out,"PREFLIGHT_OK")); cat("Preflight passed\n")
