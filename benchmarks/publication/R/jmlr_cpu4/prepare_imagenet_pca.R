# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)), "common.R"))
args <- parse_cli(); data_root <- args$data_root %||% stop("--data-root required"); out <- args$out %||% stop("--out required")
d <- load_dataset("imagenet","raw",data_root); t0 <- proc.time()[["elapsed"]]
pca <- kodamaR::KODAMA.pca(d$x,ncomp=50L,center=TRUE,scale=FALSE,backend="cpu",n.cores=4L,seed=4L)
seconds <- proc.time()[["elapsed"]]-t0; scores <- as.matrix(pca$scores %||% pca$x %||% pca)
payload <- list(scores=scores,sample_order_sha256=hash_object(seq_len(nrow(scores))),
  source_file_sha256=d$file_sha256,source_matrix_sha256=d$data_sha256,
  scores_sha256=hash_object(scores),n=nrow(scores),p=ncol(scores),seconds=seconds,
  singular_values=pca$singular_values %||% NULL,parameters=list(ncomp=50L,center=TRUE,scale=FALSE,backend="cpu",n.cores=4L,seed=4L))
atomic_save_rds(payload,out,compress=FALSE); atomic_write_csv(data.frame(n=nrow(scores),p=ncol(scores),seconds,bytes=object.size(scores),sha256=sha256(out)),paste0(out,".csv"))
