# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)), "common.R"))
args <- parse_cli(); dataset <- args$dataset %||% stop("--dataset required"); representation <- args$representation %||% stop("--representation required")
seed <- as_int(args$seed,4L); data_root <- args$data_root %||% stop("--data-root required"); out <- args$out_dir %||% stop("--out-dir required")
pca_path <- args$imagenet_pca %||% NULL; check_protocol_api(TRUE); d <- load_dataset(dataset,representation,data_root,pca_path)
t0 <- proc.time()[["elapsed"]]; graph <- kodamaR::KODAMA.graph(d$x,k=100L,metric="euclidean",backend="cpu",n.cores=4L,seed=seed,storage="matrix")
seconds <- proc.time()[["elapsed"]]-t0
# Deliberately exclude raw data and truth labels. Matrix jobs reload them independently.
payload <- list(graph=graph,dataset=dataset,representation=representation,seed=seed,n=nrow(d$x),p=ncol(d$x),
  sample_order_sha256=hash_object(seq_len(nrow(d$x))),data_sha256=d$data_sha256,dataset_file_sha256=d$file_sha256,
  graph_seconds=seconds,graph_bytes=as.numeric(object.size(graph)),created=format(Sys.time(),"%Y-%m-%dT%H:%M:%S%z"))
path <- file.path(out,"prepared_graph.rds"); atomic_save_rds(payload,path,compress=FALSE)
atomic_write_csv(data.frame(dataset,representation,seed,n=nrow(d$x),p=ncol(d$x),graph_seconds=seconds,
 graph_bytes=payload$graph_bytes,graph_file_sha256=sha256(path),data_sha256=d$data_sha256,status="success"),file.path(out,"graph_metadata.csv"))
