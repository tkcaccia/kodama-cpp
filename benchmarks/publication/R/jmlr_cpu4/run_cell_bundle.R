# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)),"common.R"))
args <- parse_cli(); bundle_id <- as_int(args$bundle_id %||% Sys.getenv("SLURM_ARRAY_TASK_ID"))
bundles <- read.csv(args$bundles %||% stop("--bundles required"),stringsAsFactors=FALSE)
cells <- read.csv(args$cells %||% stop("--cells required"),stringsAsFactors=FALSE)
members <- bundles[bundles$bundle_id==bundle_id,,drop=FALSE];if(!nrow(members))stop("Invalid bundle id")
phase <- args$phase %||% "full"
if ("cell_id" %in% names(members)) {
  selected <- cells[match(members$cell_id, cells$cell_id),,drop=FALSE]
} else {
  if(nrow(members)!=1L)stop("Legacy bundle id is duplicated")
  selected <- cells[cells$dataset==members$dataset&cells$representation==members$representation&cells$seed==members$seed,,drop=FALSE]
}
if(phase=="smoke")selected<-selected[selected$experiment=="ablation",,drop=FALSE]
runner <- file.path(dirname(normalizePath(.this_file)),"run_cell.R"); rscript <- file.path(R.home("bin"),"Rscript")
base_args <- c(paste0("--cells=",args$cells),paste0("--phase=",phase),paste0("--data-root=",args$data_root),
  paste0("--imagenet-pca=",args$imagenet_pca),paste0("--prepared-root=",args$prepared_root),paste0("--out-root=",args$out_root))
records<-list();failed<-FALSE;out<-file.path(args$status_root,phase,paste0("bundle_",bundle_id,".csv"))
for(i in seq_len(nrow(selected))){id<-selected$cell_id[[i]];started<-Sys.time();cat(sprintf("bundle=%d phase=%s cell=%d (%d/%d)\n",bundle_id,phase,id,i,nrow(selected)));status<-system2(rscript,c(runner,paste0("--cell-id=",id),base_args));records[[i]]<-data.frame(bundle_id,phase,cell_id=id,status,started=as.character(started),ended=as.character(Sys.time()));atomic_write_csv(do.call(rbind,records),out);if(status!=0L){failed<-TRUE;if(phase=="smoke")break}}
report<-do.call(rbind,records)
if(failed)quit(save="no",status=1L)
