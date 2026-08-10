.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)),"common.R"))
args<-parse_cli();bundle_id<-as_int(args$bundle_id%||%Sys.getenv("SLURM_ARRAY_TASK_ID"));bundles<-read.csv(args$bundles,stringsAsFactors=FALSE)
bundle<-bundles[bundles$graph_bundle_id==bundle_id,,drop=FALSE];if(nrow(bundle)!=1L)stop("Invalid graph bundle id")
runner<-file.path(dirname(normalizePath(.this_file)),"prepare_graph.R");rscript<-file.path(R.home("bin"),"Rscript");records<-list()
for(i in seq_along(seeds)){seed<-seeds[[i]];out<-file.path(args$prepared_root,bundle$dataset,bundle$representation,paste0("seed_",seed));started<-Sys.time();status<-system2(rscript,c(runner,paste0("--dataset=",bundle$dataset),paste0("--representation=",bundle$representation),paste0("--seed=",seed),paste0("--data-root=",args$data_root),paste0("--imagenet-pca=",args$imagenet_pca),paste0("--out-dir=",out)));records[[i]]<-data.frame(graph_bundle_id=bundle_id,dataset=bundle$dataset,representation=bundle$representation,seed,status,started=as.character(started),ended=as.character(Sys.time()));if(status!=0L){atomic_write_csv(do.call(rbind,records),file.path(args$status_root,paste0("bundle_",bundle_id,".csv")));quit(save="no",status=1L)}}
atomic_write_csv(do.call(rbind,records),file.path(args$status_root,paste0("bundle_",bundle_id,".csv")))
