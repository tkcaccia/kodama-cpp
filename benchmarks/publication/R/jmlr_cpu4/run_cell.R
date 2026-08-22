# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)), "common.R"))
args <- parse_cli(); cells <- read.csv(args$cells %||% stop("--cells required"),stringsAsFactors=FALSE)
id <- as_int(args$cell_id %||% Sys.getenv("SLURM_ARRAY_TASK_ID")); if(is.na(id)||!id%in%cells$cell_id) stop("Invalid cell id")
cell <- cells[cells$cell_id==id,,drop=FALSE]; phase <- args$phase %||% "full"; M <- if(phase=="smoke") 2L else 100L; Tcycle <- if(phase=="smoke") 2L else 100L
if(phase=="smoke" && cell$experiment != "ablation") quit(save="no",status=0L)
check_protocol_api(TRUE); pca_path <- args$imagenet_pca %||% NULL
d <- load_dataset(cell$dataset,cell$representation,args$data_root,pca_path)
graph_path <- file.path(args$prepared_root,cell$dataset,cell$representation,paste0("seed_",cell$seed),"prepared_graph.rds")
prepared <- readRDS(graph_path)
if(!identical(prepared$data_sha256,d$data_sha256)||prepared$n!=nrow(d$x)) stop("Prepared graph/data identity mismatch")
setting_path <- if (!is.na(cell$value) && nzchar(cell$value))
  paste0(cell$setting, "_", cell$value) else cell$setting
out <- file.path(args$out_root,cell$dataset,cell$representation,cell$classifier,
                 cell$experiment,setting_path,paste0("seed_",cell$seed))
if(file.exists(file.path(out,"exit_status.txt"))&&identical(readLines(file.path(out,"exit_status.txt")),"0")) quit(save="no")
dir.create(out,recursive=TRUE,showWarnings=FALSE); status <- 1L; started <- Sys.time(); warnings <- character(); stage <- "initialization"
atomic_write_csv(data.frame(cell,status="running",phase,stage=stage,started=as.character(started)),file.path(out,"progress.csv"))
withCallingHandlers(tryCatch({
  if(cell$experiment=="classic") {
    t0 <- proc.time()[["elapsed"]]
    layout <- if(cell$setting=="umap") extract_layout(fastEmbedR::umap(d$x,n_neighbors=30L,graph_mode="fuzzy",backend="cpu",seed=cell$seed)) else extract_layout(fastEmbedR::opentsne(d$x,perplexity=30,backend="cpu",seed=cell$seed))
    wall <- proc.time()[["elapsed"]]-t0; fit <- NULL; layouts <- setNames(list(layout),cell$setting)
    visual_seconds <- setNames(wall,cell$setting)
  } else {
    policy <- if(cell$experiment=="ablation") cell$setting else "full"; k <- if(cell$experiment=="knn_sensitivity") as.integer(cell$value) else 30L
    nc <- if(cell$experiment=="ncomp_sensitivity") as.integer(cell$value) else 50L
    stage <- "KODAMA.matrix"
    atomic_write_csv(data.frame(cell,status="running",phase,stage=stage,M,Tcycle,started=as.character(started)),file.path(out,"progress.csv"))
    t0 <- proc.time()[["elapsed"]]; fit <- matrix_call(d$x,prepared$graph,cell$classifier,M,Tcycle,5L,nc,100000L,if(nrow(d$x)<40000L)100L else 300L,k,cell$seed,policy,progress=TRUE,progress_file=file.path(out,"native_progress.log")); wall <- proc.time()[["elapsed"]]-t0
    stage <- "normalize KODAMA result"
    fit <- normalize_fit_result(fit, M, nrow(d$x))
    atomic_save_rds(list(best_labels=fit$best_labels,res=fit$res,acc=fit$acc),file.path(out,"labels.rds"),compress=TRUE)
    atomic_save_rds(fit$run_diagnostics,file.path(out,"run_diagnostics.rds")); atomic_save_rds(fit$cycle_diagnostics,file.path(out,"cycle_diagnostics.rds"))
    fit_stats <- fit_summary(fit)
    atomic_write_csv(fit_stats,file.path(out,"run_metrics.csv"))
    atomic_write_csv(cycle_deciles(fit),file.path(out,"cycle_deciles.csv"))
    atomic_write_csv(
      data.frame(
        stage=c(names(fit$timing),"matrix_wall"),
        seconds=c(unlist(fit$timing),wall)
      ),
      file.path(out,"matrix_timing_checkpoint.csv")
    )
    atomic_write_csv(agreement_prefix_metrics(fit$res,prepared$graph,cell$seed),file.path(out,"agreement_prefix.csv"))
    assert_visualizable_fit(fit)
    stage <- "KODAMA visualization"
    u0<-proc.time()[["elapsed"]]; umap<-extract_layout(KODAMA::KODAMA.visualization(fit,method="UMAP",k=30L,graph.mode="fuzzy",backend="cpu",n.cores=4L,seed=cell$seed)); usec<-proc.time()[["elapsed"]]-u0
    t0<-proc.time()[["elapsed"]]; opentsne<-extract_layout(KODAMA::KODAMA.visualization(fit,method="opentsne",perplexity=30,backend="cpu",n.cores=4L,seed=cell$seed)); tsec<-proc.time()[["elapsed"]]-t0
    layouts <- list(umap=umap,opentsne=opentsne); visual_seconds<-c(umap=usec,opentsne=tsec)
    layout <- layouts$umap
    stage <- "KODAMA summaries"
  }
  for(nm in names(layouts)) atomic_write_csv(data.frame(sample=seq_len(nrow(layouts[[nm]])),x=layouts[[nm]][,1],y=layouts[[nm]][,2]),file.path(out,paste0(nm,".csv")))
  embedding_rows <- do.call(rbind,lapply(names(layouts),function(nm){
    lab <- if(is.null(fit)) NULL else fit$best_labels
    q <- embedding_quality(d$x,layouts[[nm]],d$labels,cell$seed)
    e <- if(is.null(lab))c(ari=NA,nmi=NA,homogeneity=NA,completeness=NA,v_measure=NA,purity=NA,truth_geometry_metrics(layouts[[nm]],d$labels,cell$seed),kodama_label_silhouette=NA)else external_metrics(d$labels,lab,layouts[[nm]],cell$seed)
    data.frame(method=nm,seconds=unname(visual_seconds[[nm]]),t(e),t(q))
  })); atomic_write_csv(embedding_rows,file.path(out,"embedding_metrics.csv"))
  ext <- if(is.null(fit)) c(ari=NA,nmi=NA,homogeneity=NA,completeness=NA,v_measure=NA,purity=NA,silhouette_summary(layout,d$labels,seed=cell$seed)) else external_metrics(d$labels,fit$best_labels,layout,cell$seed)
  metrics <- data.frame(cell,status="success",phase,M,Tcycle,folds=if(is.null(fit))NA else 5L,n=nrow(d$x),p=ncol(d$x),workers=4L,
    requested_k=if(cell$classifier=="knn") if(cell$experiment=="knn_sensitivity")as.integer(cell$value) else 30L else NA,
    requested_ncomp=if(cell$classifier=="pls_lda") if(cell$experiment=="ncomp_sensitivity")as.integer(cell$value) else 50L else NA,
    matrix_seconds=wall,graph_seconds=prepared$graph_seconds,pipeline_seconds=wall+prepared$graph_seconds,peak_memory_mb=fit$peak_memory_mb %||% NA,
    t(ext),nonfinite_layout=sum(!is.finite(layout)),warnings=paste(unique(warnings),collapse=" | "),fallbacks="",data_sha256=d$data_sha256,
    graph_sha256=sha256(graph_path),started=as.character(started),ended=as.character(Sys.time()))
  if(!is.null(fit)) metrics <- cbind(metrics,fit_stats)
  atomic_write_csv(metrics,file.path(out,"metrics.csv")); atomic_write_csv(if(is.null(fit))data.frame(stage=paste0("visualization_",names(visual_seconds)),seconds=visual_seconds) else data.frame(stage=c(names(fit$timing),"matrix_wall",paste0("visualization_",names(visual_seconds))),seconds=c(unlist(fit$timing),wall,visual_seconds)),file.path(out,"timing.csv"))
  atomic_write_csv(data.frame(metric=c("reported_peak_mb","input_object_mb","prepared_graph_mb","fit_object_mb","layouts_mb"),
    value=c(if(is.null(fit))NA else fit$peak_memory_mb %||% NA,as.numeric(object.size(d$x))/2^20,as.numeric(object.size(prepared$graph))/2^20,
      if(is.null(fit))0 else as.numeric(object.size(fit))/2^20,as.numeric(object.size(layouts))/2^20)),file.path(out,"memory.csv"))
  atomic_write_json(list(cell=as.list(cell),phase=phase,M=M,Tcycle=Tcycle,data_sha256=d$data_sha256,graph_sha256=sha256(graph_path)),file.path(out,"manifest.json")); status <- 0L
  atomic_write_csv(data.frame(cell,status="success",phase,stage="complete",M,Tcycle,started=as.character(started),ended=as.character(Sys.time())),file.path(out,"progress.csv"))
},error=function(e) {
  failure <- data.frame(cell,status="failed",phase,stage=stage,error=conditionMessage(e),started=as.character(started),ended=as.character(Sys.time()))
  atomic_write_csv(failure,file.path(out,"metrics.csv")); atomic_write_csv(failure,file.path(out,"progress.csv"))
}),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
atomic_write_lines(as.character(status),file.path(out,"exit_status.txt")); if(status) quit(save="no",status=1L)
