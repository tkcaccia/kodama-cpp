# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)),"common.R"))
args<-parse_cli();root<-args$root%||%"/scratch/firenze/NN";code<-args$code_dir%||%file.path(root,"kodama_cpu4_jmlr")
runtime_code<-args$runtime_code_dir%||%code
release<-args$release%||%format(Sys.Date(),"%Y%m%d");run_root<-args$run_root%||%file.path(root,paste0("kodama_cpu4_jmlr_",release));output_dir<-args$output_dir%||%run_root
image<-args$image%||%file.path(root,"singularity","fastembedr_cuda.sif");data_root<-file.path(root,"Data");account<-args$account%||%"immunology";partition<-args$partition%||%"ada"
dir.create(file.path(output_dir,"slurm"),recursive=TRUE,showWarnings=FALSE);dir.create(file.path(output_dir,"logs"),recursive=TRUE,showWarnings=FALSE)
unlink(Sys.glob(file.path(output_dir,"slurm","*.sh")))
unlink(Sys.glob(file.path(output_dir,"submit_*.sh")))
cells_path<-file.path(run_root,"cells.csv");cells<-build_cell_table(load_registry(file.path(code,"datasets.csv")));atomic_write_csv(cells,file.path(output_dir,"cells.csv"))
graphs<-unique(cells[,c("dataset","representation","seed")]);graphs$graph_id<-seq_len(nrow(graphs));atomic_write_csv(graphs,file.path(output_dir,"graphs.csv"))
bundles<-graphs;names(bundles)[names(bundles)=="graph_id"]<-"bundle_id";atomic_write_csv(bundles,file.path(output_dir,"bundles.csv"))
graph_bundles<-unique(cells[,c("dataset","representation")]);graph_bundles$graph_bundle_id<-seq_len(nrow(graph_bundles));atomic_write_csv(graph_bundles,file.path(output_dir,"graph_bundles.csv"))
high_memory_dataset <- function(dataset, representation) {
  dataset == "FlowRepository" | (dataset == "imagenet" & representation == "raw")
}
graph_large <- high_memory_dataset(graph_bundles$dataset, graph_bundles$representation)
graph_standard_bundles <- graph_bundles[!graph_large,,drop=FALSE]
graph_standard_bundles$graph_bundle_id <- seq_len(nrow(graph_standard_bundles))
graph_large_bundles <- graph_bundles[graph_large,,drop=FALSE]
graph_large_bundles$graph_bundle_id <- seq_len(nrow(graph_large_bundles))
atomic_write_csv(graph_standard_bundles,file.path(output_dir,"graph_standard_bundles.csv"))
atomic_write_csv(graph_large_bundles,file.path(output_dir,"graph_large_bundles.csv"))
smoke_large <- high_memory_dataset(bundles$dataset, bundles$representation)
smoke_standard_bundles <- bundles[!smoke_large,,drop=FALSE]
smoke_standard_bundles$bundle_id <- seq_len(nrow(smoke_standard_bundles))
smoke_large_bundles <- bundles[smoke_large,,drop=FALSE]
smoke_large_bundles$bundle_id <- seq_len(nrow(smoke_large_bundles))
atomic_write_csv(smoke_standard_bundles,file.path(output_dir,"smoke_standard_bundles.csv"))
atomic_write_csv(smoke_large_bundles,file.path(output_dir,"smoke_large_bundles.csv"))

dataset_dimensions <- data.frame(
  dataset=c("COIL20","FashionMNIST","MNIST","Macosko2015_retina","MetRef","TabulaMuris","USPS","flow18","imagenet","mass41","FlowRepository"),
  n=c(1440,70000,70000,44808,873,100102,11000,1000021,1281167,965282,5220347),
  p=c(16384,784,784,50,375,50,256,11,1024,14,32), stringsAsFactors=FALSE
)
cell_cost <- function(x) {
  d <- dataset_dimensions[match(x$dataset,dataset_dimensions$dataset),,drop=FALSE]
  p <- ifelse(x$dataset=="imagenet" & x$representation=="pca50",50,d$p)
  landmarks <- pmin(100000,pmax(2,ceiling(0.75*d$n)))
  k <- ifelse(x$experiment=="knn_sensitivity",as.numeric(x$value),30)
  nc <- ifelse(x$experiment=="ncomp_sensitivity",as.numeric(x$value),50)
  ifelse(x$classifier=="pls_lda",landmarks*p^2*pmax(1,nc)/50,
    ifelse(x$classifier=="knn",landmarks*pmax(1,k),d$n*p*10))
}
balanced_members <- function(x,n_bundles) {
  if(!nrow(x))return(data.frame(bundle_id=integer(),cell_id=integer(),estimated_cost=numeric()))
  cost<-cell_cost(x);order_idx<-order(cost,decreasing=TRUE);loads<-numeric(n_bundles);assigned<-integer(nrow(x))
  for(i in order_idx){bundle<-which.min(loads);assigned[[i]]<-bundle;loads[[bundle]]<-loads[[bundle]]+cost[[i]]}
  data.frame(bundle_id=assigned,cell_id=x$cell_id,estimated_cost=cost)[order(assigned,-cost),,drop=FALSE]
}
large_mask <- high_memory_dataset(cells$dataset, cells$representation)
standard_bundles <- balanced_members(cells[!large_mask,,drop=FALSE],72L)
large_bundles <- balanced_members(cells[large_mask,,drop=FALSE],24L)
atomic_write_csv(standard_bundles,file.path(output_dir,"full_standard_bundles.csv"))
atomic_write_csv(large_bundles,file.path(output_dir,"full_large_bundles.csv"))
pca_path<-file.path(run_root,"prepared","imagenet_pca50.rds")
header<-function(name,time,mem,array=NULL,concurrency=4L)c("#!/usr/bin/env bash",paste0("#SBATCH --job-name=",name),paste0("#SBATCH --account=",account),paste0("#SBATCH --partition=",partition),"#SBATCH --nodes=1","#SBATCH --ntasks=1","#SBATCH --cpus-per-task=4",paste0("#SBATCH --mem=",mem),paste0("#SBATCH --time=",time),if(!is.null(array))paste0("#SBATCH --array=1-",array,"%",concurrency),paste0("#SBATCH --output=",run_root,"/logs/%x_%A_%a.out"),paste0("#SBATCH --error=",run_root,"/logs/%x_%A_%a.err"),"set -euo pipefail")
worker<-function(script,arguments)paste0("IMAGE=",shQuote(image)," SCRIPT=",shQuote(file.path(runtime_code,script))," bash ",shQuote(file.path(runtime_code,"run_worker.sh"))," ",arguments)
preflight<-c(header("kod_pre","02:00:00","32G"),worker("preflight.R",paste0("--image=",shQuote(image)," --data-root=",shQuote(data_root)," --out-dir=",shQuote(run_root)," --core-sha=",shQuote(args$core_sha%||%"")," --wrapper-sha=",shQuote(args$wrapper_sha%||%""))))
pca<-c(header("kod_pca","24:00:00","64G"),worker("prepare_imagenet_pca.R",paste0("--data-root=",shQuote(data_root)," --out=",shQuote(pca_path))))
graph_args<-function(bundle_file,status_name)paste0("--bundles=",shQuote(file.path(run_root,bundle_file))," --bundle-id=$SLURM_ARRAY_TASK_ID --data-root=",shQuote(data_root)," --imagenet-pca=",shQuote(pca_path)," --prepared-root=",shQuote(file.path(run_root,"prepared_graphs"))," --status-root=",shQuote(file.path(run_root,"bundle_status",status_name)))
graphs_standard<-c(header("kod_graph_std","3-00:00:00","32G",nrow(graph_standard_bundles),10L),worker("prepare_graph_bundle.R",graph_args("graph_standard_bundles.csv","graphs_standard")))
graphs_large<-c(header("kod_graph_big","3-00:00:00","64G",nrow(graph_large_bundles),2L),worker("prepare_graph_bundle.R",graph_args("graph_large_bundles.csv","graphs_large")))
bundle_args<-function(phase,out_root,bundle_file="bundles.csv",status_name="")paste0("--bundles=",shQuote(file.path(run_root,bundle_file))," --bundle-id=$SLURM_ARRAY_TASK_ID --cells=",shQuote(cells_path)," --phase=",phase," --data-root=",shQuote(data_root)," --imagenet-pca=",shQuote(pca_path)," --prepared-root=",shQuote(file.path(run_root,"prepared_graphs"))," --out-root=",shQuote(out_root)," --status-root=",shQuote(file.path(run_root,"bundle_status",status_name)))
smoke_standard<-c(header("kod_smoke_std","12:00:00","32G",nrow(smoke_standard_bundles),20L),worker("run_cell_bundle.R",bundle_args("smoke",file.path(run_root,"smoke"),"smoke_standard_bundles.csv","smoke_standard")))
smoke_large<-c(header("kod_smoke_big","12:00:00","64G",nrow(smoke_large_bundles),4L),worker("run_cell_bundle.R",bundle_args("smoke",file.path(run_root,"smoke"),"smoke_large_bundles.csv","smoke_large")))
validate<-c(header("kod_validate","01:00:00","16G"),worker("validate_smoke.R",paste0("--cells=",shQuote(cells_path)," --smoke-root=",shQuote(file.path(run_root,"smoke"))," --out=",shQuote(file.path(run_root,"smoke_validation.csv")))))
full_standard<-c(header("kod_full_std","7-00:00:00","32G",max(standard_bundles$bundle_id),20L),worker("run_cell_bundle.R",bundle_args("full_standard",file.path(run_root,"cells"),"full_standard_bundles.csv")))
full_large<-c(header("kod_full_big","7-00:00:00","64G",max(large_bundles$bundle_id),4L),worker("run_cell_bundle.R",bundle_args("full_large",file.path(run_root,"cells"),"full_large_bundles.csv")))
complete<-c(header("kod_complete","06:00:00","32G"),worker("validate_complete.R",paste0("--run-root=",shQuote(run_root))))
aggregate<-c(header("kod_agg","24:00:00","32G"),worker("aggregate.R",paste0("--run-root=",shQuote(run_root)," --data-root=",shQuote(data_root))))
scripts<-list(preflight=preflight,pca=pca,graphs_standard=graphs_standard,graphs_large=graphs_large,smoke_standard=smoke_standard,smoke_large=smoke_large,validate=validate,full_standard=full_standard,full_large=full_large,complete=complete,aggregate=aggregate)
for(nm in names(scripts)){path<-file.path(output_dir,"slurm",paste0(nm,".sh"));atomic_write_lines(scripts[[nm]],path);Sys.chmod(path,"0755")}
phase_names<-names(scripts)
for(i in seq_along(phase_names)){
  nm<-phase_names[[i]]
  submit<-c("#!/usr/bin/env bash","set -euo pipefail",paste0("RUN_ROOT=",shQuote(run_root)),paste0("sbatch \"$RUN_ROOT/slurm/",nm,".sh\""))
  path<-file.path(output_dir,sprintf("submit_%02d_%s.sh",i,nm));atomic_write_lines(submit,path);Sys.chmod(path,"0755")
}
pipeline<-c(
  "#!/usr/bin/env bash", "set -euo pipefail", paste0("RUN_ROOT=",shQuote(run_root)),
  "submit_after() { local dep=\"$1\"; shift; if [[ -n \"$dep\" ]]; then sbatch --parsable --dependency=\"afterok:$dep\" \"$@\"; else sbatch --parsable \"$@\"; fi; }",
  "join_ids() { local out=\"\"; for id in \"$@\"; do [[ -z \"$id\" ]] && continue; out=\"${out:+$out:}$id\"; done; printf '%s' \"$out\"; }",
  "PRE=$(submit_after \"\" \"$RUN_ROOT/slurm/preflight.sh\")",
  "PCA=$(submit_after \"$PRE\" \"$RUN_ROOT/slurm/pca.sh\")",
  "GRAPH_STD=$(submit_after \"$PCA\" \"$RUN_ROOT/slurm/graphs_standard.sh\")",
  "GRAPH_BIG=$(submit_after \"$PCA\" \"$RUN_ROOT/slurm/graphs_large.sh\")",
  "GRAPH_DEPS=$(join_ids \"$GRAPH_STD\" \"$GRAPH_BIG\")",
  "SMOKE_STD=$(submit_after \"$GRAPH_DEPS\" \"$RUN_ROOT/slurm/smoke_standard.sh\")",
  "SMOKE_BIG=$(submit_after \"$GRAPH_DEPS\" \"$RUN_ROOT/slurm/smoke_large.sh\")",
  "SMOKE_DEPS=$(join_ids \"$SMOKE_STD\" \"$SMOKE_BIG\")",
  "VALID=$(submit_after \"$SMOKE_DEPS\" \"$RUN_ROOT/slurm/validate.sh\")",
  "FULL_STD=$(submit_after \"$VALID\" \"$RUN_ROOT/slurm/full_standard.sh\")",
  "FULL_BIG=$(submit_after \"$VALID\" \"$RUN_ROOT/slurm/full_large.sh\")",
  "FULL_DEPS=$(join_ids \"$FULL_STD\" \"$FULL_BIG\")",
  "COMPLETE=$(submit_after \"$FULL_DEPS\" \"$RUN_ROOT/slurm/complete.sh\")",
  "AGG=$(submit_after \"$COMPLETE\" \"$RUN_ROOT/slurm/aggregate.sh\")",
  "printf 'preflight=%s pca=%s graph_standard=%s graph_large=%s smoke_standard=%s smoke_large=%s validation=%s full_standard=%s full_large=%s complete=%s aggregate=%s\\n' \"$PRE\" \"$PCA\" \"$GRAPH_STD\" \"$GRAPH_BIG\" \"$SMOKE_STD\" \"$SMOKE_BIG\" \"$VALID\" \"$FULL_STD\" \"$FULL_BIG\" \"$COMPLETE\" \"$AGG\""
)
path<-file.path(output_dir,"submit_pipeline.sh");atomic_write_lines(pipeline,path);Sys.chmod(path,"0755")
status<-c("#!/usr/bin/env bash","set -euo pipefail","squeue -u \"$USER\" -n kod_pre,kod_pca,kod_graph_std,kod_graph_big,kod_smoke_std,kod_smoke_big,kod_validate,kod_full_std,kod_full_big,kod_complete,kod_agg")
path<-file.path(output_dir,"status.sh");atomic_write_lines(status,path);Sys.chmod(path,"0755")
unlink(file.path(output_dir,c("submit_all.sh","submit_throttled.sh")))
atomic_write_json(list(run_root=run_root,output_dir=output_dir,image=image,code_dir=code,runtime_code_dir=runtime_code,account=account,partition=partition,graph_standard_jobs=nrow(graph_standard_bundles),graph_large_jobs=nrow(graph_large_bundles),smoke_standard_jobs=nrow(smoke_standard_bundles),smoke_large_jobs=nrow(smoke_large_bundles),full_standard_jobs=max(standard_bundles$bundle_id),full_large_jobs=max(large_bundles$bundle_id),smoke_policy_cells=sum(cells$experiment=="ablation"),full_cells=nrow(cells),maximum_running_kodama_jobs=20,submitted=FALSE),file.path(output_dir,"generation_manifest.json"))
cat(sprintf("Prepared %d graph datasets as %d standard and %d high-memory jobs, %d smoke bundles as %d standard and %d high-memory jobs, and %d full cells as %d standard plus %d high-cost jobs. Computation remains limited to 4 threads; only high-memory jobs request 64G. No jobs submitted.\n",nrow(graph_bundles),nrow(graph_standard_bundles),nrow(graph_large_bundles),nrow(bundles),nrow(smoke_standard_bundles),nrow(smoke_large_bundles),nrow(cells),max(standard_bundles$bundle_id),max(large_bundles$bundle_id)))
