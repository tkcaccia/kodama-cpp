# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)),"common.R"))
args<-parse_cli();root<-args$root%||%"/scratch/firenze/NN";code<-args$code_dir%||%file.path(root,"kodama_cpu4_jmlr")
release<-args$release%||%format(Sys.Date(),"%Y%m%d");run_root<-args$run_root%||%file.path(root,paste0("kodama_cpu4_jmlr_",release))
image<-args$image%||%file.path(root,"singularity","fastembedr_cuda.sif");data_root<-file.path(root,"Data");account<-args$account%||%"immunology";partition<-args$partition%||%"ada"
dir.create(file.path(run_root,"slurm"),recursive=TRUE,showWarnings=FALSE);dir.create(file.path(run_root,"logs"),recursive=TRUE,showWarnings=FALSE)
cells_path<-file.path(run_root,"cells.csv");cells<-build_cell_table(load_registry(file.path(code,"datasets.csv")));atomic_write_csv(cells,cells_path)
graphs<-unique(cells[,c("dataset","representation","seed")]);graphs$graph_id<-seq_len(nrow(graphs));atomic_write_csv(graphs,file.path(run_root,"graphs.csv"))
bundles<-graphs;names(bundles)[names(bundles)=="graph_id"]<-"bundle_id";atomic_write_csv(bundles,file.path(run_root,"bundles.csv"))
graph_bundles<-unique(cells[,c("dataset","representation")]);graph_bundles$graph_bundle_id<-seq_len(nrow(graph_bundles));atomic_write_csv(graph_bundles,file.path(run_root,"graph_bundles.csv"))
pca_path<-file.path(run_root,"prepared","imagenet_pca50.rds")
header<-function(name,time,mem,array=NULL,concurrency=4L)c("#!/usr/bin/env bash",paste0("#SBATCH --job-name=",name),paste0("#SBATCH --account=",account),paste0("#SBATCH --partition=",partition),"#SBATCH --nodes=1","#SBATCH --ntasks=1","#SBATCH --cpus-per-task=4",paste0("#SBATCH --mem=",mem),paste0("#SBATCH --time=",time),if(!is.null(array))paste0("#SBATCH --array=1-",array,"%",concurrency),paste0("#SBATCH --output=",run_root,"/logs/%x_%A_%a.out"),paste0("#SBATCH --error=",run_root,"/logs/%x_%A_%a.err"),"set -euo pipefail")
worker<-function(script,arguments)paste0("IMAGE=",shQuote(image)," SCRIPT=",shQuote(file.path(code,script))," bash ",shQuote(file.path(code,"run_worker.sh"))," ",arguments)
preflight<-c(header("kod_pre","02:00:00","32G"),worker("preflight.R",paste0("--image=",shQuote(image)," --data-root=",shQuote(data_root)," --out-dir=",shQuote(run_root)," --core-sha=",shQuote(args$core_sha%||%"")," --wrapper-sha=",shQuote(args$wrapper_sha%||%""))))
pca<-c(header("kod_pca","24:00:00","256G"),worker("prepare_imagenet_pca.R",paste0("--data-root=",shQuote(data_root)," --out=",shQuote(pca_path))))
graph<-c(header("kod_graph","3-00:00:00","256G",nrow(graph_bundles)),worker("prepare_graph_bundle.R",paste0("--bundles=",shQuote(file.path(run_root,"graph_bundles.csv"))," --bundle-id=$SLURM_ARRAY_TASK_ID --data-root=",shQuote(data_root)," --imagenet-pca=",shQuote(pca_path)," --prepared-root=",shQuote(file.path(run_root,"prepared_graphs"))," --status-root=",shQuote(file.path(run_root,"bundle_status","graphs")))))
bundle_args<-function(phase,out_root)paste0("--bundles=",shQuote(file.path(run_root,"bundles.csv"))," --bundle-id=$SLURM_ARRAY_TASK_ID --cells=",shQuote(cells_path)," --phase=",phase," --data-root=",shQuote(data_root)," --imagenet-pca=",shQuote(pca_path)," --prepared-root=",shQuote(file.path(run_root,"prepared_graphs"))," --out-root=",shQuote(out_root)," --status-root=",shQuote(file.path(run_root,"bundle_status")))
smoke<-c(header("kod_smoke","12:00:00","96G",nrow(bundles)),worker("run_cell_bundle.R",bundle_args("smoke",file.path(run_root,"smoke"))))
validate<-c(header("kod_validate","01:00:00","16G"),worker("validate_smoke.R",paste0("--cells=",shQuote(cells_path)," --smoke-root=",shQuote(file.path(run_root,"smoke"))," --out=",shQuote(file.path(run_root,"smoke_validation.csv")))))
full<-c(header("kod_full","7-00:00:00","256G",nrow(bundles)),worker("run_cell_bundle.R",bundle_args("full",file.path(run_root,"cells"))))
aggregate<-c(header("kod_agg","24:00:00","128G"),worker("aggregate.R",paste0("--run-root=",shQuote(run_root)," --data-root=",shQuote(data_root))))
scripts<-list(preflight=preflight,pca=pca,graphs=graph,smoke=smoke,validate=validate,full=full,aggregate=aggregate)
for(nm in names(scripts)){path<-file.path(run_root,"slurm",paste0(nm,".sh"));atomic_write_lines(scripts[[nm]],path);Sys.chmod(path,"0755")}
phase_names<-names(scripts)
for(i in seq_along(phase_names)){
  nm<-phase_names[[i]]
  submit<-c("#!/usr/bin/env bash","set -euo pipefail",paste0("RUN_ROOT=",shQuote(run_root)),paste0("sbatch \"$RUN_ROOT/slurm/",nm,".sh\""))
  path<-file.path(run_root,sprintf("submit_%02d_%s.sh",i,nm));atomic_write_lines(submit,path);Sys.chmod(path,"0755")
}
status<-c("#!/usr/bin/env bash","set -euo pipefail","squeue -u \"$USER\" -n kod_pre,kod_pca,kod_graph,kod_smoke,kod_validate,kod_full,kod_agg")
path<-file.path(run_root,"status.sh");atomic_write_lines(status,path);Sys.chmod(path,"0755")
unlink(file.path(run_root,c("submit_all.sh","submit_throttled.sh")))
atomic_write_json(list(run_root=run_root,image=image,account=account,partition=partition,graph_jobs=nrow(graph_bundles),smoke_jobs=nrow(bundles),full_jobs=nrow(bundles),smoke_policy_cells=sum(cells$experiment=="ablation"),full_cells=nrow(cells),maximum_kodama_jobs_in_one_phase=max(nrow(graph_bundles),nrow(bundles)),submitted=FALSE),file.path(run_root,"generation_manifest.json"))
cat(sprintf("Packed %d graph builds into %d graph jobs, %d smoke cells into %d smoke jobs, and %d full cells into %d full jobs. Run the seven submit_XX_phase.sh files in order, after each preceding phase completes. At most %d KODAMA jobs are submitted in any phase and at most four run concurrently. No jobs submitted.\n",nrow(graphs),nrow(graph_bundles),sum(cells$experiment=="ablation"),nrow(bundles),nrow(cells),nrow(bundles),max(nrow(graph_bundles),nrow(bundles))))
