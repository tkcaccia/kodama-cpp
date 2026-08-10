.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)),"common.R"))
args <- parse_cli(); root <- args$run_root %||% stop("--run-root required"); cells <- read.csv(file.path(root,"cells.csv"),stringsAsFactors=FALSE)
data_root <- args$data_root %||% "/scratch/firenze/NN/Data"; pca_path <- file.path(root,"prepared","imagenet_pca50.rds")
rows <- list(); failures <- list(); run_rows <- list(); embedding_rows <- list()
for(i in seq_len(nrow(cells))) {
  z <- cells[i,]; dir <- file.path(root,"cells",z$dataset,z$representation,z$classifier,z$experiment,z$setting,paste0("seed_",z$seed)); p <- file.path(dir,"metrics.csv")
  if(file.exists(p)) {
    m <- read.csv(p,stringsAsFactors=FALSE); rows[[length(rows)+1L]] <- m
    rp <- file.path(dir,"run_metrics.csv"); if(file.exists(rp)) run_rows[[length(run_rows)+1L]] <- cbind(z,read.csv(rp,stringsAsFactors=FALSE))
    ep <- file.path(dir,"embedding_metrics.csv"); if(file.exists(ep)) embedding_rows[[length(embedding_rows)+1L]] <- cbind(z,read.csv(ep,stringsAsFactors=FALSE))
  } else failures[[length(failures)+1L]] <- data.frame(z,status="missing",error="No metrics.csv")
}
all <- if(length(rows)) do.call(rbind,rows) else data.frame(); fail <- if(length(failures)) do.call(rbind,failures) else data.frame()
out <- file.path(root,"aggregate"); dir.create(file.path(out,"figures"),recursive=TRUE,showWarnings=FALSE)
atomic_write_csv(all,file.path(out,"all_cells.csv")); atomic_write_csv(fail,file.path(out,"failures.csv"))
atomic_write_csv(if(length(run_rows))do.call(rbind,run_rows)else data.frame(),file.path(out,"all_runs.csv"))
atomic_write_csv(if(length(embedding_rows))do.call(rbind,embedding_rows)else data.frame(),file.path(out,"all_embeddings.csv"))
if(!nrow(all)) quit(save="no")
success <- all[all$status=="success",,drop=FALSE]
if(any(duplicated(success[,c("dataset","representation","classifier","experiment","setting","seed")]))) stop("Duplicate successful cell keys")
for(field in c("data_sha256","graph_sha256")) if(any(is.na(success[[field]])|!nzchar(success[[field]]))) stop("Missing identity field: ",field)

abl <- success[success$experiment=="ablation",,drop=FALSE]; full <- abl[abl$setting=="full",,drop=FALSE]
effects <- list(); endpoints <- intersect(c("cv_median","ari","nmi","silhouette","matrix_seconds","classes_median","distinct_solutions","acceptance_rate"),names(success))
for(i in seq_len(nrow(full))) for(policy in setdiff(unique(abl$setting),"full")) {
  f <- full[i,]; a <- abl[abl$dataset==f$dataset&abl$representation==f$representation&abl$classifier==f$classifier&abl$seed==f$seed&abl$setting==policy,,drop=FALSE]
  if(nrow(a)!=1L) next
  for(metric in endpoints) effects[[length(effects)+1L]] <- data.frame(dataset=f$dataset,representation=f$representation,classifier=f$classifier,seed=f$seed,policy,metric,full=f[[metric]],ablation=a[[metric]],difference=f[[metric]]-a[[metric]])
}
effects <- if(length(effects)) do.call(rbind,effects) else data.frame(); atomic_write_csv(effects,file.path(out,"ablation_seed_effects.csv"))
if(nrow(effects)) {
  med <- aggregate(difference~dataset+representation+classifier+policy+metric,effects,median,na.rm=TRUE)
  stats <- do.call(rbind,lapply(split(med,interaction(med$classifier,med$policy,med$metric,drop=TRUE)),function(z){
    x <- z$difference[is.finite(z$difference)]; data.frame(classifier=z$classifier[[1]],policy=z$policy[[1]],metric=z$metric[[1]],datasets=length(x),median=median(x),
      wilcoxon_p=if(length(x)>1)wilcox.test(x,mu=0,exact=length(x)<50)$p.value else NA,sign_p=if(length(x))binom.test(sum(x>0),sum(x!=0),0.5)$p.value else NA,
      rank_biserial=if(length(x)) (sum(x>0)-sum(x<0))/sum(x!=0) else NA)
  }))
  stats$holm_p <- ave(stats$wilcoxon_p,interaction(stats$classifier,stats$metric),FUN=function(x)p.adjust(x,"holm")); atomic_write_csv(stats,file.path(out,"ablation_statistics.csv"))
  set.seed(20260810L); boot <- do.call(rbind,lapply(split(med,interaction(med$classifier,med$policy,med$metric,drop=TRUE)),function(z){
    x<-z$difference[is.finite(z$difference)]; b<-if(length(x))replicate(2000L,median(sample(x,length(x),replace=TRUE)))else NA_real_
    data.frame(classifier=z$classifier[[1]],policy=z$policy[[1]],metric=z$metric[[1]],estimate=median(x),ci_low=quantile(b,.025,na.rm=TRUE),ci_high=quantile(b,.975,na.rm=TRUE))
  })); atomic_write_csv(boot,file.path(out,"ablation_bootstrap_ci.csv"))
  lodo <- do.call(rbind,lapply(split(med,interaction(med$classifier,med$policy,med$metric,drop=TRUE)),function(z)do.call(rbind,lapply(unique(z$dataset),function(left)data.frame(classifier=z$classifier[[1]],policy=z$policy[[1]],metric=z$metric[[1]],left_out=left,estimate=median(z$difference[z$dataset!=left],na.rm=TRUE))))))
  atomic_write_csv(lodo,file.path(out,"ablation_leave_one_dataset_out.csv"))
}
atomic_write_csv(success[success$experiment%in%c("knn_sensitivity","ncomp_sensitivity"),],file.path(out,"predictor_sensitivity.csv"))
atomic_write_csv(success[success$dataset=="imagenet",],file.path(out,"imagenet_raw_pca50.csv"))

emb <- if(length(embedding_rows))do.call(rbind,embedding_rows)else data.frame()
if(nrow(emb)) {
  classic <- emb[emb$classifier=="classic"&emb$experiment=="classic",,drop=FALSE]
  kodama <- emb[emb$classifier%in%c("knn","pls_lda")&emb$experiment=="ablation"&emb$setting=="full",,drop=FALSE]
  quality_metrics <- intersect(c("silhouette","silhouette_class_min","db_index","calinski_harabasz","separation_ratio","dunn_sampled",
    "trust15","trust30","continuity15","continuity30","preserve15","preserve30","label_knn15","label_knn30",
    "mean_neighbor_rank_error15","mean_neighbor_rank_error30","pair_spearman","distance_pearson","stress","density_spearman",
    "density_pearson","density_log_radius_rmse","centroid_distance_correlation","rare_class_recall"),names(emb))
  comparisons <- list()
  for(i in seq_len(nrow(kodama))) {
    z<-kodama[i,]; b<-classic[classic$dataset==z$dataset&classic$representation==z$representation&classic$seed==z$seed&classic$method==z$method,,drop=FALSE]
    if(nrow(b)!=1L)next
    for(metric in quality_metrics){delta<-z[[metric]]-b[[metric]];lower_better<-metric%in%c("db_index","mean_neighbor_rank_error15","mean_neighbor_rank_error30","stress","density_log_radius_rmse");comparisons[[length(comparisons)+1L]]<-data.frame(dataset=z$dataset,representation=z$representation,classifier=z$classifier,seed=z$seed,method=z$method,metric,classic=b[[metric]],kodama=z[[metric]],difference=delta,favorable_difference=if(lower_better)-delta else delta)}
  }
  comparisons<-if(length(comparisons))do.call(rbind,comparisons)else data.frame();atomic_write_csv(comparisons,file.path(out,"classic_vs_kodama_embedding_metrics.csv"))
  if(nrow(comparisons)) {
    med<-aggregate(favorable_difference~dataset+representation+classifier+method+metric,comparisons,median,na.rm=TRUE)
    set.seed(20260810L); summary<-do.call(rbind,lapply(split(med,interaction(med$classifier,med$method,med$metric,drop=TRUE)),function(z){x<-z$favorable_difference[is.finite(z$favorable_difference)];boot<-if(length(x))replicate(2000L,median(sample(x,length(x),replace=TRUE)))else NA_real_;data.frame(classifier=z$classifier[[1]],method=z$method[[1]],metric=z$metric[[1]],datasets=length(x),median_favorable_difference=median(x),ci_low=quantile(boot,.025,na.rm=TRUE),ci_high=quantile(boot,.975,na.rm=TRUE),wilcoxon_p=if(length(x)>1)wilcox.test(x,mu=0,exact=length(x)<50)$p.value else NA,sign_p=if(length(x))binom.test(sum(x>0),sum(x!=0),.5)$p.value else NA)}))
    summary$holm_p<-ave(summary$wilcoxon_p,interaction(summary$classifier,summary$method),FUN=function(x)p.adjust(x,"holm"));atomic_write_csv(summary,file.path(out,"classic_vs_kodama_summary.csv"))
  }
}

pdf(file.path(out,"figures","runtime_quality_overview.pdf"),width=11,height=8.5)
par(mfrow=c(2,2),mar=c(4,4,2,1)); plot(success$matrix_seconds,success$ari,pch=19,col=as.integer(factor(success$classifier)),log="x",xlab="Matrix time (s, log)",ylab="ARI",main="Runtime-quality")
boxplot(matrix_seconds~classifier,data=success,log="y",ylab="Seconds",main="CPU4 matrix runtime")
boxplot(ari~classifier,data=success,ylab="ARI",main="External diagnostic")
boxplot(classes_median~classifier,data=success,ylab="Median active classes",main="Solution complexity"); dev.off()

read_xy <- function(path) { z<-read.csv(path); as.matrix(z[,c("x","y")]) }
for(ds in unique(success$dataset)) for(rep in unique(success$representation[success$dataset==ds])) for(seed in seeds) for(classifier in c("knn","pls_lda")) {
  base <- success[success$dataset==ds&success$representation==rep&success$seed==seed&success$classifier==classifier&success$experiment=="ablation",,drop=FALSE]
  if(!nrow(base)||!any(base$setting=="full")) next
  adverse_pool <- base[base$setting!="full"&is.finite(base$cv_median),,drop=FALSE]
  adverse <- if(nrow(adverse_pool)) adverse_pool$setting[[which.min(adverse_pool$cv_median)]] else NA_character_
  paths <- function(setting,method,cl=classifier,ex="ablation") file.path(root,"cells",ds,rep,cl,ex,setting,paste0("seed_",seed),paste0(method,".csv"))
  wanted <- c(paths("umap","umap","classic","classic"),paths("full","umap"),paths(adverse,"umap"),paths("opentsne","opentsne","classic","classic"),paths("full","opentsne"),paths(adverse,"opentsne"))
  if(any(!file.exists(wanted))) next
  d <- tryCatch(load_dataset(ds,rep,data_root,pca_path),error=function(e)NULL); if(is.null(d)) next
  set.seed(seed+404L); idx <- if(nrow(d$x)<=20000L)seq_len(nrow(d$x))else sort(sample.int(nrow(d$x),20000L)); cols <- as.integer(d$labels[idx])
  pdf(file.path(out,"figures",paste0(safe_name(ds),"_",safe_name(rep),"_",classifier,"_seed",seed,"_six_panel.pdf")),width=12,height=8)
  par(mfrow=c(2,3),mar=c(2,2,3,1)); titles<-c("Classic UMAP","Full KODAMA UMAP",paste("Adverse",adverse,"UMAP"),"Classic openTSNE","Full KODAMA openTSNE",paste("Adverse",adverse,"openTSNE"))
  for(j in seq_along(wanted)){xy<-read_xy(wanted[[j]]);plot(xy[idx,],pch=16,cex=.25,col=cols,axes=FALSE,xlab="",ylab="",asp=1,main=titles[[j]])}; dev.off()
}
writeLines(capture.output(sessionInfo()),file.path(out,"sessionInfo.txt")); cat("Aggregation complete; failures retained\n")
