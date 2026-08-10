# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)),"common.R"))
args <- parse_cli(); cells <- read.csv(args$cells,stringsAsFactors=FALSE); root <- args$smoke_root %||% stop("--smoke-root required")
expected <- cells[cells$experiment=="ablation",,drop=FALSE]; rows <- list(); hashes <- list()
for(i in seq_len(nrow(expected))) {
  z <- expected[i,]; path <- file.path(root,z$dataset,z$representation,z$classifier,z$experiment,z$setting,paste0("seed_",z$seed))
  metric <- file.path(path,"metrics.csv"); run <- file.path(path,"run_diagnostics.rds"); cyc <- file.path(path,"cycle_diagnostics.rds")
  ok <- file.exists(metric)&&file.exists(run)&&file.exists(cyc)&&file.exists(file.path(path,"exit_status.txt"))&&readLines(file.path(path,"exit_status.txt"))[[1]]=="0"
  reason <- ""; cv <- NA_integer_; identity <- NA_character_
  if(ok) {
    rd <- readRDS(run); cd <- readRDS(cyc); cv <- sum(rd$cv_evaluations)
    required_run <- c("landmark_rows_hash","initial_labels_hash","fold_assignments_hash","cv_evaluations")
    required_cycle <- c("proposal_size","active_classes","accepted","improving_acceptance","temperature_acceptance")
    ok <- all(required_run%in%names(rd))&&all(required_cycle%in%names(cd))&&cv==2L*(2L+1L)
    identity <- hash_object(rd[,c("landmark_rows_hash","initial_labels_hash","fold_assignments_hash")])
    if(!ok) reason <- "diagnostic schema, identity, or CV evaluation mismatch"
  } else reason <- "missing or failed smoke output"
  rows[[i]] <- data.frame(z,ok,cv_evaluations=cv,identity_hash=identity,reason)
}
report <- do.call(rbind,rows)
# For a matched dataset/classifier/seed, policy must not alter sampling/folds.
key <- interaction(report$dataset,report$representation,report$classifier,report$seed,drop=TRUE)
identity_ok <- vapply(split(report$identity_hash,key),function(x) length(unique(x[!is.na(x)]))==1L,logical(1))
atomic_write_csv(report,args$out %||% file.path(dirname(root),"smoke_validation.csv"))
if(any(!report$ok)||any(!identity_ok)) stop("Smoke validation failed; full array must not start")
atomic_write_lines("ready",file.path(dirname(root),"SMOKE_OK")); cat("Smoke schema and matched identities passed\n")
