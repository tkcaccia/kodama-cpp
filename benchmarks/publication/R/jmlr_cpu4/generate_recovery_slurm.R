# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)), "common.R"))

args <- parse_cli()
run_root <- args$run_root %||% stop("--run-root required")
code <- args$code_dir %||% dirname(normalizePath(.this_file))
runtime_code <- args$runtime_code_dir %||% code
image <- args$image %||% stop("--image required")
data_root <- args$data_root %||% file.path(dirname(run_root), "Data")
account <- args$account %||% "immunology"
partition <- args$partition %||% "ada"
cells_path <- file.path(run_root, "cells.csv")
cells <- read.csv(cells_path, stringsAsFactors = FALSE)
pca_path <- file.path(run_root, "prepared", "imagenet_pca50.rds")
prepared_root <- file.path(run_root, "prepared_graphs")
out_root <- file.path(run_root, "cells")
recovery_root <- file.path(run_root, "recovery")
slurm_root <- file.path(recovery_root, "slurm")
status_root <- file.path(recovery_root, "status")
dir.create(slurm_root, recursive = TRUE, showWarnings = FALSE)
dir.create(status_root, recursive = TRUE, showWarnings = FALSE)

setting_path <- function(cell) {
  if (!is.na(cell$value) && nzchar(cell$value)) paste0(cell$setting, "_", cell$value) else cell$setting
}
cell_output <- function(cell) file.path(
  out_root, cell$dataset, cell$representation, cell$classifier,
  cell$experiment, setting_path(cell), paste0("seed_", cell$seed)
)
cell_succeeded <- function(cell) {
  out <- cell_output(cell)
  exit <- file.path(out, "exit_status.txt")
  metric_path <- file.path(out, "metrics.csv")
  if (!file.exists(exit) || !file.exists(metric_path)) return(FALSE)
  metric <- tryCatch(
    read.csv(metric_path, stringsAsFactors = FALSE),
    error = function(e) NULL
  )
  identical(readLines(exit, warn = FALSE), "0") &&
    !is.null(metric) && nrow(metric) && metric$status[[1L]] == "success"
}

smoke_output <- function(cell) file.path(
  run_root, "smoke", cell$dataset, cell$representation, cell$classifier,
  cell$experiment, setting_path(cell), paste0("seed_", cell$seed)
)
smoke_succeeded <- function(cell) {
  out <- smoke_output(cell)
  exit <- file.path(out, "exit_status.txt")
  metric_path <- file.path(out, "metrics.csv")
  run_path <- file.path(out, "run_diagnostics.rds")
  cycle_path <- file.path(out, "cycle_diagnostics.rds")
  if (!file.exists(exit) || !file.exists(metric_path) ||
      !file.exists(run_path) || !file.exists(cycle_path)) return(FALSE)
  metric <- tryCatch(read.csv(metric_path, stringsAsFactors = FALSE),
                     error = function(e) NULL)
  identical(readLines(exit, warn = FALSE), "0") &&
    !is.null(metric) && nrow(metric) && metric$status[[1L]] == "success"
}

graphs <- unique(cells[, c("dataset", "representation", "seed")])
graph_exists <- vapply(seq_len(nrow(graphs)), function(i) {
  file.exists(file.path(
    prepared_root, graphs$dataset[[i]], graphs$representation[[i]],
    paste0("seed_", graphs$seed[[i]]), "prepared_graph.rds"
  ))
}, logical(1))
missing_graphs <- graphs[!graph_exists, , drop = FALSE]
missing_graphs$graph_id <- seq_len(nrow(missing_graphs))
atomic_write_csv(missing_graphs, file.path(recovery_root, "missing_graphs.csv"))

success <- vapply(seq_len(nrow(cells)), function(i) cell_succeeded(cells[i, , drop = FALSE]), logical(1))
failed <- cells[!success, , drop = FALSE]
atomic_write_csv(failed, file.path(recovery_root, "failed_cells.csv"))

high_memory <- function(dataset, representation) {
  dataset == "FlowRepository" || dataset == "imagenet"
}
is_large <- mapply(high_memory, failed$dataset, failed$representation)

dimensions <- data.frame(
  dataset = c("COIL20", "FashionMNIST", "MNIST", "Macosko2015_retina", "MetRef",
              "TabulaMuris", "USPS", "flow18", "imagenet", "mass41", "FlowRepository"),
  n = c(1440, 70000, 70000, 44808, 873, 100102, 11000, 1000021, 1281167, 965282, 5220347),
  p = c(16384, 784, 784, 50, 375, 50, 256, 11, 1024, 14, 32),
  stringsAsFactors = FALSE
)
cell_cost <- function(x) {
  d <- dimensions[match(x$dataset, dimensions$dataset), , drop = FALSE]
  p <- ifelse(x$dataset == "imagenet" & x$representation == "pca50", 50, d$p)
  landmarks <- pmin(100000, pmax(2, ceiling(0.75 * d$n)))
  value <- suppressWarnings(as.numeric(x$value))
  k <- ifelse(x$experiment == "knn_sensitivity", value, 30)
  nc <- ifelse(x$experiment == "ncomp_sensitivity", value, 50)
  ifelse(x$classifier == "pls_lda", landmarks * p^2 * pmax(1, nc) / 50,
         ifelse(x$classifier == "knn", landmarks * pmax(1, k), d$n * p * 10))
}
balanced <- function(x, requested) {
  if (!nrow(x)) return(data.frame(bundle_id = integer(), cell_id = integer()))
  n_bundles <- min(requested, nrow(x))
  cost <- cell_cost(x)
  loads <- numeric(n_bundles)
  assignment <- integer(nrow(x))
  for (i in order(cost, decreasing = TRUE)) {
    bundle <- which.min(loads)
    assignment[[i]] <- bundle
    loads[[bundle]] <- loads[[bundle]] + cost[[i]]
  }
  data.frame(bundle_id = assignment, cell_id = x$cell_id)[order(assignment), , drop = FALSE]
}

recovery_standard <- balanced(failed[!is_large, , drop = FALSE], 24L)
recovery_large <- balanced(failed[is_large, , drop = FALSE], 8L)
smoke_cells <- cells[cells$experiment == "ablation", , drop = FALSE]
smoke_success <- vapply(seq_len(nrow(smoke_cells)), function(i) {
  smoke_succeeded(smoke_cells[i, , drop = FALSE])
}, logical(1))
failed_smoke <- smoke_cells[!smoke_success, , drop = FALSE]
smoke_large_mask <- if (nrow(failed_smoke)) {
  mapply(high_memory, failed_smoke$dataset, failed_smoke$representation)
} else logical()
smoke_standard <- balanced(failed_smoke[!smoke_large_mask, , drop = FALSE], 30L)
smoke_large <- balanced(failed_smoke[smoke_large_mask, , drop = FALSE], 6L)
refresh_large_mask <- mapply(high_memory, cells$dataset, cells$representation)
refresh_standard <- balanced(cells[!refresh_large_mask, , drop = FALSE], 36L)
refresh_large <- balanced(cells[refresh_large_mask, , drop = FALSE], 12L)
atomic_write_csv(recovery_standard, file.path(recovery_root, "recovery_standard.csv"))
atomic_write_csv(recovery_large, file.path(recovery_root, "recovery_large.csv"))
atomic_write_csv(smoke_standard, file.path(recovery_root, "smoke_standard.csv"))
atomic_write_csv(smoke_large, file.path(recovery_root, "smoke_large.csv"))
atomic_write_csv(refresh_standard, file.path(recovery_root, "refresh_standard.csv"))
atomic_write_csv(refresh_large, file.path(recovery_root, "refresh_large.csv"))

header <- function(name, time, memory, array = NULL, concurrency = 4L) c(
  "#!/usr/bin/env bash",
  paste0("#SBATCH --job-name=", name),
  paste0("#SBATCH --account=", account),
  paste0("#SBATCH --partition=", partition),
  "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=4",
  paste0("#SBATCH --mem=", memory), paste0("#SBATCH --time=", time),
  if (!is.null(array)) paste0("#SBATCH --array=1-", array, "%", concurrency),
  paste0("#SBATCH --output=", run_root, "/logs/%x_%A_%a.out"),
  paste0("#SBATCH --error=", run_root, "/logs/%x_%A_%a.err"),
  "set -euo pipefail"
)
worker <- function(script, arguments) paste0(
  "IMAGE=", shQuote(image), " SCRIPT=", shQuote(file.path(runtime_code, script)),
  " bash ", shQuote(file.path(runtime_code, "run_worker.sh")), " ", arguments
)
write_script <- function(name, lines) {
  path <- file.path(slurm_root, paste0(name, ".sh"))
  atomic_write_lines(lines, path)
  Sys.chmod(path, "0755")
  path
}

scripts <- list()
if (nrow(missing_graphs)) {
  scripts$graphs <- write_script("graphs", c(
    header("kod_fix_graph", "3-00:00:00", "64G", nrow(missing_graphs), 3L),
    worker("prepare_missing_graph.R", paste0(
      "--graphs=", shQuote(file.path(recovery_root, "missing_graphs.csv")),
      " --graph-id=$SLURM_ARRAY_TASK_ID --data-root=", shQuote(data_root),
      " --imagenet-pca=", shQuote(pca_path), " --prepared-root=", shQuote(prepared_root)
    ))
  ))
}

smoke_args <- function(bundle_file, status_name) paste0(
  "--bundles=", shQuote(bundle_file),
  " --bundle-id=$SLURM_ARRAY_TASK_ID --cells=", shQuote(cells_path),
  " --phase=smoke --data-root=", shQuote(data_root),
  " --imagenet-pca=", shQuote(pca_path), " --prepared-root=", shQuote(prepared_root),
  " --out-root=", shQuote(file.path(run_root, "smoke")),
  " --status-root=", shQuote(file.path(status_root, status_name))
)
if (nrow(smoke_standard)) scripts$smoke_standard <- write_script("smoke_standard", c(
  header("kod_fix_smoke", "12:00:00", "16G", max(smoke_standard$bundle_id), 12L),
  worker("run_cell_bundle.R", smoke_args(file.path(recovery_root, "smoke_standard.csv"), "smoke_standard"))
))
if (nrow(smoke_large)) scripts$smoke_large <- write_script("smoke_large", c(
  header("kod_fix_smkbig", "12:00:00", "72G", max(smoke_large$bundle_id), 2L),
  worker("run_cell_bundle.R", smoke_args(file.path(recovery_root, "smoke_large.csv"), "smoke_large"))
))
scripts$validate_smoke <- write_script("validate_smoke", c(
  header("kod_fix_smkv", "01:00:00", "16G"),
  worker("validate_smoke.R", paste0(
    "--cells=", shQuote(cells_path), " --smoke-root=", shQuote(file.path(run_root, "smoke")),
    " --out=", shQuote(file.path(run_root, "smoke_validation.csv"))
  ))
))

recovery_args <- function(bundle_file, phase, status_name) paste0(
  "--bundles=", shQuote(file.path(recovery_root, bundle_file)),
  " --bundle-id=$SLURM_ARRAY_TASK_ID --cells=", shQuote(cells_path),
  " --phase=", phase, " --data-root=", shQuote(data_root),
  " --imagenet-pca=", shQuote(pca_path), " --prepared-root=", shQuote(prepared_root),
  " --out-root=", shQuote(out_root), " --status-root=", shQuote(file.path(status_root, status_name))
)
if (nrow(recovery_standard)) scripts$recovery_standard <- write_script("recovery_standard", c(
  header("kod_fix_std", "7-00:00:00", "20G", max(recovery_standard$bundle_id), 10L),
  worker("run_cell_bundle.R", recovery_args("recovery_standard.csv", "full_standard", "full_standard"))
))
if (nrow(recovery_large)) scripts$recovery_large <- write_script("recovery_large", c(
  header("kod_fix_big", "7-00:00:00", "96G", max(recovery_large$bundle_id), 2L),
  worker("run_cell_bundle.R", recovery_args("recovery_large.csv", "full_large", "full_large"))
))

refresh_args <- function(bundle_file, status_name) paste0(
  "--bundles=", shQuote(file.path(recovery_root, bundle_file)),
  " --bundle-id=$SLURM_ARRAY_TASK_ID --cells=", shQuote(cells_path),
  " --data-root=", shQuote(data_root), " --imagenet-pca=", shQuote(pca_path),
  " --out-root=", shQuote(out_root), " --status-root=", shQuote(file.path(status_root, status_name))
)
scripts$refresh_standard <- write_script("refresh_standard", c(
  header("kod_fix_metric", "2-00:00:00", "20G", max(refresh_standard$bundle_id), 10L),
  worker("refresh_metrics_bundle.R", refresh_args("refresh_standard.csv", "refresh_standard"))
))
scripts$refresh_large <- write_script("refresh_large", c(
  header("kod_fix_metbig", "2-00:00:00", "96G", max(refresh_large$bundle_id), 2L),
  worker("refresh_metrics_bundle.R", refresh_args("refresh_large.csv", "refresh_large"))
))
scripts$validate_complete <- write_script("validate_complete", c(
  header("kod_fix_check", "06:00:00", "32G"),
  worker("validate_complete.R", paste0("--run-root=", shQuote(run_root)))
))
scripts$aggregate <- write_script("aggregate", c(
  header("kod_fix_agg", "24:00:00", "32G"),
  worker("aggregate.R", paste0("--run-root=", shQuote(run_root), " --data-root=", shQuote(data_root)))
))

submit <- c(
  "#!/usr/bin/env bash", "set -euo pipefail",
  paste0("SLURM_ROOT=", shQuote(slurm_root)),
  "submit_after() { local dep=\"$1\"; shift; if [[ -n \"$dep\" ]]; then sbatch --parsable --dependency=\"afterok:$dep\" \"$@\"; else sbatch --parsable \"$@\"; fi; }",
  "join_ids() { local out=\"\"; for id in \"$@\"; do [[ -z \"$id\" ]] && continue; out=\"${out:+$out:}$id\"; done; printf '%s' \"$out\"; }"
)
if (!is.null(scripts$graphs)) submit <- c(submit, "GRAPH_JOB=$(submit_after \"\" \"$SLURM_ROOT/graphs.sh\")") else submit <- c(submit, "GRAPH_JOB=\"\"")
if (!is.null(scripts$smoke_standard)) submit <- c(submit, "SMOKE_STD=$(submit_after \"$GRAPH_JOB\" \"$SLURM_ROOT/smoke_standard.sh\")") else submit <- c(submit, "SMOKE_STD=\"\"")
if (!is.null(scripts$smoke_large)) submit <- c(submit, "SMOKE_BIG=$(submit_after \"$GRAPH_JOB\" \"$SLURM_ROOT/smoke_large.sh\")") else submit <- c(submit, "SMOKE_BIG=\"\"")
submit <- c(submit,
  "SMOKE_DEPS=$(join_ids \"$SMOKE_STD\" \"$SMOKE_BIG\")",
  "SMOKE_VALID=$(submit_after \"$SMOKE_DEPS\" \"$SLURM_ROOT/validate_smoke.sh\")"
)
if (!is.null(scripts$recovery_standard)) submit <- c(submit, "REC_STD=$(submit_after \"$SMOKE_VALID\" \"$SLURM_ROOT/recovery_standard.sh\")") else submit <- c(submit, "REC_STD=\"\"")
if (!is.null(scripts$recovery_large)) submit <- c(submit, "REC_BIG=$(submit_after \"$SMOKE_VALID\" \"$SLURM_ROOT/recovery_large.sh\")") else submit <- c(submit, "REC_BIG=\"\"")
submit <- c(submit,
  "REC_DEPS=$(join_ids \"$REC_STD\" \"$REC_BIG\")",
  "REF_STD=$(submit_after \"$REC_DEPS\" \"$SLURM_ROOT/refresh_standard.sh\")",
  "REF_BIG=$(submit_after \"$REC_DEPS\" \"$SLURM_ROOT/refresh_large.sh\")",
  "REF_DEPS=$(join_ids \"$REF_STD\" \"$REF_BIG\")",
  "CHECK_JOB=$(submit_after \"$REF_DEPS\" \"$SLURM_ROOT/validate_complete.sh\")",
  "AGG_JOB=$(submit_after \"$CHECK_JOB\" \"$SLURM_ROOT/aggregate.sh\")",
  "printf 'graphs=%s smoke_standard=%s smoke_large=%s smoke_validation=%s recovery_standard=%s recovery_large=%s refresh_standard=%s refresh_large=%s validation=%s aggregate=%s\\n' \"$GRAPH_JOB\" \"$SMOKE_STD\" \"$SMOKE_BIG\" \"$SMOKE_VALID\" \"$REC_STD\" \"$REC_BIG\" \"$REF_STD\" \"$REF_BIG\" \"$CHECK_JOB\" \"$AGG_JOB\""
)
submit_path <- file.path(recovery_root, "submit_recovery.sh")
atomic_write_lines(submit, submit_path)
Sys.chmod(submit_path, "0755")

manifest <- list(
  generated = as.character(Sys.time()), total_cells = nrow(cells), successful_cells = sum(success),
  recovery_cells = nrow(failed), missing_graphs = nrow(missing_graphs),
  successful_smoke_cells = sum(smoke_success), recovery_smoke_cells = nrow(failed_smoke),
  recovery_standard_jobs = if (nrow(recovery_standard)) max(recovery_standard$bundle_id) else 0,
  recovery_large_jobs = if (nrow(recovery_large)) max(recovery_large$bundle_id) else 0,
  refresh_standard_jobs = max(refresh_standard$bundle_id),
  refresh_large_jobs = max(refresh_large$bundle_id), submitted = FALSE
)
atomic_write_json(manifest, file.path(recovery_root, "recovery_manifest.json"))
cat("Recovery plan: ", sum(success), " successful cells retained, ", nrow(failed),
    " cells to rerun, ", nrow(missing_graphs), " graphs to build. No jobs submitted.\n", sep = "")
