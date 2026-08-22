# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)), "common.R"))

args <- parse_cli()
graphs <- read.csv(args$graphs %||% stop("--graphs required"), stringsAsFactors = FALSE)
id <- as_int(args$graph_id %||% Sys.getenv("SLURM_ARRAY_TASK_ID"))
if (is.na(id) || !id %in% graphs$graph_id) stop("Invalid graph id")
graph <- graphs[graphs$graph_id == id, , drop = FALSE]
out <- file.path(
  args$prepared_root, graph$dataset, graph$representation,
  paste0("seed_", graph$seed)
)
target <- file.path(out, "prepared_graph.rds")
if (file.exists(target)) quit(save = "no", status = 0L)

runner <- file.path(dirname(normalizePath(.this_file)), "prepare_graph.R")
status <- system2(
  file.path(R.home("bin"), "Rscript"),
  c(
    runner,
    paste0("--dataset=", graph$dataset),
    paste0("--representation=", graph$representation),
    paste0("--seed=", graph$seed),
    paste0("--data-root=", args$data_root),
    paste0("--imagenet-pca=", args$imagenet_pca),
    paste0("--out-dir=", out)
  )
)
if (status != 0L || !file.exists(target)) {
  stop("Graph preparation failed for ", graph$dataset, "/", graph$representation,
       "/seed_", graph$seed)
}
