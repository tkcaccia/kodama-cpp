# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)), "common.R"))
args <- parse_cli()
cells_path <- args$cells %||% stop("--cells required")
cells <- read.csv(cells_path, stringsAsFactors = FALSE)
runner <- file.path(dirname(normalizePath(.this_file)), "recover_cell.R")
rscript <- file.path(R.home("bin"), "Rscript")

base_args <- c(
  paste0("--cells=", cells_path),
  paste0("--data-root=", args$data_root),
  paste0("--imagenet-pca=", args$imagenet_pca),
  paste0("--prepared-root=", args$prepared_root),
  paste0("--out-root=", args$out_root)
)
records <- list()
for (i in seq_len(nrow(cells))) {
  cell <- cells[i, , drop = FALSE]
  if (cell$experiment == "classic") next
  setting_path <- if (!is.na(cell$value) && nzchar(cell$value)) {
    paste0(cell$setting, "_", cell$value)
  } else cell$setting
  out <- file.path(
    args$out_root, cell$dataset, cell$representation, cell$classifier,
    cell$experiment, setting_path, paste0("seed_", cell$seed)
  )
  labels <- file.path(out, "labels.rds")
  exit_status <- file.path(out, "exit_status.txt")
  if (!file.exists(labels)) next
  if (file.exists(exit_status) && identical(readLines(exit_status), "0")) next
  started <- Sys.time()
  cat("Recovering cell ", cell$cell_id, "\n", sep = "")
  status <- system2(
    rscript,
    c(runner, paste0("--cell-id=", cell$cell_id), base_args)
  )
  records[[length(records) + 1L]] <- data.frame(
    cell_id = cell$cell_id, status = status,
    started = as.character(started), ended = as.character(Sys.time())
  )
}
report <- if (length(records)) do.call(rbind, records) else data.frame()
atomic_write_csv(report, file.path(args$out_root, "recovery_status.csv"))
if (nrow(report) && any(report$status != 0L)) quit(save = "no", status = 1L)
cat("Checkpoint recovery complete: ", nrow(report), " cell(s)\n", sep = "")
