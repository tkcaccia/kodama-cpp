# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

.this_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))][[1L]])
source(file.path(dirname(normalizePath(.this_file)), "common.R"))
args <- parse_cli(); out <- args$out %||% file.path(suite_dir(), "cells.csv")
reg <- load_registry(); cells <- build_cell_table(reg)
atomic_write_csv(cells, out)
cat(sprintf("Wrote %d cells (%d graphs, %d smoke policy cells) to %s\n", nrow(cells), nrow(reg)*length(seeds), nrow(reg)*length(seeds)*(length(common_policies)+length(pls_policies)), out))
