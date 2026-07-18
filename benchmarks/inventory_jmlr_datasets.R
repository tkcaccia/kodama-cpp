#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

options(stringsAsFactors = FALSE)

value_or <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}

root <- normalizePath(value_or("KODAMA_DATA_ROOT", "/mnt/sata_ssd/fastEmbedR/Data"), mustWork = TRUE)
out <- value_or("KODAMA_INVENTORY_OUT", file.path(root, "kodama_dataset_inventory.csv"))
max_bytes <- as.numeric(value_or("KODAMA_INVENTORY_MAX_BYTES", "1000000000"))

files <- list.files(root, pattern = "\\.(RData|rda)$", recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
files <- sort(files)

extract_dataset <- function(e) {
  names <- ls(e, all.names = TRUE)
  if ("dataset" %in% names) {
    ds <- get("dataset", envir = e, inherits = FALSE)
    if (is.list(ds) && !is.null(ds$data) && !is.null(ds$labels)) {
      return(list(data = ds$data, labels = ds$labels, object = "dataset"))
    }
  }
  for (name in names) {
    object <- get(name, envir = e, inherits = FALSE)
    if (is.list(object) && !is.null(object$data) && !is.null(object$labels)) {
      return(list(data = object$data, labels = object$labels, object = name))
    }
  }
  data_name <- intersect(c("data", "x", "X", "u"), names)[1L]
  label_name <- intersect(c("labels", "label", "y", "class"), names)[1L]
  if (!is.na(data_name) && !is.na(label_name)) {
    return(list(
      data = get(data_name, envir = e, inherits = FALSE),
      labels = get(label_name, envir = e, inherits = FALSE),
      object = paste(data_name, label_name, sep = "+")
    ))
  }
  stop("No list with $data and $labels was found")
}

inspect_one <- function(path) {
  info <- file.info(path)
  relative <- substring(path, nchar(root) + 2L)
  if (is.finite(max_bytes) && info$size > max_bytes) {
    return(data.frame(
      dataset = sub("\\.(RData|rda)$", "", basename(path), ignore.case = TRUE),
      relative_path = relative,
      bytes = info$size,
      samples = NA_integer_,
      variables = NA_integer_,
      classes = NA_integer_,
      data_class = NA_character_,
      data_object = NA_character_,
      status = "skipped_size",
      error = ""
    ))
  }
  tryCatch({
    e <- new.env(parent = emptyenv())
    load(path, envir = e)
    ds <- extract_dataset(e)
    dims <- if (inherits(ds$data, "float32") || inherits(ds$data, "float")) {
      dim(methods::slot(ds$data, "Data"))
    } else {
      dim(ds$data)
    }
    if (length(dims) != 2L) stop("data is not two-dimensional")
    labels <- as.factor(ds$labels)
    if (length(labels) != dims[[1L]]) stop("label length does not match data rows")
    data.frame(
      dataset = sub("\\.(RData|rda)$", "", basename(path), ignore.case = TRUE),
      relative_path = relative,
      bytes = info$size,
      samples = as.integer(dims[[1L]]),
      variables = as.integer(dims[[2L]]),
      classes = nlevels(labels),
      data_class = paste(class(ds$data), collapse = "/"),
      data_object = ds$object,
      status = "complete",
      error = ""
    )
  }, error = function(e) {
    data.frame(
      dataset = sub("\\.(RData|rda)$", "", basename(path), ignore.case = TRUE),
      relative_path = relative,
      bytes = info$size,
      samples = NA_integer_,
      variables = NA_integer_,
      classes = NA_integer_,
      data_class = NA_character_,
      data_object = NA_character_,
      status = "error",
      error = conditionMessage(e)
    )
  })
}

rows <- vector("list", length(files))
for (i in seq_along(files)) {
  message(sprintf("[%d/%d] %s", i, length(files), files[[i]]))
  rows[[i]] <- inspect_one(files[[i]])
  gc(verbose = FALSE)
}

inventory <- do.call(rbind, rows)
dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
utils::write.csv(inventory, out, row.names = FALSE)
print(inventory, row.names = FALSE)
message("Inventory: ", out)
