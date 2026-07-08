#!/usr/bin/env Rscript

get_env <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (nzchar(value)) value else default
}

first_existing <- function(paths) {
  hit <- paths[file.exists(paths)][1]
  if (length(hit) == 0L || is.na(hit)) paths[1] else hit
}

load_spatial_dataset <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  if (exists("dataset", envir = env, inherits = FALSE)) {
    ds <- get("dataset", envir = env)
    return(list(data = as.matrix(ds$data), spatial = as.matrix(ds$spatial), labels = as.factor(ds$labels)))
  }
  names <- ls(env)
  data_name <- intersect(c("pca.PM", "pca_Br8100", "data", "x"), names)[1]
  spatial_name <- intersect(c("xyz", "xy_Br8100", "spatial"), names)[1]
  labels_name <- intersect(c("tissue_segments", "labels_Br8100", "labels"), names)[1]
  if (any(is.na(c(data_name, spatial_name, labels_name)))) {
    stop("Cannot identify data/spatial/labels objects in ", path)
  }
  list(
    data = as.matrix(get(data_name, envir = env)),
    spatial = as.matrix(get(spatial_name, envir = env)),
    labels = as.factor(get(labels_name, envir = env))
  )
}

write_dataset <- function(name, path, out_dir) {
  if (!file.exists(path)) {
    warning("Skipping missing dataset: ", path)
    return(invisible(FALSE))
  }
  ds <- load_spatial_dataset(path)
  utils::write.csv(ds$data, gzfile(file.path(out_dir, paste0(name, "_data.csv.gz"))), row.names = FALSE)
  utils::write.csv(ds$spatial, gzfile(file.path(out_dir, paste0(name, "_spatial.csv.gz"))), row.names = FALSE)
  utils::write.csv(data.frame(label = as.character(ds$labels)), gzfile(file.path(out_dir, paste0(name, "_labels.csv.gz"))), row.names = FALSE)
  invisible(TRUE)
}

out_dir <- get_env("KODAMA_PY_EXPORT_DIR", file.path(getwd(), "exported-spatial"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_dataset("MERFISH", get_env("MERFISH_RDATA", first_existing(c(
  file.path("data", "MERFISH.RData"),
  file.path("data", "MERFISH-input.RData")
))), out_dir)
write_dataset("Br8100", get_env("BR8100_RDATA", first_existing(c(
  file.path("data", "DLFPC-Br8100.RData"),
  file.path("data", "DLFPC-Br8100-input.RData"),
  file.path("data", "Br8100.RData")
))), out_dir)

cat("Wrote exported datasets to", out_dir, "\n")
