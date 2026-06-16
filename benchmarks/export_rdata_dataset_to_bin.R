args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: export_rdata_dataset_to_bin.R <dataset_name> <input.RData> <output_dir>", call. = FALSE)
}

dataset_name <- args[[1]]
input_path <- args[[2]]
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

e <- new.env(parent = emptyenv())
load(input_path, envir = e)

obj <- NULL
if (exists("dataset", envir = e, inherits = FALSE)) {
  obj <- get("dataset", envir = e)
} else {
  names_in_file <- ls(e)
  for (nm in names_in_file) {
    candidate <- get(nm, envir = e)
    if (is.list(candidate) && all(c("data", "labels") %in% names(candidate))) {
      obj <- candidate
      break
    }
  }
  if (is.null(obj) && all(c("data", "labels") %in% names_in_file)) {
    obj <- list(data = get("data", envir = e), labels = get("labels", envir = e))
  }
}

if (is.null(obj) || is.null(obj$data) || is.null(obj$labels)) {
  stop("Could not find a list with $data and $labels in ", input_path, call. = FALSE)
}

x <- obj$data
if (!is.matrix(x)) x <- as.matrix(x)
storage.mode(x) <- "double"

labels_raw <- obj$labels
if (is.factor(labels_raw)) {
  labels <- as.integer(labels_raw)
  label_map <- data.frame(code = seq_along(levels(labels_raw)), label = levels(labels_raw))
} else {
  labels_factor <- factor(labels_raw)
  labels <- as.integer(labels_factor)
  label_map <- data.frame(code = seq_along(levels(labels_factor)), label = levels(labels_factor))
}

if (nrow(x) != length(labels)) {
  stop("nrow(data) does not match length(labels) for ", dataset_name, call. = FALSE)
}

x_file <- file.path(output_dir, paste0(dataset_name, "_x_double_rowmajor.bin"))
y_file <- file.path(output_dir, paste0(dataset_name, "_labels_int32.bin"))
meta_file <- file.path(output_dir, paste0(dataset_name, "_metadata.csv"))
labels_file <- file.path(output_dir, paste0(dataset_name, "_label_map.csv"))

con <- file(x_file, "wb")
on.exit(close(con), add = TRUE)
writeBin(as.double(t(x)), con, size = 8, endian = "little")
close(con)

con <- file(y_file, "wb")
on.exit(close(con), add = TRUE)
writeBin(as.integer(labels), con, size = 4, endian = "little")
close(con)

write.csv(
  data.frame(
    dataset = dataset_name,
    input_path = normalizePath(input_path, mustWork = FALSE),
    n = nrow(x),
    p = ncol(x),
    n_labels = length(unique(labels)),
    x_file = normalizePath(x_file, mustWork = FALSE),
    labels_file = normalizePath(y_file, mustWork = FALSE)
  ),
  meta_file,
  row.names = FALSE
)
write.csv(label_map, labels_file, row.names = FALSE)

cat(meta_file, "\n")
