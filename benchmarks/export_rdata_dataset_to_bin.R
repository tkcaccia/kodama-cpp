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

x_double_file <- file.path(output_dir, paste0(dataset_name, "_x_double_rowmajor.bin"))
x_float32_file <- file.path(output_dir, paste0(dataset_name, "_x_float32_rowmajor.bin"))
y_file <- file.path(output_dir, paste0(dataset_name, "_labels_int32.bin"))
meta_file <- file.path(output_dir, paste0(dataset_name, "_metadata.csv"))
labels_file <- file.path(output_dir, paste0(dataset_name, "_label_map.csv"))

chunk_rows <- as.integer(Sys.getenv("KODAMA_EXPORT_CHUNK_ROWS", "10000"))
chunk_rows <- max(1L, chunk_rows)

write_matrix_file <- function(path, bytes) {
  con <- file(path, "wb")
  on.exit(close(con), add = TRUE)
  for (start in seq.int(1L, nrow(x), by = chunk_rows)) {
    end <- min(nrow(x), start + chunk_rows - 1L)
    chunk <- x[start:end, , drop = FALSE]
    writeBin(as.double(t(chunk)), con, size = bytes, endian = "little")
  }
  close(con)
  on.exit(NULL, add = FALSE)
}

write_matrix_file(x_double_file, 8)
write_matrix_file(x_float32_file, 4)

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
    x_file = normalizePath(x_double_file, mustWork = FALSE),
    x_float32_file = normalizePath(x_float32_file, mustWork = FALSE),
    labels_file = normalizePath(y_file, mustWork = FALSE)
  ),
  meta_file,
  row.names = FALSE
)
write.csv(label_map, labels_file, row.names = FALSE)

cat(meta_file, "\n")
