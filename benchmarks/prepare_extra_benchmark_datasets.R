# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

suppressPackageStartupMessages({
  options(stringsAsFactors = FALSE)
})

out_dir <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(out_dir) >= 1L) out_dir[[1L]] else "benchmark_datasets/RData"
raw_dir <- file.path(dirname(out_dir), "raw")
excluded_dir <- file.path(dirname(out_dir), "excluded_lt500")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(excluded_dir, recursive = TRUE, showWarnings = FALSE)

download <- function(url, dest) {
  if (!file.exists(dest)) {
    message("Downloading ", basename(dest))
    download.file(url, dest, mode = "wb", method = "curl", quiet = FALSE)
  }
  dest
}

save_dataset <- function(name, data, labels) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  labels <- as.factor(labels)
  ok <- stats::complete.cases(data) & !is.na(labels)
  data <- data[ok, , drop = FALSE]
  labels <- droplevels(labels[ok])
  dataset <- list(data = data, labels = labels)
  target_dir <- if (nrow(data) < 500L) excluded_dir else out_dir
  out <- file.path(target_dir, paste0(name, ".RData"))
  save(dataset, file = out, compress = "gzip")
  message(sprintf("%-28s %8d x %-6d classes=%d file=%s",
                  name, nrow(data), ncol(data), nlevels(labels), out))
  invisible(out)
}

remove_dataset <- function(name) {
  path <- file.path(out_dir, paste0(name, ".RData"))
  if (file.exists(path)) {
    unlink(path)
    message("Removed ", path)
  }
}

read_csv_no_header <- function(path) {
  read.table(path, sep = ",", header = FALSE, stringsAsFactors = FALSE)
}

prepare_letter <- function() {
  path <- download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
    file.path(raw_dir, "letter-recognition.data")
  )
  x <- read_csv_no_header(path)
  save_dataset("LetterRecognition", x[, -1], x[[1]])
}

prepare_pendigits <- function() {
  tr <- download("https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra",
                 file.path(raw_dir, "pendigits.tra"))
  te <- download("https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes",
                 file.path(raw_dir, "pendigits.tes"))
  x <- rbind(read_csv_no_header(tr), read_csv_no_header(te))
  save_dataset("PenDigits", x[, -ncol(x)], x[[ncol(x)]])
}

prepare_optdigits <- function() {
  tr <- download("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
                 file.path(raw_dir, "optdigits.tra"))
  te <- download("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",
                 file.path(raw_dir, "optdigits.tes"))
  x <- rbind(read_csv_no_header(tr), read_csv_no_header(te))
  save_dataset("OptDigits", x[, -ncol(x)], x[[ncol(x)]])
}

prepare_segmentation <- function() {
  tr <- download("https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data",
                 file.path(raw_dir, "segmentation.data"))
  te <- download("https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.test",
                 file.path(raw_dir, "segmentation.test"))
  read_seg <- function(path) {
    read.table(path, sep = ",", header = FALSE, stringsAsFactors = FALSE,
               comment.char = ";", skip = 5)
  }
  x <- rbind(read_seg(tr), read_seg(te))
  save_dataset("ImageSegmentation", x[, -1], x[[1]])
}

prepare_satimage <- function() {
  tr <- download("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn",
                 file.path(raw_dir, "sat.trn"))
  te <- download("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst",
                 file.path(raw_dir, "sat.tst"))
  x <- rbind(read.table(tr, header = FALSE), read.table(te, header = FALSE))
  save_dataset("SatImage", x[, -ncol(x)], x[[ncol(x)]])
}

prepare_har <- function() {
  zip <- download("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
                  file.path(raw_dir, "UCI_HAR_Dataset.zip"))
  ex <- file.path(raw_dir, "UCI_HAR_Dataset")
  if (!dir.exists(file.path(ex, "UCI HAR Dataset"))) {
    unzip(zip, exdir = ex)
  }
  root <- file.path(ex, "UCI HAR Dataset")
  train_x <- as.matrix(read.table(file.path(root, "train", "X_train.txt")))
  test_x <- as.matrix(read.table(file.path(root, "test", "X_test.txt")))
  train_y <- read.table(file.path(root, "train", "y_train.txt"))[[1]]
  test_y <- read.table(file.path(root, "test", "y_test.txt"))[[1]]
  act <- read.table(file.path(root, "activity_labels.txt"), stringsAsFactors = FALSE)
  labels <- act[[2]][c(train_y, test_y)]
  save_dataset("HumanActivityRecognition", rbind(train_x, test_x), labels)
}

prepare_pbmc3k_pca50 <- function() {
  h5 <- file.path(raw_dir, "scanpy-pbmc3k.h5ad")
  if (!file.exists(h5)) {
    download("https://ndownloader.figshare.com/files/30462915", h5)
  }
  py <- sprintf(
    "import h5py, numpy as np\np=%s\nout_data=%s\nout_labels=%s\nwith h5py.File(p, 'r') as f:\n    X=f['X'][:]\n    lab=f['obs/leiden'][:].astype(str)\nnp.savetxt(out_data, X, delimiter=',')\nnp.savetxt(out_labels, lab, fmt='%%s', delimiter=',')\n",
    shQuote(h5), shQuote(file.path(raw_dir, "pbmc3k_processed_data.csv")),
    shQuote(file.path(raw_dir, "pbmc3k_processed_labels.csv"))
  )
  if (!file.exists(file.path(raw_dir, "pbmc3k_processed_data.csv"))) {
    system2("python3", c("-c", shQuote(py)))
  }
  data <- as.matrix(read.csv(file.path(raw_dir, "pbmc3k_processed_data.csv"), header = FALSE))
  labels <- read.csv(file.path(raw_dir, "pbmc3k_processed_labels.csv"), header = FALSE)[[1]]
  pcs <- stats::prcomp(data, center = TRUE, scale. = FALSE, rank. = 50)$x
  save_dataset("PBMC3K_PCA50", pcs, labels)
}

prepare_sonar <- function() {
  path <- download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
    file.path(raw_dir, "sonar.all-data")
  )
  x <- read_csv_no_header(path)
  save_dataset("Sonar", x[, -ncol(x)], x[[ncol(x)]])
}

prepare_ionosphere <- function() {
  path <- download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
    file.path(raw_dir, "ionosphere.data")
  )
  x <- read_csv_no_header(path)
  save_dataset("Ionosphere", x[, -ncol(x)], x[[ncol(x)]])
}

prepare_spambase <- function() {
  path <- download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
    file.path(raw_dir, "spambase.data")
  )
  x <- read_csv_no_header(path)
  save_dataset("Spambase", x[, -ncol(x)], x[[ncol(x)]])
}

prepare_vehicle <- function() {
  base <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle"
  files <- paste0("xa", letters[1:9], ".dat")
  paths <- vapply(files, function(fn) download(file.path(base, fn), file.path(raw_dir, fn)), "")
  x <- do.call(rbind, lapply(paths, read.table, header = FALSE))
  save_dataset("Vehicle", x[, -ncol(x)], x[[ncol(x)]])
}

prepare_wine <- function() {
  path <- download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    file.path(raw_dir, "wine.data")
  )
  x <- read_csv_no_header(path)
  save_dataset("Wine", x[, -1], x[[1]])
}

prepare_glass <- function() {
  path <- download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
    file.path(raw_dir, "glass.data")
  )
  x <- read_csv_no_header(path)
  save_dataset("Glass", x[, 2:(ncol(x) - 1)], x[[ncol(x)]])
}

prepare_ecoli <- function() {
  path <- download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
    file.path(raw_dir, "ecoli.data")
  )
  x <- read.table(path, header = FALSE, stringsAsFactors = FALSE)
  save_dataset("Ecoli", x[, 2:(ncol(x) - 1)], x[[ncol(x)]])
}

prepare_yeast <- function() {
  path <- download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data",
    file.path(raw_dir, "yeast.data")
  )
  x <- read.table(path, header = FALSE, stringsAsFactors = FALSE)
  save_dataset("Yeast", x[, 2:(ncol(x) - 1)], x[[ncol(x)]])
}

prepare_pageblocks <- function() {
  path <- download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/page-blocks/page-blocks.data.Z",
    file.path(raw_dir, "page-blocks.data.Z")
  )
  tmp <- tempfile(fileext = ".Z")
  file.copy(path, tmp, overwrite = TRUE)
  on.exit(unlink(tmp), add = TRUE)
  txt <- system2("zcat", tmp, stdout = TRUE)
  x <- read.table(text = txt, header = FALSE)
  save_dataset("PageBlocks", x[, -ncol(x)], x[[ncol(x)]])
}

prepare_sklearn_small <- function() {
  py <- file.path(raw_dir, "sklearn_small_to_csv.py")
  writeLines(c(
    "from sklearn.datasets import load_breast_cancer, load_digits",
    "import numpy as np",
    "for name, loader in [('BreastCancerDiagnostic', load_breast_cancer), ('SklearnDigits8x8', load_digits)]:",
    "    d = loader()",
    "    np.savetxt(f'%s_data.csv' % name, d.data, delimiter=',')",
    "    np.savetxt(f'%s_labels.csv' % name, d.target, fmt='%s', delimiter=',')"
  ), py)
  old <- getwd()
  on.exit(setwd(old), add = TRUE)
  setwd(raw_dir)
  system2("python3", basename(py))
  for (name in c("BreastCancerDiagnostic", "SklearnDigits8x8")) {
    data <- as.matrix(read.csv(file.path(raw_dir, paste0(name, "_data.csv")), header = FALSE))
    labels <- read.csv(file.path(raw_dir, paste0(name, "_labels.csv")), header = FALSE)[[1]]
    save_dataset(name, data, labels)
  }
}

remove_dataset("PBMC3K_processed")
remove_dataset("TCGA_PANCAN_RNASeq")
remove_dataset("Waveform")

steps <- list(
  prepare_letter,
  prepare_pendigits,
  prepare_optdigits,
  prepare_segmentation,
  prepare_satimage,
  prepare_har,
  prepare_pbmc3k_pca50,
  prepare_sonar,
  prepare_ionosphere,
  prepare_spambase,
  prepare_vehicle,
  prepare_wine,
  prepare_glass,
  prepare_ecoli,
  prepare_yeast,
  prepare_pageblocks,
  prepare_sklearn_small
)

for (step in steps) {
  tryCatch(step(), error = function(e) {
    message("FAILED: ", conditionMessage(e))
  })
}
