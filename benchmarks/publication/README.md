# Publication Benchmarks

This directory contains the versioned source code used to produce benchmark
evidence for scientific publications about `kodama-cpp`. It is deliberately
separate from exploratory engineering benchmarks in the parent directory.

## Contents

- `R/`: reproducible R benchmark protocols, analyses, aggregation, and figure
  generation.
- `R/jmlr_cpu4/`: the CPU4 JMLR MLOSS protocol, including release preflight,
  reusable graph preparation, isolated evolution-policy ablations, predictor
  sensitivity experiments, ImageNet raw/PCA50 comparison, quality metrics,
  and statistical aggregation.

Only source code, compact configuration tables, and documentation belong in
Git. Datasets, generated scheduler files, logs, intermediate R objects, and
bulk results must remain outside the repository. A published result should be
linked to the exact source commit and accompanied by its generated release
manifest, session information, dataset checksums, and archived result DOI.
