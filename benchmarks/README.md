# Benchmark Notes

The first benchmark target is to compare `KNNCV` and `PLSCV` on the datasets
used by fastEmbedR Benchmark #2 and #3.

Suggested local dataset layout:

```text
/mnt/sata_ssd/fastEmbedR/Data/<dataset>/<dataset>.RData
```

The C++ library does not read RData directly. R and Python wrappers should load
datasets, scale/preprocess them, convert to row-major numeric arrays, and call
the C++ API.

For each dataset, report:

```text
dataset
n samples
p features
method: KNNCV or PLSCV
backend requested
backend used
folds
stratified
constraint policy
k for KNNCV
metric for KNNCV
index type for KNNCV
max components for PLSCV
selected components for PLSCV
classification mode for PLSCV
global accuracy
fold accuracy
confusion matrix
runtime
peak memory
```

The planned GPU benchmark should compare:

- CPU exact KNN
- FAISS CPU Flat / IVF-Flat, when available
- FAISS GPU IVF-Flat cosine / inner product
- cuVS nearest-neighbour search

