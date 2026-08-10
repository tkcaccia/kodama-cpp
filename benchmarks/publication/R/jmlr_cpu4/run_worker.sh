#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT
set -euo pipefail
: "${IMAGE:?IMAGE is required}"
: "${SCRIPT:?SCRIPT is required}"
export OMP_NUM_THREADS=4 RCPP_PARALLEL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=4 APPTAINERENV_RCPP_PARALLEL_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1
export SINGULARITYENV_OMP_NUM_THREADS=4 SINGULARITYENV_RCPP_PARALLEL_NUM_THREADS=4
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_MKL_NUM_THREADS=1
CONTAINER="$(command -v apptainer || command -v singularity || true)"
[[ -n "${CONTAINER}" ]] || { echo "apptainer/singularity was not found" >&2; exit 1; }
exec /usr/bin/time -v \
  "${CONTAINER}" exec --cleanenv \
  --env OMP_NUM_THREADS=4,RCPP_PARALLEL_NUM_THREADS=4,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,VECLIB_MAXIMUM_THREADS=1,NUMEXPR_NUM_THREADS=1 \
  --bind /scratch/firenze/NN:/scratch/firenze/NN \
  --pwd /scratch/firenze/NN \
  "${IMAGE}" /opt/conda/bin/Rscript "${SCRIPT}" "$@"
