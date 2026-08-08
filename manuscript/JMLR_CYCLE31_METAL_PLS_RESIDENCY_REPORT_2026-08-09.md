# Cycle 31: Metal PLS-LDA residency audit

## Scope

This cycle tested whether moving the complete label-aware SIMPLS recurrence to
Metal could reduce synchronization without changing SIMPLS, latent-space LDA,
the five CV folds, the fixed component count, or KODAMA evolution. MetRef used
seed 4, 655 effective landmarks, 50 requested components, and the existing
float32 implementation.

## Rejected implementations

1. A custom all-device SIMPLS recurrence, including label-aware `X'Y`, was
   numerically corrected until the physical Metal parity suite passed. It took
   60.559 seconds for `M=10`, `Tcycle=10`, compared with 13.837 seconds for the
   accepted four-worker implementation. Custom projection kernels could not
   match MPS matrix multiplication throughput.
2. A fused recurrence retained MPS for `Xr` and `X't` while encoding the small
   SIMPLS updates in one command buffer. Parity passed, but the standalone
   Metal PLS-LDA benchmark regressed from approximately 0.062 to 0.086 seconds,
   and a bounded MetRef run exceeded the accepted runtime before one M run
   completed.
3. A private-memory fold cache staged each packed fold once and reused a private
   Metal matrix. It improved the isolated PLS-LDA microbenchmark from about
   0.062 to 0.033 seconds and reduced four-worker MetRef `M=10`, `Tcycle=10`
   from 13.837 to 9.573 seconds. At `Tcycle=100`, however, completed M runs took
   approximately 24 seconds, projecting to about 600 seconds versus the
   accepted 273-second full run. The change was reverted.

All three prototypes were removed. There is no dormant alternative path or
runtime switch in the library.

## Accepted result

The retained implementation uses one thread-local resident Metal workspace and
command queue per independent M worker, cached packed fold matrices, MPS
projection, on-device PLS sufficient statistics and validation scoring, and
automatic worker selection from the device working-set budget. The full
`M=100`, `Tcycle=100` MetRef run selected 26 workers and completed
`KODAMA.matrix` in 263.568 seconds, compared with the previous accepted
273.013-second measurement (1.036x; 3.46% faster).

Quality was retained: best CV accuracy 1.000, median CV accuracy 0.993893,
best ARI 0.860185, median 25 classes, zero runs with two or fewer classes, and
KODAMA UMAP truth silhouette 0.833023. The physical Metal suite passed for KNN,
PLS-LDA, PCA, UMAP, openTSNE, normalization, and scaling. All ordinary CTest
targets passed; the sandboxed CTest Metal target was skipped and then run
directly on the physical device.

## Decision

Accept the existing resident-worker scheduler and its full-run evidence. Reject
the fully device-native recurrence, fused command-buffer recurrence, and private
fold storage. The audit shows that eliminating synchronization is not by itself
an improvement: on Apple unified memory, MPS throughput and concurrent
independent M runs are more important than forcing every small recurrence step
onto the GPU.
