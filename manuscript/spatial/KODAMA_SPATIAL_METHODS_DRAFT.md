# Spatially constrained KODAMA: methods draft

Status: working methods text for a future Journal of Translational Medicine
submission. This material is separate from the non-spatial JMLR MLOSS
submission.

## Inputs and notation

Let `X` be an `n x p` float32 feature matrix, `S` an `n x q` matrix of spatial
coordinates (`q` equal to 2 or 3), and `b_i` an optional sample or slide
identifier for observation `i`. KODAMA performs `M` independent optimization
runs. Within a run, spatial groups are atomic constraints: every observation in
one group receives the same candidate label during cross-validated accuracy
maximization.

## Separation of multiple slides

Coordinates from different slides must not be overlaid in a common coordinate
system. KODAMA first reproduces the horizontalization rule of the original R
implementation. Slides are processed in sorted identifier order. For slide
`j`, coordinate 1 is transformed as

```text
S[i,1] <- S[i,1] + a_j,  for b_i = j,
a_1 = 0,
a_(j+1) = max(S[j,1]) + 0.5 * (max(S[j,1]) - min(S[j,1])),
```

where the minimum and maximum are calculated after applying the current
slide's offset. Other coordinate axes are unchanged. A single-level sample
identifier is a no-op.

Horizontalization provides the intended geometric separation, but a finite
distance alone is not a mathematical prohibition against a cross-slide
k-means assignment. The current implementation therefore adds a hard sample
boundary. Every spatial landmark stratum and every spatial constraint is keyed
by the pair

```text
(slide identifier, within-slide spatial group).
```

Singleton spatial groups are repaired using candidate groups from the same
slide only. User-supplied constraints are also intersected with slide identity
before composition with the spatial groups. Consequently, no atomic KODAMA
optimization block can contain cells or spots from different slides,
independently of coordinate ranges, spatial resolution, or backend.

## Spatial landmark selection

If the requested landmark count is at least `n`, KODAMA applies the original
rule `L = ceil(0.75 n)`; otherwise `L` is the requested count, bounded to
`2,...,n-1`. For spatial input, KODAMA partitions the 2D or 3D coordinate range
into a regular grid with approximately `L^(1/q)` bins per axis. Grid cells are
intersected with slide identity. Quotas proportional to occupied-cell size are
then allocated with exact residual rounding, and observations are sampled
without replacement inside each stratum. This preserves coverage across the
observed tissue while retaining independent stochastic samples across the `M`
runs.

## Spatial constraint construction

Within each independent run, coordinates receive bounded random jitter whose
axis-specific scale is estimated once from the reusable spatial neighbor graph.
The jitter prevents coordinate ties from imposing arbitrary deterministic
boundaries. The jittered coordinates are partitioned into

```text
C = round(L * rho)
```

spatial groups, where the current public default is `rho = 0.4`. The 2025
preprint described `rho = 0.3`; this difference must be stated explicitly in
any comparative experiment. The standard implementation uses k-means spatial
groups. Each resulting ID is intersected with slide identity, singleton repair
is slide-local, and optional external constraints are applied without permitting
a cross-slide group.

The spatial groups constrain label proposals but do not replace the feature
matrix used by the classifier. KNN and SIMPLS plus latent-space LDA therefore
optimize cross-validated prediction from `X` while moving all observations in a
spatial group as one unit.

## Independent KODAMA evolution

For each run `m = 1,...,M`:

1. Select spatially representative landmarks.
2. Construct sample-separated atomic spatial groups.
3. Initialize feature-space labels using the requested splitting rule.
4. Calculate the initial cross-validated KNN or PLS-LDA predictions.
5. For each of `Tcycle` iterations, construct one group-level label proposal,
   perform exactly one cross-validation evaluation, and accept it according to
   the KODAMA score and temperature rule.
6. Project optimized landmark labels to non-landmarks while respecting the
   same atomic constraints.
7. Store the complete projected label vector for this run.

Runs are independent and can therefore be scheduled concurrently without
consensus information passing between them.

## KODAMA graph correction and visualization

Let `c_i^(m)` be the projected label of observation `i` in run `m`. For an edge
`(i,j)` in the reusable feature-space KNN graph, define the valid-run agreement

```text
A_ij = sum_m I(c_i^(m) = c_j^(m), both nonzero)
       / sum_m I(both labels are nonzero).
```

For positive agreement, the corrected edge dissimilarity is

```text
d*_ij = (1 + d_ij) / A_ij^2.
```

Edges with no valid equal-label run receive infinite dissimilarity. The
corrected graph is sorted row-wise and passed to the library's UMAP or openTSNE
implementation. Spatial coordinates are not used as UMAP or openTSNE
initialization. The default visualization initialization is calculated from
the raw feature matrix, with backend-matched PCA scaling for the selected
embedding method.

## Reproducibility and backend contract

The input matrix, spatial coordinates, landmarks, classifier buffers, and graph
operations use float32 internally. CPU, CUDA, and Metal execute the same
horizontalization, hard sample boundary, spatial-group definition, evolution
policy, and graph-correction mathematics. Backend comparisons should report
the seed, `M`, `Tcycle`, landmark count, splitting, `rho`, classifier parameters,
fold count, graph neighbors, and visualization parameters.

## Required validation for the spatial manuscript

- Assert zero mixed-slide constraint groups for every run and dataset.
- Report per-slide landmark coverage and spatial-group counts.
- Compare `rho = 0.3` and `rho = 0.4` without selecting by truth labels.
- Demonstrate that results are invariant to arbitrary translations of an
  individual slide before horizontalization.
- Report CPU/CUDA/Metal agreement using identical seeds and inputs.
- Distinguish geometric visualization metrics from biological interpretation;
  truth labels must not participate in KODAMA optimization or run selection.
