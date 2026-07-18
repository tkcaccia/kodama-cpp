# Release checklist

This checklist freezes the software snapshot reviewed with the JMLR MLOSS
submission. Do not enter a DOI or release commit in the manuscript before the
corresponding artifact exists.

## Source identity

- [ ] Confirm the worktree contains only intended release changes.
- [ ] Update `CHANGELOG.md`, `CITATION.cff`, and the wrapper versions.
- [ ] Confirm the `v0.1.0` public API with `tests/test_public_api_0_1.cpp`.
- [ ] Create the signed or annotated tag `v0.1.0`.
- [ ] Record the full tagged commit in the manuscript and cover letter.

## Validation

- [ ] Run the dependency-free CPU build and CTest suite.
- [ ] Run the Apple Metal build and CTest suite on macOS.
- [ ] Run the CUDA build and CTest suite on the release CUDA machine.
- [ ] Run `R CMD check --as-cran` on the R wrapper source tarball.
- [ ] Build a Python wheel and run pytest in a clean environment.
- [ ] Run `benchmarks/run_jmlr_release_validation.sh`.
- [ ] Run the final classic-versus-KODAMA UMAP/openTSNE comparison.
- [ ] Preserve commands, environment metadata, CSV results, and plots.

## Licensing and archive

- [ ] Run `bash tools/check_license_headers.sh`.
- [ ] Confirm the residual provenance condition documented in `PROVENANCE.md`.
- [ ] Confirm that the release contains no hidden VCS, macOS resource-fork, or
  benchmark-output files.
- [ ] Run `tools/make_release_archive.sh v0.1.0`.
- [ ] Verify the generated SHA-256 digest.
- [ ] Deposit the tagged archive in an archival repository.
- [ ] Add the assigned DOI to `CITATION.cff`, the manuscript, and cover letter.

## Submission

- [ ] Compile the four-page JMLR MLOSS paper with `latexmk`.
- [ ] Confirm that description content ends by page 4; references may follow.
- [ ] Attach the detailed technical supplement separately.
- [ ] Obtain every coauthor's submission consent.
- [ ] Complete funding and competing-interest declarations.
- [ ] Supply 3-5 conflict-free action editors and 3-5 conflict-free reviewers.
- [ ] Attach the prior KODAMA publications and describe the software delta.
