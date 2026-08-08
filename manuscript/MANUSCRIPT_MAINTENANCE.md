# Manuscript Maintenance Rule

Any accepted modification to the KODAMA C++/CUDA library that changes the
algorithm, public API, supported backends, dependencies, performance claims,
benchmark scope, or accepted implementation strategy must be reflected in the
JMLR manuscript draft.

Update targets:

- `kodama_cpp_jmlr_mloss.tex` and PDF
- `kodama_cpp_jmlr_manuscript.docx`
- `kodama_cpp_jmlr_technical_supplement.tex` and PDF
- `kodama_cpp_jmlr_self_review.docx`, when the change affects reviewer-facing
  limitations, claims, or remaining work
- `build_kodama_manuscript.py` and `build_jmlr_submission.py`, so all generated
  publication files remain reproducible

Do not describe rejected experimental paths as supported features.
MERFISH and Br8100 are internal regression fixtures only. They must not appear
in the JMLR manuscript, supplement, cover letter, readiness report, or
publication figures. Submission-facing visualization evidence must come from
the non-spatial benchmark archive and compare KODAMA with classic UMAP and
openTSNE on matched datasets, seeds, parameters, and backends.
Do not enter a release DOI, tag, commit, benchmark value, author consent,
funding statement, conflict declaration, editor, or reviewer before it has
been produced or confirmed.
