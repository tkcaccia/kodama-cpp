# Manuscript Maintenance Rule

Any accepted modification to the KODAMA C++/CUDA library that changes the
algorithm, public API, supported backends, dependencies, performance claims,
benchmark scope, or accepted implementation strategy must be reflected in the
JMLR manuscript draft.

Update targets:

- `kodama_cpp_jmlr_manuscript.docx`
- `kodama_cpp_jmlr_self_review.docx`, when the change affects reviewer-facing
  limitations, claims, or remaining work
- `build_kodama_manuscript.py`, so the DOCX files remain reproducible

Do not describe rejected experimental paths as supported features.
