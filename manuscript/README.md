# Manuscript build

The JMLR MLOSS submission is split deliberately:

- `kodama_cpp_jmlr_mloss.tex` and its PDF are the four-page software
  description, with a fifth references-only page;
- `kodama_cpp_jmlr_manuscript.docx`,
  `kodama_cpp_jmlr_technical_supplement.tex`, and the corresponding PDF retain
  the full method, pseudocode, validation protocol, provenance, and tables;
- `kodama_cpp_jmlr_cover_letter.docx`/PDF and
  `kodama_cpp_jmlr_submission_readiness.docx`/PDF are submission support files.

Generate the main paper and cover materials with:

```sh
python3 build_jmlr_submission.py
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=tex_build_mloss kodama_cpp_jmlr_mloss.tex
```

Generate the detailed supplement with:

```sh
python3 build_kodama_manuscript.py
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=tex_build kodama_cpp_jmlr_technical_supplement.tex
```

`jmlr2e.sty` comes from the official
[JMLR style-file repository](https://github.com/JmlrOrg/jmlr-style-file).
The inspected upstream repository does not contain a standalone license file,
so `jmlr2e.sty` is a manuscript-build input rather than part of the kodama-cpp
software release archive pending confirmation of redistribution terms.
`grfext.sty` is a generated copy of the `grfext` package and remains under the
LaTeX Project Public License 1.3c or later, as stated in its header. Neither
style file is linked into the kodama-cpp runtime library.
