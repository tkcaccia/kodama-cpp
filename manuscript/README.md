# Manuscript build

The JMLR MLOSS submission is split deliberately:

- `kodama_cpp_jmlr_mloss.tex` and its PDF are the four-page software
  description, with a fifth references-only page;
- `kodama_cpp_jmlr_technical_supplement.tex` and its corresponding PDF retain
  the exact method, pseudocode, architecture/backend contracts, release
  protocol, decisive evidence, limitations, and provenance. Dated engineering
  history and rejected prototypes remain in `JMLR_CYCLE*.md` and benchmark
  directories rather than in the publication supplement;
- `kodama_cpp_jmlr_cover_letter.docx`/PDF and
  `kodama_cpp_jmlr_submission_readiness.docx`/PDF are submission support files.

Generate the main paper and cover materials with:

```sh
python3 build_jmlr_submission.py
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=tex_build_mloss kodama_cpp_jmlr_mloss.tex
```

Generate the condensed technical supplement directly with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=tex_build_supplement kodama_cpp_jmlr_technical_supplement.tex
```

`build_kodama_manuscript.py` generates the editable DOCX and self-review
materials. It does not own the condensed LaTeX supplement.

`jmlr2e.sty` comes from the official
[JMLR style-file repository](https://github.com/JmlrOrg/jmlr-style-file).
The inspected upstream repository does not contain a standalone license file,
so `jmlr2e.sty` is used only as a manuscript-build input and is excluded from
the kodama-cpp software release archive.
`grfext.sty` is a generated copy of the `grfext` package and remains under the
LaTeX Project Public License 1.3c or later, as stated in its header. Neither
style file is linked into the kodama-cpp runtime library.

## Non-spatial submission payload

The `manuscript` directory is a working area and also contains internal
engineering reports. It is not itself the upload payload. Build the JMLR MLOSS
package only through the allowlisted builder:

```sh
python3 build_submission_package.py
```

The command creates `submission-package/`, copies only the main paper,
technical supplement, cover letter, bibliography, required styles,
architecture figure, and non-spatial ImageNet figure, writes `SHA256SUMS`, and
audits text in TeX, PDF, DOCX, BibTeX, and style files. The build fails if it
finds spatial-analysis language or internal spatial benchmark identifiers.
Internal regression datasets, plots, cycle reports, and backend diagnostics
remain outside this directory.
