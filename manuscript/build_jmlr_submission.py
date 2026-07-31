#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

import build_kodama_manuscript as supplement


ROOT = Path(__file__).resolve().parent
MAIN_TEX = ROOT / "kodama_cpp_jmlr_mloss.tex"
COVER_DOCX = ROOT / "kodama_cpp_jmlr_cover_letter.docx"
COVER_TXT = ROOT / "kodama_cpp_jmlr_cover_letter.txt"
READINESS_DOCX = ROOT / "kodama_cpp_jmlr_submission_readiness.docx"

SOFTWARE_VERSION = "0.1.0"
REVIEWED_BASELINE = "a32f71844fb0860f8faec0169c2850f800a68da4"
R_WRAPPER_BASELINE = "4d5506bde8918162dae0ad1d19c4f6f7a4df64ba"


def build_main_tex() -> None:
    text = dedent(
        r"""
        \documentclass[twoside,11pt]{article}
        \usepackage{jmlr2e}
        \usepackage{amsmath}
        \usepackage{booktabs}
        \usepackage{graphicx}
        \usepackage{lastpage}
        \usepackage{tabularx}
        \usepackage{url}
        \hypersetup{hidelinks}
        \setlength{\textfloatsep}{7pt plus 2pt minus 2pt}
        \setlength{\floatsep}{6pt plus 2pt minus 2pt}
        \setlength{\intextsep}{6pt plus 2pt minus 2pt}
        \setlength{\abovecaptionskip}{3pt}
        \setlength{\belowcaptionskip}{0pt}
        \renewcommand{\arraystretch}{0.96}

        \jmlrheading{1}{2026}{1--\pageref{LastPage}}{7/26}{}{26-0000}{Kassim et al.}
        \ShortHeadings{kodama-cpp}{Kassim et al.}
        \firstpageno{1}

        \begin{document}

        \title{kodama-cpp: Cross-Validated Accuracy Maximization on CPU, CUDA, and Apple Metal}

        \author{%
        \name Moussa Kassim$^{1,2,\ast}$ \email moussa.kassim@icgeb.org\\
        \name Martin Ocharo$^{1,2,\ast}$ \email martin.ocharo@icgeb.org\\
        \name Dalia Ahmed$^{1}$ \email dalia.ahmed@icgeb.org\\
        \name Dupe Ojo$^{1}$ \email dupe.ojo@icgeb.org\\
        \name Alessia Vignoli$^{3,4}$ \email vignoli@cerm.unifi.it\\
        \name Leonardo Tenori$^{3,4,\dagger}$ \email tenori@cerm.unifi.it\\
        \name Stefano Cacciatore$^{1,2,\dagger}$ \email stefano.cacciatore@icgeb.org\\
        \addr $^1$Bioinformatics Unit, International Centre for Genetic Engineering and Biotechnology (ICGEB), Cape Town 7925, South Africa\\
        \addr $^2$Department of Integrative Biomedical Sciences, Institute of Infectious Disease \& Molecular Medicine (IDM), University of Cape Town, Cape Town 7925, South Africa\\
        \addr $^3$Department of Chemistry ``Ugo Schiff'', University of Florence, Sesto Fiorentino, Italy\\
        \addr $^4$Magnetic Resonance Center (CERM), University of Florence, Sesto Fiorentino, Italy\\
        \addr $^{\ast}$Moussa Kassim and Martin Ocharo contributed equally.\\
        \addr $^{\dagger}$Leonardo Tenori and Stefano Cacciatore are co-corresponding authors.}

        \editor{To be assigned}
        \maketitle

        \begin{abstract}
        KODAMA discovers latent structure by maximizing held-out label predictability. This standalone C++17 implementation is shared by thin R and Python wrappers. Native multicore CPU, NVIDIA CUDA, and Apple Metal backends provide float32 KNN and SIMPLS--LDA without silent fallback. The library also introduces prediction-guided label evolution and exact-quota landmark projection. Tests distinguish numerical consistency and systems performance from external-label diagnostics.
        \end{abstract}

        \begin{keywords}
        KODAMA, cross-validation, unsupervised learning, heterogeneous computing, manifold learning
        \end{keywords}

        \section{Introduction}

        KODAMA treats labels as optimization variables: useful labels can be reproduced on held-out samples. An ensemble of optimized vectors then corrects neighborhood dissimilarities \citep{cacciatore2014kodama,cacciatore2017kodama}. Unlike stability analysis or prediction strength, predictability is inside the search and no observed labels anchor it \citep{benhur2002stability,tibshirani2005predictionstrength}.

        Repeated cross-validation makes reusable state a systems concern. Our four contributions are \textbf{(1)} a typed C++ core shared by R and Python; \textbf{(2)} CPU, CUDA, and Metal under one float32 contract; \textbf{(3)} prediction-guided evolution with adaptive proposals, degeneracy guards, and error-scaled cooling; and \textbf{(4)} exact-quota landmark projection, formalizing the principle introduced by \citet{abdelshafy2025landmarks}.

        \clearpage
        \section{KODAMA objective and algorithm}

        For data $X\in\mathbb{R}^{n\times p}$, labels $y$, folds $\Pi$, and classifier family $\mathcal F$, let
        \begin{equation}
          A(y;X,\mathcal F,\Pi)=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\!\left[y_i=\widehat y_i^{(-\Pi(i))}\right].
        \end{equation}
        KODAMA searches for high $A$ using either KNN or SIMPLS followed by latent-space LDA \citep{dejong1993simpls}. Folds stay fixed within each run, and supplied constraint groups move atomically. Each independent run draws exactly $L$ landmarks by population-proportional quotas from a coarse matrix partition; randomized systematic rounding fills the residual quota without replacement. A separate partition initializes the labels, so sampling strata are not optimization targets.

        The classifiers share an evolution \emph{scaffold}, not an identical state policy. Both use fixed folds, prediction-guided grouped proposals, the common transition matrix, one new CV pass per proposal, error-scaled cooling, and independent $M$ runs. Let $p_k$ be the class proportions, $D=1-\sum_kp_k^2$, $H=-\sum_kp_k\log p_k$, $K_{\rm eff}=e^H$, and $R=\max\{0,\log[K/\max(1,K_{\rm eff})]\}$. State selection uses
        \begin{equation}
          S_{\mathcal F}(y)=A(y)\sqrt D-\mathbf{1}_{\{\mathcal F={\rm PLS\mbox{-}LDA}\}}(1-A(y))
          \max\!\left\{\frac{H}{\log n},\frac{R}{1+R}\right\}.
        \end{equation}
        Standard \texttt{KODAMA.matrix} applies transition-driven coarsening and the subtraction only to PLS--LDA; KNN uses the common diversity term alone. This reflects that LDA fits a mean per active class, whereas KNN has no analogous global class fit. Existing sensitivity results motivate, but do not isolate, this policy; no universal improvement is claimed.

        The selected landmark vector is projected to all samples with the same classifier. If $a_{ij}$ is the fraction of projected runs agreeing on graph edge $(i,j)$, KODAMA returns $d'_{ij}=(1+d_{ij})/a_{ij}^{2}$ and removes zero-agreement edges. Exact proposal, temperature, quota, and coarsening formulas are given in the supplement.

        \begin{figure}[h!]
        \hrule\vspace{2pt}
        \begin{minipage}{0.98\linewidth}
        \footnotesize
        \textbf{Algorithm 1: KODAMA ensemble}\\
        \textbf{Input:} $X,\mathcal F,M,T,L,K_0,\Pi,s$.\quad \textbf{Output:} $G,C,A_{\rm best},Z_U,Z_T$.
        \begin{tabbing}
        \quad\=\quad\=\quad\=\kill
        $X_{32}\leftarrow\textsc{Float32}(X)$; $(G_0,Z_U,Z_T)\leftarrow\textsc{KODAMA.Graph}(X_{32})$ once\\
        \textbf{for} $r=1,\ldots,M$ \textbf{do}\\
        \> $I_r\leftarrow\textsc{ExactQuotaSample}(\textsc{Partition}(X,K_0),L,s+r)$\\
        \> $y\leftarrow\textsc{InitialPartition}(X_{I_r},K_0)$; $(A,\widehat y)\leftarrow\textsc{CV}(\mathcal F,X_{I_r},y,\Pi_r)$\\
        \> \textbf{for} $t=1,\ldots,T$ \textbf{do}\\
        \>\> $y'\leftarrow\textsc{GuidedProposal}(y,\widehat y,t,T)$; if $\mathcal F={\rm PLS\mbox{-}LDA}$, apply transition coarsening\\
        \>\> $(A',\widehat y')\leftarrow\textsc{CV}(\mathcal F,X_{I_r},y',\Pi_r)$; update by $S_{\mathcal F}$ and cooling\\
        \> \textbf{end for}\\
        \> $C_{\cdot r}\leftarrow\textsc{Project}(\mathcal F,X_{I_r},y_{\rm best},X)$\\
        \textbf{end for}\\
        $G\leftarrow\textsc{AgreementCorrectionInPlace}(G_0,C)$; \textbf{return} $G,C,A_{\rm best},Z_U,Z_T$
        \end{tabbing}
        \end{minipage}
        \vspace{-4pt}\hrule
        \caption{Compact pseudocode. The scaffold is common; the indicated coarsening and score term are PLS--LDA-specific.}
        \label{alg:kodama}
        \end{figure}

        \clearpage
        \section{Standalone heterogeneous implementation}

        \begin{figure}[h!]
        \centering
        \includegraphics[width=0.68\linewidth]{kodama_cpp_architecture.png}
        \caption{The C++17 core owns algorithm state; R and Python are thin adapters. For a fixed classifier, backends share folds, policy, component semantics, and typed results.}
        \label{fig:architecture}
        \end{figure}

        The typed C++ API owns folds, proposals, classifiers, landmark projection, graph correction, and metadata; wrappers only convert host objects and results. CPU supplies parallel HNSW construction over contiguous float32 storage \citep{malkov2020hnsw}, with a deterministic connected base seed before lock-protected concurrent insertion and querying, plus SIMPLS--LDA, PCA, graph operations, and embeddings. CUDA supplies exact/IVF KNN, batched k-means, label-aware SIMPLS--LDA, PCA, and embeddings. Metal supplies exact/IVF KNN, k-means, MPS-assisted SIMPLS--LDA, and PCA. Accelerator IVF construction and persistent indices retain training data and index state on-device. During KODAMA, one resident full-data graph and persistent worker-lane buffers retain fold layouts, labels, projections, voting state, and PLS--LDA workspaces across proposal cycles; contents that depend on a new landmark sample are refreshed in place once per $M$ run. Unsupported backends raise an error rather than silently falling back.

        PLS--LDA forms class cross-products without dense one-hot responses, compacts active labels, reuses fold workspaces, and evaluates the requested feasible component count without model selection. \texttt{KODAMA.graph} prepares one full-data neighbor graph and separately scaled backend-matched UMAP/openTSNE PCA starts. Its result owns only graph arrays, starts, and metadata: it never retains the raw matrix. \texttt{KODAMA.matrix} accepts four explicit forms: the prepared object alone, that object plus caller-supplied raw features, a bare paired $n\times k$ index/distance graph, or raw features; raw input invokes the same preparation internally. Graph-only execution uses disclosed self-tuning normalized-Laplacian geometry, including the PLS--LDA spectral surrogate, whereas supplying raw features retains ordinary data-input PLS--LDA. Fixed-graph KNN reuses supplied neighborhoods and is distinguished from data-native KNN rather than asserted to reproduce its stochastic trajectory. Final compaction and agreement correction mutate the single graph owner in place. The API also exposes both CV kernels, both core optimizers, PCA, and float32 UMAP/openTSNE \citep{mcinnes2018umap,vandermaaten2008tsne}. Explicit visualization starts take precedence, then backend-matched stored starts, then raw-data recomputation and graph-only fallbacks; provenance is reported. Derivations, safeguards, pseudocode, evidence, and API contracts are in the supplement.

        \section{Validation, availability, and scope}

        \begin{table}[h!]
        \caption{Matched runtime and held-out accuracy. Times are seconds; KODAMA reports median raw CV accuracy for $M=T=100$.}
        \label{tab:validation}
        \centering\scriptsize
        \begin{tabular}{lllrrrr}
        \toprule
        Scope & Data & Accel. & CPU & Device & Speedup & Accuracy \\
        \midrule
        KNNCV & MNIST & CUDA & 76.769 & 4.753 & 16.2 & .974/.973 \\
        PLSLDACV & MetRef & CUDA & 1.294 & 0.308 & 4.20 & .992/.993 \\
        KODAMA PLS--LDA & MetRef & CUDA & 341.648 & 270.408 & 1.26 & .9924/.9924 \\
        KNNCV & MetRef & Metal & 11.145 & 0.026 & 425.0 & .827/.827 \\
        PLSLDACV & MetRef & Metal & 0.790 & 0.202 & 3.90 & .991/.990 \\
        \bottomrule
        \end{tabular}
        \end{table}

        Table~\ref{tab:validation} separates kernels from the complete KODAMA call. CUDA and Metal preserved closely matched held-out accuracy, while end-to-end acceleration was smaller because repeated orchestration and graph work remain. Finite-precision ties may select different stochastic trajectories, so numerical consistency is assessed by contracts and paired metrics rather than bitwise label identity. Extended runtime, memory, landmark, $M/T$, visualization, and prior-version comparisons are reported in the supplement.

        The fixed-$M=100$ sensitivity study also shows why classifier-independent scoring is not claimed: at $T=100$, median active-class counts were 9 versus 24 on MetRef and 32 versus 71 on USPS for KNN versus PLS--LDA. These observations establish different fragmentation regimes, but because they are not penalty-on/off ablations they do not establish the causal benefit of the PLS--LDA term.

        CTest and wrapper suites exercise float32 numerics, strict backend identity, graph and PCA contracts, backend-matched visualization initialization, landmark counts, and the candidate-\texttt{0.1.0} API. A controlled CPU openTSNE comparison against the pinned fastEmbedR source was coordinate-identical. GitHub Actions covers Linux/macOS CPU, native Metal, wrappers, coverage, and API documentation; CUDA is validated on a dedicated NVIDIA host. The audited CPU snapshot reached 63.58\% line and 57.03\% branch coverage.

        The MIT-licensed core is available at \url{https://github.com/tkcaccia/kodama-cpp}; the R wrapper is at \url{https://github.com/tkcaccia/kodama-r}, and tested Python wrapper source accompanies the core pending its standalone repository. Repeated CV remains the dominant cost. Graph and classifier workspaces are device resident, but landmark selection, proposal decisions, and compact result collection remain host orchestrated; approximate-neighbor ties can alter trajectories, and CUDA graph construction currently supports $k\leq256$.

        \newpage
        \bibliography{kodama_cpp_refs}

        \end{document}
        """
    ).strip() + "\n"
    MAIN_TEX.write_text(text, encoding="utf-8")


def add_label_value(doc: Document, label: str, value: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(2)
    label_run = paragraph.add_run(f"{label}: ")
    label_run.bold = True
    paragraph.add_run(value)


def build_cover_letter() -> None:
    doc = Document()
    supplement.apply_styles(doc)
    supplement.apply_compact_memo_styles(doc)
    section = doc.sections[0]
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(9.4)
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.space_after = Pt(2)
    heading = doc.styles["Heading 1"]
    heading.font.size = Pt(12.5)
    heading.paragraph_format.space_before = Pt(6)
    heading.paragraph_format.space_after = Pt(2)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Cover Letter: JMLR Machine Learning Open Source Software")
    run.bold = True
    run.font.size = Pt(13.5)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("kodama-cpp: Cross-Validated Accuracy Maximization on CPU, CUDA, and Apple Metal").italic = True

    doc.add_paragraph("Dear Editors of the Journal of Machine Learning Research,")
    doc.add_paragraph(
        "We submit kodama-cpp for consideration in the Machine Learning Open Source Software track. "
        "The software is a standalone C++17 implementation of KODAMA with explicit CPU, NVIDIA CUDA, "
        "and Apple Metal backends and thin R and Python wrappers."
    )

    doc.add_heading("Submission identity", level=1)
    add_label_value(doc, "Open-source license", "MIT project code with compatible retained third-party terms; see PROVENANCE.md and THIRD_PARTY_NOTICES.md")
    add_label_value(doc, "Project URL", "https://github.com/tkcaccia/kodama-cpp")
    add_label_value(doc, "R wrapper URL", "https://github.com/tkcaccia/kodama-r")
    add_label_value(
        doc,
        "Python binding source",
        "https://github.com/tkcaccia/kodama-cpp/tree/main/split-repos/kodama-python "
        "(standalone repository publication pending)",
    )
    add_label_value(doc, "Software version", f"{SOFTWARE_VERSION} release candidate")
    add_label_value(doc, "Reviewed baseline", REVIEWED_BASELINE)
    add_label_value(doc, "Reviewed R-wrapper baseline", R_WRAPPER_BASELINE)
    add_label_value(doc, "Final release commit", "AUTHOR ACTION REQUIRED: insert the full v0.1.0 tagged commit")
    add_label_value(
        doc,
        "Archived source",
        "AUTHOR ACTION REQUIRED: deposit the tagged source archive, insert its URL and checksum, and add a DOI if assigned",
    )

    doc.add_heading("Significance and software contribution", level=1)
    doc.add_paragraph(
        "The submission preserves KODAMA's cross-validated label-predictability signal while making "
        "four contributions: a standalone C++17 core for thin R/Python wrappers; native and observable "
        "CPU/CUDA/Metal execution; guided label evolution based on out-of-fold predictions and class "
        "transitions; and an exact-quota landmark-projection strategy that generalizes the landmark "
        "idea introduced in the KODAMA bioRxiv work. The repository contains installation instructions, "
        "API examples, explicit pseudocode, tests, benchmark drivers, a public API snapshot, and a "
        "machine-checkable licensing/provenance audit."
    )

    doc.add_heading("Prior publications and delta", level=1)
    doc.add_paragraph(
        "The KODAMA method was published by Cacciatore, Luchinat, and Tenori in PNAS (2014), and the "
        "historical R package was described by Cacciatore et al. in Bioinformatics (2017). Prior "
        "publication of the method is disclosed. The landmark-projection principle was subsequently "
        "introduced in the 2025 KODAMA bioRxiv preprint. This submission concerns its general formalization "
        "and the new standalone software: a C++17 core independent of R, native CUDA and Metal backends, "
        "guided search mechanics, exact-quota landmark selection, float32 label-aware SIMPLS/LDA, "
        "package-owned neighbor search, strict backend metadata, and separately maintained R/Python bindings. "
        "The detailed delta is recorded in docs/previous-version-delta.md. The prior papers and preprint "
        "will be supplied as supplementary material."
    )

    doc.add_heading("Evidence of use and openness", level=1)
    doc.add_paragraph(
        "KODAMA has an established public predecessor: a peer-reviewed method paper, a peer-reviewed R-package "
        "paper, a maintained public R package, documentation, and public source history. The CRANlogs public API "
        "reports 48,480 KODAMA downloads from 2017-01-01 through 2026-07-26, including 4,526 during "
        "2025-07-27--2026-07-26 (https://cranlogs.r-pkg.org/downloads/total/2017-01-01:2026-07-26/KODAMA and "
        "https://cranlogs.r-pkg.org/downloads/total/2025-07-27:2026-07-26/KODAMA). These counts concern the "
        "predecessor R package, not the new implementation. kodama-cpp itself is a new repository and, on "
        "2026-07-27, had zero GitHub stars, forks, releases, and public issues; these figures are disclosed rather "
        "than presented as adoption. AUTHOR ACTION REQUIRED: refresh the dated counts and add any independently "
        "verifiable user or downstream-package evidence available immediately before submission. "
        "The repository provides an issue tracker, contribution guide, changelog, API-stability policy, reproducible "
        "tests, and release checklist."
    )

    doc.add_heading("Required author declarations", level=1)
    supplement.add_bullets(
        doc,
        [
            "Coauthor consent: AUTHOR ACTION REQUIRED. Confirm that all seven authors know of and consent to this submission.",
            "Funding supporting this work during the previous 36 months: AUTHOR ACTION REQUIRED.",
            "Competing interests and recent collaborations with JMLR action editors: AUTHOR ACTION REQUIRED.",
            "Suggested action editors: AUTHOR ACTION REQUIRED. Supply 3-5 conflict-free JMLR action editors with brief relevance statements.",
            "Suggested reviewers: AUTHOR ACTION REQUIRED. Supply 3-5 conflict-free reviewers with brief relevance statements.",
        ],
    )

    doc.add_heading("Keywords", level=1)
    doc.add_paragraph("KODAMA; cross-validation; unsupervised learning; heterogeneous computing; manifold learning")

    doc.add_paragraph("Sincerely,")
    doc.add_paragraph(
        "Leonardo Tenori and Stefano Cacciatore\n"
        "Co-corresponding authors\n"
        "tenori@cerm.unifi.it; stefano.cacciatore@icgeb.org"
    )
    doc.save(COVER_DOCX)

    cover_text = "\n\n".join(
        paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()
    )
    COVER_TXT.write_text(cover_text + "\n", encoding="utf-8")


def build_readiness_report() -> None:
    doc = Document()
    supplement.apply_styles(doc)
    supplement.apply_compact_memo_styles(doc)
    supplement.add_title(
        doc,
        "JMLR MLOSS reviewer comments and response status",
        "kodama-cpp release candidate 0.1.0",
        "Internal submission-readiness report",
    )
    doc.add_paragraph(
        "This report separates completed manuscript/repository revisions from evidence that cannot be "
        "represented as complete until the remaining benchmark matrix, tagged archive, and author declarations are available."
    )
    rows = [
        ("Legacy KNN comparison", "A matched MetRef KNN benchmark compares the KNN-capable KODAMA 2.4.1 release with kodama-cpp single-core CPU, four-core CPU, and CUDA paths. It is not presented as the current CRAN baseline.", "Completed on chiamaka"),
        ("Current CRAN comparison", "A separately labeled MetRef M=20/Tcycle=20 systems check compares KODAMA 3.3 CPU4 with kodama-cpp PLS-LDA CPU4. It reports unfavorable results unchanged and explicitly states classifier non-parity and the CUDA k limit.", "Completed on chiamaka"),
        ("Frozen public API", "Version 0.1.0 macros, SemVer policy, CMake package version, wrapper versions, and a compile-link API snapshot test are present.", "Completed locally"),
        ("Version/tag/archive", "Release checklist and clean-tag archive script prevent an uncommitted or dirty artifact from being cited; a DOI is optional.", "External tag and deposit pending"),
        ("License provenance", "Pinned upstream snapshots, per-component licensing, SPDX tests, retained licenses, and the Metal Apache exception are documented.", "Engineering audit complete; coauthor contribution confirmation remains"),
        ("MLOSS page budget", "The detailed manuscript is retained as a technical supplement; a separate official-style main paper targets four description pages plus references.", "Completed; four description pages plus references verified"),
        ("CUDA release build", "Commit a32f718 passed all four configured license, numerical/core, public-API, and float32 smoke tests in the CUDA 13.0 build on chiamaka.", "Completed on chiamaka"),
        ("Continuous integration", "GitHub Actions builds and tests the CPU core on Linux/macOS, native Metal on macOS, and the R/Python wrappers on Ubuntu. Separate coverage and API-documentation workflows passed for a32f718.", "Completed at a32f718"),
        ("User documentation", "A compact C++/R/Python walkthrough and generated Doxygen API are published through GitHub Pages.", "Completed and URL verified"),
        ("Active user community", "The cover letter reports dated CRANlogs evidence for the predecessor R package and explicitly discloses that the new library has no public adoption metrics yet.", "Independent kodama-cpp usage evidence pending"),
        ("Python publication", "The tested Python wrapper is included in the public core repository and exercised by CI.", "Standalone public repository pending"),
        ("Coverage depth", "LLVM reports 63.58% line and 57.03% branch CPU coverage; hardware-specific tests cover Metal and manually validated CUDA paths.", "Substantial state-machine, graph, PLS, and CUDA audit expansion pending"),
        ("M/Tcycle rationale", "Named MetRef/USPS sweeps report CV accuracy, ARI, active classes, runtime, and agreement-graph convergence; M=100 is justified by ensemble precision rather than external-label tuning.", "Completed on CUDA"),
        ("Final ablations", "The release driver fixes M=100, Tcycle=100, landmarks, k, ncomp, splitting, graph correction, backend, wrappers, and external metrics before inspection.", "Five-dataset CUDA panel complete; current-CRAN CPU4 check complete; exact current-CRAN CUDA match excluded because k=500 exceeds the native CUDA limit"),
        ("Visualization claims", "Classic and KODAMA UMAP/openTSNE use the same implementation/settings. The completed five-dataset panel retains positive and negative results, and the paper explicitly declines a universal improvement claim.", "Completed for MetRef, PBMC3K PCA50, OptDigits, USPS, and Macosko2015 retina"),
        ("Cover letter", "A track-specific draft includes prior-publication disclosure, license, URL, version, software delta, and explicit author-only fields.", "Draft complete; author fields pending"),
        ("Release sustainability", "README, changelog, contribution guide, API policy, prior-version delta, validation protocol, and release checklist are included.", "Completed locally"),
    ]
    supplement.add_table(doc, ("Comment", "Response", "Status"), rows, [1.55, 4.15, 1.25])
    doc.save(READINESS_DOCX)


def main() -> None:
    supplement.build_bib()
    build_main_tex()
    build_cover_letter()
    build_readiness_report()
    print(MAIN_TEX)
    print(COVER_DOCX)
    print(COVER_TXT)
    print(READINESS_DOCX)


if __name__ == "__main__":
    main()
