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
REVIEWED_BASELINE = "9da48ee03d4ce76d50adf7cae0189b8ea7d65c62"
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

        \jmlrheading{1}{2026}{1--\pageref{LastPage}}{7/26}{}{26-0000}{Kassim et al.}
        \ShortHeadings{kodama-cpp}{Kassim et al.}
        \firstpageno{1}

        \begin{document}

        \title{kodama-cpp: Cross-Validated Accuracy Maximization on CPU, CUDA, and Apple Metal}

        \author{%
        \name Moussa Kassim$^{1,2}$ \email moussa.kassim@icgeb.org\\
        \name Martin Ocharo$^{1,2}$ \email martin.ocharo@icgeb.org\\
        \name Dalia Ahmed$^{1}$ \email dalia.ahmed@icgeb.org\\
        \name Dupe Ojo$^{1}$ \email dupe.ojo@icgeb.org\\
        \name Alessia Vignoli$^{3,4}$ \email vignoli@cerm.unifi.it\\
        \name Leonardo Tenori$^{3,4}$ \email tenori@cerm.unifi.it\\
        \name Stefano Cacciatore$^{1,2}$ \email stefano.cacciatore@icgeb.org\\
        \addr $^1$Bioinformatics Unit, International Centre for Genetic Engineering and Biotechnology (ICGEB), Cape Town 7925, South Africa\\
        \addr $^2$Department of Integrative Biomedical Sciences, Institute of Infectious Disease \& Molecular Medicine (IDM), University of Cape Town, Cape Town 7925, South Africa\\
        \addr $^3$Department of Chemistry ``Ugo Schiff'', University of Florence, Sesto Fiorentino, Italy\\
        \addr $^4$Magnetic Resonance Center (CERM), University of Florence, Sesto Fiorentino, Italy}

        \editor{To be assigned}
        \maketitle

        \begin{abstract}
        KODAMA searches for latent structure by maximizing the held-out predictability of evolving labels. We present \texttt{kodama-cpp}, a standalone C++17 implementation for multicore CPU, NVIDIA CUDA, and Apple Metal. One float32 core owns folds, labels, workspaces, and graph outputs, and exposes KNN or SIMPLS followed by latent-space LDA. Backend kernels provide neighbor search, k-means, label-aware cross-products, and reusable workspaces without runtime dependence on R, Python, FAISS, cuVS, RAFT, or Armadillo. The current standard specifies grouped proposals and degeneracy guards while retaining the classifier objective, but not identical label trajectories. Matrix/graph APIs and thin wrappers also expose PCA and UMAP/openTSNE. We validate primitive parity, end-to-end runtime, and external-label diagnostics.
        \end{abstract}

        \begin{keywords}
        KODAMA, cross-validation, unsupervised learning, heterogeneous computing, manifold learning
        \end{keywords}

        \section{Introduction}

        KODAMA treats labels as optimization variables: a labeling is useful when a classifier trained on some samples reproduces it on held-out samples. The original method used an ensemble of optimized vectors to correct dissimilarities before visualization \citep{cacciatore2014kodama,cacciatore2017kodama}. Unlike cluster stability, prediction strength, or semi-supervised learning, KODAMA places held-out predictability inside label search without observed labels as anchors \citep{benhur2002stability,tibshirani2005predictionstrength,chapelle2006semisupervised}.

        Repeated fitting makes KODAMA a systems problem because graphs, folds, encodings, and projections recur. \texttt{kodama-cpp} consolidates them in one float32 state machine. The held-out signal and KNN/PLS--LDA classifiers are inherited; grouped proposals, guarded acceptance, and sparse agreement graphs are current policies evaluated separately.

        \section{KODAMA objective and search}

        For data $X\in\mathbb{R}^{n\times p}$, labels $y$, folds $\Pi$, and classifier family $\mathcal F$, let
        \begin{equation}
          A(y;X,\mathcal F,\Pi)=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\!\left[y_i=\widehat y_i^{(-\Pi(i))}\right].
        \end{equation}
        KODAMA searches for label vectors with high $A$; it does not interpret $A$ as agreement with external truth. The classifier is KNN or SIMPLS plus LDA in the requested feasible latent dimension \citep{dejong1993simpls}. Folds remain fixed within a run, and samples sharing a supplied constraint identifier form an atomic proposal group.

        Run $r$ uses seed $s+r$ and exactly $L<n$ landmarks. If the request is at least $n$, the historical $L=\lceil0.75n\rceil$ rule is retained. Matrix-only input is partitioned into $K_0=\texttt{splitting}$ coarse k-means strata (100 for $n<40000$, 300 otherwise). Stratum $c$, with $n_c$ samples, receives expected quota $q_c=Ln_c/n$: $\lfloor q_c\rfloor$ samples are drawn without replacement and the remaining slots are allocated by randomized systematic rounding of the fractional quotas. This replaces the previous $L$-center landmark k-means. With auxiliary 2D/3D coordinates, an automatic regular grid uses $\lceil L^{1/d}\rceil$ bins per axis and applies the same quota rule to occupied cells in grid order; no cell is guaranteed a landmark, so isolated spots can be omitted without a density threshold. In both paths, a separate expression-space \texttt{splitting} k-means on the selected landmarks preserves the established initial-label construction and keeps sampling strata distinct from labels being optimized. Held-out predictions then propose grouped relabelings whose maximum size decreases smoothly,
        \begin{equation}
          q_{\max}(t)=1+\left\lfloor(G-1)\left[1-p_t^2(3-2p_t)\right]\right\rfloor,
          \qquad p_t=\frac{t}{T+1},
        \end{equation}
        with uniform $q\in\{1,\ldots,q_{\max}(t)\}$. A group replacement is sampled from member predictions by multiplicity. The transition matrix $N_{ab}=\#\{i:y_i=a,\widehat y_i=b\}$ permits absorption only if $N_{ab}-n_an_b/n>0$ and $N_{ab}>N_{aa}$; one-class proposals are rejected. PLS--LDA also uses the transition-coarsening budget specified in the supplement.

        The proposed vector receives exactly one new CV pass. Let $p_k=n_k/n$, $D=1-\sum_kp_k^2$, $H=-\sum_kp_k\log p_k$, $K_{\rm eff}=e^H$, and $R=\max\{0,\log[K/\max(1,K_{\rm eff})]\}$. The state-selection score is
        \begin{equation}
          S(y)=A(y)\sqrt D-\mathbf{1}_{\{\mathcal F={\rm PLS\mbox{-}LDA}\}}(1-A(y))
          \max\!\left\{\frac{H}{\log n},\frac{R}{1+R}\right\}.
        \end{equation}
        Best labels maximize $S$, with paired raw $A$ returned as \texttt{accbest}. Improvements are accepted. Worse proposals have probability $P_{\rm acc}=\exp[(S_{\rm new}-S_{\rm cur})/\tau_t]$. Here $\tau_t=\max\{10^{-9},\,0.10(1-A_{\rm cur})(1-t/T)\}$. The final cycle is greedy up to the floor. Search uses labels and CV predictions, never reference labels.

        After $T$ cycles, labels are projected from landmarks. For each original graph edge $(i,j)$, $a_{ij}$ is the fraction of valid runs agreeing on its endpoints. The contract is $d'_{ij}=(1+d_{ij})/a_{ij}^2$; zero agreement removes the edge and rows are resorted. Since this is not scale invariant, comparisons fix metric and preprocessing.

        \begin{quote}\small
        \textbf{One independent run.} Initialize landmarks, labels, folds, and one CV prediction. For $t=1,\ldots,T$, propose grouped/transition moves, evaluate once by CV, update the best-by-$S$ state, and update the current state by cooling. Return the best-by-$S$ labels and their paired raw accuracy. Repeat for $M$ independent seeds, project labels, and reweight the shared graph by agreement.
        \end{quote}

        \section{Standalone heterogeneous implementation}

        Figure~\ref{fig:architecture} shows ownership. CPU provides package-owned HNSW \citep{malkov2020hnsw}, SIMPLS/LDA, PCA, graph operations, and embeddings. CUDA provides exact/IVF KNN, batched k-means, label-aware SIMPLS/LDA, PCA, and embeddings; Metal provides exact/IVF KNN, k-means, MPS-assisted SIMPLS/LDA, and PCA. For IVF construction on both accelerators, assignments, centroid accumulation and empty-cluster repair, list counting, prefix offsets, and identifier scatter remain on-device. A move-only resident-index API retains the float32 training matrix and trained IVF state across self or external-query calls without a global cache. Unavailable accelerators raise errors instead of falling back.

        \begin{figure}[t]
        \centering
        \includegraphics[width=0.74\linewidth]{kodama_cpp_architecture.png}
        \caption{The C++17 core owns KODAMA state and exposes one typed API to R and Python. Backend-specific kernels share the same folds, proposals, component-count semantics, and result contract.}
        \label{fig:architecture}
        \end{figure}

        PLS--LDA computes $X^\top Y$ from class sums, compacts active labels, and reuses fold indices and workspaces. It evaluates the requested feasible component count without internal selection. Results record predictions, folds, confusion matrices, ensembles, resources, parameters, and backend.

        For latent training scores $T\in\mathbb{R}^{n\times q}$, class counts $n_c$, means $\mu_c$, and $C$ active classes, the pooled within-class covariance is
        \begin{equation}
          \Sigma=\frac{T^\top T-\sum_{c=1}^{C}n_c\mu_c\mu_c^\top}{\max(1,n-C)}.
        \end{equation}
        LDA solves $\Sigma w_c=\mu_c$ by Cholesky and triangular substitution. A fixed scale-normalized ridge sequence advances only after failure; it is a safeguard, not tuning. Prediction uses $\delta_c(t)=t^\top w_c-\tfrac12\mu_c^\top w_c+\log(n_c/n)$; the sequence is in the supplement.

        KNN can consume supplied neighbors; graph-only PLS--LDA uses a self-tuning normalized Laplacian and is a spectral surrogate, not data-input parity. Pinned UMAP/openTSNE kernels consume float32 CSR graphs \citep{mcinnes2018umap,vandermaaten2008tsne}. The R wrapper is at \url{https://github.com/tkcaccia/kodama-r}; tested Python source is under \texttt{split-repos/kodama-python}, pending standalone publication.

        \section{Validation and scope}

        CTest and wrappers check numerical, float32, backend, graph, PCA, visualization, and candidate-\texttt{0.1.0} API contracts. At commit \texttt{9da48ee}, GitHub Actions passed Linux/macOS CPU, Metal, R/Python, coverage, and documentation jobs. CPU coverage was 63.58\% line and 57.03\% branch, motivating more state-machine tests. Table~\ref{tab:validation} reports within-platform measurements.

        \begin{table}[t]
        \caption{Selected implementation validation. Accuracy is CPU/accelerator; runtime is seconds. The complete KODAMA row reports median raw CV accuracy.}
        \label{tab:validation}
        \centering\scriptsize
        \begin{tabular}{llrrrr}
        \toprule
        Platform & Dataset/kernel & CPU & Accel. & Speedup & Accuracy \\
        \midrule
        CUDA & MNIST KNNCV & 76.769 & 4.753 & 16.2 & .974/.973 \\
        CUDA & MetRef KODAMA, $M=T=100$ & 341.648 & 270.408 & 1.26 & .9924/.9924 \\
        CUDA & MetRef PLSLDACV & 1.294 & 0.308 & 4.20 & .992/.993 \\
        Metal & MetRef KNNCV & 11.145 & 0.026 & 425.0 & .827/.827 \\
        Metal & MetRef PLSLDACV & 0.790 & 0.202 & 3.90 & .991/.990 \\
        \bottomrule
        \end{tabular}
        \end{table}

        Five-run MetRef PLSLDACV medians gave 3.90x Metal and 4.20x CUDA speedups over four-thread hosts; accuracy differed by at most one of 873 samples. A matched current-commit $M=T=100$ run took 341.648/270.408 seconds (1.26x) and 351/730 MB peak host memory. Median raw CV accuracy was .9924 on both; finite precision selected different runs, so label equality is not claimed.

        In a deterministic 50,000-by-32 float32 systems microbenchmark with five timed searches after one warm-up, retaining one IVF index reduced 1,000-query rebuild-plus-search time from 0.1179 to 0.0306 seconds on Metal and from 0.0318 to 0.0092 seconds on CUDA (3.86x and 3.45x), with neighbor overlap 1.000. All-row self-search gains were 1.07x and 1.12x because the repeated search itself dominated construction. These timings validate state reuse rather than dataset-level quality.

        Matched MetRef KNN ($M=T=100$) took 610.5 seconds in KODAMA R 2.4.1 and 965.3/235.5/2.36 seconds in kodama-cpp CPU1/CPU4/CUDA. The supplement retains a labeled, non-parity current-R PLS--LDA check.

        On one-million-row flow18, exact-quota sampling selected 750,016 landmarks from 300 coarse strata in 0.321 seconds. A CUDA KNN run with $M=1,T=10$ took 362.35 seconds and reached raw CV accuracy .9639; 335.08 seconds were spent once on the 100-neighbor global graph, isolating graph construction rather than landmark selection as the remaining large-data bottleneck.

        Raising $T$ from 20 to 100 improved median accuracy and fragmentation for USPS and MetRef PLS--LDA, while MetRef KNN remained over-coarsened. At $M=100$, worst-case agreement-edge standard error is .05; every $M=50$ graph correlated at least .9888 with $M=100$. ARI and silhouette changes were nonmonotone.

        \section{Availability and limitations}

        The MIT candidate is at \url{https://github.com/tkcaccia/kodama-cpp}; Metal-derived files retain \texttt{MIT AND Apache-2.0} notices. Commits \texttt{9da48ee}/\texttt{4d5506b} are audited core/R baselines; a tag and archive remain pre-submission actions. Repeated CV dominates, $M$ runs are not fully device resident, approximate ties may vary, and CUDA graph $k\leq256$.

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
        "The submission preserves KODAMA's cross-validated label-predictability objective while moving "
        "its repeated work into one typed float32 core. It contributes package-owned nearest-neighbor, "
        "SIMPLS/LDA, graph, and visualization paths; strict and observable CPU/CUDA/Metal execution; "
        "matrix and supplied-graph entry points; and a common API for R and Python. The repository "
        "contains installation instructions, API examples, tests, benchmark drivers, a public API "
        "snapshot, and a machine-checkable licensing/provenance audit."
    )

    doc.add_heading("Prior publications and delta", level=1)
    doc.add_paragraph(
        "The KODAMA method was published by Cacciatore, Luchinat, and Tenori in PNAS (2014), and the "
        "historical R package was described by Cacciatore et al. in Bioinformatics (2017). Prior "
        "publication of the method is disclosed. This submission concerns the new standalone software: "
        "a C++17 core independent of R, native CUDA and Metal backends, float32 and label-aware SIMPLS/LDA, "
        "package-owned neighbor search, graph-input APIs, strict backend metadata, and separately maintained "
        "R/Python bindings. The detailed delta is recorded in docs/previous-version-delta.md. The two prior "
        "papers will be supplied as supplementary material."
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
    doc.add_paragraph("Stefano Cacciatore\nCorresponding author\nstefano.cacciatore@icgeb.org")
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
        ("CUDA release build", "An isolated commit-9da48ee CUDA 13.0 build targeting compute capability 12.0 passed all three configured license, numerical/core, and public-API tests; the installed R wrapper passed testthat.", "Completed on chiamaka"),
        ("Continuous integration", "GitHub Actions builds and tests the CPU core on Linux/macOS, native Metal on macOS, and the R/Python wrappers on Ubuntu. A separate LLVM workflow reports CPU coverage.", "Completed at 9da48ee"),
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
