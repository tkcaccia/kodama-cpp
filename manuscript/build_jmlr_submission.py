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
REVIEWED_BASELINE = "0e019c1c2d371e330355afa9ca7fa3b18761de0a"


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
        KODAMA searches for latent structure by maximizing the held-out predictability of evolving labels. We present \texttt{kodama-cpp}, a standalone C++17 implementation that preserves this objective across multicore CPU, NVIDIA CUDA, and Apple Metal. One float32 core owns folds, labels, classifier workspaces, and graph outputs. It exposes KNN and SIMPLS followed by latent-space LDA; backend implementations provide neighbor search, k-means, label-aware cross-products, reusable workspaces, and strict backend identity without runtime dependence on R, Python, FAISS, cuVS, RAFT, or Armadillo. Independent runs may sample different landmarks, and each proposal receives exactly one cross-validation evaluation. The matrix/graph API also provides PCA and UMAP/openTSNE, with thin R and Python wrappers. Validation separates classifier parity, kernel and complete runtimes, and external-label diagnostics.
        \end{abstract}

        \begin{keywords}
        KODAMA, cross-validation, unsupervised learning, heterogeneous computing, manifold learning
        \end{keywords}

        \section{Introduction}

        KODAMA treats a sample-label vector as an optimization variable rather than observed truth. A candidate labeling is useful when a classifier trained on some samples can reproduce it on held-out samples. The original method and R package established this cross-validated accuracy principle and used an ensemble of optimized label vectors to correct pairwise dissimilarities before visualization \citep{cacciatore2014kodama,cacciatore2017kodama}. The method is therefore related to prediction-strength and stability validation, but differs by placing held-out predictability inside the label search itself \citep{tibshirani2005predictionstrength}.

        Repeated fitting also makes KODAMA a systems problem: graphs, folds, class encodings, and projections recur across proposals. \texttt{kodama-cpp} moves this work into one float32 core with a common state machine and R/Python contract. It is a portable implementation of established mathematics, not a replacement objective.

        \section{KODAMA objective and search}

        For data $X\in\mathbb{R}^{n\times p}$, labels $y$, folds $\Pi$, and classifier family $F$, let
        \begin{equation}
          A(y;X,F,\Pi)=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\!\left[y_i=\widehat y_i^{(-\Pi(i))}\right].
        \end{equation}
        KODAMA searches for label vectors with high $A$; it does not interpret $A$ as external accuracy. The current implementation uses either KNN or SIMPLS plus LDA in the requested feasible latent dimension \citep{dejong1993simpls}. Fold assignments remain fixed within a run, and constrained samples move as one group.

        Run $r$ uses seed $s+r$, up to $L$ landmarks, and \texttt{splitting} k-means labels. If $L\ge n$, the historical rule $\lceil0.75n\rceil$ is retained; initial classes default to 100 for $n<40000$ and 300 otherwise. Held-out predictions propose grouped relabelings whose maximum size decreases smoothly,
        \begin{equation}
          q_{\max}(t)=1+\left\lfloor(G-1)\left[1-p_t^2(3-2p_t)\right]\right\rfloor,
          \qquad p_t=\frac{t+1}{T+1},
        \end{equation}
        with $q$ sampled uniformly from $1,\ldots,q_{\max}(t)$. Transition proposals absorb source labels preferentially predicted as a target without permitting one class; PLS--LDA also coarsens unstable fragments.

        The proposed vector receives exactly one new CV pass. Raw $A$ is reported, while the default acceptance score guards against trivial predictability:
        \begin{equation}
          S(y)=A(y)\sqrt{1-\sum_k p_k^2}-(1-A(y))P(y),
        \end{equation}
        where $p_k$ is a class proportion and $P=0$ for KNN; for PLS--LDA, $P$ is the larger of normalized label entropy and an effective-class fragmentation term. A proposal improving the best $S$ is retained. The current chain can also accept a worse proposal with probability $\exp[(S_{new}-S_{cur})/\tau_t]$, where $\tau_t=0.10(1-A_{cur})(1-t/T)$. These grouped, guarded search dynamics extend the historical implementation but use only CV predictions, never reference labels.

        After $T$ cycles, labels are projected from landmarks to all samples. Across runs, an original graph edge $(i,j)$ has agreement $a_{ij}$ equal to the fraction of valid runs assigning its endpoints the same label. Its corrected distance is $d'_{ij}=(1+d_{ij})/a_{ij}^2$; zero agreement removes the edge. This sparse graph is the KODAMA representation supplied to visualization.

        \begin{quote}\small
        \textbf{One independent run.} Initialize landmarks, labels, folds, and one CV prediction. For $t=1,\ldots,T$, propose grouped/transition moves, evaluate once by CV, update the best by $S$, and update the current state by cooling. Return the best raw accuracy and labels. Repeat for $M$ independent runs, project labels, and reweight the shared graph by agreement.
        \end{quote}

        \section{Standalone heterogeneous implementation}

        Figure~\ref{fig:architecture} shows the ownership boundary. CPU provides package-owned HNSW \citep{malkov2020hnsw}, SIMPLS/LDA, PCA, graph operations, and UMAP/openTSNE. CUDA provides exact/IVF KNN, batched k-means, label-aware SIMPLS/LDA, PCA, and embedding kernels; Metal provides exact/IVF KNN, k-means, MPS-assisted SIMPLS/LDA, and PCA. Unavailable accelerators raise errors rather than silently executing CPU code.

        \begin{figure}[t]
        \centering
        \includegraphics[width=0.78\linewidth]{kodama_cpp_architecture.png}
        \caption{The C++17 core owns KODAMA state and exposes one typed API to R and Python. Backend-specific kernels share the same folds, proposals, component-count semantics, and result contract.}
        \label{fig:architecture}
        \end{figure}

        PLS--LDA avoids dense one-hot responses by computing $X^\top Y$ from class sums and compacting active labels each cycle. Fold indices and workspaces are reused. The requested feasible component count is evaluated without internal selection. Results record predictions, folds, confusion matrices, label ensembles, resources, search parameters, and the executing backend.

        The alternative graph API lets KNN consume supplied neighbors; graph-only PLS--LDA uses a documented self-tuning normalized Laplacian. PCA is auxiliary. Pinned fastEmbedR UMAP/openTSNE kernels use matched classic/corrected contracts and direct float32 CSR graphs \citep{mcinnes2018umap,vandermaaten2008tsne}.

        \section{Validation and scope}

        CTest checks numerical, float32, backend, graph, and frozen \texttt{0.1.0} API contracts; wrappers test matrix, graph, PCA, and visualization calls. GitHub Actions passed CPU jobs on Linux and macOS, native Metal, R/Python, coverage, and documentation. LLVM CPU coverage was 63.58\% by line and 57.03\% by branch; accelerators retain hardware-specific tests. Table~\ref{tab:validation} reports within-platform measurements.

        \begin{table}[t]
        \caption{Selected implementation validation. Accuracy is CPU/accelerator; runtime is seconds.}
        \label{tab:validation}
        \centering\small
        \begin{tabular}{llrrrr}
        \toprule
        Platform & Dataset/kernel & CPU & Accel. & Speedup & Accuracy \\
        \midrule
        CUDA & MNIST KNNCV & 76.769 & 4.753 & 16.2 & .974/.973 \\
        CUDA & MetRef PLSLDACV & 3.196 & 0.118 & 27.2 & .992/.992 \\
        CUDA & USPS PLSLDACV & 3.760 & 0.158 & 23.8 & .684/.687 \\
        Metal & MetRef KNNCV & 11.145 & 0.026 & 425.0 & .827/.827 \\
        Metal & MetRef PLSLDACV & 3.395 & 1.054 & 3.2 & .992/.992 \\
        \bottomrule
        \end{tabular}
        \end{table}

        At matched MetRef KNN settings ($M=T=100$, \texttt{splitting}=50), legacy KODAMA R 2.4.1 took 610.5 seconds versus 965.3, 235.5, and 2.36 seconds for kodama-cpp CPU1, CPU4, and CUDA. A short non-parity check ($M=T=20$, 655 landmarks, graph $k=500$) took 344.914 seconds for KODAMA 3.3 CPU4 and 822.989 seconds for kodama-cpp PLS--LDA CPU4; selected ARI was .7004/.6430. CUDA is omitted because $k=500$ exceeds its limit of 256.

        Raw accuracy, proposal score, and external diagnostics remain separate because predictability can be too coarse. Raising $T$ from 20 to 100 improved median accuracy and fragmentation for USPS and MetRef PLS--LDA, but MetRef KNN stayed over-coarsened. At $M=100$, worst-case agreement-edge standard error is .05; all $M=50$ graphs correlated at least .9888 with $M=100$. ARI was nonmonotone and untuned. A five-dataset panel (largest $n=44{,}808$) retained positive and negative silhouette changes.

        \section{Availability and limitations}

        Version \texttt{0.1.0} is MIT licensed at \url{https://github.com/tkcaccia/kodama-cpp}; retained notices include a Metal Apache-2.0 exception. Provenance and API tests are archived, and documentation is at \url{https://tkcaccia.github.io/kodama-cpp/}; the final tag/DOI remain pending. Repeated CV is the main cost, and $M$ runs are not fully device-resident. Approximate ties may vary. CUDA graph construction supports $k\leq256$ and raises otherwise. Graph-only PLS--LDA is a spectral surrogate, not data-input PLS--LDA.

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
    add_label_value(doc, "Software version", SOFTWARE_VERSION)
    add_label_value(doc, "Reviewed baseline", REVIEWED_BASELINE)
    add_label_value(doc, "Final release commit", "AUTHOR ACTION REQUIRED: insert the full v0.1.0 tagged commit")
    add_label_value(doc, "Archival DOI", "AUTHOR ACTION REQUIRED: insert after depositing the tagged source archive")

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
        "paper, a maintained public R package, documentation, and public source history. kodama-cpp itself is a "
        "new repository, so the cover letter should not inflate its current star or contributor count. "
        "AUTHOR ACTION REQUIRED: add a dated CRAN/R-universe download count and current repository metrics immediately "
        "before submission. The repository provides an issue tracker, contribution guide, changelog, API-stability "
        "policy, reproducible tests, and release checklist."
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

    COVER_TXT.write_text(
        "JMLR MLOSS cover letter for kodama-cpp\n\n"
        "This plain-text companion is generated with the formatted DOCX. Fields marked AUTHOR ACTION REQUIRED "
        "must be completed before submission: final v0.1.0 commit, archival DOI, current community metrics, "
        "coauthor consent, funding, competing interests, 3-5 action editors, and 3-5 reviewers.\n",
        encoding="utf-8",
    )


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
        ("Version/tag/DOI", "Release checklist and clean-tag archive script prevent an uncommitted or dirty artifact from being cited.", "External tag and deposit pending"),
        ("License provenance", "Pinned upstream snapshots, per-component licensing, SPDX tests, retained licenses, and the Metal Apache exception are documented.", "Engineering audit complete; coauthor contribution confirmation remains"),
        ("MLOSS page budget", "The detailed manuscript is retained as a technical supplement; a separate official-style main paper targets four description pages plus references.", "Completed; four description pages plus references verified"),
        ("CUDA release build", "An isolated CUDA 13.2 release-candidate build passed all four configured CTests, including license, core, public-API, and float32 install-consumer checks.", "Completed on chiamaka"),
        ("Continuous integration", "GitHub Actions builds and tests the CPU core on Linux/macOS, native Metal on macOS, and the R/Python wrappers on Ubuntu. A separate LLVM workflow reports CPU coverage.", "Completed at 0e019c1"),
        ("User documentation", "A compact C++/R/Python walkthrough and generated Doxygen API are published through GitHub Pages.", "Completed and URL verified"),
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
