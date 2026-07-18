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
REVIEWED_BASELINE = "0eb2261ec10c62082950acb6f7b1c9f98c4617a4"


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

        \jmlrheading{1}{2026}{1--\pageref{LastPage}}{7/26}{}{26-0000}{Kassim, Ocharo, Vignoli, Tenori, and Cacciatore}
        \ShortHeadings{kodama-cpp}{Kassim et al.}
        \firstpageno{1}

        \begin{document}

        \title{kodama-cpp: Cross-Validated Accuracy Maximization on CPU, CUDA, and Apple Metal}

        \author{%
        \name Moussa Kassim$^{1,2}$, Martin Ocharo$^{1,2}$, Alessia Vignoli$^{3,4}$,\\
        Leonardo Tenori$^{3,4}$, and Stefano Cacciatore$^{1,2}$
        \email stefano.cacciatore@icgeb.org\\
        \addr $^1$Bioinformatics Unit, International Center for Genetic Engineering and Biotechnology, Cape Town 7925, South Africa\\
        \addr $^2$Department of Integrative Biomedical Sciences, Institute of Infectious Disease \& Molecular Medicine, University of Cape Town, Cape Town 7925, South Africa\\
        \addr $^3$Department of Chemistry ``Ugo Schiff'', University of Florence, Sesto Fiorentino, Italy\\
        \addr $^4$Magnetic Resonance Center (CERM), University of Florence, Sesto Fiorentino, Italy}

        \editor{To be assigned}
        \maketitle

        \begin{abstract}
        KODAMA searches for latent structure by maximizing the held-out predictability of an evolving label vector. We present \texttt{kodama-cpp}, a standalone C++17 implementation that preserves this objective while reorganizing its repeated numerical work for multicore CPU, NVIDIA CUDA, and Apple Metal. One typed core owns float32 data, folds, labels, classifier workspaces, and graph outputs. KODAMA optimization exposes two classifiers: KNN and SIMPLS followed by latent-space LDA. Backend-specific implementations provide nearest-neighbor search, k-means initialization, label-aware SIMPLS cross-products, reusable workspaces, and strict backend identity without linking R, Python, FAISS, cuVS, RAFT, or Armadillo. Independent optimization runs can start from different sampled landmarks, and every proposal cycle performs exactly one cross-validation evaluation. The public API supports data matrices, supplied neighbor graphs, graph construction, float32 randomized PCA, and UMAP/openTSNE visualization; thin R and Python wrappers call the same interface. Validation separates classifier parity, kernel runtime, complete KODAMA runtime, and external-label diagnostics so that internal cross-validated accuracy is not mistaken for ground-truth recovery.
        \end{abstract}

        \begin{keywords}
        KODAMA, cross-validation, unsupervised learning, heterogeneous computing, manifold learning
        \end{keywords}

        \section{Introduction}

        KODAMA treats a sample-label vector as an optimization variable rather than observed truth. A candidate labeling is useful when a classifier trained on some samples can reproduce it on held-out samples. The original method and R package established this cross-validated accuracy principle and used an ensemble of optimized label vectors to correct pairwise dissimilarities before visualization \citep{cacciatore2014kodama,cacciatore2017kodama}. The method is therefore related to prediction-strength and stability validation, but differs by placing held-out predictability inside the label search itself \citep{tibshirani2005predictionstrength}.

        Repeated classifier fitting makes KODAMA a systems problem as well as a statistical one. Neighbor graphs, fold matrices, class encodings, projections, and evolving labels recur across many proposal cycles. \texttt{kodama-cpp} moves these operations into a standalone float32 core, keeps the KODAMA state machine common across backends, and exposes the same contract to R and Python. The contribution is a new portable implementation of established mathematics, not a replacement learning objective.

        \section{KODAMA objective and search}

        For data $X\in\mathbb{R}^{n\times p}$, labels $y$, folds $\Pi$, and classifier family $F$, let
        \begin{equation}
          A(y;X,F,\Pi)=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\!\left[y_i=\widehat y_i^{(-\Pi(i))}\right].
        \end{equation}
        KODAMA searches for label vectors with high $A$; it does not interpret $A$ as external accuracy. The current implementation uses either KNN or SIMPLS plus LDA in the requested feasible latent dimension \citep{dejong1993simpls}. Fold assignments remain fixed within a run, and constrained samples move as one group.

        Each of $M$ runs uses seed $s+r$, selects up to $L$ landmarks, and initializes \texttt{splitting} labels by k-means. When the requested $L\ge n$, the historical rule $\lceil0.75n\rceil$ is retained. The default initial class count is 100 for $n<40000$ and 300 otherwise. At cycle $t$, the previous held-out predictions propose group relabelings. The largest proposal size decreases smoothly from broad to local moves,
        \begin{equation}
          q_{\max}(t)=1+\left\lfloor(G-1)\left[1-p_t^2(3-2p_t)\right]\right\rfloor,
          \qquad p_t=\frac{t+1}{T+1},
        \end{equation}
        and $q$ groups are sampled uniformly from $1,\ldots,q_{\max}(t)$. Class-transition proposals can absorb one or more source labels whose held-out predictions preferentially transition into a target, without permitting a one-class result. PLS--LDA additionally applies transition-driven coarsening when fragmented classes are unstable.

        The proposed vector receives exactly one new CV pass. Raw $A$ is reported, while the default acceptance score guards against trivial predictability:
        \begin{equation}
          S(y)=A(y)\sqrt{1-\sum_k p_k^2}-(1-A(y))P(y),
        \end{equation}
        where $p_k$ is a class proportion and $P=0$ for KNN; for PLS--LDA, $P$ is the larger of normalized label entropy and an effective-class fragmentation term. A proposal improving the best $S$ is retained. The current chain can also accept a worse proposal with probability $\exp[(S_{new}-S_{cur})/\tau_t]$, where $\tau_t=0.10(1-A_{cur})(1-t/T)$. These grouped, guarded search dynamics extend the historical implementation but use only CV predictions, never reference labels.

        After $T$ cycles, labels are projected from landmarks to all samples. Across runs, an original graph edge $(i,j)$ has agreement $a_{ij}$ equal to the fraction of valid runs assigning its endpoints the same label. Its corrected distance is $d'_{ij}=(1+d_{ij})/a_{ij}^2$; zero agreement removes the edge. This sparse graph is the KODAMA representation supplied to visualization.

        \begin{quote}\small
        \textbf{One independent run.} Initialize landmarks, labels, folds, and one CV prediction; for $t=1,\ldots,T$: propose grouped and class-transition moves from the current predictions; evaluate the proposal once by CV; update the best state by $S$ and the current state by the cooling rule. Return the best raw accuracy and labels. Repeat independently for $r=1,\ldots,M$, project labels, and reweight the shared graph by run-wise agreement.
        \end{quote}

        \section{Standalone heterogeneous implementation}

        Figure~\ref{fig:architecture} shows the ownership boundary. CPU provides package-owned HNSW \citep{malkov2020hnsw}, SIMPLS/LDA, PCA, graph operations, and UMAP/openTSNE. CUDA provides exact/IVF KNN, batched k-means, label-aware SIMPLS/LDA, PCA, and embedding kernels; Metal provides exact/IVF KNN, k-means, MPS-assisted SIMPLS/LDA, and PCA. Unavailable accelerators raise errors rather than silently executing CPU code.

        \begin{figure}[t]
        \centering
        \includegraphics[width=0.78\linewidth]{kodama_cpp_architecture.png}
        \caption{The C++17 core owns KODAMA state and exposes one typed API to R and Python. Backend-specific kernels share the same folds, proposals, component-count semantics, and result contract.}
        \label{fig:architecture}
        \end{figure}

        Dense one-hot responses are avoided in PLS--LDA: $X^\top Y$ is computed from class-wise feature sums, and compact active labels are mapped to contiguous codes each cycle. Fold indices and workspaces are reused. The requested component count is evaluated wherever mathematically feasible; it is not selected by internal validation. Results include predictions, folds, confusion matrices, label ensembles, timings, memory, search parameters, and the backend that executed.

        Graph input is an alternative API. KNN consumes supplied indices and distances; graph-only PLS--LDA uses one documented self-tuning normalized-Laplacian transform. PCA is an auxiliary primitive, not part of the KODAMA objective. Pinned fastEmbedR UMAP/openTSNE kernels use matched contracts for classic and corrected graphs, with direct float32 CSR construction and explicit binary or fuzzy UMAP edges \citep{mcinnes2018umap,vandermaaten2008tsne}.

        \section{Validation and scope}

        Validation separates kernels from end-to-end runtime. CTest checks predictions, folds, requested components, PCA invariants, strict backend metadata, float32 inputs, graph utilities, and the frozen \texttt{0.1.0} API; wrappers test matrix, graph, PCA, and visualization calls. Table~\ref{tab:validation} reports within-platform measurements.

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

        At matched MetRef KNN settings ($M=T=100$, \texttt{splitting}=50), the legacy KNN-capable KODAMA R 2.4.1 took 610.5 seconds; kodama-cpp CPU1, CPU4, and CUDA took 965.3, 235.5, and 2.36 seconds. All reached best raw $A=1$. This is neither a current-CRAN nor a trajectory-parity comparison.

        An easily predicted labeling may still be too coarse, so raw accuracy, proposal score, and external diagnostics are separated. At $M=100$, increasing $T$ from 20 to 100 improved median accuracy and reduced fragmentation on USPS and MetRef PLS--LDA; MetRef KNN remained over-coarsened. $M$ instead controls ensemble precision: worst-case agreement-edge standard error is 0.05 at $M=100$, and all four $M=50$ graphs correlated at least 0.9888 with their $M=100$ counterparts. ARI was nonmonotone and was not tuned. A five-dataset panel, including 44,808 samples in its largest matrix, showed classifier- and dataset-dependent silhouette changes; the supplement retains all positive and negative results.

        \section{Availability and limitations}

        Version \texttt{0.1.0} is MIT licensed at \url{https://github.com/tkcaccia/kodama-cpp}; adapted portions retain upstream notices and the Metal Apache-2.0 exception. Provenance pins source snapshots, and release tests compile-link the API. The final commit and DOI await archival deposition.

        Repeated CV fitting remains the main cost, and $M$-run orchestration is not fully device-resident. Approximate search may change neighbor ties within documented recall tolerances. Graph-only PLS--LDA is a spectral surrogate, not equivalent to data-input PLS--LDA; strict backend metadata exposes these boundaries.

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
            "Coauthor consent: AUTHOR ACTION REQUIRED. Confirm that all five authors know of and consent to this submission.",
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
        ("Current CRAN comparison", "KODAMA 3.3 is the current CRAN release. Its deprecated FUN argument is ignored and its automatic PLS-DA route is not classifier-parity with kodama-cpp latent-space PLS-LDA, so it requires a separately labeled end-to-end comparison.", "Benchmark pending"),
        ("Frozen public API", "Version 0.1.0 macros, SemVer policy, CMake package version, wrapper versions, and a compile-link API snapshot test are present.", "Completed locally"),
        ("Version/tag/DOI", "Release checklist and clean-tag archive script prevent an uncommitted or dirty artifact from being cited.", "External tag and deposit pending"),
        ("License provenance", "Pinned upstream snapshots, per-component licensing, SPDX tests, retained licenses, and the Metal Apache exception are documented.", "Engineering audit complete; coauthor contribution confirmation remains"),
        ("MLOSS page budget", "The detailed manuscript is retained as a technical supplement; a separate official-style main paper targets four description pages plus references.", "Completed; four description pages plus references verified"),
        ("CUDA release build", "An isolated CUDA 13.2 release-candidate build passed all four configured CTests, including license, core, public-API, and float32 install-consumer checks.", "Completed on chiamaka"),
        ("Continuous integration", "Manual CPU, CUDA, Metal, R, and Python checks are recorded, but no cross-platform GitHub Actions workflow or measured coverage report is yet committed.", "Release blocker"),
        ("User documentation", "Installation and wrapper references exist; a compact end-to-end C++/R/Python tutorial and browsable generated API site are still needed.", "Release blocker"),
        ("M/Tcycle rationale", "Named MetRef/USPS sweeps report CV accuracy, ARI, active classes, runtime, and agreement-graph convergence; M=100 is justified by ensemble precision rather than external-label tuning.", "Completed on CUDA"),
        ("Final ablations", "The release driver fixes M=100, Tcycle=100, landmarks, k, ncomp, splitting, graph correction, backend, wrappers, and external metrics before inspection.", "Five-dataset CUDA panel complete, including the 44,808-sample scale case; matched-backend rows pending"),
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
