from __future__ import annotations

import shutil
from pathlib import Path
from textwrap import dedent

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "kodama_cpp_jmlr_manuscript.docx"
SELF_REVIEW = ROOT / "kodama_cpp_jmlr_self_review.docx"
TEX = ROOT / "kodama_cpp_jmlr_mloss.tex"
BIB = ROOT / "kodama_cpp_refs.bib"
ARCH_FIGURE = ROOT / "kodama_cpp_architecture.png"
FIGMA_ARCH_FIGURE = ROOT / "kodama_cpp_architecture_figma.png"
SENSITIVITY_FIGURE = ROOT / "kodama_m_tcycle_sensitivity.png"


TOKENS = {
    "font": "Calibri",
    "body_size": 11,
    "h1_size": 16,
    "h2_size": 13,
    "h3_size": 12,
    "heading_blue": RGBColor(0x2E, 0x74, 0xB5),
    "heading_dark": RGBColor(0x1F, 0x4D, 0x78),
    "title": RGBColor(0x0B, 0x25, 0x45),
    "muted": RGBColor(0x55, 0x55, 0x55),
    "table_fill": "E8EEF5",
    "callout_fill": "F4F6F9",
    "border": "B7C7D9",
}


ABSTRACT = (
    "KODAMA searches for latent structure by maximizing the cross-validated predictability "
    "of an evolving label vector. We present kodama-cpp, a standalone C++17/CUDA library that "
    "moves this accuracy-maximization core out of the R package layer while preserving the "
    "methodological contract established by the original KODAMA literature. The library exposes "
    "typed KNN and PLS-LDA optimization paths, KNN and PLS-LDA cross-validation kernels, KODAMA "
    "matrix construction from either data matrices or supplied KNN graphs, graph construction, "
    "embedding helpers, and clustering utilities. Its "
    "implementation uses float32 internal storage, FAISS/cuVS nearest-neighbor infrastructure, "
    "high-recall CPU HNSW search, CUDA KNN search, label-aware SIMPLS PLS-LDA, reusable fold "
    "buffers, and independent M-cycle execution. The result is a wrapper-independent numerical "
    "backend intended to support reproducible R and Python interfaces, with validation reported "
    "separately for the internal cross-validation objective, external label agreement, runtime, "
    "and visualization quality."
)


SECTIONS = [
    (
        "1. Introduction",
        [
            (
                "High-dimensional data analysis often begins with clustering or dimensionality "
                "reduction, but neither operation directly asks whether the discovered groups are "
                "predictable from the data. KODAMA takes a different view: a useful labeling of "
                "samples should be reproducible by a classifier under cross-validation. The labels "
                "are not treated as ground truth; they are optimization variables whose quality is "
                "measured by held-out prediction."
            ),
            (
                "The original KODAMA paper introduced this idea as knowledge discovery by accuracy "
                "maximization, and the subsequent R package made it available as an unsupervised "
                "and semi-supervised feature-extraction tool for noisy high-dimensional data. The "
                "Bioinformatics package paper describes the algorithm as repeated random label "
                "initialization, iterative maximization of cross-validated accuracy by label "
                "swapping, and construction of a dissimilarity matrix from the resulting label "
                "vectors. The R documentation exposes the same logic through M independent "
                "iterative processes, Tcycle optimization steps, KNN or PLS-DA classifiers, and "
                "separate KODAMA.matrix and KODAMA.visualization functions."
            ),
            (
                "kodama-cpp addresses this problem by moving the KODAMA numerical core into a "
                "standalone C++/CUDA library. The library currently focuses on two classifier "
                "families for the KODAMA optimization layer, KNN and PLS-LDA, because they provide "
                "complementary inductive biases: local neighborhood consistency and low-rank "
                "linear discrimination in a latent component space. The implementation is "
                "designed so R and Python wrappers can call the same backend without reimplementing "
                "the mathematics."
            ),
            (
                "We deliberately frame this contribution as a software-methods paper, not as a "
                "claim that KODAMA replaces clustering, semi-supervised learning, or manifold "
                "learning. Cross-validated accuracy is the internal optimization criterion used "
                "to construct candidate label vectors. External labels, when available, are used "
                "only for diagnostics such as adjusted Rand index, silhouette, local purity, and "
                "runtime-quality tradeoffs."
            ),
        ],
    ),
    (
        "2. KODAMA objective",
        [
            (
                "Let X in R^(n x p) denote the input matrix, c in {1,...,K}^n a candidate label "
                "vector, F a classifier family, and Pi a fold assignment. For each validation "
                "sample i, the classifier is trained on samples not in Pi(i) and predicts c_i from "
                "x_i. The empirical objective is the held-out accuracy A(c; X, F, Pi). KODAMA "
                "searches for a labeling c* that maximizes this quantity, turning unsupervised "
                "structure discovery into a self-guided supervised prediction problem."
            ),
            (
                "One KODAMA run starts from an initial partition. In the historical R interface, "
                "this can be one label per sample or a clustering-derived vector W; a fraction of "
                "samples can be selected for each run, with 75% as the documented default. The "
                "current C++ interface keeps the same idea but exposes starting labels, splitting, "
                "landmarks, and fixed or constrained groups as typed inputs. During each cycle, "
                "misclassified samples or constrained groups generate candidate label moves. A "
                "move is accepted if it improves the objective, or under a temperature rule that "
                "allows exploration early in the run."
            ),
            (
                "The M runs are independent. This independence is important both statistically, "
                "because it averages over random initialization and proposal order, and "
                "computationally, because CPU workers or CUDA streams can process runs without "
                "changing the objective. The final KODAMA representation is obtained from the "
                "ensemble of optimized label vectors: pairs of samples that repeatedly receive "
                "compatible labels are pulled closer in the derived graph or dissimilarity, while "
                "unstable pairs are weakened."
            ),
        ],
    ),
    (
        "3. Implementation architecture",
        [
            (
                "The core library exposes C++ functions rather than wrapper-specific entry points. "
                "A MatrixView accepts double or float inputs; internally, the analysis matrix is "
                "stored in float32 buffers to match GPU kernels and reduce memory traffic. The "
                "public API is grouped into cross-validation kernels, core label optimization, "
                "KODAMA matrix construction, graph/embedding utilities, and clustering routines."
            ),
            (
                "KNN kernels use FAISS/cuVS infrastructure. The CPU path uses FAISS HNSW with a "
                "high-recall target, while the CUDA path uses GPU nearest-neighbor search for the "
                "same voting objective. In the KODAMA optimizer, neighbor work is organized so the "
                "search structure can be reused where the mathematics permits. This is the main "
                "difference from repeatedly launching wrapper-level nearest-neighbor searches."
            ),
            (
                "PLS-LDA uses a SIMPLS strategy following the fastPLS implementation rather than "
                "a simplified SVD approximation. CUDA PLS-LDA avoids dense one-hot response "
                "matrices by computing class-label cross-products directly when possible. The "
                "requested component count is treated as the evaluated component count whenever "
                "mathematically feasible, rather than being replaced by an internal model-selection "
                "shortcut."
            ),
            (
                "For graph inputs, where neighbor indices and distances are supplied by the caller, "
                "KODAMA keeps the KNN optimizer directly on the supplied graph and avoids rebuilding "
                "the neighbor search. If the PLS-LDA optimizer is requested without the original "
                "feature matrix, the graph is converted into a PLS-compatible float32 representation "
                "with the standard self-tuning normalized Laplacian transform: local row-wise distance "
                "scales define edge weights, the graph is symmetrized and degree-normalized, and "
                "randomized power iterations produce features for the same PLS-LDA core. This is the "
                "only supported graph-to-feature path in the public API."
            ),
            (
                "The implementation deliberately keeps classifier paths clean. KODAMA optimization "
                "is exposed as KNN or PLS-LDA; auxiliary cross-validation kernels exist for testing "
                "and benchmarking, but the KODAMA optimizer is not a collection of benchmark-specific "
                "branches. The graph, embedding, and clustering utilities are similarly separated "
                "from the label-evolution loop."
            ),
        ],
    ),
    (
        "4. Relationship to public KODAMA literature",
        [
            (
                "The contribution of kodama-cpp is not a new definition of KODAMA. It is a "
                "reorganization of the published method into a reusable numerical system. The "
                "2014 PNAS paper is the methodological reference for accuracy maximization. The "
                "2017 Bioinformatics paper is the software reference for the R package, including "
                "KNN and PLS-DA classifiers, repeated suboptimal runs, and construction of a "
                "dissimilarity matrix from the optimized label vectors. The R documentation is the "
                "public reference for defaults such as M = 100, Tcycle = 20, 75% sample selection, "
                "and the output fields dissimilarity, accuracy, proximity, classification vectors, "
                "scores, entropy, and landmarks."
            ),
            (
                "The main differences are architectural and computational. The R package owns a "
                "complete R workflow and uses R-package dependencies for data handling, nearest "
                "neighbors, and visualization. kodama-cpp owns the numerical kernels and exposes "
                "them through C++ types so that R and Python wrappers can share one backend. This "
                "makes the comparison with the previous version a compatibility question: the "
                "objective and output contract should match, while implementation details such as "
                "float32 storage, FAISS/cuVS search, CUDA kernels, and wrapper-independent graph "
                "objects are new."
            ),
        ],
    ),
    (
        "5. Evaluation",
        [
            (
                "We evaluate kodama-cpp at three levels. First, kernel benchmarks isolate KNNCV "
                "and PLSLDACV from the full KODAMA matrix pipeline so that nearest-neighbor search "
                "and PLS-LDA fitting can be interpreted separately. Second, core optimizer "
                "benchmarks measure repeated label-vector evolution over multiple seeds. Third, "
                "a matrix-level smoke benchmark measures an end-to-end KODAMA.matrix call and "
                "reports both cross-validated accuracy and external-label diagnostics."
            ),
            (
                "The evaluation is intentionally separated into these layers because KODAMA has "
                "several scaling regimes. KNN acceleration is dominated by neighbor search and "
                "voting. PLS-LDA acceleration is dominated by repeated SIMPLS and LDA work across "
                "folds and cycles. The final graph and visualization stages depend primarily on "
                "the sparse graph size. Reporting the layers separately makes it possible to "
                "attribute a speedup to the part of the implementation that produced it."
            ),
            (
                "The evaluation also separates optimization from validation. The label vector is "
                "chosen by internal cross-validation accuracy, but this score is not interpreted "
                "as external biological, chemical, or semantic truth. When reference labels are "
                "available, they are held outside the optimizer and reported after the fact. This "
                "distinction follows the broader cautionary literature on model selection, feature "
                "selection bias, and circular analysis."
            ),
            (
                "All timings in this section are wall-clock seconds from the benchmark drivers, "
                "and all accuracy values are computed on held-out cross-validation folds. CPU "
                "kernel measurements use the configured single-thread CPU path unless explicitly "
                "labeled otherwise; CUDA measurements use the same CUDA/FAISS runtime used by the "
                "test suite. The same protocol defines the larger wrapper-level comparison run, "
                "and the measurements below are directly produced by the current code and are not "
                "placeholders."
            ),
        ],
    ),
    (
        "6. Availability and reproducibility",
        [
            (
                "kodama-cpp is intended for release under the MIT License at "
                "https://github.com/tkcaccia/kodama-cpp. The recommended project split is a "
                "standalone C++/CUDA repository, an R wrapper repository, and a Python wrapper "
                "repository. This keeps wrapper code thin and makes the numerical library reusable "
                "outside either language ecosystem."
            ),
            (
                "Reproducibility is supported through CMake builds, CPU and CUDA tests, wrapper "
                "smoke tests, benchmark scripts, and explicit recording of backend parameters. "
                "The C++ core does not depend on R data readers; wrapper packages are responsible "
                "for translating host-language objects into contiguous matrices before calling the "
                "library."
            ),
        ],
    ),
    (
        "7. Limitations and future work",
        [
            (
                "The current implementation prioritizes KNN and PLS-LDA because they are central "
                "to the KODAMA optimization principle and can be implemented cleanly on both CPU "
                "and CUDA. Additional classifiers should be added only when they preserve the "
                "cross-validated accuracy objective and can be tested without changing the method "
                "for a specific benchmark."
            ),
            (
                "The main engineering limitation is that KODAMA performs many cross-validation "
                "fits. This makes memory locality, buffer reuse, label compaction, and GPU "
                "scheduling as important as asymptotic complexity. Future work will focus on "
                "persistent GPU-resident fold workspaces, better batching of independent M cycles, "
                "and broader continuous-integration coverage across Linux and macOS."
            ),
        ],
    ),
]


CONTRIBUTION_ROWS = [
    (
        "Formalization",
        "States KODAMA as cross-validated label predictability, with M independent runs and a derived graph or dissimilarity.",
    ),
    (
        "Standalone implementation",
        "Moves the numerical core into C++17/CUDA so R and Python wrappers can share one backend.",
    ),
    (
        "Classifier-specific kernels",
        "Provides clean KNN and PLS-LDA optimization paths instead of wrapper-level branching.",
    ),
    (
        "GPU-ready numerics",
        "Uses float32 internal buffers, CUDA nearest-neighbor search, and label-aware SIMPLS PLS-LDA.",
    ),
    (
        "Reproducible API",
        "Returns typed predictions, accuracies, labels, graphs, timings, memory, and backend parameters.",
    ),
]


API_ROWS = [
    (
        "Cross-validation",
        "KNNCV, PLSLDACV",
        "Fold assignment, prediction, fold accuracies, confusion matrices, timing, memory, and backend metadata.",
    ),
    (
        "Core optimization",
        "CoreKNN, CorePLSLDA",
        "Independent label-evolution runs that maximize KNN or PLS-LDA cross-validated accuracy.",
    ),
    (
        "KODAMA matrix",
        "KODAMA.matrix / KODAMAMatrix",
        "Landmark selection, splitting, M runs, optimized label vectors, KODAMA dissimilarity, and graph output.",
    ),
    (
        "Graph-input KODAMA",
        "KODAMA.matrix.graph / KODAMAMatrixFromGraph",
        "Runs the same KNN or PLS-LDA KODAMA optimizer from supplied neighbor indices/distances; KNN reuses the graph directly and PLS-LDA uses self-tuning Laplacian features when the data matrix is unavailable.",
    ),
    (
        "Visualization",
        "KODAMA.visualization",
        "UMAP and openTSNE embeddings from standard or KODAMA-corrected neighbor graphs.",
    ),
    (
        "Graph and clustering",
        "makeSNNGraph equivalent, Louvain, Leiden, random walk",
        "Neighbor graph construction and clustering by resolution or target cluster count.",
    ),
]


COMPATIBILITY_ROWS = [
    (
        "Objective",
        "KODAMA maximizes cross-validated accuracy of an evolving label vector.",
        "Same objective; exposed through CoreKNN and CorePLSLDA.",
    ),
    (
        "Independent runs",
        "R documentation reports M iterative processes and Tcycle optimization cycles.",
        "KODAMAMatrixOptions exposes runs and cycles; runs are independent for CPU/CUDA scheduling.",
    ),
    (
        "Sample selection",
        "The documented FUN_SAM default selects 75% of samples per iterative process.",
        "Landmark and splitting controls preserve the same sampling role while exposing it in typed options.",
    ),
    (
        "Classifiers",
        "The R package documents KNN and PLS-DA classifiers.",
        "The KODAMA optimizer exposes KNN and SIMPLS + PLS-LDA; component count is explicit.",
    ),
    (
        "Output contract",
        "The R interface returns dissimilarity, accuracy, proximity, label vectors, scores, entropy, and landmarks.",
        "The C++ result returns optimized labels, accuracies, graphs, timings, memory, and backend metadata for wrappers.",
    ),
    (
        "Architecture",
        "The R package is built around R workflows and R dependencies.",
        "The core is R/Python independent, CMake-installable, and accelerated where available.",
    ),
]


RELATED_WORK_PARAGRAPHS = [
    (
        "KODAMA is related to, but distinct from, semi-supervised learning and weak supervision. "
        "Classical semi-supervised graph methods assume that at least some labels are observed and "
        "then propagate them under smoothness assumptions, as in Gaussian-field label propagation "
        "and local-global consistency. Pseudo-labeling and weak-supervision systems also create or "
        "aggregate imperfect labels, but they typically use those labels to train a downstream "
        "predictor. KODAMA reverses the emphasis: the label vector itself is the optimization "
        "object, and its quality is the ability of a chosen classifier to reproduce it under "
        "held-out folds."
    ),
    (
        "The closest evaluation literature is cluster validation by stability or prediction. "
        "Methods such as stability-based clustering and prediction strength ask whether a grouping "
        "is reproducible under perturbation or held-out prediction. KODAMA uses the same general "
        "principle of reproducibility, but internalizes it as the search objective over label "
        "vectors. This is why the benchmark reports the internal CV score together with adjusted "
        "Rand index, silhouette, local purity, and active class count rather than presenting CV "
        "accuracy alone as proof of external correctness."
    ),
    (
        "The manuscript also follows warnings from the model-selection and circular-analysis "
        "literature. Cross-validation can be biased when the same data are used both to choose a "
        "model and to estimate final performance; feature selection outside the resampling loop is "
        "a classic example. In kodama-cpp, the internal CV score is intentionally the search "
        "criterion, so external claims must come from separate diagnostics, transparent parameter "
        "sensitivity, and comparisons with standard UMAP and t-SNE on the same datasets."
    ),
]


RELATED_WORK_ROWS = [
    (
        "Semi-supervised graph learning",
        "Label propagation and local-global consistency start from observed labels and smooth them on a graph.",
        "KODAMA starts from synthetic or supplied candidate labels and optimizes their cross-validated predictability.",
    ),
    (
        "Pseudo-labeling and weak supervision",
        "Pseudo-labeling and data programming create imperfect labels to train predictive models.",
        "KODAMA treats the label vector as the object being discovered; the classifier is the measuring instrument.",
    ),
    (
        "Cluster stability and prediction strength",
        "Stability methods evaluate whether clusters survive perturbation or prediction on held-out samples.",
        "KODAMA turns held-out predictability into an optimization objective, then reports external diagnostics separately.",
    ),
    (
        "Manifold visualization",
        "UMAP and t-SNE optimize low-dimensional layouts from neighbor or affinity structures.",
        "KODAMA supplies a corrected graph to the same visualization step; it does not replace those embedding objectives.",
    ),
    (
        "Open-source ML software",
        "JMLR software papers emphasize clear APIs, reproducibility, tests, licensing, and benchmark transparency.",
        "kodama-cpp separates the numerical core from R/Python wrappers and records backend parameters and validation evidence.",
    ),
]


VALIDATION_ROWS = [
    (
        "Correctness",
        "CTest passed on the CUDA build; tests cover constrained folds, prediction sizes, confusion matrices, requested component counts, and CPU/CUDA parity checks.",
    ),
    (
        "Numerics",
        "Float32 MatrixView smoke tests and CPU/CUDA tests exercise KNN and PLS-LDA paths with typed backend metadata.",
    ),
    (
        "Performance",
        "Kernel-level KNNCV/PLSLDACV timings are separated from core optimizer and KODAMA.matrix timings in the benchmark outputs.",
    ),
    (
        "Release",
        "CMake install targets, wrapper build scripts, benchmark drivers, and third-party license notices are present; archival DOI and coverage report are release tasks.",
    ),
]


INSTALLATION_ROWS = [
    (
        "C++ CPU",
        "cmake -S kodama-cpp -B build -DKODAMA_ENABLE_CUDA=OFF; cmake --build build -j; cmake --install build --prefix <prefix>",
    ),
    (
        "C++ CUDA",
        "export CONDA_PREFIX=<cuda-faiss-env>; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH; cmake -S kodama-cpp -B build-cuda -DKODAMA_ENABLE_CUDA=ON; cmake --build build-cuda -j",
    ),
    (
        "R wrapper",
        "KODAMA_CPP_ROOT=<kodama-cpp> KODAMA_CPP_BUILD_DIR=<build-or-build-cuda> R CMD INSTALL kodama-r",
    ),
    (
        "Python wrapper",
        "python -m pip install kodama-python --config-settings=cmake.define.KODAMA_CPP_ROOT=<kodama-cpp> --config-settings=cmake.define.KODAMA_CPP_BUILD_DIR=<build-or-build-cuda>",
    ),
    (
        "Runtime check",
        "Run ctest for the core library; in R call KODAMA.diagnostics(); in Python import kodama and run the package smoke tests.",
    ),
]


LICENSE_DEPENDENCY_ROWS = [
    ("kodama-cpp", "MIT license intended for the standalone core release."),
    ("KODAMA and fastPLS-derived code", "Authored by the same group; release terms can be aligned with the MIT core before publication."),
    ("FAISS / HNSW", "Nearest-neighbor dependency used for CPU and CUDA KNN search; benchmark metadata records index parameters."),
    ("CUDA / cuVS runtime", "Optional acceleration backend; CPU builds remain available without CUDA."),
    ("R and Python wrappers", "Thin language bindings in separate repositories, released under licenses compatible with the core."),
]


WRAPPER_VALIDATION_ROWS = [
    ("C++ core, local CPU", "cmake --build build -j and ctest passed 2/2 tests on macOS."),
    ("C++ core, CUDA", "chiamaka CUDA build succeeded and ctest passed 2/2 configured tests."),
    ("R wrapper, local CPU", "R CMD build followed by R CMD check --as-cran --no-manual --no-vignettes passed on the source tarball with only the expected new-submission NOTE under LC_ALL=C."),
    ("Python wrapper, local CPU", "Temporary virtual-environment install against the local CPU build passed pytest: 4/4 tests."),
]


TCYCLE_SENSITIVITY_ROWS = [
    (
        "Sensitivity fixture 1",
        "KNN",
        "0.900 -> 0.906",
        "0.844 -> 0.897",
        "4 -> 3",
        "23.4 -> 23.7",
    ),
    (
        "Sensitivity fixture 1",
        "PLS-LDA",
        "0.898 -> 0.911",
        "0.683 -> 0.838",
        "9 -> 4",
        "113.7 -> 527.9",
    ),
    (
        "Sensitivity fixture 2",
        "KNN",
        "0.773 -> 0.863",
        "0.701 -> 0.809",
        "23 -> 9",
        "47.0 -> 51.7",
    ),
    (
        "Sensitivity fixture 2",
        "PLS-LDA",
        "0.482 -> 0.872",
        "0.465 -> 0.829",
        "67 -> 5",
        "143.9 -> 596.7",
    ),
]


M_SENSITIVITY_ROWS = [
    (
        "Sensitivity fixture 1",
        "KNN",
        "0.906 / 0.905 / 0.906",
        "0.897 / 0.897 / 0.897",
        "0.385 / 0.382 / 0.385",
        "5.3 / 12.2 / 23.7",
    ),
    (
        "Sensitivity fixture 1",
        "PLS-LDA",
        "0.907 / 0.910 / 0.911",
        "0.895 / 0.845 / 0.838",
        "0.377 / 0.378 / 0.381",
        "102.1 / 265.9 / 527.9",
    ),
    (
        "Sensitivity fixture 2",
        "KNN",
        "0.832 / 0.863 / 0.863",
        "0.809 / 0.809 / 0.809",
        "0.413 / 0.352 / 0.352",
        "15.7 / 26.2 / 51.7",
    ),
    (
        "Sensitivity fixture 2",
        "PLS-LDA",
        "0.871 / 0.872 / 0.872",
        "0.812 / 0.815 / 0.829",
        "0.268 / 0.319 / 0.319",
        "116.3 / 301.0 / 596.7",
    ),
]


PARAMETER_SENSITIVITY_PARAGRAPHS = [
    (
        "Because KODAMA has two nested stochastic controls, M and Tcycle should not be justified by visual quality alone. Tcycle controls the depth of label-vector evolution inside one run, whereas M controls how many independent local optima enter the final ensemble. We therefore ran a cross-shaped CUDA sensitivity experiment on two representative datasets and both KODAMA classifiers: fixing M = 100 while varying Tcycle in {20, 50, 100}, and fixing Tcycle = 100 while varying M in {20, 50, 100}. The experiment used landmarks = 100000, splitting = 100, ncomp = 50, and knn.k = 30."
    ),
    (
        "The Tcycle sweep supports Tcycle = 100 as the final-analysis setting. On the harder benchmark, the PLS-LDA path changed from a fragmented state at Tcycle = 20, with median 67 classes and best CV accuracy 0.482, to median 5 classes and best CV accuracy 0.872 at Tcycle = 100. The KNN path also improved on the same benchmark, from best CV accuracy 0.773 to 0.863 and from median 23 classes to 9. The easier benchmark showed smaller best-accuracy changes, but still showed improved median accuracy and reduced fragmentation for PLS-LDA."
    ),
    (
        "The M sweep gives a different interpretation. Increasing M mostly improves the probability of seeing a good independent solution and stabilizes the ensemble; it is not expected to monotonically improve every single-run diagnostic. In these experiments, M = 50 was already close to M = 100 for several best-accuracy values, but M = 100 either tied or improved the best CV accuracy and provided the conservative ensemble size used for final figures. Smaller M remains appropriate for interactive exploration, while M = 100 is the reproducible benchmark setting."
    ),
]


PILOT_EXPERIMENT_PARAGRAPHS = [
    (
        "The benchmark run used copied float32 RData datasets spanning n = 873 to 5,220,347 samples and p = 11 to 16,384 variables. The CUDA machine first rebuilt the project and reran the CTest suite; both configured tests passed. The current table is deliberately kernel-focused: it is meant to validate implementation claims and identify scaling regimes before the wrapper-level release benchmark is frozen."
    ),
    (
        "The kernel-level experiment isolates KNNCV and PLSLDACV from the full KODAMA matrix pipeline. CUDA KNN matched CPU accuracy within about 0.001 on COIL20, MNIST, USPS, and mass41 while giving 3.8x to 99.4x speedups once the dataset was large enough to amortize GPU overhead. On the small MetRef KNN task, CPU remained faster because setup overhead dominated. CUDA PLSLDACV matched CPU exactly on MetRef and was 27.2x faster; on USPS it was 23.8x faster with a small positive accuracy difference."
    ),
    (
        "The KODAMA.matrix pilot on MetRef shows why cross-validated accuracy is reported together with label-quality diagnostics. The KNN optimizer reached very high CV accuracy, but the best label vectors collapsed to few classes and had low ARI. The PLS-LDA optimizer had lower CV accuracy but much stronger external-label agreement. We therefore report CV accuracy, number of active classes, ARI or another label-agreement statistic when available, and the downstream embedding compactness as complementary diagnostics."
    ),
    (
        "The wrapper smoke test also identified a packaging requirement: the conda libstdc++ runtime used by the CUDA/FAISS environment must be visible before R starts. The R wrapper installation notes describe the required environment variables, and the configure script links the conda libstdc++ runtime when CONDA_PREFIX is set."
    ),
]


BENCHMARK_PROTOCOL_ROWS = [
    ("Date", "2026-07-06 UTC"),
    ("GPU", "NVIDIA GeForce RTX 5060 Ti, 16 GB device memory, driver 595.71.05"),
    ("Build validation", "CUDA build succeeded; CTest passed 2/2 configured tests in 1.95 s"),
    ("Runtime", "CUDA/FAISS/cuVS conda runtime used by the benchmark and test suite"),
    ("Data format", "RData lists exported to contiguous float32 row-major matrices"),
    ("CPU setting", "Single-thread CPU kernels unless otherwise stated"),
    ("CV kernels", "KNNCV and PLSLDACV measured independently from the full matrix pipeline"),
    ("Core optimizer", "Three seeds per dataset for CoreKNN and CorePLSLDA medians"),
    ("Reported metrics", "Wall time, peak memory where available, CV accuracy, ARI, and active class count"),
]


PILOT_DATASET_ROWS = [
    ("COIL20", "1,440", "16,384", "20"),
    ("FashionMNIST", "70,000", "784", "10"),
    ("FlowRepository", "5,220,347", "32", "13"),
    ("MNIST", "70,000", "784", "10"),
    ("Macosko2015_retina", "44,808", "50", "12"),
    ("MetRef", "873", "375", "22"),
    ("TabulaMuris", "70,118", "50", "56"),
    ("USPS", "11,000", "256", "10"),
    ("flow18", "1,000,021", "11", "16"),
    ("imagenet", "1,281,167", "1,024", "1,000"),
    ("mass41", "965,282", "14", "17"),
]


EVALUATION_GUARDRAIL_ROWS = [
    (
        "Circularity",
        "CV accuracy is the internal objective, not an external truth score.",
        "Report ARI, silhouette, local purity, active classes, and standard embedding baselines after optimization.",
    ),
    (
        "Overclaiming",
        "Present kodama-cpp as a KODAMA implementation and acceleration library, not a new universal learning paradigm.",
        "State methodological continuity with the 2014 and 2017 KODAMA publications and limit novelty claims to architecture and numerics.",
    ),
    (
        "Parameter dependence",
        "M, Tcycle, landmarks, splitting, KNN k, graph k, and PLS components can affect results.",
        "Use predefined sensitivity experiments and report parameter settings in every benchmark table and plot.",
    ),
    (
        "Visualization bias",
        "A visually appealing layout is not itself evidence of label quality.",
        "Run classic UMAP/openTSNE and KODAMA-corrected UMAP/openTSNE with the same implementation and record compactness metrics.",
    ),
    (
        "Runtime attribution",
        "End-to-end speedups can hide which kernel produced the benefit.",
        "Separate CV kernels, core optimizer, KODAMA.matrix, graph construction, and visualization timings.",
    ),
    (
        "Reproducibility",
        "Wrapper behavior must match the standalone core.",
        "Record CMake settings, backend metadata, R CMD check, pytest, CTest, seeds, and benchmark command lines.",
    ),
]


ABLATION_MATRIX_ROWS = [
    (
        "Classifier",
        "KNN versus PLS-LDA",
        "Tests whether local-neighbor and latent-linear predictability give complementary label vectors.",
    ),
    (
        "Graph correction",
        "Standard graph versus KODAMA-corrected graph",
        "Isolates whether the optimized label ensemble improves downstream UMAP/openTSNE structure.",
    ),
    (
        "Search depth",
        "Tcycle = 20, 50, 100",
        "Measures whether additional proposal cycles improve CV accuracy or reduce fragmentation.",
    ),
    (
        "Ensemble size",
        "M = 20, 50, 100",
        "Measures whether more independent runs stabilize the final graph and best-run diagnostics.",
    ),
    (
        "Landmark and splitting controls",
        "Predeclared values, not per-dataset visual tuning",
        "Checks sensitivity of the initial label space and directly optimized sample subset.",
    ),
    (
        "Backend",
        "Single-core CPU, 4-core MultiCPU, and CUDA where available",
        "Separates mathematical equivalence from scheduling and hardware acceleration.",
    ),
    (
        "Wrapper parity",
        "C++ core, R wrapper, Python wrapper",
        "Ensures language bindings are thin and do not introduce independent numerical behavior.",
    ),
]


PILOT_CV_ROWS = [
    ("COIL20", "KNNCV", "5.455", "1.435", "3.8x", "0.917/0.916", "-0.000695", "-"),
    ("MNIST", "KNNCV", "76.769", "4.753", "16.2x", "0.974/0.973", "-0.000928", "-"),
    ("MetRef", "KNNCV", "0.163", "0.297", "0.6x", "0.821/0.814", "-0.006873", "-"),
    ("MetRef", "PLSLDACV", "3.196", "0.118", "27.2x", "0.992/0.992", "0.000000", "50"),
    ("USPS", "KNNCV", "2.273", "0.429", "5.3x", "0.688/0.688", "-0.000819", "-"),
    ("USPS", "PLSLDACV", "3.760", "0.158", "23.8x", "0.684/0.687", "+0.002727", "50"),
    ("mass41", "KNNCV", "366.883", "3.692", "99.4x", "0.912/0.912", "-0.000066", "-"),
]


PILOT_CORE_ROWS = [
    ("MetRef", "CoreKNN", "0.147", "0.122", "1.2x", "0.963/0.947", "0.271/0.313", "16/16"),
    ("MetRef", "CorePLSLDA", "16.597", "0.626", "26.5x", "0.885/0.838", "0.357/0.366", "53/48"),
    ("USPS", "CoreKNN", "2.282", "0.242", "9.4x", "0.969/0.968", "0.216/0.240", "59/53"),
    ("USPS", "CorePLSLDA", "20.189", "1.950", "10.4x", "0.938/0.910", "0.140/0.146", "80/77"),
]


KODAMA_BACKEND_COMPARISON_PARAGRAPHS = [
    (
        "The final benchmark table for KODAMA should compare the full matrix pipeline under "
        "a multicore CPU backend and a CUDA backend, not only the isolated CV kernels. This is "
        "important because KODAMA.matrix includes landmark selection, M independent label-search "
        "runs, final graph construction, and KODAMA graph correction. The comparison must "
        "therefore report both runtime and label quality for the same M, Tcycle, splitting, "
        "landmarks, classifier, and random seed."
    ),
    (
        "The current pilot evidence below measures the KODAMA core optimizer on the CUDA "
        "workstation and shows the speed range expected before final graph construction. The "
        "full-matrix driver tools/run_multicpu_vs_gpu_kodama_matrix.R records the same comparison "
        "end to end with n.cores stored explicitly. In this benchmark definition, MultiCPU rows "
        "use backend = cpu and n.cores = 4, while GPU rows use backend = cuda and the automatic "
        "CUDA worker scheduler."
    ),
]


KODAMA_BACKEND_COMPARISON_ROWS = [
    ("MetRef", "KNN", "CPU", "0.147", "0.963", "0.271", "16", "core optimizer"),
    ("MetRef", "KNN", "CUDA", "0.122", "0.947", "0.313", "16", "core optimizer"),
    ("MetRef", "PLS-LDA", "CPU", "16.597", "0.885", "0.357", "53", "core optimizer"),
    ("MetRef", "PLS-LDA", "CUDA", "0.626", "0.838", "0.366", "48", "core optimizer"),
    ("USPS", "KNN", "CPU", "2.282", "0.969", "0.216", "59", "core optimizer"),
    ("USPS", "KNN", "CUDA", "0.242", "0.968", "0.240", "53", "core optimizer"),
    ("USPS", "PLS-LDA", "CPU", "20.189", "0.938", "0.140", "80", "core optimizer"),
    ("USPS", "PLS-LDA", "CUDA", "1.950", "0.910", "0.146", "77", "core optimizer"),
]


PILOT_MATRIX_ROWS = [
    ("KNN", "6.849", "3.019", "0.990840/0.979389", "0.123581/0.046694", "3", "2-28"),
    ("PLS-LDA", "23.795", "23.748", "0.975573/0.946565", "0.795916/0.681493", "35", "21-42"),
]


VISUALIZATION_COMPARISON_PARAGRAPHS = [
    (
        "Because the final KODAMA object is often interpreted through a two-dimensional layout, "
        "we added an explicit visualization comparison against standard UMAP and openTSNE. The "
        "driver runs the same embedding implementation on either the ordinary neighbor graph or "
        "the KODAMA-corrected graph, then reports truth-label silhouette, local label purity, "
        "KODAMA label ARI, active classes, and wall-clock time. This makes the visualization "
        "comparison separate from the CV objective and prevents the evaluation from relying on "
        "a single favorable small dataset."
    ),
    (
        "The first local CPU preview used KNN KODAMA with M = 4 and Tcycle = 20 on MetRef, "
        "COIL20, and a 3,000-sample USPS subset. It is a harness validation rather than the "
        "final CUDA benchmark: KODAMA KNN achieved high CV accuracy on all three datasets, but "
        "the KODAMA-corrected layouts did not uniformly improve truth-label silhouette at these "
        "short settings. This result supports the manuscript policy of reporting CV accuracy, "
        "label agreement, active classes, and classic-vs-KODAMA visualization metrics together."
    ),
]


VISUALIZATION_COMPARISON_ROWS = [
    ("MetRef", "UMAP", "0.051", "-0.157", "0.533/0.439", "0.052", "3", "0.916"),
    ("MetRef", "openTSNE", "0.002", "-0.028", "0.579/0.499", "0.052", "3", "0.916"),
    ("COIL20", "UMAP", "0.494", "0.229", "0.824/0.764", "0.461", "20", "59.469"),
    ("COIL20", "openTSNE", "0.268", "0.168", "0.803/0.780", "0.461", "20", "59.469"),
    ("USPS subset", "UMAP", "0.189", "0.011", "0.660/0.573", "0.239", "23", "23.741"),
    ("USPS subset", "openTSNE", "0.148", "0.040", "0.662/0.609", "0.239", "23", "23.741"),
]


KODAMA_PARAMETER_ROWS = [
    (
        "M / runs",
        "Number of independent KODAMA runs. KODAMAMatrixOptions::runs defaults to 100. Run r uses seed seed + r and writes one length-n label vector to the result matrix.",
    ),
    (
        "Tcycle / cycles",
        "Number of proposal/evaluation cycles inside each run. KODAMAMatrixOptions::cycles defaults to 20 for compatibility with the historical interface; the benchmark-quality setting used in this manuscript is 100. CoreOptions accepts the same role when the core optimizer is called directly.",
    ),
    (
        "landmarks",
        "Maximum number of samples optimized directly in one run. The default is 10000; if the requested value is at least n, the implementation uses ceil(0.75 n), then caps the value to [2, n-1].",
    ),
    (
        "splitting",
        "Initial number of label classes for each run. If not supplied, splitting is 100 for n < 40000 and 300 otherwise. The run initializes labels by k-means with min(splitting, number of landmarks) clusters.",
    ),
    (
        "constrain",
        "Optional group vector. Empty means one movable group per sample. Non-empty constraints keep all members of a group in the same CV fold and make the group the unit of label proposal.",
    ),
    (
        "fixed",
        "Optional binary vector. Fixed samples are included in CV but are not relabeled by proposal moves.",
    ),
    (
        "graph_neighbors",
        "Number of neighbors retained in the final sparse graph. The actual k is min(graph_neighbors, landmarks, floor(0.75 n - 1)) after enforcing k >= 1.",
    ),
]


LABEL_SEARCH_DETAIL_PARAGRAPHS = [
    (
        "The object optimized by KODAMA is the label vector itself. At any time the algorithm holds a candidate vector y on the landmark samples. A classifier is not used to predict an external truth; instead it is trained to reproduce y under cross-validation. A label vector is therefore good when the labels assigned to held-out samples are predictable from the measured variables. The raw accuracy attached to a candidate, acc(y), is the fraction of landmark samples for which the held-out prediction pred_i(y) equals the candidate label y_i."
    ),
    (
        "The first CV pass is run on the initial label vector. This gives both an initial accuracy and a prediction vector. The prediction vector is then used as a proposal guide. If a sample or constrained group repeatedly receives a different predicted label from the classifier, that predicted label is evidence that the current label vector is not aligned with the structure that the classifier can reproduce. KODAMA uses this information to propose a new label vector rather than drawing completely blind random swaps."
    ),
    (
        "Within a cycle, the algorithm chooses a proposal base. In evolutionary mode this is the current accepted vector; otherwise it is the best vector found so far. It samples a number of movable groups according to the adaptive proposal-size schedule. Early cycles can alter many groups, allowing coarse movement through label space. Later cycles alter fewer groups, turning the search into local refinement. For each sampled group, the replacement label is drawn from the empirical distribution of the previous CV predictions among the non-fixed members of that group."
    ),
    (
        "The class-transition step aggregates the same information at label level. It builds a transition matrix N_ab counting samples whose current label is a and whose held-out prediction is b. Large off-diagonal counts indicate that the classifier consistently maps one candidate class into another. KODAMA may therefore propose merging unstable source labels into predicted destination labels, while rejecting moves that collapse the solution to a single class or do not reduce unstable fragmentation."
    ),
    (
        "After these proposal moves, the proposed vector is evaluated by exactly one new CV pass. The resulting raw accuracy is stored for the best accepted vector, while the acceptance score may include generic guards against degenerate labelings. In the current implementation, guarded diversity multiplies accuracy by sqrt(1 - sum_k p_k^2), where p_k is the class proportion. Optional coarsening subtracts a parsimony term when fragmentation remains high. These score terms steer the search away from trivial or excessively fragmented vectors without changing the underlying CV classifier."
    ),
    (
        "The best vector is updated whenever the proposal score improves. In evolutionary mode, a separate current vector is also maintained. A worse proposal can become the current vector with probability exp((score_new - score_current) / tau_t), where tau_t cools toward zero as cycles progress and is also smaller when current accuracy is already high. This simulated-annealing-style rule permits early exploration but makes late-cycle changes conservative. Repeating this process for M independent runs gives an ensemble of high-accuracy label vectors rather than relying on a single local optimum."
    ),
]


SCORE_INTERPRETATION_PARAGRAPHS = [
    (
        "It is useful to distinguish three quantities that appear in the implementation. The first is the raw held-out CV accuracy A(y), which is the KODAMA objective reported for a label vector. The second is the proposal acceptance score S(y), which may multiply or penalize A(y) to avoid degenerate label vectors during stochastic search. The third is an external diagnostic, such as ARI or embedding compactness, which is never optimized by KODAMA but is reported when reference labels are available."
    ),
    (
        "This distinction is necessary because a high raw CV accuracy alone can be achieved by a collapsed or overly coarse label vector. Such a vector may be easy for the classifier to reproduce but poor as a representation of latent structure. The guarded diversity factor and class-coarsening penalties are therefore search guards: they change which proposals are accepted, not the definition of the held-out classifier or the reported raw CV accuracy."
    ),
    (
        "The same distinction also clarifies how to compare KNN and PLS-LDA. KNN can produce very stable local partitions with high raw accuracy, whereas PLS-LDA can favor labelings that are predictable in a low-dimensional discriminant space. For that reason, the benchmark reports both raw CV accuracy and label-quality diagnostics, rather than treating any single scalar as a complete measure of KODAMA quality."
    ),
]


REPRODUCIBLE_ALGORITHM_STEPS = [
    (
        "For each run r = 1,...,M, create a random generator with seed seed + r. Copy the input matrix into float32 working storage. Select up to landmarks representative samples by running k-means on all rows with landmarks centers for 10 iterations, then sampling one representative from each non-empty cluster."
    ),
    (
        "Initialize the label vector on the selected landmarks. If starting labels are supplied, subset them to the landmarks and apply constrained-majority relabeling when constraints are present. Otherwise run k-means on the landmark matrix with init_k = max(2, min(splitting, number of landmarks)) for 10 iterations."
    ),
    (
        "Build CV folds on the landmark set. With no constrain vector, samples are assigned to folds directly. With a constrain vector, each group is assigned as a whole; stratified folds use the group-majority label and non-stratified folds shuffle groups. KODAMA.matrix currently sets the internal CV folds to non-stratified for both KNN and PLS-LDA core paths."
    ),
    (
        "At cycle t, choose the proposal base. In evolutionary-chain mode, the base label vector and prediction vector are the current accepted state; otherwise they are the current best state. These carried predictions drive the label proposal, and the proposed label vector is then evaluated by one CV pass."
    ),
    (
        "Sample proposal groups. Let G be the number of groups. With adaptive proposal size enabled, p_t = (t + 1) / (Tcycle + 1), s_t = p_t^2 (3 - 2 p_t), theta_t = 1 - s_t, and q_max(t) = 1 + floor((G - 1) theta_t). Draw q uniformly from {1,...,q_max(t)} and sample q groups without replacement. Without adaptive proposal size, q is uniform on {1,...,G}."
    ),
    (
        "For each sampled group, collect the non-fixed members. Their candidate replacement labels are the previous CV predictions for those members. The replacement label is drawn from the empirical frequency of those predicted labels, and all eligible members of the group are relabeled together."
    ),
    (
        "Apply class-level transition proposals. The transition matrix has entries N_ab = #{i: current label a, CV prediction b}. The many-to-one proposal selects targets with positive surplus over the independence expectation n_a n_b / n and relabels one or more unstable source classes into the selected target, while rejecting moves that collapse to one class or fail to reduce the class count. For PLS-LDA, an additional coarsening move may merge small unstable classes according to the same transition matrix."
    ),
    (
        "Evaluate the proposed labels with exactly one CV pass. Let A_t be the fraction of landmark samples whose proposed label equals the held-out CV prediction. The implementation records the raw accuracy attached to the selected best vector, while the default KODAMA.matrix acceptance score guards against degenerate labelings by multiplying A_t by sqrt(1 - sum_k p_k^2), where p_k is the current class proportion. When class coarsening is enabled, an additional parsimony term based on label entropy and effective class count is subtracted."
    ),
    (
        "Update the best vector whenever the proposal score improves. In evolutionary-chain mode, also update the current vector if the score improves the current score, or with probability exp((score - current_score) / tau_t), where tau_t = max(1e-9, 0.10 max(0, 1 - current_accuracy) (1 - (t + 1) / Tcycle))."
    ),
    (
        "After Tcycle cycles, project the optimized landmark labels back to all samples. For the KNN path, each non-landmark receives the majority label among its labeled landmark neighbors in the global graph. For the PLS-LDA path, the final PLS-LDA model predicts the non-landmarks. If constraints are present, the final labels are replaced by the majority label inside each constraint group."
    ),
]


GRAPH_CONSTRUCTION_PARAGRAPHS = [
    (
        "The C++ implementation stores the final KODAMA representation as a sparse neighbor graph rather than materializing a dense n by n matrix by default. First, it constructs a global KNN graph on the original samples using the selected metric and graph_neighbors. This unmodified graph is returned as base_knn."
    ),
    (
        "Let c_i^(r) be the final label of sample i from run r. For each retained edge (i,j) with original distance d_ij, compute V_ij as the number of runs where both endpoint labels are nonzero, and S_ij as the number of those valid runs where the two labels agree. If V_ij = 0 or S_ij = 0, the corrected edge distance is set to infinity. Otherwise a_ij = S_ij / V_ij and d'_ij = (1 + d_ij) / a_ij^2. Each neighbor row is then sorted by d'_ij. The resulting graph is the KODAMA-corrected graph used by KODAMA.visualization."
    ),
    (
        "A dense dissimilarity can be derived from the same agreement statistic by evaluating the formula for all sample pairs. The library keeps the sparse graph form because downstream UMAP, openTSNE, and graph clustering only need local neighborhoods, and the sparse form is the scalable object for CPU and CUDA backends."
    ),
]


SINGLE_RUN_PSEUDOCODE_LINES = [
    "Algorithm 1. One independent KODAMA run",
    "",
    "Input:",
    "  X_L: landmark matrix for this run",
    "  y0: initial label vector on landmarks",
    "  groups: movable units, either samples or constrained groups",
    "  fixed: optional fixed-label mask",
    "  folds: CV fold assignment",
    "  F: classifier in {KNN, PLS-LDA}",
    "  Tcycle: proposal/evaluation cycles",
    "",
    "Output:",
    "  y_best: optimized labels for this run",
    "  pred_best: CV predictions attached to y_best",
    "  acc_best: raw CV accuracy of y_best",
    "",
    "pred0 <- CrossValidate(F, X_L, y0, folds)",
    "acc0 <- Accuracy(y0, pred0)",
    "score0 <- ObjectiveScore(y0, acc0)",
    "y_best <- y0; pred_best <- pred0; acc_best <- acc0; score_best <- score0",
    "y_current <- y0; pred_current <- pred0; acc_current <- acc0; score_current <- score0",
    "",
    "for t = 1,...,Tcycle do",
    "  if evolutionary_search then",
    "    y_base <- y_current; pred_base <- pred_current",
    "  else",
    "    y_base <- y_best; pred_base <- pred_best",
    "  end if",
    "",
    "  y_prop <- y_base",
    "  qmax <- ProposalBudget(t, Tcycle, number_of_groups)",
    "  q <- UniformInteger(1, qmax)",
    "  selected_groups <- SampleWithoutReplacement(groups, q)",
    "",
    "  for each group g in selected_groups do",
    "    E <- {i in members(g): fixed[i] != 1}",
    "    if E is empty then continue",
    "    replacement <- SampleEmpirical(pred_base[E])",
    "    y_prop[E] <- replacement",
    "  end for",
    "",
    "  y_prop <- ClassTransitionProposals(y_prop, pred_base, fixed)",
    "  pred_prop <- CrossValidate(F, X_L, y_prop, folds)  # exactly one CV pass",
    "  acc_prop <- Accuracy(y_prop, pred_prop)",
    "  score_prop <- ObjectiveScore(y_prop, acc_prop)",
    "",
    "  if score_prop > score_best then",
    "    y_best <- y_prop; pred_best <- pred_prop",
    "    acc_best <- acc_prop; score_best <- score_prop",
    "  end if",
    "",
    "  if evolutionary_search and",
    "     Accept(score_prop, score_current, acc_current, t, Tcycle, rng) then",
    "    y_current <- y_prop; pred_current <- pred_prop",
    "    acc_current <- acc_prop; score_current <- score_prop",
    "  end if",
    "",
    "  if acc_prop == 1 then break",
    "end for",
    "",
    "return y_best, pred_best, acc_best",
]


KODAMA_PSEUDOCODE_LINES = [
    "Algorithm 2. KODAMA.matrix ensemble",
    "",
    "Input:",
    "  X: n by p data matrix",
    "  F: classifier in {KNN, PLS-LDA}",
    "  M: number of independent runs",
    "  Tcycle: proposal/evaluation cycles per run",
    "  landmarks, splitting, graph_neighbors",
    "  optional starting_labels, constrain, fixed",
    "",
    "Output:",
    "  C: M by n matrix of optimized label vectors",
    "  A: length-M vector of best raw CV accuracies",
    "  G_K: KODAMA-corrected neighbor graph",
    "",
    "X32 <- float32(X)",
    "G0 <- KNNGraph(X32, graph_neighbors)",
    "for r = 1,...,M do",
    "  rng <- RNG(seed + r)",
    "  L <- SelectLandmarks(X32, landmarks, rng)",
    "  groups <- ConstrainedGroups(constrain[L])",
    "            or one singleton group per landmark",
    "  y0 <- InitialLabels(X32[L], starting_labels[L], splitting, groups, rng)",
    "  folds <- AssignCVFolds(groups, y0)",
    "",
    "  pred0 <- CrossValidate(F, X32[L], y0, folds)",
    "  acc0 <- Accuracy(y0, pred0)",
    "  score0 <- ObjectiveScore(y0, acc0)",
    "  y_best <- y0; pred_best <- pred0; acc_best <- acc0; score_best <- score0",
    "  y_current <- y0; pred_current <- pred0; acc_current <- acc0; score_current <- score0",
    "",
    "  for t = 1,...,Tcycle do",
    "    if evolutionary_search then",
    "      y_base <- y_current; pred_base <- pred_current",
    "    else",
    "      y_base <- y_best; pred_base <- pred_best",
    "    end if",
    "",
    "    y_prop <- y_base",
    "    qmax <- ProposalBudget(t, Tcycle, number_of_groups)",
    "    q <- UniformInteger(1, qmax)",
    "    for each group g in SampleWithoutReplacement(groups, q) do",
    "      E <- {i in members(g): fixed[i] != 1}",
    "      if E is not empty then",
    "        label <- SampleEmpirical(pred_base[E])",
    "        y_prop[E] <- label",
    "      end if",
    "    end for",
    "",
    "    y_prop <- ClassTransitionProposals(y_prop, pred_base, fixed)",
    "    pred_prop <- CrossValidate(F, X32[L], y_prop, folds)  # one CV pass",
    "    acc_prop <- Accuracy(y_prop, pred_prop)",
    "    score_prop <- ObjectiveScore(y_prop, acc_prop)",
    "",
    "    if score_prop > score_best then",
    "      y_best <- y_prop; pred_best <- pred_prop",
    "      acc_best <- acc_prop; score_best <- score_prop",
    "    end if",
    "",
    "    if evolutionary_search and",
    "       Accept(score_prop, score_current, acc_current, t, Tcycle, rng) then",
    "      y_current <- y_prop; pred_current <- pred_prop",
    "      acc_current <- acc_prop; score_current <- score_prop",
    "    end if",
    "",
    "    if acc_prop == 1 then break",
    "  end for",
    "",
    "  C[r, ] <- ProjectLabelsToAllSamples(y_best, X32, L, F, groups, G0)",
    "  A[r] <- acc_best",
    "end for",
    "",
    "G_K <- ReweightGraphByLabelAgreement(G0, C)",
    "return C, A, G_K",
    "",
    "ObjectiveScore(y, a):",
    "  score <- a",
    "  if guarded_diversity then",
    "    p_l <- class proportions in y",
    "    score <- score * sqrt(1 - sum_l p_l^2)",
    "  end if",
    "  if class_coarsening then",
    "    score <- score - (1 - a) * Parsimony(y)",
    "  end if",
    "  return score",
    "",
    "Accept(s_new, s_old, a_old, t, Tcycle, rng):",
    "  if s_new >= s_old then return TRUE",
    "  tau <- max(1e-9, 0.10 * max(0, 1 - a_old) * (1 - t / Tcycle))",
    "  return Uniform(0, 1, rng) < exp((s_new - s_old) / tau)",
]


GRAPH_INPUT_PSEUDOCODE_LINES = [
    "Algorithm 3. KODAMA.matrix.graph from supplied neighbors",
    "",
    "Input:",
    "  I, D: n by k neighbor indices and distances",
    "  optional X: original n by p data matrix",
    "  F: classifier in {KNN, PLS-LDA}",
    "  M, Tcycle, landmarks, splitting",
    "  optional starting_labels, constrain, fixed",
    "",
    "Output:",
    "  C: M by n matrix of optimized label vectors",
    "  A: length-M vector of best raw CV accuracies",
    "  G_K: KODAMA-corrected graph on the supplied neighborhoods",
    "",
    "G0 <- NormalizeExternalGraph(I, D)",
    "if X is supplied then",
    "  X_work <- float32(X)",
    "else if F == PLS-LDA then",
    "  X_work <- SelfTuningLaplacianFeatures(G0, ncomp)",
    "else",
    "  X_work <- empty",
    "end if",
    "",
    "for r = 1,...,M do",
    "  rng <- RNG(seed + r)",
    "  L <- SelectLandmarksFromGraphOrData(G0, X_work, landmarks, rng)",
    "  groups <- ConstrainedGroups(constrain[L])",
    "            or one singleton group per landmark",
    "  y0 <- InitialLabelsFromGraphOrData(G0[L], X_work[L], starting_labels[L],",
    "                                    splitting, groups, rng)",
    "  folds <- AssignCVFolds(groups, y0)",
    "",
    "  if F == KNN then",
    "    y_best, pred_best, acc_best <- Algorithm1(G0[L], y0, groups, fixed[L], folds, KNN, Tcycle)",
    "  else",
    "    y_best, pred_best, acc_best <- Algorithm1(X_work[L], y0, groups, fixed[L], folds, PLS-LDA, Tcycle)",
    "  end if",
    "",
    "  C[r, ] <- ProjectLabelsToAllSamples(y_best, X_work, L, F, groups, G0)",
    "  A[r] <- acc_best",
    "end for",
    "",
    "G_K <- ReweightGraphByLabelAgreement(G0, C)",
    "return C, A, G_K",
]


IMPLEMENTATION_EVIDENCE_ROWS = [
    (
        "Float32 internal buffers",
        "MatrixView float overload in include/kodama/kodama.hpp; float32 PLS-LDA and CUDA workspaces in src/plscv.cpp.",
        "tests/test_cv.cpp checks Float32 MatrixView, float32 KNNCV/PLSLDACV CPU and CUDA outputs, backend metadata, and accuracy thresholds.",
        "The pilot suite loaded copied float32 RData files and records CPU/CUDA runtime and accuracy for kernel and core paths.",
    ),
    (
        "High-recall CPU HNSW",
        "FAISS IndexHNSWFlat paths in src/knncv.cpp and src/core.cpp; hnsw_target_recall defaults to 0.99 and hnsw_tune_k to 50.",
        "tests/test_cv.cpp asserts FAISS HNSW default use, target recall 0.99, tune k 50, valid ef parameters, constrained folds, and accuracy.",
        "KNNCV/CoreKNN benchmark rows should compare HNSW with the previous R nearest-neighbor path and report recall/runtime tradeoffs.",
    ),
    (
        "CUDA nearest-neighbor search",
        "KNNCV_CUDA in src/knncv.cpp and precompute_knn_cv_cuda in src/core.cpp use FAISS GPU IVF-Flat with recorded nlist/nprobe.",
        "CUDA tests assert backend=CUDA, prediction and fold sizes, constrained folds, float32 dispatch, and accuracy for KNNCV/CoreKNN.",
        "Pilot KNNCV rows report 3.8x to 99.4x speedups on larger datasets with accuracy changes within about 0.001.",
    ),
    (
        "Label-aware SIMPLS PLS-LDA",
        "src/plscv.cpp uses label/class inputs in fit_pls_components_labels_float and fit_pls_components_cuda_labels_float; CUDA LDA uses label-sum kernels.",
        "tests/test_cv.cpp checks PLSLDACV CPU/CUDA sizes, constrained folds, selected component reporting, backend metadata, and accuracy thresholds.",
        "Pilot PLSLDACV rows report 27.2x speedup on MetRef with identical accuracy and 23.8x speedup on USPS.",
    ),
    (
        "Reusable fold and data buffers",
        "PLSFoldXCacheF in src/plscv.cpp caches fold assignments, train/validation indices, scaled matrices, and CUDA Gram buffers; CoreKNN precomputes fold neighbors.",
        "tests/test_cv.cpp checks CoreKNN/CorePLSLDA result sizes, worker/scheduler metadata, and that optimization does not decrease initial CV accuracy.",
        "Pilot CorePLSLDA rows report 26.5x speedup on MetRef and 10.4x speedup on USPS for the CUDA core path.",
    ),
    (
        "Graph-input KODAMA",
        "KODAMAMatrixFromGraph and KODAMAMatrixFromGraphData in src/kodama_matrix.cpp accept supplied neighbor indices/distances; the KNN path reuses the graph and the PLS-LDA graph-only path builds self-tuning Laplacian features.",
        "R and Python wrapper tests exercise matrix_graph with KNN and PLS-LDA, and C++ tests cover graph construction, clustering, and visualization outputs.",
        "Graph-input benchmarks compare graph-supplied KODAMA against data-input KODAMA using the same M and Tcycle settings before accepting the graph-input path as an optional API.",
    ),
]


NOVELTY_ROWS = [
    (
        "Standalone core",
        "Moves KODAMA computation from R orchestration into an R/Python-independent C++17/CUDA library.",
    ),
    (
        "Classifier-specific paths",
        "Keeps KNN and PLS-LDA implementations separate so each can reuse the right buffers and data structures.",
    ),
    (
        "Float32 execution",
        "Stores analysis matrices and GPU workspaces in single precision while accepting double inputs from wrappers.",
    ),
    (
        "Label-aware SIMPLS",
        "Computes PLS-LDA cross-products from labels directly and evaluates the requested feasible component count.",
    ),
    (
        "High-recall neighbor search",
        "Uses FAISS/cuVS and CPU HNSW infrastructure for fast KNN cross-validation and graph construction.",
    ),
    (
        "Independent M cycles",
        "Keeps KODAMA runs independent so CPU workers and CUDA lanes can be scheduled without changing the objective.",
    ),
]


REFERENCES = [
    (
        "Cacciatore, S., Luchinat, C., and Tenori, L. Knowledge discovery by accuracy maximization. "
        "Proceedings of the National Academy of Sciences, 111(14), 5117-5122, 2014."
    ),
    (
        "Cacciatore, S., Tenori, L., Luchinat, C., Bennett, P. R., and MacIntyre, D. A. "
        "KODAMA: an R package for knowledge discovery and data mining. Bioinformatics, "
        "33(4), 621-623, 2017."
    ),
    (
        "Cacciatore, S. and Tenori, L. KODAMA: Knowledge Discovery by Accuracy Maximization. "
        "R package documentation, CRAN."
    ),
    (
        "Chapelle, O., Scholkopf, B., and Zien, A. Semi-Supervised Learning. MIT Press, 2006."
    ),
    (
        "Zhu, X., Ghahramani, Z., and Lafferty, J. Semi-supervised learning using Gaussian fields "
        "and harmonic functions. Proceedings of ICML, 2003."
    ),
    (
        "Zhou, D., Bousquet, O., Lal, T. N., Weston, J., and Scholkopf, B. Learning with local "
        "and global consistency. Advances in Neural Information Processing Systems, 2003."
    ),
    (
        "Lee, D.-H. Pseudo-label: The simple and efficient semi-supervised learning method for "
        "deep neural networks. ICML Workshop on Challenges in Representation Learning, 2013."
    ),
    (
        "Ratner, A., De Sa, C., Wu, S., Selsam, D., and Re, C. Data programming: Creating large "
        "training sets, quickly. Advances in Neural Information Processing Systems, 2016."
    ),
    (
        "Ben-Hur, A., Elisseeff, A., and Guyon, I. A stability based method for discovering "
        "structure in clustered data. Pacific Symposium on Biocomputing, 2002."
    ),
    (
        "Tibshirani, R. and Walther, G. Cluster validation by prediction strength. Journal of "
        "Computational and Graphical Statistics, 14(3), 511-528, 2005."
    ),
    (
        "Hubert, L. and Arabie, P. Comparing partitions. Journal of Classification, 2, "
        "193-218, 1985."
    ),
    (
        "Rousseeuw, P. J. Silhouettes: a graphical aid to the interpretation and validation of "
        "cluster analysis. Journal of Computational and Applied Mathematics, 20, 53-65, 1987."
    ),
    (
        "Kohavi, R. A study of cross-validation and bootstrap for accuracy estimation and model "
        "selection. Proceedings of IJCAI, 1995."
    ),
    (
        "Ambroise, C. and McLachlan, G. J. Selection bias in gene extraction on the basis of "
        "microarray gene-expression data. Proceedings of the National Academy of Sciences, "
        "99(10), 6562-6566, 2002."
    ),
    (
        "Varma, S. and Simon, R. Bias in error estimation when using cross-validation for model "
        "selection. BMC Bioinformatics, 7, 91, 2006."
    ),
    (
        "Kriegeskorte, N., Simmons, W. K., Bellgowan, P. S. F., and Baker, C. I. Circular "
        "analysis in systems neuroscience: the dangers of double dipping. Nature Neuroscience, "
        "12, 535-540, 2009."
    ),
    (
        "de Jong, S. SIMPLS: an alternative approach to partial least squares regression. "
        "Chemometrics and Intelligent Laboratory Systems, 18(3), 251-263, 1993."
    ),
    (
        "Johnson, J., Douze, M., and Jegou, H. Billion-scale similarity search with GPUs. "
        "IEEE Transactions on Big Data, 2019."
    ),
    (
        "Malkov, Y. A. and Yashunin, D. A. Efficient and robust approximate nearest neighbor "
        "search using Hierarchical Navigable Small World graphs. IEEE Transactions on Pattern "
        "Analysis and Machine Intelligence, 2020."
    ),
    (
        "McInnes, L., Healy, J., and Melville, J. UMAP: Uniform Manifold Approximation and "
        "Projection for dimension reduction. arXiv:1802.03426, 2018."
    ),
    (
        "van der Maaten, L. and Hinton, G. Visualizing data using t-SNE. Journal of Machine "
        "Learning Research, 9, 2579-2605, 2008."
    ),
    (
        "Linderman, G. C., Rachh, M., Hoskins, J. G., Steinerberger, S., and Kluger, Y. "
        "Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. "
        "Nature Methods, 16, 243-245, 2019."
    ),
    (
        "Blondel, V. D., Guillaume, J.-L., Lambiotte, R., and Lefebvre, E. Fast unfolding "
        "of communities in large networks. Journal of Statistical Mechanics: Theory and "
        "Experiment, 2008."
    ),
    (
        "Traag, V. A., Waltman, L., and van Eck, N. J. From Louvain to Leiden: guaranteeing "
        "well-connected communities. Scientific Reports, 9, 5233, 2019."
    ),
    (
        "Sonnenburg, S., Braun, M. L., Ong, C. S., Bengio, S., Bottou, L., Holmes, G., LeCun, Y., "
        "Muller, K.-R., Pereira, F., Rasmussen, C. E., Ratsch, G., Scholkopf, B., Smola, A., "
        "Vincent, P., Weston, J., and Williamson, R. C. The need for open source software in "
        "machine learning. Journal of Machine Learning Research, 8, 2443-2466, 2007."
    ),
    (
        "Pineau, J., Vincent-Lamarre, P., Sinha, K., Lariviere, V., Beygelzimer, A., d'Alche-Buc, "
        "F., Fox, E., and Larochelle, H. Improving reproducibility in machine learning research. "
        "Journal of Machine Learning Research, 22(164), 1-20, 2021."
    ),
]


def set_run_font(run, size: float | None = None, bold: bool | None = None, color=None) -> None:
    run.font.name = TOKENS["font"]
    run._element.rPr.rFonts.set(qn("w:ascii"), TOKENS["font"])
    run._element.rPr.rFonts.set(qn("w:hAnsi"), TOKENS["font"])
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_paragraph_shading(paragraph, fill: str) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    shd = p_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        p_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, bottom=80, start=120, end=120) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in {"top": top, "bottom": bottom, "start": start, "end": end}.items():
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="D0D7DE", size="4") -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        node = borders.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), size)
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), color)


def set_table_width(table, widths_in) -> None:
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in table.rows:
        tr_pr = row._tr.get_or_add_trPr()
        if tr_pr.find(qn("w:cantSplit")) is None:
            tr_pr.append(OxmlElement("w:cantSplit"))
        for idx, width in enumerate(widths_in):
            cell = row.cells[idx]
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)


def apply_styles(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = TOKENS["font"]
    normal.font.size = Pt(TOKENS["body_size"])
    normal._element.rPr.rFonts.set(qn("w:ascii"), TOKENS["font"])
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), TOKENS["font"])
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.18

    for name, size, color, before, after in [
        ("Heading 1", TOKENS["h1_size"], TOKENS["heading_blue"], 16, 8),
        ("Heading 2", TOKENS["h2_size"], TOKENS["heading_blue"], 12, 6),
        ("Heading 3", TOKENS["h3_size"], TOKENS["heading_dark"], 8, 4),
    ]:
        style = styles[name]
        style.font.name = TOKENS["font"]
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style._element.rPr.rFonts.set(qn("w:ascii"), TOKENS["font"])
        style._element.rPr.rFonts.set(qn("w:hAnsi"), TOKENS["font"])
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True


def apply_compact_memo_styles(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.3)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.08
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[name]
        style.paragraph_format.space_before = Pt(8)
        style.paragraph_format.space_after = Pt(3)


def add_title(doc: Document, title: str, subtitle: str, authors: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(title)
    set_run_font(run, 22, True, TOKENS["title"])

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(subtitle)
    set_run_font(run, 11)
    run.italic = True

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(authors)
    set_run_font(run, 10, False, TOKENS["muted"])


def add_callout(doc: Document, label: str, text: str) -> None:
    table = doc.add_table(rows=1, cols=1)
    set_table_width(table, [6.45])
    set_table_borders(table, color=TOKENS["border"])
    cell = table.cell(0, 0)
    set_cell_shading(cell, TOKENS["callout_fill"])
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(label + ": ")
    set_run_font(run, 10.5, True, RGBColor(0x1F, 0x3A, 0x5F))
    run = p.add_run(text)
    set_run_font(run, 10.5)


def add_bullets(doc: Document, items) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(item)
        set_run_font(run)


def new_numbering_id(doc: Document, style_id: str = "ListNumber") -> int:
    numbering = doc.part.numbering_part.element
    abstract_num_id = None
    for abstract in numbering.findall(qn("w:abstractNum")):
        for p_style in abstract.iter(qn("w:pStyle")):
            if p_style.get(qn("w:val")) == style_id:
                abstract_num_id = abstract.get(qn("w:abstractNumId"))
                break
        if abstract_num_id is not None:
            break
    if abstract_num_id is None:
        abstract_num_id = "0"

    existing = [
        int(num.get(qn("w:numId")))
        for num in numbering.findall(qn("w:num"))
        if num.get(qn("w:numId")) is not None
    ]
    num_id = (max(existing) + 1) if existing else 1

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract = OxmlElement("w:abstractNumId")
    abstract.set(qn("w:val"), str(abstract_num_id))
    num.append(abstract)
    override = OxmlElement("w:lvlOverride")
    override.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:startOverride")
    start.set(qn("w:val"), "1")
    override.append(start)
    num.append(override)
    numbering.append(num)
    return num_id


def apply_numbering_id(paragraph, num_id: int) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    num_pr = p_pr.find(qn("w:numPr"))
    if num_pr is None:
        num_pr = OxmlElement("w:numPr")
        p_pr.append(num_pr)
    ilvl = num_pr.find(qn("w:ilvl"))
    if ilvl is None:
        ilvl = OxmlElement("w:ilvl")
        num_pr.append(ilvl)
    ilvl.set(qn("w:val"), "0")
    num_id_node = num_pr.find(qn("w:numId"))
    if num_id_node is None:
        num_id_node = OxmlElement("w:numId")
        num_pr.append(num_id_node)
    num_id_node.set(qn("w:val"), str(num_id))


def add_numbered(doc: Document, items) -> None:
    num_id = new_numbering_id(doc)
    for item in items:
        p = doc.add_paragraph(style="List Number")
        apply_numbering_id(p, num_id)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(item)
        set_run_font(run)


def add_pseudocode_block(doc: Document, lines) -> None:
    for idx, line in enumerate(lines):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.16)
        paragraph.paragraph_format.right_indent = Inches(0.16)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 0.88
        paragraph.paragraph_format.keep_together = False
        set_paragraph_shading(paragraph, "F7F9FC")
        run = paragraph.add_run(line if line else " ")
        run.font.name = "Courier New"
        run._element.rPr.rFonts.set(qn("w:ascii"), "Courier New")
        run._element.rPr.rFonts.set(qn("w:hAnsi"), "Courier New")
        run.font.size = Pt(7.2)
        if line.startswith("Algorithm "):
            run.font.bold = True
            run.font.color.rgb = TOKENS["title"]
        if idx == len(lines) - 1:
            paragraph.paragraph_format.space_after = Pt(6)


def format_table_text(table, font_size=8.5) -> None:
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.1
                for run in p.runs:
                    set_run_font(run, font_size)


def add_table(doc: Document, headers, rows, widths, font_size=8.5) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_borders(table)
    set_table_width(table, widths)
    for idx, text in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.text = text
        set_cell_shading(cell, TOKENS["table_fill"])
        for p in cell.paragraphs:
            for run in p.runs:
                set_run_font(run, font_size, True)
    for row in rows:
        cells = table.add_row().cells
        for idx, text in enumerate(row):
            cells[idx].text = text
    format_table_text(table, font_size=font_size)


def build_architecture_figure() -> None:
    if FIGMA_ARCH_FIGURE.exists():
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.open(FIGMA_ARCH_FIGURE).convert("RGB")
            if img.size == (1800, 1120):
                draw = ImageDraw.Draw(img)
                # Repair the exported Figma label where the central connector crosses the text.
                draw.rounded_rectangle(
                    (610, 662, 1210, 744),
                    radius=16,
                    fill=(248, 251, 255),
                    outline=(193, 211, 234),
                    width=2,
                )
                def load_font(candidates, size):
                    for candidate in candidates:
                        if Path(candidate).exists():
                            return ImageFont.truetype(candidate, size)
                    return ImageFont.load_default()

                regular_font_candidates = [
                    "/Library/Fonts/Arial.ttf",
                    "/System/Library/Fonts/Supplemental/Arial.ttf",
                    "/System/Library/Fonts/Supplemental/Helvetica.ttf",
                ]
                bold_font_candidates = [
                    "/Library/Fonts/Arial Bold.ttf",
                    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                    "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
                ]
                label_font = load_font(regular_font_candidates, 26)
                utility_title_font = load_font(bold_font_candidates, 28)
                utility_font = load_font(regular_font_candidates, 22)

                draw.rounded_rectangle(
                    (510, 760, 1290, 902),
                    radius=20,
                    fill=(244, 242, 255),
                    outline=(118, 101, 223),
                    width=3,
                )
                utility_title = "graph, embedding, and clustering utilities"
                utility_subtitle = "KNN/SNN graph  |  UMAP/openTSNE  |  Louvain/Leiden/random walk"
                for text, font_obj, y_pos in (
                    (utility_title, utility_title_font, 807),
                    (utility_subtitle, utility_font, 849),
                ):
                    bbox = draw.textbbox((0, 0), text, font=font_obj)
                    draw.text(
                        (900 - (bbox[2] - bbox[0]) / 2, y_pos - (bbox[3] - bbox[1]) / 2),
                        text,
                        font=font_obj,
                        fill=(11, 37, 69),
                    )
                label = "KNNCV + PLSLDACV kernels"
                bbox = draw.textbbox((0, 0), label, font=label_font)
                draw.text(
                    (900 - (bbox[2] - bbox[0]) / 2, 703 - (bbox[3] - bbox[1]) / 2),
                    label,
                    font=label_font,
                    fill=(11, 37, 69),
                )
            img.save(ARCH_FIGURE)
        except Exception:
            shutil.copyfile(FIGMA_ARCH_FIGURE, ARCH_FIGURE)
        return

    from PIL import Image, ImageDraw, ImageFont

    scale = 2
    width, height = 1800, 1120
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)

    def font(size, bold=False):
        candidates = [
            "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return ImageFont.truetype(candidate, size * scale)
        return ImageFont.load_default()

    title_font = font(42, True)
    box_font = font(30, True)
    small_font = font(24)
    tiny_font = font(21)

    def xy(coords):
        return tuple(int(v * scale) for v in coords)

    def centered_text(box, lines, fill, title=False, small=False):
        x0, y0, x1, y1 = xy(box)
        line_font = box_font if title else small_font if small else tiny_font
        rendered = []
        total_h = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=line_font)
            rendered.append((line, bbox[2] - bbox[0], bbox[3] - bbox[1]))
            total_h += bbox[3] - bbox[1]
        total_h += int(8 * scale) * (len(lines) - 1)
        y = y0 + ((y1 - y0) - total_h) // 2
        for line, tw, th in rendered:
            draw.text((x0 + ((x1 - x0) - tw) // 2, y), line, font=line_font, fill=fill)
            y += th + int(8 * scale)

    def box(coords, fill, outline, lines, text_fill=(20, 35, 55), radius=22, title=False, small=False):
        draw.rounded_rectangle(xy(coords), radius=radius * scale, fill=fill, outline=outline, width=3 * scale)
        centered_text(coords, lines, text_fill, title=title, small=small)

    def arrow(start, end, color=(85, 98, 115), width_px=5):
        sx, sy = start
        ex, ey = end
        draw.line(xy((sx, sy, ex, ey)), fill=color, width=width_px * scale)
        import math
        angle = math.atan2(ey - sy, ex - sx)
        length = 18
        spread = 0.42
        p1 = (ex - length * math.cos(angle - spread), ey - length * math.sin(angle - spread))
        p2 = (ex - length * math.cos(angle + spread), ey - length * math.sin(angle + spread))
        draw.polygon([xy((ex, ey))[0:2], xy(p1)[0:2], xy(p2)[0:2]], fill=color)

    # Palette chosen to keep the figure readable when printed in grayscale.
    navy = (11, 37, 69)
    blue = (232, 241, 252)
    blue_line = (91, 139, 190)
    green = (232, 246, 239)
    green_line = (84, 150, 111)
    amber = (252, 243, 224)
    amber_line = (196, 143, 60)
    gray = (246, 248, 251)
    gray_line = (180, 193, 208)

    draw.text(xy((70, 48)), "kodama-cpp architecture", font=title_font, fill=navy)
    draw.text(
        xy((70, 105)),
        "Standalone C++17/CUDA numerical core shared by R and Python wrappers",
        font=small_font,
        fill=(75, 85, 100),
    )

    box((185, 170, 595, 285), blue, blue_line, ["R wrapper", "thin language binding"], small=True)
    box((1205, 170, 1615, 285), blue, blue_line, ["Python wrapper", "thin language binding"], small=True)

    draw.rounded_rectangle(xy((585, 320, 1215, 560)), radius=22 * scale, fill=gray, outline=gray_line, width=3 * scale)
    core_title = "kodama-cpp C++17 core"
    core_subtitle = "MatrixView + typed options/results"
    for y, text, used_font, fill in [
        (346, core_title, box_font, navy),
        (382, core_subtitle, small_font, (75, 85, 100)),
    ]:
        bbox = draw.textbbox((0, 0), text, font=used_font)
        draw.text(xy((900 - ((bbox[2] - bbox[0]) / (2 * scale)), y)), text, font=used_font, fill=fill)
    box((655, 425, 905, 525), (255, 255, 255), gray_line, ["KODAMA.matrix", "M independent runs"], small=True)
    box((930, 425, 1145, 525), (255, 255, 255), gray_line, ["CoreKNN", "CorePLSLDA"], small=True)

    box((90, 385, 485, 650), green, green_line, ["CPU backend", "FAISS HNSW KNN", "SIMPLS PLS-LDA", "OpenMP work sharing"], small=True)
    box((1315, 385, 1710, 650), amber, amber_line, ["CUDA backend", "FAISS/cuVS KNN", "label-aware SIMPLS", "GPU workspaces"], small=True)

    box((585, 690, 1215, 875), blue, blue_line, ["Graph, embedding, and clustering utilities", "KNN/SNN graph  |  UMAP/openTSNE  |  Louvain/Leiden/random walk"], small=True)
    box((565, 925, 1235, 1025), gray, gray_line, ["Outputs", "labels, accuracy traces, graphs, embeddings", "timings, memory, backend metadata"], small=True)

    arrow((595, 228), (730, 320))
    arrow((1205, 228), (1070, 320))
    arrow((585, 450), (485, 505), color=green_line)
    arrow((1215, 450), (1315, 505), color=amber_line)
    arrow((900, 560), (900, 690), color=blue_line)
    arrow((900, 875), (900, 925))

    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(ARCH_FIGURE)


def build_docx() -> None:
    build_architecture_figure()
    doc = Document()
    apply_styles(doc)
    add_title(
        doc,
        "kodama-cpp: A C++/CUDA Library for KODAMA Cross-Validated Accuracy Maximization",
        "Manuscript for the Journal of Machine Learning Research Machine Learning Open Source Software track",
        "Stefano Cacciatore and contributors",
    )

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(ABSTRACT)

    for heading, paragraphs in SECTIONS:
        level = 1 if heading[0].isdigit() else 2
        doc.add_heading(heading, level=level)
        for paragraph in paragraphs:
            doc.add_paragraph(paragraph)
        if heading == "1. Introduction":
            doc.add_heading("Main contributions", level=2)
            add_table(doc, ("Contribution", "Role in kodama-cpp"), CONTRIBUTION_ROWS, [1.75, 4.7])
        if heading == "2. KODAMA objective":
            add_callout(
                doc,
                "Objective",
                "c* = argmax_c A(c; X, F, Pi), where A is cross-validated accuracy under the chosen classifier and fold assignment.",
            )
            doc.add_heading("Label-vector search mechanics", level=2)
            for paragraph in LABEL_SEARCH_DETAIL_PARAGRAPHS:
                doc.add_paragraph(paragraph)
            doc.add_heading("Raw accuracy, acceptance score, and diagnostics", level=2)
            for paragraph in SCORE_INTERPRETATION_PARAGRAPHS:
                doc.add_paragraph(paragraph)
            doc.add_heading("Reproducible KODAMA.matrix procedure", level=2)
            doc.add_paragraph(
                "The public KODAMA.matrix routine is reproducible from the following option definitions and run-level procedure. "
                "The description below names the C++ options because the R and Python wrappers should expose the same contract."
            )
            add_table(doc, ("Option", "Role in the algorithm"), KODAMA_PARAMETER_ROWS, [1.35, 5.1], font_size=8.0)
            doc.add_heading("Run-level optimization rule", level=3)
            add_numbered(doc, REPRODUCIBLE_ALGORITHM_STEPS)
            doc.add_heading("Final graph and dissimilarity", level=3)
            for paragraph in GRAPH_CONSTRUCTION_PARAGRAPHS:
                doc.add_paragraph(paragraph)
        if heading == "3. Implementation architecture":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(4)
            p.add_run().add_picture(str(ARCH_FIGURE), width=Inches(6.45))
            caption = doc.add_paragraph()
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption.paragraph_format.space_after = Pt(8)
            run = caption.add_run(
                "Figure 1. Architecture of kodama-cpp. The C++17 core owns the numerical kernels, "
                "CPU/CUDA backends, graph utilities, and typed outputs; R and Python remain thin wrappers."
            )
            set_run_font(run, 9.2, False, TOKENS["muted"])
            doc.add_heading("Public API summary", level=2)
            add_table(doc, ("Layer", "Public API", "Purpose"), API_ROWS, [1.35, 1.75, 3.35])
        if heading == "4. Relationship to public KODAMA literature":
            doc.add_heading("Compatibility with the R implementation", level=2)
            add_table(doc, ("Aspect", "Public R/literature contract", "kodama-cpp"), COMPATIBILITY_ROWS, [1.25, 2.65, 2.55])
            doc.add_heading("Relationship to semi-supervised learning and cluster validation", level=2)
            for paragraph in RELATED_WORK_PARAGRAPHS:
                doc.add_paragraph(paragraph)
            add_table(
                doc,
                ("Literature area", "Typical goal", "Role of KODAMA"),
                RELATED_WORK_ROWS,
                [1.45, 2.35, 2.65],
                font_size=7.8,
            )
        if heading == "6. Availability and reproducibility":
            doc.add_heading("Installation paths", level=2)
            add_table(doc, ("Target", "Command or check"), INSTALLATION_ROWS, [1.1, 5.35], font_size=7.4)
            doc.add_page_break()
            doc.add_heading("Wrapper validation", level=2)
            add_table(doc, ("Target", "Validation result"), WRAPPER_VALIDATION_ROWS, [1.5, 4.95], font_size=8.0)
            doc.add_heading("License and dependency notes", level=2)
            add_table(doc, ("Component", "Release note"), LICENSE_DEPENDENCY_ROWS, [1.8, 4.65], font_size=8.0)
        if heading == "5. Evaluation":
            doc.add_heading("Benchmark protocol and data coverage", level=2)
            add_table(doc, ("Item", "Value"), BENCHMARK_PROTOCOL_ROWS, [1.45, 5.0], font_size=8.0)
            add_table(
                doc,
                ("Dataset", "Samples", "Variables", "Classes"),
                PILOT_DATASET_ROWS,
                [1.8, 1.25, 1.25, 1.25],
                font_size=7.6,
            )
            doc.add_page_break()
            doc.add_heading("Evaluation guardrails and ablations", level=2)
            add_table(
                doc,
                ("Concern", "Manuscript position", "Required evidence"),
                EVALUATION_GUARDRAIL_ROWS,
                [1.15, 2.25, 3.05],
                font_size=7.5,
            )
            add_table(
                doc,
                ("Ablation", "Comparison", "Question answered"),
                ABLATION_MATRIX_ROWS,
                [1.2, 1.9, 3.35],
                font_size=7.5,
            )
            doc.add_heading("Pilot experiments on the CUDA workstation", level=2)
            for paragraph in PILOT_EXPERIMENT_PARAGRAPHS:
                doc.add_paragraph(paragraph)
            add_table(
                doc,
                ("Dataset", "Kernel", "CPU s", "CUDA s", "Speedup", "Acc CPU/CUDA", "Delta", "Comp."),
                PILOT_CV_ROWS,
                [0.75, 0.75, 0.7, 0.7, 0.65, 1.65, 0.7, 0.45],
                font_size=6.4,
            )
            doc.add_heading("Core optimizer pilot medians", level=3)
            add_table(
                doc,
                ("Dataset", "Core", "CPU s", "CUDA s", "Speedup", "Acc CPU/CUDA", "ARI CPU/CUDA", "Classes"),
                PILOT_CORE_ROWS,
                [0.78, 0.95, 0.72, 0.72, 0.65, 1.0, 1.0, 0.63],
                font_size=6.7,
            )
            doc.add_heading("MultiCPU versus GPU KODAMA comparison", level=3)
            for paragraph in KODAMA_BACKEND_COMPARISON_PARAGRAPHS:
                doc.add_paragraph(paragraph)
            add_table(
                doc,
                ("Dataset", "Classifier", "Backend", "Seconds", "CV acc", "ARI", "Classes", "Scope"),
                KODAMA_BACKEND_COMPARISON_ROWS,
                [0.75, 0.75, 0.65, 0.65, 0.65, 0.55, 0.55, 1.0],
                font_size=6.9,
            )
            doc.add_heading("KODAMA.matrix MetRef smoke benchmark", level=3)
            add_table(
                doc,
                ("Classifier", "Elapsed s", "Runtime s", "Best/median CV acc", "Best/median ARI", "Med. classes", "Class range"),
                PILOT_MATRIX_ROWS,
                [0.85, 0.75, 0.75, 1.25, 1.2, 0.8, 0.85],
                font_size=7.0,
            )
            doc.add_heading("Classic versus KODAMA visualization pilot", level=3)
            for paragraph in VISUALIZATION_COMPARISON_PARAGRAPHS:
                doc.add_paragraph(paragraph)
            add_table(
                doc,
                ("Dataset", "Embedding", "Sil. classic", "Sil. KODAMA", "Purity classic/KODAMA", "ARI", "Classes", "KODAMA s"),
                VISUALIZATION_COMPARISON_ROWS,
                [0.8, 0.75, 0.72, 0.75, 1.35, 0.55, 0.55, 0.65],
                font_size=6.8,
            )
            doc.add_page_break()
            doc.add_heading("Sensitivity of M and Tcycle", level=2)
            for paragraph in PARAMETER_SENSITIVITY_PARAGRAPHS:
                doc.add_paragraph(paragraph)
            add_table(
                doc,
                ("Dataset", "Classifier", "Best acc T20 -> T100", "Median acc T20 -> T100", "Median classes", "Seconds"),
                TCYCLE_SENSITIVITY_ROWS,
                [0.8, 0.8, 1.25, 1.35, 1.0, 0.9],
                font_size=7.6,
            )
            add_table(
                doc,
                ("Dataset", "Classifier", "Best acc M20/M50/M100", "Median acc M20/M50/M100", "Best-run ARI", "Seconds"),
                M_SENSITIVITY_ROWS,
                [0.8, 0.8, 1.4, 1.4, 1.1, 1.0],
                font_size=7.4,
            )
            if SENSITIVITY_FIGURE.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(4)
                p.add_run().add_picture(str(SENSITIVITY_FIGURE), width=Inches(6.45))
                caption = doc.add_paragraph()
                caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                caption.paragraph_format.space_after = Pt(8)
                run = caption.add_run(
                    "Figure 2. Sensitivity of KODAMA quality and runtime to Tcycle and M. "
                    "Tcycle = 100 is supported by convergence of the label-vector search; "
                    "M = 100 is used as a conservative independent-run ensemble size."
                )
                set_run_font(run, 9.0, False, TOKENS["muted"])
            doc.add_page_break()
            doc.add_heading("Implementation claims and evidence", level=2)
            add_table(
                doc,
                ("Feature claim", "Where implemented", "Test proving it", "Benchmark proving benefit"),
                IMPLEMENTATION_EVIDENCE_ROWS,
                [1.15, 1.8, 1.85, 1.65],
                font_size=7.4,
            )
            doc.add_heading("Release-validation evidence", level=2)
            add_table(doc, ("Area", "Current evidence or release task"), VALIDATION_ROWS, [1.45, 5.0])

    doc.add_page_break()
    doc.add_heading("Algorithm 1: one independent M run", level=1)
    doc.add_paragraph(
        "The optimization unit of KODAMA is one independent run inside the M-run ensemble. "
        "The pseudocode below isolates that unit: it starts from one landmark label vector, "
        "uses the previous CV predictions to propose label changes, evaluates the proposal "
        "with exactly one CV pass, and returns the best label vector found by that run."
    )
    add_pseudocode_block(doc, SINGLE_RUN_PSEUDOCODE_LINES)

    doc.add_page_break()
    doc.add_heading("Algorithm 2: KODAMA.matrix ensemble pseudocode", level=1)
    doc.add_paragraph(
        "The matrix-level routine repeats Algorithm 1 independently M times, projects each "
        "optimized run back to all samples, and builds the final KODAMA-corrected graph from "
        "agreement across the optimized label vectors."
    )
    add_pseudocode_block(doc, KODAMA_PSEUDOCODE_LINES)

    doc.add_page_break()
    doc.add_heading("Algorithm 3: KODAMA.matrix.graph pseudocode", level=1)
    doc.add_paragraph(
        "The graph-input routine keeps the same label-evolution objective while replacing "
        "the initial neighbor search with caller-supplied indices and distances. KNN uses "
        "the supplied graph directly; PLS-LDA uses self-tuning Laplacian features only when "
        "the original feature matrix is not supplied."
    )
    add_pseudocode_block(doc, GRAPH_INPUT_PSEUDOCODE_LINES)

    doc.add_heading("References", level=1)
    for ref in REFERENCES:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run(ref)
        set_run_font(run, 9.2)

    footer = doc.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("kodama-cpp JMLR MLOSS manuscript")
    doc.save(MANUSCRIPT)


def build_self_review() -> None:
    doc = Document()
    apply_styles(doc)
    apply_compact_memo_styles(doc)
    add_title(
        doc,
        "Reviewer-style self-assessment of the kodama-cpp JMLR manuscript",
        "Internal review memo and addressed revisions",
        "Prepared with the current manuscript.",
    )
    doc.add_heading("Reviewer Summary", level=1)
    doc.add_paragraph(
        "The manuscript now reads as a software-methods submission rather than a checklist. It formalizes KODAMA as cross-validated accuracy maximization, separates raw CV accuracy from proposal acceptance scores and external diagnostics, compares the C++/CUDA library with the public R-package literature, and distinguishes methodological continuity from implementation novelty. "
        "The KODAMA.matrix procedure is specified with label-vector search mechanics, proposal, temperature, M/Tcycle, landmark, splitting, constraint, graph-correction rules, and explicit pseudocode at both the single-run and matrix-ensemble levels. The revision adds literature context from semi-supervised learning, weak supervision, cluster validation, circular-analysis warnings, visualization, and open-source ML reproducibility. Implementation claims are mapped to source locations, tests, and pilot benchmark evidence. The remaining release tasks are archival DOI, coverage reporting, and the larger wrapper-level comparison run once remote CUDA access is available."
    )
    doc.add_heading("Major Comments", level=1)
    add_numbered(
        doc,
        [
            "Extend the benchmark table with the larger wrapper-level comparison against the R implementation.",
            "Freeze the public API and cite a versioned release, commit hash, and archival DOI.",
            "Complete the license compatibility audit for KODAMA, fastPLS, fastEmbedR, FAISS, and CUDA-adjacent libraries.",
            "Convert the final source into the official JMLR LaTeX style and keep the MLOSS page budget in mind.",
            "Run the final ablation matrix across the frozen release datasets rather than treating the current local CPU preview as the full benchmark.",
        ],
    )
    doc.add_heading("Revisions Applied", level=1)
    add_bullets(
        doc,
        [
            "Removed application-specific discussion and focused the paper on the KODAMA principle and implementation.",
            "Removed unsupported experimental paths and described only KNN and PLS-LDA as KODAMA classifiers.",
            "Added theory framing for cross-validated label predictability.",
            "Expanded the description of how high cross-validated accuracy label vectors are found: CV predictions guide group proposals, class-transition proposals aggregate unstable label movement, one CV pass evaluates each proposal, and a cooling acceptance rule maintains exploration.",
            "Expanded the mathematical description into a reproducible KODAMA.matrix procedure, including M runs, Tcycle, landmark selection, splitting, constrained groups, proposal rules, temperature acceptance, and KODAMA graph correction.",
            "Added KODAMA pseudocode that makes the initial CV evaluation, one-CV-per-cycle proposal loop, acceptance rule, projection step, and final graph reweighting explicit.",
            "Added a focused single-run pseudocode block showing one independent M run as the stochastic unit repeated by the ensemble.",
            "Added a public-literature comparison against the original KODAMA paper, the Bioinformatics R-package paper, and the documented R interface.",
            "Added a related-work section positioning KODAMA relative to semi-supervised graph learning, pseudo-labeling, weak supervision, clustering stability, prediction strength, UMAP, t-SNE, and open-source ML reproducibility.",
            "Added a compact architecture figure showing R/Python wrappers, the C++17 core, CPU and CUDA backends, graph/embedding/clustering utilities, and typed outputs.",
            "Added implementation details on float32 storage, label-aware SIMPLS, FAISS/cuVS, CPU HNSW, independent M cycles, typed outputs, and release validation.",
            "Added an implementation-evidence matrix linking each claim to source files, tests, and measured benchmark rows.",
            "Added evaluation guardrails that explicitly address circularity, parameter dependence, visualization bias, runtime attribution, and wrapper reproducibility.",
            "Added an ablation matrix specifying the required classifier, graph-correction, M/Tcycle, landmark/splitting, backend, and wrapper-parity comparisons.",
            "Added exact benchmark protocol, dataset coverage, and installation commands for the C++ core plus R and Python wrappers.",
            "Added an explicit distinction between raw CV accuracy, proposal acceptance score, and external diagnostics.",
            "Inserted completed pilot results from the CUDA workstation: CV kernel speedups, CoreKNN/CorePLSLDA optimization medians, and the MetRef KODAMA.matrix smoke benchmark.",
            "Added a reproducible M/Tcycle sensitivity experiment on two representative benchmarks for KNN and PLS-LDA, including quantitative tables and a curve figure.",
            "Removed draft-status boxes and footer language from the manuscript.",
        ],
    )
    doc.add_heading("Remaining Work", level=1)
    add_bullets(
        doc,
        [
            "Run the larger wrapper-level comparison once remote CUDA authentication is available.",
            "Prepare the cover letter for the MLOSS track.",
            "Archive the reviewed software version.",
        ],
    )
    doc.save(SELF_REVIEW)


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = text
    for key, value in replacements.items():
        out = out.replace(key, value)
    return out


def build_tex() -> None:
    body = []
    body.append(r"\documentclass[twoside,11pt]{article}")
    body.append(r"\usepackage{jmlr2e}")
    body.append(r"\usepackage{booktabs}")
    body.append(r"\usepackage{tabularx}")
    body.append(r"\usepackage{amsmath}")
    body.append(r"\usepackage{graphicx}")
    body.append(r"\usepackage{url}")
    body.append("")
    body.append(r"\jmlrheading{1}{2026}{1-4}{1/26}{7/26}{Cacciatore et al.}")
    body.append(r"\ShortHeadings{kodama-cpp}{Cacciatore et al.}")
    body.append(r"\firstpageno{1}")
    body.append("")
    body.append(r"\begin{document}")
    body.append(r"\title{kodama-cpp: A C++/CUDA Library for KODAMA Cross-Validated Accuracy Maximization}")
    body.append(r"\author{\name Stefano Cacciatore \email stefano.cacciatore@icgeb.org \\ \addr Bioinformatics Unit, International Centre for Genetic Engineering and Biotechnology}")
    body.append(r"\editor{To be assigned}")
    body.append(r"\maketitle")
    body.append(r"\begin{abstract}")
    body.append(tex_escape(ABSTRACT))
    body.append(r"\end{abstract}")
    body.append(r"\begin{keywords}")
    body.append("KODAMA, unsupervised learning, cross-validation, CUDA, C++ library, nearest neighbors, PLS-LDA")
    body.append(r"\end{keywords}")
    body.append("")
    for heading, paragraphs in SECTIONS:
        title = heading.split(". ", 1)[1] if ". " in heading else heading
        body.append(r"\section{" + tex_escape(title) + "}")
        for paragraph in paragraphs:
            body.append(tex_escape(paragraph))
            body.append("")
        if heading == "1. Introduction":
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Main contributions of kodama-cpp.}")
            body.append(r"\begin{tabular}{ll}")
            body.append(r"\toprule")
            body.append(r"Contribution & Role in kodama-cpp \\")
            body.append(r"\midrule")
            for contribution, detail in CONTRIBUTION_ROWS:
                body.append(f"{tex_escape(contribution)} & {tex_escape(detail)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append(r"\end{table}")
            body.append("")
        if heading == "2. KODAMA objective":
            body.append(r"\begin{equation}")
            body.append(r"c^\star = \arg\max_c \; A(c; X, F, \Pi),")
            body.append(r"\end{equation}")
            body.append("where A is the held-out accuracy of classifier family F under fold assignment Pi.")
            body.append("")
            body.append(r"\subsection{Label-vector search mechanics}")
            for paragraph in LABEL_SEARCH_DETAIL_PARAGRAPHS:
                body.append(tex_escape(paragraph))
                body.append("")
            body.append(r"\subsection{Raw accuracy, acceptance score, and diagnostics}")
            for paragraph in SCORE_INTERPRETATION_PARAGRAPHS:
                body.append(tex_escape(paragraph))
                body.append("")
            body.append(r"\subsection{Reproducible KODAMA.matrix procedure}")
            body.append(tex_escape(
                "The public KODAMA.matrix routine is reproducible from the following option definitions and run-level procedure. "
                "The description names the C++ options because the R and Python wrappers should expose the same contract."
            ))
            body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{KODAMA.matrix options that determine the optimization.}")
            body.append(r"\small")
            body.append(r"\begin{tabularx}{\linewidth}{p{0.18\linewidth}X}")
            body.append(r"\toprule")
            body.append(r"Option & Role in the algorithm \\")
            body.append(r"\midrule")
            for option, role in KODAMA_PARAMETER_ROWS:
                body.append(f"{tex_escape(option)} & {tex_escape(role)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabularx}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\subsubsection{Run-level optimization rule}")
            body.append(r"\begin{enumerate}")
            for item in REPRODUCIBLE_ALGORITHM_STEPS:
                body.append(r"\item " + tex_escape(item))
            body.append(r"\end{enumerate}")
            body.append("")
            body.append(r"\subsubsection{Final graph and dissimilarity}")
            for paragraph in GRAPH_CONSTRUCTION_PARAGRAPHS:
                body.append(tex_escape(paragraph))
                body.append("")
        if heading == "3. Implementation architecture":
            body.append(r"\begin{figure}[h]")
            body.append(r"\centering")
            body.append(r"\includegraphics[width=\linewidth]{kodama_cpp_architecture.png}")
            body.append(r"\caption{Architecture of kodama-cpp. The C++17 core owns the numerical kernels, CPU/CUDA backends, graph utilities, and typed outputs; R and Python remain thin wrappers.}")
            body.append(r"\end{figure}")
            body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Public API groups in kodama-cpp.}")
            body.append(r"\begin{tabular}{lll}")
            body.append(r"\toprule")
            body.append(r"Layer & Public API & Purpose \\")
            body.append(r"\midrule")
            for layer, api, purpose in API_ROWS:
                body.append(f"{tex_escape(layer)} & {tex_escape(api)} & {tex_escape(purpose)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append(r"\end{table}")
            body.append("")
        if heading == "4. Relationship to public KODAMA literature":
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Compatibility with the public KODAMA R interface and literature.}")
            body.append(r"\begin{tabular}{lll}")
            body.append(r"\toprule")
            body.append(r"Aspect & Public R/literature contract & kodama-cpp \\")
            body.append(r"\midrule")
            for aspect, prior, current in COMPATIBILITY_ROWS:
                body.append(f"{tex_escape(aspect)} & {tex_escape(prior)} & {tex_escape(current)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\subsection{Relationship to semi-supervised learning and cluster validation}")
            for paragraph in RELATED_WORK_PARAGRAPHS:
                body.append(tex_escape(paragraph))
                body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Literature positioning of KODAMA.}")
            body.append(r"\small")
            body.append(r"\begin{tabularx}{\linewidth}{p{0.22\linewidth}X X}")
            body.append(r"\toprule")
            body.append(r"Literature area & Typical goal & Role of KODAMA \\")
            body.append(r"\midrule")
            for area, goal, role in RELATED_WORK_ROWS:
                body.append(f"{tex_escape(area)} & {tex_escape(goal)} & {tex_escape(role)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabularx}")
            body.append(r"\end{table}")
            body.append("")
        if heading == "6. Availability and reproducibility":
            body.append(r"\subsection{Installation paths}")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Installation and runtime checks for the standalone core and wrappers.}")
            body.append(r"\small")
            body.append(r"\begin{tabularx}{\linewidth}{p{0.18\linewidth}X}")
            body.append(r"\toprule")
            body.append(r"Target & Command or check \\")
            body.append(r"\midrule")
            for target, command in INSTALLATION_ROWS:
                body.append(f"{tex_escape(target)} & {tex_escape(command)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabularx}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\subsection{Wrapper validation}")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Local and CUDA validation results for the core and wrapper packages.}")
            body.append(r"\small")
            body.append(r"\begin{tabularx}{\linewidth}{p{0.25\linewidth}X}")
            body.append(r"\toprule")
            body.append(r"Target & Validation result \\")
            body.append(r"\midrule")
            for target, result in WRAPPER_VALIDATION_ROWS:
                body.append(f"{tex_escape(target)} & {tex_escape(result)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabularx}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\subsection{License and dependency notes}")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Release notes for core components and external dependencies.}")
            body.append(r"\small")
            body.append(r"\begin{tabularx}{\linewidth}{p{0.28\linewidth}X}")
            body.append(r"\toprule")
            body.append(r"Component & Release note \\")
            body.append(r"\midrule")
            for component, note in LICENSE_DEPENDENCY_ROWS:
                body.append(f"{tex_escape(component)} & {tex_escape(note)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabularx}")
            body.append(r"\end{table}")
            body.append("")
        if heading == "5. Evaluation":
            body.append(r"\subsection{Benchmark protocol and data coverage}")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Benchmark protocol for the current evaluation.}")
            body.append(r"\begin{tabularx}{\linewidth}{p{0.22\linewidth}X}")
            body.append(r"\toprule")
            body.append(r"Item & Value \\")
            body.append(r"\midrule")
            for item, value in BENCHMARK_PROTOCOL_ROWS:
                body.append(f"{tex_escape(item)} & {tex_escape(value)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabularx}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Datasets copied into the benchmark data root for the current evaluation.}")
            body.append(r"\begin{tabular}{lrrr}")
            body.append(r"\toprule")
            body.append(r"Dataset & Samples & Variables & Classes \\")
            body.append(r"\midrule")
            for dataset, n, p, classes in PILOT_DATASET_ROWS:
                body.append(f"{tex_escape(dataset)} & {tex_escape(n)} & {tex_escape(p)} & {tex_escape(classes)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\subsection{Evaluation guardrails and ablations}")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Evaluation safeguards used to avoid over-interpreting the internal CV objective.}")
            body.append(r"\small")
            body.append(r"\begin{tabularx}{\linewidth}{p{0.16\linewidth}X X}")
            body.append(r"\toprule")
            body.append(r"Concern & Manuscript position & Required evidence \\")
            body.append(r"\midrule")
            for concern, position, evidence in EVALUATION_GUARDRAIL_ROWS:
                body.append(f"{tex_escape(concern)} & {tex_escape(position)} & {tex_escape(evidence)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabularx}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Ablation matrix for the release benchmark.}")
            body.append(r"\small")
            body.append(r"\begin{tabularx}{\linewidth}{p{0.18\linewidth}p{0.26\linewidth}X}")
            body.append(r"\toprule")
            body.append(r"Ablation & Comparison & Question answered \\")
            body.append(r"\midrule")
            for ablation, comparison, question in ABLATION_MATRIX_ROWS:
                body.append(f"{tex_escape(ablation)} & {tex_escape(comparison)} & {tex_escape(question)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabularx}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\subsection{Pilot experiments on the CUDA workstation}")
            for paragraph in PILOT_EXPERIMENT_PARAGRAPHS:
                body.append(tex_escape(paragraph))
                body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Pilot CPU/CUDA cross-validation kernel measurements. Accuracy is reported as CPU/CUDA.}")
            body.append(r"\small")
            body.append(r"\begin{tabular}{llllllll}")
            body.append(r"\toprule")
            body.append(r"Dataset & Kernel & CPU s & CUDA s & Speedup & Acc CPU/CUDA & Delta & Comp. \\")
            body.append(r"\midrule")
            for dataset, kernel, cpu_s, cuda_s, speedup, acc_pair, delta, comps in PILOT_CV_ROWS:
                body.append(
                    f"{tex_escape(dataset)} & {tex_escape(kernel)} & {tex_escape(cpu_s)} & "
                    f"{tex_escape(cuda_s)} & {tex_escape(speedup)} & {tex_escape(acc_pair)} & "
                    f"{tex_escape(delta)} & {tex_escape(comps)} \\\\"
                )
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\subsubsection{Core optimizer pilot medians}")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Pilot medians for the CoreKNN and CorePLSLDA optimization paths. Accuracy and ARI are reported as CPU/CUDA.}")
            body.append(r"\small")
            body.append(r"\begin{tabular}{llllllll}")
            body.append(r"\toprule")
            body.append(r"Dataset & Core & CPU s & CUDA s & Speedup & Acc & ARI & Classes \\")
            body.append(r"\midrule")
            for dataset, core, cpu_s, cuda_s, speedup, acc_pair, ari_pair, classes in PILOT_CORE_ROWS:
                body.append(
                    f"{tex_escape(dataset)} & {tex_escape(core)} & {tex_escape(cpu_s)} & "
                    f"{tex_escape(cuda_s)} & {tex_escape(speedup)} & {tex_escape(acc_pair)} & "
                    f"{tex_escape(ari_pair)} & {tex_escape(classes)} \\\\"
                )
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\subsubsection{MultiCPU versus GPU KODAMA comparison}")
            for paragraph in KODAMA_BACKEND_COMPARISON_PARAGRAPHS:
                body.append(tex_escape(paragraph))
                body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{KODAMA core optimizer CPU/CUDA pilot. The full KODAMA.matrix MultiCPU/GPU driver records the same columns end to end.}")
            body.append(r"\small")
            body.append(r"\begin{tabular}{llllllll}")
            body.append(r"\toprule")
            body.append(r"Dataset & Classifier & Backend & Seconds & CV acc & ARI & Classes & Scope \\")
            body.append(r"\midrule")
            for dataset, classifier, backend, seconds, acc, ari, classes, scope in KODAMA_BACKEND_COMPARISON_ROWS:
                body.append(
                    f"{tex_escape(dataset)} & {tex_escape(classifier)} & {tex_escape(backend)} & "
                    f"{tex_escape(seconds)} & {tex_escape(acc)} & {tex_escape(ari)} & "
                    f"{tex_escape(classes)} & {tex_escape(scope)} \\\\"
                )
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Pilot KODAMA.matrix CUDA smoke benchmark on MetRef with M = 20 and Tcycle = 20.}")
            body.append(r"\small")
            body.append(r"\begin{tabular}{lllllll}")
            body.append(r"\toprule")
            body.append(r"Classifier & Elapsed s & Runtime s & CV acc & ARI & Med. classes & Class range \\")
            body.append(r"\midrule")
            for classifier, elapsed, runtime, acc_pair, ari_pair, median_classes, class_range in PILOT_MATRIX_ROWS:
                body.append(
                    f"{tex_escape(classifier)} & {tex_escape(elapsed)} & {tex_escape(runtime)} & "
                    f"{tex_escape(acc_pair)} & {tex_escape(ari_pair)} & {tex_escape(median_classes)} & "
                    f"{tex_escape(class_range)} \\\\"
                )
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\subsubsection{Classic versus KODAMA visualization pilot}")
            for paragraph in VISUALIZATION_COMPARISON_PARAGRAPHS:
                body.append(tex_escape(paragraph))
                body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Local CPU preview comparing standard UMAP/openTSNE with KODAMA-KNN corrected graphs. Silhouette and purity are computed against external labels; ARI and class count refer to the best KODAMA label vector.}")
            body.append(r"\small")
            body.append(r"\begin{tabular}{llllllll}")
            body.append(r"\toprule")
            body.append(r"Dataset & Embedding & Sil. classic & Sil. KODAMA & Purity & ARI & Classes & KODAMA s \\")
            body.append(r"\midrule")
            for dataset, embedding, sil_classic, sil_kodama, purity, ari, classes, seconds in VISUALIZATION_COMPARISON_ROWS:
                body.append(
                    f"{tex_escape(dataset)} & {tex_escape(embedding)} & {tex_escape(sil_classic)} & "
                    f"{tex_escape(sil_kodama)} & {tex_escape(purity)} & {tex_escape(ari)} & "
                    f"{tex_escape(classes)} & {tex_escape(seconds)} \\\\"
                )
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\subsection{Sensitivity of M and Tcycle}")
            for paragraph in PARAMETER_SENSITIVITY_PARAGRAPHS:
                body.append(tex_escape(paragraph))
                body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Tcycle sensitivity at fixed M = 100. Accuracy entries show Tcycle = 20 to Tcycle = 100.}")
            body.append(r"\small")
            body.append(r"\begin{tabular}{llllll}")
            body.append(r"\toprule")
            body.append(r"Dataset & Classifier & Best acc & Median acc & Median classes & Seconds \\")
            body.append(r"\midrule")
            for dataset, classifier, best_acc, median_acc, median_classes, seconds in TCYCLE_SENSITIVITY_ROWS:
                body.append(
                    f"{tex_escape(dataset)} & {tex_escape(classifier)} & {tex_escape(best_acc)} & "
                    f"{tex_escape(median_acc)} & {tex_escape(median_classes)} & {tex_escape(seconds)} \\\\"
                )
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{M sensitivity at fixed Tcycle = 100. Entries show M = 20 / 50 / 100.}")
            body.append(r"\small")
            body.append(r"\begin{tabular}{llllll}")
            body.append(r"\toprule")
            body.append(r"Dataset & Classifier & Best acc & Median acc & Best-run ARI & Seconds \\")
            body.append(r"\midrule")
            for dataset, classifier, best_acc, median_acc, best_ari, seconds in M_SENSITIVITY_ROWS:
                body.append(
                    f"{tex_escape(dataset)} & {tex_escape(classifier)} & {tex_escape(best_acc)} & "
                    f"{tex_escape(median_acc)} & {tex_escape(best_ari)} & {tex_escape(seconds)} \\\\"
                )
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\begin{figure}[h]")
            body.append(r"\centering")
            body.append(r"\includegraphics[width=\linewidth]{kodama_m_tcycle_sensitivity.png}")
            body.append(r"\caption{Sensitivity of KODAMA quality and runtime to Tcycle and M. Tcycle = 100 is supported by convergence of the label-vector search; M = 100 is used as a conservative independent-run ensemble size.}")
            body.append(r"\end{figure}")
            body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Implementation claims and required evidence.}")
            body.append(r"\small")
            body.append(r"\begin{tabularx}{\linewidth}{p{0.17\linewidth}X X X}")
            body.append(r"\toprule")
            body.append(r"Feature claim & Where implemented & Test proving it & Benchmark proving benefit \\")
            body.append(r"\midrule")
            for feature, implemented, test, benchmark in IMPLEMENTATION_EVIDENCE_ROWS:
                body.append(
                    f"{tex_escape(feature)} & {tex_escape(implemented)} & "
                    f"{tex_escape(test)} & {tex_escape(benchmark)} \\\\"
                )
            body.append(r"\bottomrule")
            body.append(r"\end{tabularx}")
            body.append(r"\end{table}")
            body.append("")
            body.append(r"\begin{table}[h]")
            body.append(r"\caption{Release-validation evidence tracked by the project.}")
            body.append(r"\begin{tabular}{ll}")
            body.append(r"\toprule")
            body.append(r"Area & Current evidence or release task \\")
            body.append(r"\midrule")
            for area, evidence in VALIDATION_ROWS:
                body.append(f"{tex_escape(area)} & {tex_escape(evidence)} \\\\")
            body.append(r"\bottomrule")
            body.append(r"\end{tabular}")
            body.append(r"\end{table}")
            body.append("")
    body.append(r"\section{KODAMA pseudocode}")
    body.append(tex_escape(
        "Algorithm 1 isolates one independent run inside the M-run ensemble. It starts from one "
        "landmark label vector, uses the previous CV predictions to propose label changes, "
        "evaluates each proposal with exactly one CV pass, and returns the best label vector "
        "found by that run."
    ))
    body.append("")
    body.append(r"\begin{small}")
    body.append(r"\begin{verbatim}")
    body.extend(SINGLE_RUN_PSEUDOCODE_LINES)
    body.append(r"\end{verbatim}")
    body.append(r"\end{small}")
    body.append("")
    body.append(r"\subsection{KODAMA.matrix ensemble}")
    body.append(tex_escape(
        "Algorithm 2 repeats Algorithm 1 independently M times, projects each optimized run "
        "back to all samples, and builds the final KODAMA-corrected graph from agreement "
        "across the optimized label vectors."
    ))
    body.append("")
    body.append(r"\begin{small}")
    body.append(r"\begin{verbatim}")
    body.extend(KODAMA_PSEUDOCODE_LINES)
    body.append(r"\end{verbatim}")
    body.append(r"\end{small}")
    body.append("")
    body.append(r"\subsection{KODAMA.matrix.graph from supplied neighbors}")
    body.append(tex_escape(
        "Algorithm 3 keeps the same label-evolution objective while replacing the initial "
        "neighbor search with caller-supplied indices and distances. KNN uses the supplied "
        "graph directly; PLS-LDA uses self-tuning Laplacian features only when the original "
        "feature matrix is not supplied."
    ))
    body.append("")
    body.append(r"\begin{small}")
    body.append(r"\begin{verbatim}")
    body.extend(GRAPH_INPUT_PSEUDOCODE_LINES)
    body.append(r"\end{verbatim}")
    body.append(r"\end{small}")
    body.append("")
    body.append(r"\acks{The authors thank contributors to the KODAMA, fastPLS, and fastEmbedR software projects.}")
    body.append(r"\nocite{*}")
    body.append(r"\bibliography{kodama_cpp_refs}")
    body.append(r"\end{document}")
    TEX.write_text("\n".join(body) + "\n")


def build_bib() -> None:
    BIB.write_text(
        dedent(
            r"""
            @article{cacciatore2014kodama,
              title={Knowledge discovery by accuracy maximization},
              author={Cacciatore, Stefano and Luchinat, Claudio and Tenori, Leonardo},
              journal={Proceedings of the National Academy of Sciences},
              volume={111},
              number={14},
              pages={5117--5122},
              year={2014}
            }

            @manual{kodamaR,
              title={KODAMA: Knowledge Discovery by Accuracy Maximization},
              author={Cacciatore, Stefano and Tenori, Leonardo},
              note={R package documentation, CRAN},
              year={2026}
            }

            @article{cacciatore2017kodama,
              title={KODAMA: an R package for knowledge discovery and data mining},
              author={Cacciatore, Stefano and Tenori, Leonardo and Luchinat, Claudio and Bennett, Phillip R. and MacIntyre, David A.},
              journal={Bioinformatics},
              volume={33},
              number={4},
              pages={621--623},
              year={2017},
              doi={10.1093/bioinformatics/btw705}
            }

            @book{chapelle2006semisupervised,
              title={Semi-Supervised Learning},
              author={Chapelle, Olivier and Sch{\"o}lkopf, Bernhard and Zien, Alexander},
              publisher={MIT Press},
              year={2006}
            }

            @inproceedings{zhu2003gaussianfields,
              title={Semi-supervised learning using Gaussian fields and harmonic functions},
              author={Zhu, Xiaojin and Ghahramani, Zoubin and Lafferty, John},
              booktitle={Proceedings of the Twentieth International Conference on Machine Learning},
              pages={912--919},
              year={2003}
            }

            @inproceedings{zhou2003localglobal,
              title={Learning with local and global consistency},
              author={Zhou, Dengyong and Bousquet, Olivier and Lal, Thomas Navin and Weston, Jason and Sch{\"o}lkopf, Bernhard},
              booktitle={Advances in Neural Information Processing Systems},
              year={2003}
            }

            @inproceedings{lee2013pseudolabel,
              title={Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks},
              author={Lee, Dong-Hyun},
              booktitle={ICML Workshop on Challenges in Representation Learning},
              year={2013}
            }

            @inproceedings{ratner2016dataprogramming,
              title={Data programming: Creating large training sets, quickly},
              author={Ratner, Alexander and De Sa, Christopher and Wu, Sen and Selsam, Daniel and R{\'e}, Christopher},
              booktitle={Advances in Neural Information Processing Systems},
              year={2016}
            }

            @inproceedings{benhur2002stability,
              title={A stability based method for discovering structure in clustered data},
              author={Ben-Hur, Asa and Elisseeff, Andre and Guyon, Isabelle},
              booktitle={Pacific Symposium on Biocomputing},
              pages={6--17},
              year={2002}
            }

            @article{tibshirani2005predictionstrength,
              title={Cluster validation by prediction strength},
              author={Tibshirani, Robert and Walther, Guenther},
              journal={Journal of Computational and Graphical Statistics},
              volume={14},
              number={3},
              pages={511--528},
              year={2005}
            }

            @article{hubert1985comparing,
              title={Comparing partitions},
              author={Hubert, Lawrence and Arabie, Phipps},
              journal={Journal of Classification},
              volume={2},
              pages={193--218},
              year={1985}
            }

            @article{rousseeuw1987silhouettes,
              title={Silhouettes: A graphical aid to the interpretation and validation of cluster analysis},
              author={Rousseeuw, Peter J.},
              journal={Journal of Computational and Applied Mathematics},
              volume={20},
              pages={53--65},
              year={1987}
            }

            @inproceedings{kohavi1995crossvalidation,
              title={A study of cross-validation and bootstrap for accuracy estimation and model selection},
              author={Kohavi, Ron},
              booktitle={Proceedings of the Fourteenth International Joint Conference on Artificial Intelligence},
              pages={1137--1143},
              year={1995}
            }

            @article{ambroise2002selectionbias,
              title={Selection bias in gene extraction on the basis of microarray gene-expression data},
              author={Ambroise, Christophe and McLachlan, Geoffrey J.},
              journal={Proceedings of the National Academy of Sciences},
              volume={99},
              number={10},
              pages={6562--6566},
              year={2002}
            }

            @article{varma2006bias,
              title={Bias in error estimation when using cross-validation for model selection},
              author={Varma, Sudhir and Simon, Richard},
              journal={BMC Bioinformatics},
              volume={7},
              pages={91},
              year={2006}
            }

            @article{kriegeskorte2009circular,
              title={Circular analysis in systems neuroscience: the dangers of double dipping},
              author={Kriegeskorte, Nikolaus and Simmons, W. Kyle and Bellgowan, Patrick S. F. and Baker, Chris I.},
              journal={Nature Neuroscience},
              volume={12},
              pages={535--540},
              year={2009}
            }

            @article{dejong1993simpls,
              title={SIMPLS: an alternative approach to partial least squares regression},
              author={de Jong, Sijmen},
              journal={Chemometrics and Intelligent Laboratory Systems},
              volume={18},
              number={3},
              pages={251--263},
              year={1993}
            }

            @article{johnson2019faiss,
              title={Billion-scale similarity search with GPUs},
              author={Johnson, Jeff and Douze, Matthijs and Jegou, Herve},
              journal={IEEE Transactions on Big Data},
              year={2019}
            }

            @article{malkov2020hnsw,
              title={Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs},
              author={Malkov, Yu. A. and Yashunin, D. A.},
              journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
              year={2020}
            }

            @article{mcinnes2018umap,
              title={UMAP: Uniform Manifold Approximation and Projection for dimension reduction},
              author={McInnes, Leland and Healy, John and Melville, James},
              journal={arXiv:1802.03426},
              year={2018}
            }

            @article{vandermaaten2008tsne,
              title={Visualizing data using t-SNE},
              author={van der Maaten, Laurens and Hinton, Geoffrey},
              journal={Journal of Machine Learning Research},
              volume={9},
              pages={2579--2605},
              year={2008}
            }

            @article{linderman2019fitsne,
              title={Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data},
              author={Linderman, George C. and Rachh, Manas and Hoskins, Jeremy G. and Steinerberger, Stefan and Kluger, Yuval},
              journal={Nature Methods},
              volume={16},
              pages={243--245},
              year={2019}
            }

            @article{blondel2008louvain,
              title={Fast unfolding of communities in large networks},
              author={Blondel, Vincent D. and Guillaume, Jean-Loup and Lambiotte, Renaud and Lefebvre, Etienne},
              journal={Journal of Statistical Mechanics: Theory and Experiment},
              year={2008}
            }

            @article{traag2019leiden,
              title={From Louvain to Leiden: guaranteeing well-connected communities},
              author={Traag, Vincent A. and Waltman, Ludo and van Eck, Nees Jan},
              journal={Scientific Reports},
              volume={9},
              pages={5233},
              year={2019}
            }

            @article{sonnenburg2007opensource,
              title={The need for open source software in machine learning},
              author={Sonnenburg, S{\"o}ren and Braun, Mikio L. and Ong, Cheng Soon and Bengio, Samy and Bottou, Leon and Holmes, Geoffrey and LeCun, Yann and M{\"u}ller, Klaus-Robert and Pereira, Fernando and Rasmussen, Carl Edward and R{\"a}tsch, Gunnar and Sch{\"o}lkopf, Bernhard and Smola, Alexander and Vincent, Pascal and Weston, Jason and Williamson, Robert C.},
              journal={Journal of Machine Learning Research},
              volume={8},
              pages={2443--2466},
              year={2007}
            }

            @article{pineau2021reproducibility,
              title={Improving reproducibility in machine learning research},
              author={Pineau, Joelle and Vincent-Lamarre, Philippe and Sinha, Koustuv and Lariviere, Vincent and Beygelzimer, Alina and d'Alche-Buc, Florence and Fox, Emily and Larochelle, Hugo},
              journal={Journal of Machine Learning Research},
              volume={22},
              number={164},
              pages={1--20},
              year={2021}
            }
            """
        ).strip()
        + "\n"
    )


def main() -> None:
    build_docx()
    build_self_review()
    build_tex()
    build_bib()
    print(MANUSCRIPT)
    print(SELF_REVIEW)
    print(TEX)
    print(BIB)


if __name__ == "__main__":
    main()
