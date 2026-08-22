# KODAMA enables self-guided weakly supervised learning for spatial transcriptomics across tissues, platforms and scales

Ebtesam A. Abdel-Shafy^1,2,*^, Moussa Kassim^1,*^, Alessia Vignoli^3,4^, Farag Mamdouh^5^, Svitlana Tyekucheva^6^, Dalia Ahmed^1^, Dupe Ojo^1^, Brendon Price^7^, Fabio Socciarelli^8^, Nancy Paola Duarte-Delgado^1,9^, Martin Ocharo^1^, Chiamaka Jessica Okeke^1^, Luca Triboli^10^, David A. MacIntyre^11,12^, Massimo Loda^8,13-16^, Devanand Sarkar^17-19^, Dinesh Gupta^20^, Silvano Piazza^21,22^, Luiz Fernando Zerbini^23^, Leonardo Tenori^3,24,+^, Stefano Cacciatore^1,+^

1. Bioinformatics Unit, International Centre for Genetic Engineering and Biotechnology, Cape Town 7925, South Africa.
2. National Research Centre, Cairo, Egypt.
3. Department of Chemistry "Ugo Schiff", University of Florence, 50019 Sesto Fiorentino, Italy.
4. Magnetic Resonance Center (CERM), University of Florence, 50019 Sesto Fiorentino, Italy.
5. Genetics and Genetic Engineering Specification, Zoology Department, Faculty of Science, Benha University, Egypt.
6. Department of Data Science, Dana-Farber Cancer Institute, Boston, MA, USA.
7. Division of Anatomical Pathology, University of Cape Town and National Health Laboratory Service, Cape Town, South Africa.
8. Department of Pathology and Laboratory Medicine, Weill Cornell Medicine, New York, NY, USA.
9. Instituto de Genetica Humana, Facultad de Medicina, Pontificia Universidad Javeriana, Bogota, Colombia.
10. Cancer Cell Signaling Group, International Centre for Genetic Engineering and Biotechnology, Trieste, Italy.
11. March of Dimes Prematurity Research Centre, Imperial College London, London, UK.
12. Robinson Research Institute, University of Adelaide, Adelaide, Australia.
13. Sandra and Edward Meyer Cancer Center, Weill Cornell Medicine, New York, NY, USA.
14. Department of Oncologic Pathology, Dana-Farber Cancer Institute and Harvard Medical School, Boston, MA, USA.
15. University of Oxford, Nuffield Department of Surgical Sciences, Oxford, UK.
16. [Affiliation 16 to be verified against the author submission record.]
17. Department of Cellular, Molecular and Genetic Medicine, Virginia Commonwealth University, Richmond, VA, USA.
18. Massey Comprehensive Cancer Center, Virginia Commonwealth University, Richmond, VA, USA.
19. VCU Institute of Molecular Medicine, Richmond, VA, USA.
20. Translational Bioinformatics Group, International Centre for Genetic Engineering and Biotechnology, New Delhi, India.
21. Computational Biology Group, International Centre for Genetic Engineering and Biotechnology, Trieste, Italy.
22. Bioinformatics Facility, Department of Cellular, Computational and Integrative Biology, University of Trento, Trento, Italy.
23. Cancer Genetics Group, International Centre for Genetic Engineering and Biotechnology, Cape Town, South Africa.
24. Consorzio Interuniversitario Risonanze Magnetiche MetalloProteine, 50019 Sesto Fiorentino, Italy.

*Ebtesam A. Abdel-Shafy and Moussa Kassim contributed equally. +Leonardo Tenori and Stefano Cacciatore jointly supervised the work and are corresponding authors.*

Correspondence: Leonardo Tenori, tenori@cerm.unifi.it; Stefano Cacciatore, stefano.cacciatore@icgeb.org.

## Abstract

### Background

Spatial transcriptomics measures molecular phenotypes while retaining their tissue coordinates, but analysis is complicated by high dimensionality, technical noise, non-linear tissue organization, multiple sections and rapidly increasing sample size. Existing workflows commonly impose a single partition or append spatial information during preprocessing, which can obscure gradients, nested domains and alternative locally coherent solutions.

### Methods

We developed a spatially constrained implementation of KODAMA, a self-guided weakly supervised learning framework that evolves internally generated latent labels by maximizing cross-validated prediction accuracy. Biological annotations are not supplied to the optimization. Independent runs select spatially representative landmarks, define indivisible local spatial groups and refine their labels using either K-nearest-neighbor or SIMPLS plus latent-space linear discriminant analysis. Multi-slide coordinates are horizontalized before graph construction, and every spatial stratum and constraint is additionally keyed by slide identity, formally preventing optimization groups from crossing slide boundaries. The ensemble of run-specific labels corrects a reusable feature-space neighbor graph, which is visualized with UMAP or openTSNE. CPU, CUDA and Metal backends execute the same float32 algorithmic contract.

### Results

Across image-based, array-based and high-resolution spatial transcriptomics data, KODAMA recovered tissue organization, continuous expression gradients and reproducible subdivisions that were not consistently represented by conventional embeddings or single clustering solutions. The framework supported three-dimensional measurements, coordinated analysis of multiple sections and large Visium HD data. In an explicit four-slide DLPFC validation, no spatial constraint group crossed a slide boundary in any independent run. [Final comparative effect sizes, uncertainty intervals and runtime results will be inserted after the frozen benchmark analysis.]

### Conclusions

Spatially constrained KODAMA provides a general feature-learning representation that combines molecular predictability with local spatial coherence without using annotations during optimization. Its ensemble output can support visualization, clustering, trajectory analysis and histopathological interpretation across heterogeneous spatial transcriptomics technologies.

## Keywords

Spatial transcriptomics; dimensionality reduction; weakly supervised learning; cross-validation; tissue architecture; CUDA; Metal; multi-sample integration

## Background

Spatial transcriptomics (ST) links molecular profiles to tissue position, making spatial domain identification and representation learning central analytical tasks. The difficulty is not simply to obtain a contiguous partition. Tissues can contain sharp boundaries, continuous gradients, nested structures and spatially separated regions with related molecular states. A useful representation should therefore retain molecular relationships while using local spatial context without assuming that one partition is the unique biological truth. This problem becomes more demanding in multi-section and high-resolution studies, where coordinates may overlap numerically between physically independent slides and the number of observations can reach millions.

Existing methods encode spatial context through distinct and valuable statistical assumptions. Bayesian approaches such as BayesSpace and BASS model neighborhood dependence while estimating spatial domains [3,4]. PRECAST jointly estimates embeddings, clusters and cross-slide alignment [5]. BANKSY augments each observation with features derived from its spatial neighborhood [6]. Graph-learning approaches including SpaGCN, STAGATE, GraphST, SPACEL and STAIG combine molecular measurements, spatial graphs and, in some implementations, histology through graph convolution, attention, contrastive learning or adversarial objectives [7-11]. These approaches should not be reduced to a single category: they differ in whether spatial information enters a probabilistic prior, an augmented feature space, a graph encoder, an integration model or an image-informed objective. General-purpose t-SNE and UMAP remain useful for visualizing molecular neighborhoods, but they do not by themselves define how tissue proximity should influence representation learning.

KODAMA addresses a complementary problem. The original method was introduced as unsupervised feature extraction by accuracy maximization and later released as an R package [1,2]. It uses a supervised classifier internally, but its standard unsupervised operation receives no biological class labels. Instead, temporary labels are generated within the algorithm, modified stochastically and evaluated by out-of-fold prediction. Cross-validated accuracy is therefore an internal measure of how reproducibly a candidate organization can be inferred from the molecular features, not agreement with known tissue annotations. Multiple independent solutions are retained and converted into pairwise agreement evidence, so the output is a corrected dissimilarity or neighbor graph rather than a claim that one optimized label vector is ground truth.

The spatial extension preserves this principle while changing the unit of label evolution. Local groups derived from tissue coordinates are atomic proposal units: all observations in a group can change latent label together, but the spatial group identifier is never used as the prediction target. Initial latent classes are generated from molecular similarity, histological annotations are not supplied during optimization, and spatial coordinates are not appended to the classifier features or used to initialize the final UMAP or openTSNE display. Thus, spatial context regularizes which label changes are admissible, whereas molecular cross-validation determines whether a proposed organization is retained. This differs from directly smoothing expression, optimizing a graph-reconstruction loss or training on externally supplied pseudo-labels. Spatial message passing, when explicitly used, is a separate preprocessing operation and is not a methodological novelty of spatial KODAMA.

This formulation also defines the limits of the method. Horizontal translation and sample-keyed constraints prevent local groups from crossing physically independent slides, but they do not estimate or remove molecular batch effects. Multi-slide analyses therefore require appropriate expression preprocessing or a dedicated integration method when batch correction is scientifically necessary. Likewise, the present implementation supports two internal predictors, K-nearest neighbors (KNN) and SIMPLS followed by latent-space linear discriminant analysis (PLS-LDA); it does not rely on support vector machines.

The contribution of spatial KODAMA is consequently not a new class of machine learning or a replacement for every spatial clustering model. It is an annotation-free, ensemble representation strategy in which local tissue constraints act on stochastic label proposals and repeated molecular predictability reshapes a reusable neighbor graph. The present study has four aims: to formalize this mechanism and its objective; to isolate the contribution of spatial constraints and optional spatial feature weighting through ablation; to compare KODAMA with statistical and graph-learning ST methods across tissues, technologies and perturbations; and to quantify computational scalability using matched CPU, CUDA and Metal implementations. Biological claims are evaluated post hoc and separately from the unsupervised optimization.

## Methods

### Study design and datasets

The study is a retrospective computational analysis of publicly available spatial transcriptomics datasets. The principal evaluations comprise: (i) a three-dimensional mouse preoptic hypothalamus MERFISH dataset; (ii) twelve human dorsolateral prefrontal cortex (DLPFC) Visium sections; (iii) human prostate Visium tissues; and (iv) a human colorectal cancer Visium HD dataset. Dataset accession links, original publications, sample counts, variables retained, preprocessing, ground-truth annotations and exclusion criteria will be consolidated in Table 1 and the Availability of data and materials statement.

No ground-truth tissue label is supplied to KODAMA during optimization, hyperparameter selection or visualization. Histological annotations are used only for post hoc evaluation and biological interpretation.

### Optional spatial message-passing preprocessing

For spatial single-cell datasets, an optional preprocessing operation reproduces the `passing.message` function distributed with KODAMAextra. This operation is included for computational completeness and is not claimed as a methodological contribution of this study. Let `N_k(i,b)` be the self-inclusive set of the `k` nearest spatial neighbors of observation `i` within slide or tissue `b`, and let

`d_max,b = max_i max_{j in N_k(i,b)} d(S_i,S_j)`.

The message-passed value of molecular variable `v` is

`X_message[i,v] = sum_{j in N_k(i,b)} X[j,v] exp(-d(S_i,S_j) / d_max,b)`.

The sum is intentionally not divided by the sum of the weights, matching the original KODAMAextra mathematics; the default is `k = 15`. When all selected distances in a slide are zero, the continuous limiting convention assigns weight one. If a sample/slide vector is supplied, each tissue receives an independent neighbor search and distance scale, so no message can pass between physically distinct tissues even when their coordinate ranges overlap.

The standalone implementation stores expression values in float32, replaces `Rnanoflann::nn` with the exact two- or three-dimensional grid search used by the KODAMA library, and evaluates the weighted sums with parallel C++ row kernels. Equal-distance ties are resolved deterministically by row index. Message passing is applied only in analyses explicitly labelled as such; otherwise KODAMA receives the stated molecular input without this smoothing step.

### Optional multi-slide spatial feature screening

The software also provides an independently written, MIT-licensed spatial
feature screen that does not link to or include SPARK-X. Within each slide,
coordinates are range standardized and expanded into eleven fixed low-rank
bases: one linear basis, five smooth Gaussian coordinate bases and five
periodic sine-cosine bases. Every basis is centered and orthonormalized. For
centered feature vector `x_v` and basis matrix `Q_h`, the explained spatial
fraction is

`R2[v,h] = ||Q_h' x_v||2^2 / ||x_v||2^2`.

Under the fixed-basis Gaussian null this fraction is evaluated with the beta
law `Beta(rank(Q_h)/2, (n-rank(Q_h)-1)/2)`. Basis-level probabilities are
combined by the Cauchy rule. Slides are never pooled during basis construction;
the output reports each slide separately, an across-slide combined probability,
and a ranking by the strongest slide-specific evidence. Variables that are
identically zero in any slide can be excluded, matching the filtering contract
of `multi_SPARKX`. Projection arithmetic is float32 in a multicore CPU
implementation; tail probabilities and BH adjustment are double precision.
CUDA and Metal routes were removed because they did not improve end-to-end
runtime sufficiently to justify additional backend complexity.

Preliminary external benchmarking against SPARK-X on 10X breast cancer, 10X
mouse brain and SeqFISH embryo data produced rank correlations of 0.875, 0.936
and 0.791, respectively. Four-core CPU speedups were 19-36 fold. These
measurements establish screening similarity,
not numerical identity; the complete protocol, timings and plots are retained
in `benchmarks/spatial-feature-selection/` and will be repeated in the release
benchmark.

### Overview of spatially constrained KODAMA

Let `X` denote an `n x p` float32 molecular feature matrix and `S` an `n x q` spatial coordinate matrix, where `q` is two or three. KODAMA performs `M` independent runs. Each run selects landmark observations, constructs local spatial constraint groups, initializes latent labels from molecular similarity, evolves those labels through cross-validated prediction, projects optimized labels to non-landmarks and stores the complete run-specific label vector. The `M` solutions are then aggregated as pairwise label-agreement evidence on a reusable molecular KNN graph.

The current public implementation supports two classifiers. KNN predicts held-out labels from neighboring training observations. PLS-LDA fits SIMPLS using integer-label class sums and performs LDA in the latent component space. Both classifiers use the same proposal and acceptance framework and the same spatial constraints; classifier-specific score terms are reported explicitly where applicable.

### Landmark selection

Landmarks restrict repeated classifier optimization to a representative subset. When the requested landmark number is greater than or equal to `n`, KODAMA applies the original rule `L = ceiling(0.75n)`; otherwise the requested number is bounded to `2,...,n-1`.

For spatial data, each coordinate axis is divided into approximately `L^(1/q)` regular bins. Occupied grid cells are intersected with slide identity, proportional quotas are assigned with exact residual rounding, and observations are sampled without replacement within each stratum. This provides broad spatial coverage while preserving stochastic independence between runs. For non-spatial inputs, the current library uses a shared coarse molecular k-means atlas; that pathway is not part of the spatial mechanism evaluated here.

Every spatial constraint block contributes at least one landmark. Residual
landmarks are then allocated within the occupied strata. When the reusable
neighbor graph does not initially contain enough landmark-only neighbors for a
landmark row, candidate expansion continues until a valid landmark neighbor is
found; the implementation does not silently replace an empty row with an
arbitrary observation.

### Multi-slide horizontalization and hard separation

Independent tissue sections can have overlapping coordinate ranges. If sample identifiers `b_i` are provided, slides are processed in sorted identifier order and the first coordinate is translated. For slide `j`,

`S[i,1] <- S[i,1] + a_j` for `b_i = j`, with `a_1 = 0` and

`a_(j+1) = max(S[j,1]) + 0.5[max(S[j,1]) - min(S[j,1])]`,

where the range is calculated after applying the current slide offset. The remaining coordinate axes are unchanged. This reproduces the multi-sample rule in the original KODAMA implementation.

Geometric translation alone cannot formally exclude every cross-slide k-means assignment. KODAMA therefore applies a second, exact boundary: every landmark stratum and spatial constraint is represented by the pair `(slide identifier, within-slide group)`. Singleton groups can be reassigned only to groups from the same slide. User-supplied constraints are intersected with slide identity before they are composed with spatial groups. Consequently, no atomic optimization group can contain observations from different slides, regardless of coordinate scale, resolution or execution backend.

### Spatial constraint construction

Within each run, bounded coordinate jitter resolves ties. Axis-specific jitter scales are estimated once from the reusable spatial neighbor graph. The jittered coordinates are partitioned into

`C = round(L rho)`

groups, where `rho` is the spatial resolution. The public and benchmark default
is fixed at `rho = 0.3`; it is not selected separately for individual datasets.
Standard analyses use k-means spatial constraints. Every resulting group is
intersected with slide identity and becomes an indivisible unit during label
evolution. A connected-region construction is available as an experimental
sensitivity option, but controlled MERFISH and DLPFC validation did not support
it as the production default.

### Cross-validated label evolution

Initial landmark labels are generated by molecular k-means using `splitting`
initial classes. Spatial blocks are assigned intact to `F` fixed
cross-validation folds by greedily balancing their sample counts; no block is
split between training and validation. With current labels `c`, the classifier
returns out-of-fold predictions and accuracy `A(c)`. At every temperature
cycle, KODAMA constructs one group-level label proposal using sample-level
prediction transitions and adaptive proposal size, evaluates the proposed
vector exactly once by cross-validation, and accepts an improving proposal or
a non-improving proposal according to the temperature rule. The classifier
transition evidence is therefore not reduced to one majority prediction per
block, while the proposed move remains atomic for that block. Single-class
proposals are rejected. The best-scoring label vector observed during the run
is retained.

Spatial constraints change the proposal unit, not the classifier input: all observations within one local group move together, but prediction remains based on `X`. The `M` runs use independent seeds, landmarks and evolution trajectories and exchange no consensus information. This independence enables parallel scheduling and avoids forcing convergence to a single partition.

[The exact acceptance probability and classifier-specific objective terms will be inserted from the frozen release equations before submission.]

### Projection to non-landmarks

The optimized landmark labels train the selected classifier for projection to non-landmark observations. Projection respects the same spatial constraints, so all observations assigned to one atomic group receive a consistent projected label. Only projected labels and run diagnostics are required for subsequent graph correction.

### Ensemble graph correction

Let `c_i^(m)` be the projected label of observation `i` in run `m`. For an edge `(i,j)` in the reusable molecular KNN graph, label agreement is

`G_ij = sum_m I(c_i^(m) = c_j^(m), both nonzero) / sum_m I(both labels nonzero)`.

For positive agreement, the KODAMA-corrected dissimilarity is

`d*_ij = (1 + d_ij) / G_ij^2`.

Edges without any valid equal-label run receive infinite dissimilarity. Corrected neighbors are sorted row-wise. This operation uses all `M` label vectors; it does not select a run using histological truth.

Downstream clustering is performed directly on this corrected graph. The
production benchmark uses one deterministic exact-`K` Pons-Latapy random-walk
hierarchy for every dataset; UMAP and openTSNE are visualization methods and do
not determine cluster membership. Reference annotations supply `K` only in the
explicit external clustering benchmark and are otherwise withheld.

### Visualization

The corrected graph is embedded using the library's UMAP or openTSNE implementation. UMAP uses the fuzzy graph mode by default. PCA initialization is calculated from the original molecular input and matched to the selected execution backend. Spatial coordinates are never used to initialize UMAP or openTSNE. This separation prevents the final display from reproducing tissue geometry by construction.

### Implementation and reproducibility

Core computations use float32. CPU, CUDA and Metal backends implement the same landmark, constraint, evolution and graph-correction mathematics. KODAMA.graph prepares a reusable KNN graph and PCA initialization; KODAMA.matrix consumes raw data, the prepared graph or both; KODAMA.visualization reuses these products without rebuilding the graph. Experiments must report backend, seed, `M`, `Tcycle`, folds, landmarks, splitting, classifier settings, graph neighbors, `rho` and visualization settings.

### Comparative methods and statistical analysis

Comparisons include conventional PCA, UMAP and openTSNE and established spatial methods used in the original study, including BayesSpace, BASS, BANKSY and PRECAST where compatible with the data. Clustering methods and target cluster counts must be prespecified. Evaluation should report adjusted Rand index where reference domains are available, silhouette and Davies-Bouldin indices, neighborhood preservation, spatial continuity, per-class compactness, runtime and peak memory. Repeated stochastic analyses should report medians and dispersion across seeds; paired comparisons should be used when identical data splits and seeds permit pairing.

### Biological analyses

Differential expression, pathway enrichment, spatially variable gene analysis, histological image analysis, trajectory construction and geometry export will be retained from the original study but separated from the core KODAMA algorithm. Their software versions, filtering criteria, multiple-testing corrections and analysis-specific parameters will be consolidated in the Supplementary Methods.

### Use of generative artificial intelligence

Generative AI-assisted tools were used during manuscript restructuring and language editing. All algorithmic descriptions, analyses, interpretations and references were reviewed and verified by the authors. [Revise this disclosure to match the journal's policy and the final author workflow.]

## Results

### KODAMA captures anatomical structure in three-dimensional spatial single-cell data

The MERFISH evaluation will present the three-dimensional tissue model, KODAMA representations, clustering comparisons and quantitative spatial-continuity results as a single coherent result. The revised text will distinguish representation quality from clustering accuracy and will report uncertainty across stochastic runs.

### Multi-slide KODAMA identifies reproducible and section-specific DLPFC organization

The twelve DLPFC sections will be analyzed jointly using explicit slide identifiers. Horizontalization prevents coordinate overlap, while hard sample-keyed constraints guarantee that local blocks do not span sections. The Results will report both the integrated KODAMA representation and per-section spatial maps, including cortical layers, newly observed subdivisions and technical outlier regions.

As an implementation validation on the four Br8100 slides, both KNN and
PLS-LDA completed 100 independent runs without a mixed-slide constraint group.
Using the fixed CUDA PLS-LDA protocol (`M = 100`, `Tcycle = 100`, 50
components, 100 graph neighbors, `rho = 0.3`), direct exact-seven clustering of
the all-M corrected graph produced slide-level ARI values of 0.519, 0.461,
0.435 and 0.435 for sections 151673--151676, respectively (pooled ARI 0.459).
The corresponding five MERFISH sections produced ARI values of 0.609, 0.617,
0.585, 0.519 and 0.496 (pooled ARI 0.559). These are implementation-validation
measurements from the frozen-seed control and will not replace the planned
multi-seed comparative analysis.

### KODAMA reveals molecular gradients and heterogeneous regions in prostate tissue

The prostate analysis will retain the original trajectory and differential-expression findings but reorganize them around a prespecified question: whether the KODAMA representation supports continuous molecular variation across histologically related regions. Marker and enrichment results will include effect sizes, multiple-testing correction and sample-level provenance.

### KODAMA scales to high-resolution colorectal cancer data

The Visium HD analysis will report the number of retained spots, variables, landmarks, graph size, runtime, peak memory and backend. Molecular gradients and candidate markers will be presented separately from computational scalability to avoid conflating biological and engineering conclusions.

### Comparative performance across technologies

A consolidated table will compare KODAMA-KNN, KODAMA-PLS-LDA, conventional embeddings and compatible spatial methods across datasets. The primary endpoints, number of repetitions and statistical comparisons will be fixed before the final benchmark is run. Visual examples will be selected using algorithmic criteria or fixed seeds rather than histological labels.

## Discussion

Spatial KODAMA differs from methods that append coordinates to molecular features or return one optimized partition. Spatial coordinates define the atomic scope of a candidate label change, while cross-validated molecular prediction determines whether the change is retained. The ensemble of independent solutions is then used to reshape a local molecular graph. This division of roles provides local coherence without making tissue coordinates the final embedding geometry.

The multi-slide mechanism is important for integrated studies. Horizontalization preserves the original behavior expected by KODAMA users, while the sample-keyed constraint intersection provides a formal guarantee that was not supplied by geometric spacing alone. Equivalent molecular states may still receive the same final latent label across slides; what is prohibited is a single spatial constraint block containing observations from different slides.

The framework is intended for representation learning rather than automatic recovery of a unique biological truth. High cross-validated accuracy does not itself establish biological validity, and lower spatial resolution does not necessarily imply a preferred number of domains. Interpretation therefore requires examination of the ensemble, external biological evidence and sensitivity to prespecified parameters. The final manuscript will emphasize effect sizes and uncertainty instead of relying only on selected two-dimensional plots.

Limitations include dependence on the initial molecular representation, the computational cost of repeated cross-validation, sensitivity of downstream clustering to graph and resolution choices, and incomplete reference annotations in several ST datasets. The present evaluation is retrospective and primarily uses public datasets. Prospective pathological validation and independent clinical cohorts will be needed before translational or diagnostic use.

## Conclusions

Spatially constrained KODAMA combines molecular predictability with local tissue coherence through independent cross-validated label evolution. The method handles multiple slides without coordinate overlap or cross-slide spatial blocks, supports two- and three-dimensional measurements and produces graph representations suitable for visualization and downstream biological analysis. The revised implementation and evaluation provide a foundation for studying heterogeneous tissue organization across spatial transcriptomics technologies while keeping algorithmic learning separate from reference annotations.

## List of abbreviations

AMP: after message-passing; ARI: adjusted Rand index; CUDA: Compute Unified Device Architecture; DLPFC: dorsolateral prefrontal cortex; KNN: K-nearest neighbors; LDA: linear discriminant analysis; MERFISH: multiplexed error-robust fluorescence in situ hybridization; PCA: principal component analysis; PLS: partial least squares; ST: spatial transcriptomics; t-SNE: t-distributed stochastic neighbor embedding; UMAP: uniform manifold approximation and projection.

## Declarations

### Ethics approval and consent to participate

[To be completed. State that the computational study used public deidentified datasets and reproduce the original studies' ethics approvals where human tissue data require them.]

### Consent for publication

Not applicable, subject to verification that no identifiable individual-level image or clinical information is included.

### Availability of data and materials

All datasets analyzed in this study are publicly available. The final statement will provide persistent repository identifiers for the MERFISH, DLPFC, prostate and Visium HD datasets. Analysis code and frozen configuration files will be archived with a versioned software release and persistent identifier.

### Competing interests

The authors declare that they have no competing interests. [Confirm with every author before submission.]

### Funding

[Insert all grants and state the role of each funder.]

### Authors' contributions

[Prepare a CRediT-based statement using author initials. All authors must approve the final manuscript.]

### Acknowledgements

The authors thank the University of Cape Town's ICTS High Performance Computing team for providing a high-performance computing facility for this study (https://ucthpc.uct.ac.za/).

[Add any study-specific acknowledgements from the 2025 preprint after author confirmation.]

### Authors' information

Not applicable.

## References

1. Cacciatore S, Luchinat C, Tenori L. Knowledge discovery by accuracy maximization. Proc Natl Acad Sci U S A. 2014;111:5117-5122. doi:10.1073/pnas.1220873111.
2. Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA. KODAMA: an R package for knowledge discovery and data mining. Bioinformatics. 2017;33:621-623. doi:10.1093/bioinformatics/btw705.
3. Zhao E, Stone MR, Ren X, et al. Spatial transcriptomics at subspot resolution with BayesSpace. Nat Biotechnol. 2021;39:1375-1384. doi:10.1038/s41587-021-00935-2.
4. Li Z, Zhou X. BASS: multi-scale and multi-sample analysis enables accurate cell type clustering and spatial domain detection in spatial transcriptomic studies. Genome Biol. 2022;23:168. doi:10.1186/s13059-022-02734-7.
5. Liu W, Liao X, Luo Z, et al. Probabilistic embedding, clustering, and alignment for integrating spatial transcriptomics data with PRECAST. Nat Commun. 2023;14:296. doi:10.1038/s41467-023-35947-w.
6. Singhal V, Chou N, Lee J, et al. BANKSY unifies cell typing and tissue domain segmentation for scalable spatial omics data analysis. Nat Genet. 2024;56:431-441. doi:10.1038/s41588-024-01664-3.
7. Hu J, Li X, Coleman K, et al. SpaGCN: integrating gene expression, spatial location and histology to identify spatial domains and spatially variable genes by graph convolutional network. Nat Methods. 2021;18:1342-1351. doi:10.1038/s41592-021-01255-8.
8. Dong K, Zhang S. Deciphering spatial domains from spatially resolved transcriptomics with an adaptive graph attention auto-encoder. Nat Commun. 2022;13:1739. doi:10.1038/s41467-022-29439-6.
9. Long Y, Ang KS, Li M, et al. Spatially informed clustering, integration, and deconvolution of spatial transcriptomics with GraphST. Nat Commun. 2023;14:1155. doi:10.1038/s41467-023-36796-3.
10. Xu H, Wang S, Fang M, et al. SPACEL: deep learning-based characterization of spatial transcriptome architectures. Nat Commun. 2023;14:7603. doi:10.1038/s41467-023-43220-3.
11. Yang Y, Cui Y, Zeng X, et al. STAIG: spatial transcriptomics analysis via image-aided graph contrastive learning for domain exploration and alignment-free integration. Nat Commun. 2025;16:1067. doi:10.1038/s41467-025-56276-0.

[Complete the Vancouver bibliography by migrating and verifying the remaining biological, dataset, software and statistical references from the preprint.]

## Figure legends

[Reorder the original figures to follow the revised Results: Figure 1, algorithm and study design; Figure 2, three-dimensional MERFISH; Figure 3, multi-slide DLPFC; Figure 4, prostate; Figure 5, Visium HD; Figure 6, cross-platform quantitative comparison. Figure titles must not exceed 15 words and legends must not exceed 300 words.]
