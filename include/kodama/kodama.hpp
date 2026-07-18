// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include "kodama/version.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kodama {

enum class Backend {
  Auto,
  CPU,
  CUDA,
  Metal
};

enum class DistanceMetric {
  Cosine,
  InnerProduct,
  Euclidean
};

enum class KNNIndexType {
  NativeHNSW,
  CudaExact,
  CudaIVFFlat,
  MetalExact,
  MetalIVFFlat
};

enum class PLSMode {
  PLS_DA,
  PLS_LDA
};

enum class CoreClassifier {
  PLS_LDA,
  KNN
};

enum class GraphWeightType {
  SNN,
  Distance,
  Adaptive,
  Binary
};

enum class GraphFeatureMode {
  LaplacianSelfTuning
};

enum class UMAPGraphMode {
  Binary,
  Fuzzy
};

enum class MatrixValueType {
  Float64,
  Float32
};

struct MatrixView {
  const void* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;
  MatrixValueType value_type = MatrixValueType::Float64;

  MatrixView() = default;

  MatrixView(const double* data_ptr, std::size_t n_rows, std::size_t n_cols)
      : data(data_ptr), rows(n_rows), cols(n_cols), value_type(MatrixValueType::Float64) {}

  MatrixView(const float* data_ptr, std::size_t n_rows, std::size_t n_cols)
      : data(data_ptr), rows(n_rows), cols(n_cols), value_type(MatrixValueType::Float32) {}

  float value_float(std::size_t i, std::size_t j) const {
    const std::size_t offset = i * cols + j;
    if (value_type == MatrixValueType::Float32) {
      return static_cast<const float*>(data)[offset];
    }
    return static_cast<float>(static_cast<const double*>(data)[offset]);
  }

  double operator()(std::size_t i, std::size_t j) const {
    const std::size_t offset = i * cols + j;
    if (value_type == MatrixValueType::Float32) {
      return static_cast<double>(static_cast<const float*>(data)[offset]);
    }
    return static_cast<const double*>(data)[offset];
  }
};

struct FoldOptions {
  int folds = 10;
  bool stratified = true;
  std::uint64_t seed = 1;
};

struct KNNOptions {
  FoldOptions cv;
  int k = 10;
  DistanceMetric metric = DistanceMetric::Cosine;
  Backend backend = Backend::CPU;
  KNNIndexType index_type = KNNIndexType::NativeHNSW;
  int ivf_nlist = 0;
  int ivf_nprobe = 0;
  int hnsw_m = 0;
  int hnsw_ef_construction = 0;
  int hnsw_ef_search = 0;
  int hnsw_tune_k = 50;
  double hnsw_target_recall = 0.99;
  int gpu_device = 0;
  int n_threads = 1;
};

struct PLSOptions {
  FoldOptions cv;
  int max_components = 10;
  int fixed_components = 0;
  bool center = true;
  bool scale = true;
  Backend backend = Backend::CPU;
  int gpu_device = 0;
  int n_threads = 1;
};

struct CorePLSLDAOptions {
  FoldOptions cv;
  int max_components = 10;
  int fixed_components = 0;
  bool center = true;
  bool scale = true;
  Backend backend = Backend::CPU;
  int gpu_device = 0;
  int n_threads = 1;

  CorePLSLDAOptions() = default;

  CorePLSLDAOptions& operator=(const PLSOptions& options) {
    cv = options.cv;
    max_components = options.max_components;
    fixed_components = options.fixed_components;
    center = options.center;
    scale = options.scale;
    backend = options.backend;
    gpu_device = options.gpu_device;
    n_threads = options.n_threads;
    return *this;
  }

  operator PLSOptions() const {
    PLSOptions out;
    out.cv = cv;
    out.max_components = max_components;
    out.fixed_components = fixed_components;
    out.center = center;
    out.scale = scale;
    out.backend = backend;
    out.gpu_device = gpu_device;
    out.n_threads = n_threads;
    return out;
  }
};

struct FoldResult {
  int fold = 0;
  int n_train = 0;
  int n_validation = 0;
  double accuracy = 0.0;
};

struct ConfusionMatrix {
  std::vector<int> labels;
  std::vector<int> counts;
  std::size_t n_labels = 0;

  int operator()(std::size_t truth, std::size_t predicted) const {
    return counts[truth * n_labels + predicted];
  }
};

struct KNNParametersUsed {
  Backend backend = Backend::CPU;
  KNNIndexType index_type = KNNIndexType::NativeHNSW;
  DistanceMetric metric = DistanceMetric::Cosine;
  int k = 10;
  int ivf_nlist = 0;
  int ivf_nprobe = 0;
  double ivf_pilot_recall = 0.0;
  int hnsw_m = 0;
  int hnsw_ef_construction = 0;
  int hnsw_ef_search = 0;
  int hnsw_tune_k = 50;
  double hnsw_target_recall = 0.99;
  int gpu_device = 0;
  int n_threads = 1;
};

struct KNNCVResult {
  std::vector<int> predicted_labels;
  std::vector<int> true_labels;
  std::vector<int> fold_assignments;
  std::vector<FoldResult> folds;
  double global_accuracy = 0.0;
  ConfusionMatrix confusion;
  double runtime_seconds = 0.0;
  double peak_memory_mb = 0.0;
  KNNParametersUsed parameters;
};

struct PLSParametersUsed {
  Backend backend = Backend::CPU;
  PLSMode mode = PLSMode::PLS_DA;
  int max_components = 10;
  int selected_components = 1;
  int fixed_components = 0;
  bool center = true;
  bool scale = true;
  int gpu_device = 0;
  int n_threads = 1;
};

struct PLSCVResult {
  std::vector<int> predicted_labels;
  std::vector<int> true_labels;
  std::vector<int> fold_assignments;
  std::vector<FoldResult> folds;
  std::vector<double> accuracy_by_components;
  int selected_components = 1;
  double global_accuracy = 0.0;
  ConfusionMatrix confusion;
  double runtime_seconds = 0.0;
  double peak_memory_mb = 0.0;
  PLSParametersUsed parameters;
};

struct CoreOptions {
  int cycles = 100;
  CoreClassifier classifier = CoreClassifier::PLS_LDA;
  bool shake = false;
  std::uint64_t seed = 1;
  bool auto_class_coarsening = false;
  bool many_to_one_absorption = false;
  bool evolutionary_search = false;
  bool guarded_diversity = false;
  bool adaptive_proposal_size = true;
  CorePLSLDAOptions pls;
  KNNOptions knn;
};

struct CoreResult {
  std::vector<int> clbest;
  std::vector<int> clbest_dirty;
  std::vector<int> cvpredbest;
  double accbest = 0.0;
  double scorebest = 0.0;
  std::vector<double> vect_acc;
  std::vector<double> vect_score;
  int cycles_completed = 0;
  bool success = false;
  double runtime_seconds = 0.0;
  double peak_memory_mb = 0.0;
};

struct NeighborGraph {
  std::vector<int> indices;
  std::vector<float> distances;
  int neighbors = 0;
};

struct GraphClusterOptions {
  Backend backend = Backend::CPU;
  GraphWeightType weight_type = GraphWeightType::Distance;
  DistanceMetric metric = DistanceMetric::Euclidean;
  int k = 30;
  int n_threads = 1;
  int n_iterations = 10;
  int random_walk_steps = 4;
  int target_clusters = 0;
  int gpu_device = 0;
  double prune = 0.0;
  bool mutual = false;
};

struct GraphClusterResult {
  std::vector<int> membership;
  double modularity = 0.0;
  int n_communities = 0;
  int n_vertices = 0;
  int n_edges = 0;
  int target_clusters = 0;
  int target_gap = 0;
  bool target_exact = true;
  double runtime_seconds = 0.0;
  Backend backend = Backend::CPU;
};

struct KODAMAMatrixOptions {
  int runs = 100;
  int cycles = 20;
  int components = 50;
  int landmarks = 10000;
  int splitting = 0;
  int graph_neighbors = 100;
  int n_threads = 1;
  int spatial_cols = 0;
  double spatial_resolution = 0.4;
  bool spatial_graph_mix = false;
  int spatial_constraint_mode = 0;
  std::uint64_t seed = 1234;
  DistanceMetric metric = DistanceMetric::Euclidean;
  Backend backend = Backend::CPU;
  CoreClassifier classifier = CoreClassifier::KNN;
  bool progress = false;
  bool apply_kodama_dissimilarity = true;
  GraphFeatureMode graph_feature_mode = GraphFeatureMode::LaplacianSelfTuning;
  int graph_feature_components = 0;
  int graph_feature_steps = 3;
  std::vector<float> spatial;
  KNNOptions knn;
  CorePLSLDAOptions pls;
};

struct KODAMAMatrixResult {
  std::vector<double> acc;
  std::vector<double> v;
  std::vector<int> res;
  std::vector<int> res_constrain;
  NeighborGraph base_knn;
  NeighborGraph knn;
  int runs = 0;
  int samples = 0;
  int cycles = 0;
  int n_threads = 1;
  Backend backend = Backend::CPU;
  bool gpu_auto_workers = false;
  bool gpu_scheduler_enabled = false;
  int gpu_scheduler_lanes = 0;
  int gpu_sm_count = 0;
  double gpu_free_memory_mb = 0.0;
  double gpu_total_memory_mb = 0.0;
  double gpu_worker_memory_estimate_mb = 0.0;
  double runtime_seconds = 0.0;
  double input_copy_seconds = 0.0;
  double graph_feature_seconds = 0.0;
  double spatial_precompute_seconds = 0.0;
  double graph_seconds = 0.0;
  double spatial_graph_seconds = 0.0;
  double optimization_wall_seconds = 0.0;
  double optimization_sum_seconds = 0.0;
  double dissimilarity_seconds = 0.0;
  double peak_memory_mb = 0.0;
};

struct UMAPOptions {
  int n_components = 2;
  int n_epochs = 200;
  int n_neighbors = 30;
  int negative_sample_rate = 5;
  double learning_rate = 1.0;
  double min_dist = 0.01;
  double repulsion_strength = 1.0;
  int spectral_n_iter = 20;
  int n_threads = 1;
  int seed = 1234;
  int gpu_device = 0;
  UMAPGraphMode graph_mode = UMAPGraphMode::Binary;
  std::vector<float> init;
};

struct OpenTSNEOptions {
  int n_components = 2;
  int n_neighbors = 0;
  double perplexity = 30.0;
  double theta = 0.5;
  int early_exaggeration_iter = 250;
  int n_iter = 500;
  double early_exaggeration = 12.0;
  double exaggeration = 1.0;
  double learning_rate = 0.0;
  bool learning_rate_auto = true;
  double initial_momentum = 0.8;
  double final_momentum = 0.8;
  double min_gain = 0.01;
  double max_step_norm = 5.0;
  int n_threads = 1;
  int seed = 4;
  int gpu_device = 0;
  std::vector<float> init;
};

struct EmbeddingResult {
  std::vector<float> embedding;
  int samples = 0;
  int components = 2;
  Backend backend = Backend::CPU;
  double runtime_seconds = 0.0;
};

struct PCAOptions {
  int n_components = 2;
  bool center = true;
  bool scale = false;
  int oversample = -1;
  int power_iterations = -1;
  int n_threads = 1;
  std::uint64_t seed = 4;
  int gpu_device = 0;
  Backend backend = Backend::CPU;
};

struct PCAResult {
  std::vector<float> scores;
  std::vector<float> loadings;
  std::vector<float> singular_values;
  std::vector<float> sdev;
  std::vector<float> variance;
  std::vector<float> variance_explained;
  std::vector<float> cumulative_variance_explained;
  std::vector<float> center;
  std::vector<float> scale;
  int samples = 0;
  int variables = 0;
  int components = 0;
  int oversample = 0;
  int power_iterations = 0;
  Backend backend = Backend::CPU;
  double total_variance = 0.0;
  double runtime_seconds = 0.0;
};

KNNCVResult KNNCV(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options = KNNOptions()
);

KNNCVResult KNNCV_CPU(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options = KNNOptions()
);

KNNCVResult KNNCV_CUDA(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options = KNNOptions()
);

KNNCVResult KNNCV_METAL(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options = KNNOptions()
);

PLSCVResult PLSDACV(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

PLSCVResult PLSLDACV(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

PLSCVResult PLSDACV_CPU(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

PLSCVResult PLSLDACV_CPU(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

PLSCVResult PLSDACV_CUDA(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

PLSCVResult PLSLDACV_CUDA(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

PLSCVResult PLSLDACV_METAL(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options = PLSOptions()
);

std::vector<int> PLSLDAPredict_CPU(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options = PLSOptions()
);

std::vector<int> PLSLDAPredict_CUDA(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options = PLSOptions()
);

std::vector<int> PLSLDAPredict_METAL(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options = PLSOptions()
);

std::vector<int> PLSLDAPredict(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options = PLSOptions()
);

CoreResult core_cpp(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CorePLSLDA(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CoreKNN(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CorePLSLDA_CPU(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CoreKNN_CPU(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CorePLSLDA_CUDA(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CoreKNN_CUDA(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CorePLSLDA_METAL(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CoreKNN_METAL(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult CoreKNNGraph_CPU(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

CoreResult Core(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options = CoreOptions()
);

KODAMAMatrixResult KODAMAMatrix_CPU(
  MatrixView x,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrix_CUDA(
  MatrixView x,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrix_METAL(
  MatrixView x,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrix(
  MatrixView x,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

std::vector<float> KODAMAGraphFeatures_CPU(
  const NeighborGraph& graph,
  int samples,
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraph_CPU(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraphData_CPU(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraphData_CUDA(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraphData_METAL(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraphData(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraph_CUDA(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraph_METAL(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

KODAMAMatrixResult KODAMAMatrixFromGraph(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

EmbeddingResult KODAMAUMAP_CUDA(
  const NeighborGraph& graph,
  const UMAPOptions& options = UMAPOptions()
);

EmbeddingResult KODAMAUMAP_CPU(
  const NeighborGraph& graph,
  const UMAPOptions& options = UMAPOptions()
);

EmbeddingResult KODAMAOpenTSNE_CUDA(
  const NeighborGraph& graph,
  const OpenTSNEOptions& options = OpenTSNEOptions()
);

EmbeddingResult KODAMAOpenTSNE_CPU(
  const NeighborGraph& graph,
  const OpenTSNEOptions& options = OpenTSNEOptions()
);

PCAResult PCA(
  MatrixView x,
  const PCAOptions& options = PCAOptions()
);

PCAResult PCA_CPU(
  MatrixView x,
  const PCAOptions& options = PCAOptions()
);

PCAResult PCA_CUDA(
  MatrixView x,
  const PCAOptions& options = PCAOptions()
);

PCAResult PCA_METAL(
  MatrixView x,
  const PCAOptions& options = PCAOptions()
);

NeighborGraph KODAMAKNNGraph_CPU(
  MatrixView x,
  const GraphClusterOptions& options = GraphClusterOptions()
);

NeighborGraph KODAMAKNNGraph_CUDA(
  MatrixView x,
  const GraphClusterOptions& options = GraphClusterOptions()
);

NeighborGraph KODAMAKNNGraph_METAL(
  MatrixView x,
  const GraphClusterOptions& options = GraphClusterOptions()
);

NeighborGraph KODAMAKNNGraph(
  MatrixView x,
  const GraphClusterOptions& options = GraphClusterOptions()
);

GraphClusterResult KODAMAGraphCluster_CPU(
  const NeighborGraph& graph,
  int samples,
  const GraphClusterOptions& options = GraphClusterOptions()
);

GraphClusterResult KODAMAGraphCluster(
  const NeighborGraph& graph,
  int samples,
  const GraphClusterOptions& options = GraphClusterOptions()
);

GraphClusterResult KODAMAEmbeddingGraphCluster(
  MatrixView embedding,
  const NeighborGraph& graph,
  const GraphClusterOptions& options = GraphClusterOptions()
);

GraphClusterResult KODAMAEmbeddingCluster(
  MatrixView embedding,
  const GraphClusterOptions& options = GraphClusterOptions()
);

const char* to_string(Backend backend);
const char* to_string(DistanceMetric metric);
const char* to_string(KNNIndexType index_type);
const char* to_string(PLSMode mode);
const char* to_string(CoreClassifier classifier);
const char* to_string(GraphWeightType weight_type);
const char* to_string(GraphFeatureMode mode);
bool MetalAvailable();

}  // namespace kodama
