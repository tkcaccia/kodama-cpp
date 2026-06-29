#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kodama {

enum class Backend {
  Auto,
  CPU,
  CUDA
};

enum class DistanceMetric {
  Cosine,
  InnerProduct,
  Euclidean
};

enum class KNNIndexType {
  FaissIVFFlat,
  FaissHNSWFlat,
  CuvsIVFFlat
};

enum class PLSMode {
  PLS_DA,
  PLS_LDA,
  PLS_CKNN
};

enum class CoreClassifier {
  PLS_LDA,
  KNN
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
  KNNIndexType index_type = KNNIndexType::FaissHNSWFlat;
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
  int cknn_k = 3;
  int cknn_top_m = 20;
  double cknn_tau = 0.2;
  double cknn_alpha = 0.5;
  bool center = true;
  bool scale = true;
  Backend backend = Backend::CPU;
  int gpu_device = 0;
  int n_threads = 1;
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
  KNNIndexType index_type = KNNIndexType::FaissHNSWFlat;
  DistanceMetric metric = DistanceMetric::Cosine;
  int k = 10;
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
  int cknn_k = 3;
  int cknn_top_m = 20;
  double cknn_tau = 0.2;
  double cknn_alpha = 0.5;
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
  int target_classes = 0;
  double class_count_penalty = 0.0;
  double imbalance_penalty = 0.0;
  bool auto_class_coarsening = false;
  PLSOptions pls;
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

struct KODAMAMatrixOptions {
  int runs = 100;
  int cycles = 20;
  int components = 50;
  int landmarks = 10000;
  int splitting = 0;
  int graph_neighbors = 100;
  int n_threads = 1;
  int spatial_cols = 0;
  double spatial_resolution = 0.3;
  bool spatial_graph_mix = false;
  int spatial_constraint_mode = 0;
  std::uint64_t seed = 1234;
  DistanceMetric metric = DistanceMetric::Euclidean;
  Backend backend = Backend::CPU;
  CoreClassifier classifier = CoreClassifier::KNN;
  bool progress = false;
  std::vector<float> spatial;
  KNNOptions knn;
  PLSOptions pls;
};

struct KODAMAMatrixResult {
  std::vector<double> acc;
  std::vector<double> v;
  std::vector<int> res;
  std::vector<int> res_constrain;
  NeighborGraph knn;
  int runs = 0;
  int samples = 0;
  int cycles = 0;
  int n_threads = 1;
  double runtime_seconds = 0.0;
  double peak_memory_mb = 0.0;
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

PLSCVResult PLSCKNNCV(
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

PLSCVResult PLSCKNNCV_CPU(
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

PLSCVResult PLSCKNNCV_CUDA(
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

KODAMAMatrixResult KODAMAMatrix(
  MatrixView x,
  const std::vector<int>& starting_labels = std::vector<int>(),
  const std::vector<int>& constrain = std::vector<int>(),
  const std::vector<int>& fixed = std::vector<int>(),
  const KODAMAMatrixOptions& options = KODAMAMatrixOptions()
);

const char* to_string(Backend backend);
const char* to_string(DistanceMetric metric);
const char* to_string(KNNIndexType index_type);
const char* to_string(PLSMode mode);
const char* to_string(CoreClassifier classifier);

}  // namespace kodama
