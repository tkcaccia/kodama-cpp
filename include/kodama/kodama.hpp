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
  InnerProduct
};

enum class KNNIndexType {
  FaissIVFFlat,
  CuvsIVFFlat
};

enum class PLSMode {
  PLS_DA,
  PLS_LDA
};

struct MatrixView {
  const double* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;

  double operator()(std::size_t i, std::size_t j) const {
    return data[i * cols + j];
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
  KNNIndexType index_type = KNNIndexType::FaissIVFFlat;
  int ivf_nlist = 0;
  int ivf_nprobe = 0;
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
  KNNIndexType index_type = KNNIndexType::FaissIVFFlat;
  DistanceMetric metric = DistanceMetric::Cosine;
  int k = 10;
  int ivf_nlist = 0;
  int ivf_nprobe = 0;
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

const char* to_string(Backend backend);
const char* to_string(DistanceMetric metric);
const char* to_string(KNNIndexType index_type);
const char* to_string(PLSMode mode);

}  // namespace kodama
