#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <set>
#include <vector>

#include "kodama/kodama.hpp"

namespace {

struct ToyData {
  std::vector<double> x;
  std::vector<int> y;
  std::vector<int> constrain;
  int n = 0;
  int p = 0;
};

ToyData make_toy_data() {
  ToyData d;
  const int classes = 3;
  const int n_per_class = 50;
  d.p = 6;
  d.n = classes * n_per_class;
  d.x.assign(static_cast<std::size_t>(d.n * d.p), 0.0);
  d.y.assign(static_cast<std::size_t>(d.n), 0);
  d.constrain.assign(static_cast<std::size_t>(d.n), 0);

  std::mt19937_64 rng(44);
  std::normal_distribution<double> noise(0.0, 0.25);
  for (int c = 0; c < classes; ++c) {
    for (int i = 0; i < n_per_class; ++i) {
      const int row = c * n_per_class + i;
      d.y[static_cast<std::size_t>(row)] = 10 + c;
      d.constrain[static_cast<std::size_t>(row)] = row / 2;
      for (int j = 0; j < d.p; ++j) {
        const double signal = (j == c || j == c + 3) ? 2.5 : -0.5;
        d.x[static_cast<std::size_t>(row * d.p + j)] = signal + noise(rng);
      }
    }
  }
  return d;
}

void check_constrained_folds(const std::vector<int>& constrain, const std::vector<int>& folds) {
  for (std::size_t i = 0; i < constrain.size(); ++i) {
    for (std::size_t j = i + 1; j < constrain.size(); ++j) {
      if (constrain[i] == constrain[j] && folds[i] != folds[j]) {
        throw std::runtime_error("Constraint group was split across folds.");
      }
    }
  }
}

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

double direct_agreement(const std::vector<int>& a, const std::vector<int>& b) {
  require(a.size() == b.size(), "agreement size mismatch.");
  int ok = 0;
  for (std::size_t i = 0; i < a.size(); ++i) ok += a[i] == b[i] ? 1 : 0;
  return static_cast<double>(ok) / static_cast<double>(a.size());
}

std::vector<int> make_noisy_labels(const std::vector<int>& labels) {
  std::vector<int> out = labels;
  const std::vector<int> classes = {10, 11, 12};
  for (std::size_t i = 0; i < out.size(); i += 11) {
    const auto it = std::find(classes.begin(), classes.end(), out[i]);
    const int pos = it == classes.end() ? 0 : static_cast<int>(it - classes.begin());
    out[i] = classes[static_cast<std::size_t>((pos + 1) % static_cast<int>(classes.size()))];
  }
  return out;
}

void check_pls_result(
  const kodama::PLSCVResult& result,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  int expected_components
) {
  require(result.predicted_labels.size() == labels.size(), "PLSCV prediction size mismatch.");
  require(result.fold_assignments.size() == labels.size(), "PLSCV fold size mismatch.");
  check_constrained_folds(constrain, result.fold_assignments);
  require(result.selected_components == expected_components, "PLSCV did not report requested component count.");
  require(result.parameters.selected_components == expected_components, "PLSCV parameters did not report requested component count.");
}

}  // namespace

int main() {
  ToyData d = make_toy_data();
  kodama::MatrixView view{d.x.data(), static_cast<std::size_t>(d.n), static_cast<std::size_t>(d.p)};
  std::vector<float> xf(d.x.begin(), d.x.end());
  kodama::MatrixView fview{xf.data(), static_cast<std::size_t>(d.n), static_cast<std::size_t>(d.p)};
  require(fview.value_type == kodama::MatrixValueType::Float32, "Float32 MatrixView did not record float32 storage.");
  require(std::abs(fview(0, 0) - static_cast<double>(xf[0])) < 1e-6, "Float32 MatrixView returned unexpected value.");

  kodama::KNNOptions knn;
  knn.cv.folds = 5;
  knn.cv.seed = 1;
  knn.k = 7;
  knn.metric = kodama::DistanceMetric::Cosine;
  kodama::KNNCVResult kres = kodama::KNNCV(view, d.y, d.constrain, knn);
  require(kres.predicted_labels.size() == d.y.size(), "KNNCV prediction size mismatch.");
  require(kres.fold_assignments.size() == d.y.size(), "KNNCV fold size mismatch.");
  check_constrained_folds(d.constrain, kres.fold_assignments);
  require(kres.global_accuracy > 0.95, "KNNCV accuracy unexpectedly low.");
  require(kres.confusion.n_labels == 3, "KNNCV confusion matrix label count mismatch.");
  require(kres.parameters.index_type == kodama::KNNIndexType::FaissHNSWFlat, "KNNCV CPU did not use FAISS HNSW by default.");
  require(kres.parameters.hnsw_tune_k == 50, "KNNCV CPU HNSW tune k was not 50.");
  require(std::abs(kres.parameters.hnsw_target_recall - 0.99) < 1e-12, "KNNCV CPU HNSW target recall was not 0.99.");
  require(kres.parameters.hnsw_m > 0, "KNNCV CPU HNSW m was not recorded.");
  require(kres.parameters.hnsw_ef_construction >= kres.parameters.hnsw_m, "KNNCV CPU HNSW efConstruction was invalid.");
  require(kres.parameters.hnsw_ef_search >= 50, "KNNCV CPU HNSW efSearch was invalid.");

  kodama::KNNCVResult fkres = kodama::KNNCV_CPU(fview, d.y, d.constrain, knn);
  require(fkres.predicted_labels.size() == d.y.size(), "Float32 KNNCV prediction size mismatch.");
  check_constrained_folds(d.constrain, fkres.fold_assignments);
  require(fkres.global_accuracy > 0.95, "Float32 KNNCV accuracy unexpectedly low.");

  kodama::KNNCVResult fdispatch_kres = kodama::KNNCV(fview, d.y, d.constrain, knn);
  require(fdispatch_kres.predicted_labels.size() == d.y.size(), "Float32 generic KNNCV prediction size mismatch.");
  check_constrained_folds(d.constrain, fdispatch_kres.fold_assignments);
  require(fdispatch_kres.global_accuracy > 0.95, "Float32 generic KNNCV accuracy unexpectedly low.");

#if defined(KODAMA_ENABLE_CUDA)
  kodama::KNNOptions cuda_knn = knn;
  cuda_knn.backend = kodama::Backend::CUDA;
  cuda_knn.ivf_nlist = 8;
  cuda_knn.ivf_nprobe = 4;
  kodama::KNNCVResult cuda_kres = kodama::KNNCV_CUDA(view, d.y, d.constrain, cuda_knn);
  require(cuda_kres.parameters.backend == kodama::Backend::CUDA, "CUDA KNNCV did not report CUDA backend.");
  require(cuda_kres.predicted_labels.size() == d.y.size(), "CUDA KNNCV prediction size mismatch.");
  require(cuda_kres.fold_assignments.size() == d.y.size(), "CUDA KNNCV fold size mismatch.");
  check_constrained_folds(d.constrain, cuda_kres.fold_assignments);
  require(cuda_kres.global_accuracy > 0.95, "CUDA KNNCV accuracy unexpectedly low.");
  kodama::KNNCVResult float_cuda_kres = kodama::KNNCV_CUDA(fview, d.y, d.constrain, cuda_knn);
  require(float_cuda_kres.parameters.backend == kodama::Backend::CUDA, "Float32 CUDA KNNCV did not report CUDA backend.");
  require(float_cuda_kres.predicted_labels.size() == d.y.size(), "Float32 CUDA KNNCV prediction size mismatch.");
  require(float_cuda_kres.fold_assignments.size() == d.y.size(), "Float32 CUDA KNNCV fold size mismatch.");
  check_constrained_folds(d.constrain, float_cuda_kres.fold_assignments);
  require(float_cuda_kres.global_accuracy > 0.95, "Float32 CUDA KNNCV accuracy unexpectedly low.");
#endif

  kodama::PLSOptions pls;
  pls.cv.folds = 5;
  pls.cv.seed = 1;
  pls.max_components = 4;
  kodama::PLSCVResult pres = kodama::PLSDACV(view, d.y, d.constrain, pls);
  check_pls_result(pres, d.y, d.constrain, 4);
  require(pres.accuracy_by_components.size() == 4, "PLSCV component accuracy size mismatch.");
  require(pres.global_accuracy > 0.90, "PLS-DA accuracy unexpectedly low.");

  kodama::PLSCVResult fpres = kodama::PLSDACV_CPU(fview, d.y, d.constrain, pls);
  check_pls_result(fpres, d.y, d.constrain, 4);
  require(fpres.accuracy_by_components.size() == 4, "Float32 PLSCV component accuracy size mismatch.");
  require(fpres.global_accuracy > 0.90, "Float32 PLS-DA accuracy unexpectedly low.");

  kodama::PLSCVResult fdispatch_pres = kodama::PLSDACV(fview, d.y, d.constrain, pls);
  check_pls_result(fdispatch_pres, d.y, d.constrain, 4);
  require(fdispatch_pres.global_accuracy > 0.90, "Float32 generic PLS-DA accuracy unexpectedly low.");

  kodama::PLSCVResult lres = kodama::PLSLDACV(view, d.y, d.constrain, pls);
  check_pls_result(lres, d.y, d.constrain, 4);
  require(lres.global_accuracy > 0.60, "PLS-LDA accuracy unexpectedly low.");

  kodama::PLSCVResult flres = kodama::PLSLDACV_CPU(fview, d.y, d.constrain, pls);
  check_pls_result(flres, d.y, d.constrain, 4);
  require(flres.global_accuracy > 0.60, "Float32 PLS-LDA accuracy unexpectedly low.");

  kodama::PLSCVResult fdispatch_lres = kodama::PLSLDACV(fview, d.y, d.constrain, pls);
  check_pls_result(fdispatch_lres, d.y, d.constrain, 4);
  require(fdispatch_lres.global_accuracy > 0.60, "Float32 generic PLS-LDA accuracy unexpectedly low.");

  kodama::PLSCVResult ckres = kodama::PLSCKNNCV(view, d.y, d.constrain, pls);
  check_pls_result(ckres, d.y, d.constrain, 4);
  require(ckres.parameters.mode == kodama::PLSMode::PLS_CKNN, "PLS-cKNN did not report cKNN mode.");
  require(ckres.global_accuracy > 0.60, "PLS-cKNN accuracy unexpectedly low.");

  kodama::PLSCVResult fckres = kodama::PLSCKNNCV_CPU(fview, d.y, d.constrain, pls);
  check_pls_result(fckres, d.y, d.constrain, 4);
  require(fckres.parameters.mode == kodama::PLSMode::PLS_CKNN, "Float32 PLS-cKNN did not report cKNN mode.");
  require(fckres.global_accuracy > 0.60, "Float32 PLS-cKNN accuracy unexpectedly low.");

  kodama::PLSCVResult fdispatch_ckres = kodama::PLSCKNNCV(fview, d.y, d.constrain, pls);
  check_pls_result(fdispatch_ckres, d.y, d.constrain, 4);
  require(fdispatch_ckres.parameters.mode == kodama::PLSMode::PLS_CKNN, "Float32 generic PLS-cKNN did not report cKNN mode.");
  require(fdispatch_ckres.global_accuracy > 0.60, "Float32 generic PLS-cKNN accuracy unexpectedly low.");

  std::vector<int> noisy = make_noisy_labels(d.y);
  const double initial_agreement = direct_agreement(noisy, d.y);
  std::vector<int> fixed(static_cast<std::size_t>(d.n), 0);

  kodama::CoreOptions core_pls;
  core_pls.cycles = 12;
  core_pls.seed = 17;
  core_pls.classifier = kodama::CoreClassifier::PLS_LDA;
  core_pls.pls = pls;
  core_pls.pls.cv.seed = 17;
  kodama::PLSCVResult noisy_lres = kodama::PLSLDACV(view, noisy, d.constrain, core_pls.pls);
  const double initial_pls_acc = noisy_lres.global_accuracy;
  kodama::CoreResult core_lres = kodama::CorePLSLDA_CPU(view, noisy, d.constrain, fixed, core_pls);
  require(core_lres.clbest.size() == noisy.size(), "Core PLS-LDA clbest size mismatch.");
  require(core_lres.cvpredbest.size() == noisy.size(), "Core PLS-LDA cvpredbest size mismatch.");
  require(core_lres.vect_acc.size() == static_cast<std::size_t>(core_pls.cycles), "Core PLS-LDA vect_acc size mismatch.");
  require(core_lres.vect_score.size() == static_cast<std::size_t>(core_pls.cycles), "Core PLS-LDA vect_score size mismatch.");
  require(core_lres.cycles_completed >= 1, "Core PLS-LDA did not run any cycles.");
  require(core_lres.accbest >= initial_pls_acc, "Core PLS-LDA decreased best CV accuracy.");
  require(std::abs(core_lres.scorebest - core_lres.accbest) < 1e-12, "Core PLS-LDA default score should match accuracy.");
  require(direct_agreement(core_lres.clbest, d.y) >= initial_agreement, "Core PLS-LDA reduced label agreement.");

  kodama::PLSCVResult float_noisy_lres = kodama::PLSLDACV_CPU(fview, noisy, d.constrain, core_pls.pls);
  const double float_initial_pls_acc = float_noisy_lres.global_accuracy;
  kodama::CoreResult fcore_lres = kodama::CorePLSLDA_CPU(fview, noisy, d.constrain, fixed, core_pls);
  require(fcore_lres.clbest.size() == noisy.size(), "Float32 Core PLS-LDA clbest size mismatch.");
  require(fcore_lres.cvpredbest.size() == noisy.size(), "Float32 Core PLS-LDA cvpredbest size mismatch.");
  require(fcore_lres.vect_acc.size() == static_cast<std::size_t>(core_pls.cycles), "Float32 Core PLS-LDA vect_acc size mismatch.");
  require(fcore_lres.vect_score.size() == static_cast<std::size_t>(core_pls.cycles), "Float32 Core PLS-LDA vect_score size mismatch.");
  require(fcore_lres.cycles_completed >= 1, "Float32 Core PLS-LDA did not run any cycles.");
  require(fcore_lres.accbest >= float_initial_pls_acc, "Float32 Core PLS-LDA decreased best CV accuracy.");
  require(std::abs(fcore_lres.scorebest - fcore_lres.accbest) < 1e-12, "Float32 Core PLS-LDA default score should match accuracy.");
  require(direct_agreement(fcore_lres.clbest, d.y) >= initial_agreement, "Float32 Core PLS-LDA reduced label agreement.");

  kodama::CoreResult fdispatch_core_lres = kodama::CorePLSLDA(fview, noisy, d.constrain, fixed, core_pls);
  require(fdispatch_core_lres.clbest.size() == noisy.size(), "Float32 generic Core PLS-LDA clbest size mismatch.");
  require(fdispatch_core_lres.cycles_completed >= 1, "Float32 generic Core PLS-LDA did not run any cycles.");
  require(fdispatch_core_lres.accbest >= float_initial_pls_acc, "Float32 generic Core PLS-LDA decreased best CV accuracy.");

  kodama::CoreResult fcore_cpp_lres = kodama::core_cpp(fview, noisy, d.constrain, fixed, core_pls);
  require(fcore_cpp_lres.clbest.size() == noisy.size(), "Float32 core_cpp PLS-LDA clbest size mismatch.");
  require(fcore_cpp_lres.cycles_completed >= 1, "Float32 core_cpp PLS-LDA did not run any cycles.");
  require(fcore_cpp_lres.accbest >= float_initial_pls_acc, "Float32 core_cpp PLS-LDA decreased best CV accuracy.");

  kodama::CoreOptions core_knn;
  core_knn.cycles = 8;
  core_knn.seed = 19;
  core_knn.classifier = kodama::CoreClassifier::KNN;
  core_knn.knn = knn;
  core_knn.knn.cv.seed = 19;
  kodama::KNNCVResult noisy_kres = kodama::KNNCV(view, noisy, d.constrain, core_knn.knn);
  const double initial_knn_acc = noisy_kres.global_accuracy;
  kodama::CoreResult core_kres = kodama::CoreKNN_CPU(view, noisy, d.constrain, fixed, core_knn);
  require(core_kres.clbest.size() == noisy.size(), "Core KNN clbest size mismatch.");
  require(core_kres.cvpredbest.size() == noisy.size(), "Core KNN cvpredbest size mismatch.");
  require(core_kres.vect_acc.size() == static_cast<std::size_t>(core_knn.cycles), "Core KNN vect_acc size mismatch.");
  require(core_kres.vect_score.size() == static_cast<std::size_t>(core_knn.cycles), "Core KNN vect_score size mismatch.");
  require(core_kres.cycles_completed >= 1, "Core KNN did not run any cycles.");
  require(core_kres.accbest >= initial_knn_acc, "Core KNN decreased best CV accuracy.");
  require(std::abs(core_kres.scorebest - core_kres.accbest) < 1e-12, "Core KNN default score should match accuracy.");
  require(direct_agreement(core_kres.clbest, d.y) >= initial_agreement, "Core KNN reduced label agreement.");

  kodama::CoreResult fcore_kres = kodama::CoreKNN_CPU(fview, noisy, d.constrain, fixed, core_knn);
  require(fcore_kres.clbest.size() == noisy.size(), "Float32 Core KNN clbest size mismatch.");
  require(fcore_kres.cvpredbest.size() == noisy.size(), "Float32 Core KNN cvpredbest size mismatch.");
  require(fcore_kres.cycles_completed >= 1, "Float32 Core KNN did not run any cycles.");
  require(fcore_kres.accbest >= initial_knn_acc, "Float32 Core KNN decreased best CV accuracy.");

  kodama::CoreResult fdispatch_core_kres = kodama::CoreKNN(fview, noisy, d.constrain, fixed, core_knn);
  require(fdispatch_core_kres.clbest.size() == noisy.size(), "Float32 generic Core KNN clbest size mismatch.");
  require(fdispatch_core_kres.cycles_completed >= 1, "Float32 generic Core KNN did not run any cycles.");
  require(fdispatch_core_kres.accbest >= initial_knn_acc, "Float32 generic Core KNN decreased best CV accuracy.");

  kodama::CoreResult fcore_cpp_kres = kodama::Core(fview, noisy, d.constrain, fixed, core_knn);
  require(fcore_cpp_kres.clbest.size() == noisy.size(), "Float32 Core dispatcher KNN clbest size mismatch.");
  require(fcore_cpp_kres.cycles_completed >= 1, "Float32 Core dispatcher KNN did not run any cycles.");
  require(fcore_cpp_kres.accbest >= initial_knn_acc, "Float32 Core dispatcher KNN decreased best CV accuracy.");

  kodama::CoreOptions penalized_knn = core_knn;
  penalized_knn.target_classes = 3;
  penalized_knn.class_count_penalty = 0.01;
  penalized_knn.imbalance_penalty = 0.01;
  penalized_knn.auto_class_coarsening = true;
  kodama::CoreResult penalized_kres = kodama::CoreKNN_CPU(view, noisy, d.constrain, fixed, penalized_knn);
  require(penalized_kres.clbest.size() == noisy.size(), "Penalized Core KNN clbest size mismatch.");
  require(penalized_kres.vect_score.size() == static_cast<std::size_t>(penalized_knn.cycles), "Penalized Core KNN vect_score size mismatch.");
  require(penalized_kres.scorebest <= penalized_kres.accbest + 1e-12, "Penalized Core KNN score should not exceed accuracy.");

  kodama::KODAMAMatrixOptions km_options;
  km_options.runs = 2;
  km_options.cycles = 3;
  km_options.landmarks = 45;
  km_options.splitting = 6;
  km_options.n_threads = 2;
  km_options.seed = 23;
  km_options.metric = kodama::DistanceMetric::Euclidean;
  km_options.classifier = kodama::CoreClassifier::KNN;
  km_options.knn.k = 10;
  km_options.knn.n_threads = 1;
  kodama::KODAMAMatrixResult km_res = kodama::KODAMAMatrix_CPU(fview, std::vector<int>(), std::vector<int>(), fixed, km_options);
  require(km_res.runs == km_options.runs, "KODAMAMatrix run count mismatch.");
  require(km_res.samples == d.n, "KODAMAMatrix sample count mismatch.");
  require(km_res.cycles == km_options.cycles, "KODAMAMatrix cycle count mismatch.");
  require(km_res.acc.size() == static_cast<std::size_t>(km_options.runs), "KODAMAMatrix acc size mismatch.");
  require(km_res.v.size() == static_cast<std::size_t>(km_options.runs * km_options.cycles), "KODAMAMatrix trace size mismatch.");
  require(km_res.res.size() == static_cast<std::size_t>(km_options.runs * d.n), "KODAMAMatrix result label size mismatch.");
  require(km_res.res_constrain.size() == static_cast<std::size_t>(km_options.runs * d.n), "KODAMAMatrix constrain size mismatch.");
  require(km_res.knn.neighbors > 0, "KODAMAMatrix HNSW neighbor count was not recorded.");
  require(km_res.knn.indices.size() == static_cast<std::size_t>(d.n * km_res.knn.neighbors), "KODAMAMatrix neighbor index size mismatch.");
  require(km_res.knn.distances.size() == km_res.knn.indices.size(), "KODAMAMatrix neighbor distance size mismatch.");
  require(km_res.knn.indices.front() >= 1, "KODAMAMatrix neighbor indices should be one-based for R compatibility.");

#if defined(KODAMA_ENABLE_CUDA)
  kodama::PLSOptions cuda_pls = pls;
  cuda_pls.backend = kodama::Backend::CUDA;
  kodama::PLSCVResult cuda_pres = kodama::PLSDACV_CUDA(view, d.y, d.constrain, cuda_pls);
  require(cuda_pres.parameters.backend == kodama::Backend::CUDA, "CUDA PLS-DA did not report CUDA backend.");
  check_pls_result(cuda_pres, d.y, d.constrain, 4);
  require(cuda_pres.global_accuracy > 0.90, "CUDA PLS-DA accuracy unexpectedly low.");
  kodama::PLSCVResult float_cuda_pres = kodama::PLSDACV_CUDA(fview, d.y, d.constrain, cuda_pls);
  require(float_cuda_pres.parameters.backend == kodama::Backend::CUDA, "Float32 CUDA PLS-DA did not report CUDA backend.");
  check_pls_result(float_cuda_pres, d.y, d.constrain, 4);
  require(float_cuda_pres.global_accuracy > 0.90, "Float32 CUDA PLS-DA accuracy unexpectedly low.");
  kodama::PLSCVResult cuda_lres = kodama::PLSLDACV_CUDA(view, d.y, d.constrain, cuda_pls);
  require(cuda_lres.parameters.backend == kodama::Backend::CUDA, "CUDA PLS-LDA did not report CUDA backend.");
  check_pls_result(cuda_lres, d.y, d.constrain, 4);
  require(cuda_lres.global_accuracy > 0.60, "CUDA PLS-LDA accuracy unexpectedly low.");
  kodama::PLSCVResult float_cuda_lres = kodama::PLSLDACV_CUDA(fview, d.y, d.constrain, cuda_pls);
  require(float_cuda_lres.parameters.backend == kodama::Backend::CUDA, "Float32 CUDA PLS-LDA did not report CUDA backend.");
  check_pls_result(float_cuda_lres, d.y, d.constrain, 4);
  require(float_cuda_lres.global_accuracy > 0.60, "Float32 CUDA PLS-LDA accuracy unexpectedly low.");
  kodama::PLSCVResult cuda_ckres = kodama::PLSCKNNCV_CUDA(view, d.y, d.constrain, cuda_pls);
  require(cuda_ckres.parameters.backend == kodama::Backend::CUDA, "CUDA PLS-cKNN did not report CUDA backend.");
  require(cuda_ckres.parameters.mode == kodama::PLSMode::PLS_CKNN, "CUDA PLS-cKNN did not report cKNN mode.");
  check_pls_result(cuda_ckres, d.y, d.constrain, 4);
  require(cuda_ckres.global_accuracy > 0.60, "CUDA PLS-cKNN accuracy unexpectedly low.");
  kodama::PLSCVResult float_cuda_ckres = kodama::PLSCKNNCV_CUDA(fview, d.y, d.constrain, cuda_pls);
  require(float_cuda_ckres.parameters.backend == kodama::Backend::CUDA, "Float32 CUDA PLS-cKNN did not report CUDA backend.");
  require(float_cuda_ckres.parameters.mode == kodama::PLSMode::PLS_CKNN, "Float32 CUDA PLS-cKNN did not report cKNN mode.");
  check_pls_result(float_cuda_ckres, d.y, d.constrain, 4);
  require(float_cuda_ckres.global_accuracy > 0.60, "Float32 CUDA PLS-cKNN accuracy unexpectedly low.");

  kodama::CoreOptions cuda_core_pls = core_pls;
  cuda_core_pls.pls.backend = kodama::Backend::CUDA;
  kodama::CoreResult cuda_core_lres = kodama::CorePLSLDA_CUDA(fview, noisy, d.constrain, fixed, cuda_core_pls);
  require(cuda_core_lres.clbest.size() == noisy.size(), "Float32 CUDA Core PLS-LDA clbest size mismatch.");
  require(cuda_core_lres.cycles_completed >= 1, "Float32 CUDA Core PLS-LDA did not run any cycles.");
  require(cuda_core_lres.accbest >= float_initial_pls_acc, "Float32 CUDA Core PLS-LDA decreased best CV accuracy.");

  kodama::CoreOptions cuda_core_knn = core_knn;
  cuda_core_knn.knn.backend = kodama::Backend::CUDA;
  cuda_core_knn.knn.ivf_nlist = 8;
  cuda_core_knn.knn.ivf_nprobe = 4;
  kodama::CoreResult cuda_core_kres = kodama::CoreKNN_CUDA(fview, noisy, d.constrain, fixed, cuda_core_knn);
  require(cuda_core_kres.clbest.size() == noisy.size(), "Float32 CUDA Core KNN clbest size mismatch.");
  require(cuda_core_kres.cycles_completed >= 1, "Float32 CUDA Core KNN did not run any cycles.");
  require(cuda_core_kres.accbest >= initial_knn_acc, "Float32 CUDA Core KNN decreased best CV accuracy.");
#endif

  std::cout << "All kodama-cpp CV tests passed.\n";
  return 0;
}
