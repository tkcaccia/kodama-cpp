#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <set>
#include <utility>
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

void check_spatial_grid_graph() {
  const int n = 10000;
  const int p = 2;
  const int k = 3;
  std::vector<float> x(static_cast<std::size_t>(n) * p, 0.0f);
  const std::vector<float> near_points = {
    0.0f, 0.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    2.0f, 0.0f,
    2.0f, 2.0f,
    5.0f, 5.0f,
    6.0f, 5.0f,
    5.0f, 6.0f
  };
  std::copy(near_points.begin(), near_points.end(), x.begin());
  for (int i = 8; i < n; ++i) {
    x[static_cast<std::size_t>(i) * p] = 1000.0f + static_cast<float>(i);
    x[static_cast<std::size_t>(i) * p + 1] = 2000.0f + static_cast<float>(i % 97);
  }
  kodama::MatrixView view{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)};
  kodama::GraphClusterOptions options;
  options.k = k;
  options.metric = kodama::DistanceMetric::Euclidean;
  options.n_threads = 2;

  const kodama::NeighborGraph graph = kodama::KODAMAKNNGraph_CPU(view, options);
  require(graph.neighbors == k, "Spatial grid graph neighbor count mismatch.");
  require(graph.indices.size() == static_cast<std::size_t>(n * k), "Spatial grid graph index size mismatch.");
  require(graph.distances.size() == graph.indices.size(), "Spatial grid graph distance size mismatch.");

  kodama::NeighborGraph cluster_graph;
  cluster_graph.neighbors = 2;
  cluster_graph.indices = {
    2, 3, 1, 3, 1, 2,
    5, 6, 4, 6, 4, 5
  };
  cluster_graph.distances.assign(cluster_graph.indices.size(), 0.1f);
  kodama::GraphClusterOptions cluster_options;
  cluster_options.backend = kodama::Backend::CPU;
  cluster_options.random_walk_steps = 2;
  cluster_options.n_iterations = 10;
  const kodama::GraphClusterResult clustered = kodama::KODAMAGraphCluster_CPU(
    cluster_graph,
    6,
    cluster_options
  );
  require(clustered.membership.size() == 6, "Random-walk clustering membership size mismatch.");
  require(clustered.n_communities == 2, "Random-walk clustering did not preserve two disconnected groups.");
  require(clustered.backend == kodama::Backend::CPU, "Random-walk clustering did not report CPU backend.");

  for (int i = 0; i < 8; ++i) {
    std::vector<std::pair<float, int>> expected;
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      const float dx = x[static_cast<std::size_t>(i) * p] - x[static_cast<std::size_t>(j) * p];
      const float dy = x[static_cast<std::size_t>(i) * p + 1] - x[static_cast<std::size_t>(j) * p + 1];
      expected.emplace_back(dx * dx + dy * dy, j);
    }
    std::sort(expected.begin(), expected.end());
    for (int j = 0; j < k; ++j) {
      const std::size_t offset = static_cast<std::size_t>(i) * k + j;
      require(graph.indices[offset] == expected[static_cast<std::size_t>(j)].second + 1, "Spatial grid graph nearest-neighbor order mismatch.");
      const float distance = std::sqrt(expected[static_cast<std::size_t>(j)].first);
      require(std::abs(graph.distances[offset] - distance) < 1e-5f, "Spatial grid graph distance mismatch.");
    }
  }

#if defined(KODAMA_ENABLE_CUDA)
  kodama::GraphClusterOptions cuda_options = options;
  cuda_options.backend = kodama::Backend::CUDA;
  const kodama::NeighborGraph cuda_graph = kodama::KODAMAKNNGraph_CUDA(view, cuda_options);
  require(cuda_graph.indices == graph.indices, "CUDA spatial grid graph indices differ from CPU.");
  require(cuda_graph.distances.size() == graph.distances.size(), "CUDA spatial grid graph distance size mismatch.");
  for (std::size_t i = 0; i < graph.distances.size(); ++i) {
    require(std::abs(cuda_graph.distances[i] - graph.distances[i]) < 1e-5f, "CUDA spatial grid graph distances differ from CPU.");
  }
  bool rejected_cuda_clustering = false;
  try {
    (void)kodama::KODAMAGraphCluster(cluster_graph, 6, cuda_options);
  } catch (const std::runtime_error&) {
    rejected_cuda_clustering = true;
  }
  require(rejected_cuda_clustering, "Random-walk clustering silently mixed CUDA and CPU backends.");
#endif
}

}  // namespace

int main() {
  check_spatial_grid_graph();

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
  require(kres.parameters.index_type == kodama::KNNIndexType::NativeHNSW, "KNNCV CPU did not use native HNSW by default.");
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
  require(cuda_kres.parameters.index_type == kodama::KNNIndexType::CudaIVFFlat, "CUDA KNNCV did not report native IVF-Flat search.");
  require(cuda_kres.parameters.ivf_nlist == 8, "CUDA KNNCV did not report the requested IVF list count.");
  require(cuda_kres.parameters.ivf_nprobe == 4, "CUDA KNNCV did not report the requested IVF probe count.");
  require(cuda_kres.parameters.ivf_pilot_recall > 0.0, "CUDA KNNCV did not report pilot recall.");
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

  kodama::KNNOptions exact_cuda_knn = cuda_knn;
  exact_cuda_knn.index_type = kodama::KNNIndexType::CudaExact;
  kodama::KNNCVResult exact_cuda_kres = kodama::KNNCV_CUDA(fview, d.y, d.constrain, exact_cuda_knn);
  require(exact_cuda_kres.parameters.index_type == kodama::KNNIndexType::CudaExact, "CUDA KNNCV did not report native exact search.");
  require(exact_cuda_kres.global_accuracy > 0.95, "Exact CUDA KNNCV accuracy unexpectedly low.");
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

  kodama::CoreOptions coarsened_knn = core_knn;
  coarsened_knn.auto_class_coarsening = true;
  kodama::CoreResult coarsened_kres = kodama::CoreKNN_CPU(view, noisy, d.constrain, fixed, coarsened_knn);
  require(coarsened_kres.clbest.size() == noisy.size(), "Coarsened Core KNN clbest size mismatch.");
  require(coarsened_kres.vect_score.size() == static_cast<std::size_t>(coarsened_knn.cycles), "Coarsened Core KNN vect_score size mismatch.");
  require(coarsened_kres.scorebest <= coarsened_kres.accbest + 1e-12, "Coarsened Core KNN score should not exceed accuracy.");

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

  kodama::UMAPOptions umap_options;
  umap_options.n_neighbors = std::min(12, km_res.knn.neighbors);
  umap_options.n_epochs = 5;
  umap_options.n_threads = 2;
  umap_options.seed = 9;
  kodama::EmbeddingResult umap_res = kodama::KODAMAUMAP_CPU(km_res.knn, umap_options);
  require(umap_res.samples == d.n, "CPU UMAP sample count mismatch.");
  require(umap_res.components == 2, "CPU UMAP component count mismatch.");
  require(umap_res.embedding.size() == static_cast<std::size_t>(d.n * 2), "CPU UMAP embedding size mismatch.");
  for (float value : umap_res.embedding) require(std::isfinite(value), "CPU UMAP produced a non-finite value.");

  kodama::OpenTSNEOptions tsne_options;
  tsne_options.n_neighbors = std::min(12, km_res.knn.neighbors);
  tsne_options.perplexity = 3.0;
  tsne_options.early_exaggeration_iter = 2;
  tsne_options.n_iter = 2;
  tsne_options.n_threads = 2;
  tsne_options.seed = 11;
  kodama::EmbeddingResult tsne_res = kodama::KODAMAOpenTSNE_CPU(km_res.knn, tsne_options);
  require(tsne_res.samples == d.n, "CPU openTSNE sample count mismatch.");
  require(tsne_res.components == 2, "CPU openTSNE component count mismatch.");
  require(tsne_res.embedding.size() == static_cast<std::size_t>(d.n * 2), "CPU openTSNE embedding size mismatch.");
  for (float value : tsne_res.embedding) require(std::isfinite(value), "CPU openTSNE produced a non-finite value.");

  kodama::KODAMAMatrixOptions km_pls_options = km_options;
  km_pls_options.classifier = kodama::CoreClassifier::PLS_LDA;
  km_pls_options.components = 3;
  km_pls_options.pls.n_threads = 1;
  kodama::KODAMAMatrixResult km_pls_res = kodama::KODAMAMatrix_CPU(fview, std::vector<int>(), std::vector<int>(), fixed, km_pls_options);
  require(km_pls_res.runs == km_pls_options.runs, "PLS-LDA KODAMAMatrix run count mismatch.");
  require(km_pls_res.samples == d.n, "PLS-LDA KODAMAMatrix sample count mismatch.");
  require(km_pls_res.cycles == km_pls_options.cycles, "PLS-LDA KODAMAMatrix cycle count mismatch.");
  require(km_pls_res.acc.size() == static_cast<std::size_t>(km_pls_options.runs), "PLS-LDA KODAMAMatrix acc size mismatch.");
  require(km_pls_res.v.size() == static_cast<std::size_t>(km_pls_options.runs * km_pls_options.cycles), "PLS-LDA KODAMAMatrix trace size mismatch.");
  require(km_pls_res.res.size() == static_cast<std::size_t>(km_pls_options.runs * d.n), "PLS-LDA KODAMAMatrix result label size mismatch.");
  require(km_pls_res.n_threads == km_pls_options.n_threads, "PLS-LDA KODAMAMatrix CPU worker count mismatch.");
  require(!km_pls_res.gpu_scheduler_enabled, "PLS-LDA KODAMAMatrix CPU unexpectedly enabled the CUDA scheduler.");

  kodama::KODAMAMatrixOptions km_graph_options = km_options;
  km_graph_options.runs = 1;
  km_graph_options.cycles = 2;
  km_graph_options.landmarks = 40;
  km_graph_options.graph_neighbors = std::min(20, km_res.base_knn.neighbors);
  km_graph_options.knn.k = 5;
  km_graph_options.apply_kodama_dissimilarity = true;
  kodama::KODAMAMatrixResult km_graph_knn_res = kodama::KODAMAMatrixFromGraph_CPU(
    km_res.base_knn,
    d.n,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    km_graph_options
  );
  require(km_graph_knn_res.samples == d.n, "Graph-input KODAMAMatrix KNN sample count mismatch.");
  require(km_graph_knn_res.res.size() == static_cast<std::size_t>(d.n), "Graph-input KNN result size mismatch.");
  require(km_graph_knn_res.graph_feature_seconds >= 0.0, "Graph-input KNN feature timing missing.");
  require(km_graph_knn_res.graph_seconds >= 0.0, "Graph-input KNN graph timing missing.");
  require(km_graph_knn_res.knn.indices.size() == static_cast<std::size_t>(d.n * km_graph_knn_res.knn.neighbors), "Graph-input KNN graph size mismatch.");

  kodama::KODAMAMatrixResult km_graph_data_knn_res = kodama::KODAMAMatrixFromGraphData_CPU(
    fview,
    km_res.base_knn,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    km_graph_options
  );
  require(km_graph_data_knn_res.samples == d.n, "Graph+data KODAMAMatrix KNN sample count mismatch.");
  require(km_graph_data_knn_res.res.size() == static_cast<std::size_t>(d.n), "Graph+data KNN result size mismatch.");
  require(km_graph_data_knn_res.graph_feature_seconds == 0.0, "Graph+data KNN should not build graph features.");
  require(km_graph_data_knn_res.knn.indices.size() == static_cast<std::size_t>(d.n * km_graph_data_knn_res.knn.neighbors), "Graph+data KNN graph size mismatch.");

  kodama::KODAMAMatrixOptions km_graph_pls_options = km_graph_options;
  km_graph_pls_options.classifier = kodama::CoreClassifier::PLS_LDA;
  km_graph_pls_options.components = 3;
  km_graph_pls_options.graph_feature_components = 5;
  km_graph_pls_options.graph_feature_steps = 2;
  kodama::KODAMAMatrixResult km_graph_pls_res = kodama::KODAMAMatrixFromGraph_CPU(
    km_res.base_knn,
    d.n,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    km_graph_pls_options
  );
  require(km_graph_pls_res.samples == d.n, "Graph-input PLS-LDA sample count mismatch.");
  require(km_graph_pls_res.res.size() == static_cast<std::size_t>(d.n), "Graph-input PLS-LDA result size mismatch.");
  require(km_graph_pls_res.graph_feature_seconds >= 0.0, "Graph-input PLS-LDA feature timing missing.");

  std::vector<float> graph_laplacian_features = kodama::KODAMAGraphFeatures_CPU(km_res.base_knn, d.n, [&]() {
    kodama::KODAMAMatrixOptions opts = km_graph_pls_options;
    opts.graph_feature_components = 4;
    opts.graph_feature_steps = 4;
    return opts;
  }());
  require(graph_laplacian_features.size() == static_cast<std::size_t>(d.n * 4), "Self-tuning graph feature size mismatch.");
  for (float value : graph_laplacian_features) require(std::isfinite(value), "Self-tuning graph features contain non-finite values.");

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

  kodama::KODAMAMatrixOptions cuda_km_options = km_options;
  cuda_km_options.backend = kodama::Backend::CUDA;
  cuda_km_options.runs = 1;
  cuda_km_options.cycles = 1;
  cuda_km_options.n_threads = 1;
  cuda_km_options.knn.backend = kodama::Backend::CUDA;
  cuda_km_options.knn.index_type = kodama::KNNIndexType::CudaExact;
  kodama::KODAMAMatrixResult cuda_km_res = kodama::KODAMAMatrix_CUDA(
    fview,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    cuda_km_options
  );
  require(cuda_km_res.backend == kodama::Backend::CUDA, "CUDA KODAMAMatrix did not report CUDA backend.");
  require(cuda_km_res.samples == d.n, "CUDA KODAMAMatrix sample count mismatch.");
  require(cuda_km_res.res.size() == static_cast<std::size_t>(d.n), "CUDA KODAMAMatrix result size mismatch.");
  require(cuda_km_res.knn.indices.size() == static_cast<std::size_t>(d.n * cuda_km_res.knn.neighbors), "CUDA KODAMAMatrix graph size mismatch.");
#endif

#if defined(KODAMA_ENABLE_METAL)
  require(kodama::MetalAvailable(), "Metal build did not find an Apple Metal device.");
  kodama::KNNOptions metal_knn = knn;
  metal_knn.backend = kodama::Backend::Metal;
  metal_knn.index_type = kodama::KNNIndexType::MetalExact;
  kodama::KNNCVResult metal_kres = kodama::KNNCV_METAL(fview, d.y, d.constrain, metal_knn);
  require(metal_kres.parameters.backend == kodama::Backend::Metal, "Metal KNNCV did not report Metal backend.");
  require(metal_kres.parameters.index_type == kodama::KNNIndexType::MetalExact, "Metal KNNCV did not report exact Metal search.");
  check_constrained_folds(d.constrain, metal_kres.fold_assignments);
  require(metal_kres.global_accuracy > 0.95, "Metal KNNCV accuracy unexpectedly low.");

  kodama::PLSOptions metal_pls = pls;
  metal_pls.backend = kodama::Backend::Metal;
  kodama::PLSCVResult metal_lres = kodama::PLSLDACV_METAL(fview, d.y, d.constrain, metal_pls);
  require(metal_lres.parameters.backend == kodama::Backend::Metal, "Metal PLS-LDA did not report Metal backend.");
  check_pls_result(metal_lres, d.y, d.constrain, 4);
  require(metal_lres.global_accuracy > 0.60, "Metal PLS-LDA accuracy unexpectedly low.");
  require(std::abs(metal_lres.global_accuracy - flres.global_accuracy) < 0.10, "Metal PLS-LDA diverged from CPU accuracy.");

  kodama::CoreOptions metal_core_knn = core_knn;
  metal_core_knn.knn.backend = kodama::Backend::Metal;
  metal_core_knn.knn.index_type = kodama::KNNIndexType::MetalExact;
  kodama::CoreResult metal_core_kres = kodama::CoreKNN_METAL(fview, noisy, d.constrain, fixed, metal_core_knn);
  require(metal_core_kres.clbest.size() == noisy.size(), "Metal Core KNN clbest size mismatch.");
  require(metal_core_kres.cycles_completed >= 1, "Metal Core KNN did not run any cycles.");

  kodama::CoreOptions metal_core_pls = core_pls;
  metal_core_pls.pls.backend = kodama::Backend::Metal;
  kodama::CoreResult metal_core_lres = kodama::CorePLSLDA_METAL(fview, noisy, d.constrain, fixed, metal_core_pls);
  require(metal_core_lres.clbest.size() == noisy.size(), "Metal Core PLS-LDA clbest size mismatch.");
  require(metal_core_lres.cycles_completed >= 1, "Metal Core PLS-LDA did not run any cycles.");
#endif

  std::cout << "All kodama-cpp CV tests passed.\n";
  return 0;
}
