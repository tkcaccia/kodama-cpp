#include <cstdint>
#include <algorithm>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kodama/kodama.hpp"

namespace py = pybind11;

namespace {

kodama::DistanceMetric parse_metric(const std::string& metric) {
  if (metric == "euclidean" || metric == "l2") return kodama::DistanceMetric::Euclidean;
  if (metric == "cosine") return kodama::DistanceMetric::Cosine;
  if (metric == "inner_product" || metric == "ip") return kodama::DistanceMetric::InnerProduct;
  throw std::invalid_argument("Unsupported metric: " + metric);
}

kodama::Backend parse_backend(const std::string& backend) {
  if (backend == "cpu") return kodama::Backend::CPU;
  if (backend == "cuda") return kodama::Backend::CUDA;
  throw std::invalid_argument("Unsupported backend: " + backend);
}

kodama::CoreClassifier parse_classifier(const std::string& classifier) {
  if (classifier == "knn") return kodama::CoreClassifier::KNN;
  if (classifier == "pls_lda") return kodama::CoreClassifier::PLS_LDA;
  throw std::invalid_argument("Unsupported classifier: " + classifier);
}

kodama::GraphFeatureMode parse_graph_feature_mode(const std::string& mode) {
  if (mode == "laplacian_self_tuning") return kodama::GraphFeatureMode::LaplacianSelfTuning;
  throw std::invalid_argument("Unsupported graph feature mode: " + mode);
}

kodama::GraphClusterMethod parse_graph_cluster_method(const std::string& method) {
  if (method == "louvain") return kodama::GraphClusterMethod::Louvain;
  if (method == "leiden") return kodama::GraphClusterMethod::Leiden;
  if (method == "random_walk" || method == "random_walking") return kodama::GraphClusterMethod::RandomWalking;
  throw std::invalid_argument("Unsupported graph clustering method: " + method);
}

kodama::GraphWeightType parse_graph_weight_type(const std::string& weight) {
  if (weight == "snn") return kodama::GraphWeightType::SNN;
  if (weight == "distance") return kodama::GraphWeightType::Distance;
  if (weight == "adaptive") return kodama::GraphWeightType::Adaptive;
  if (weight == "binary") return kodama::GraphWeightType::Binary;
  throw std::invalid_argument("Unsupported graph weight type: " + weight);
}

std::vector<int> optional_int_vector(const py::object& value) {
  if (value.is_none()) return {};
  py::array_t<int, py::array::c_style | py::array::forcecast> array(value);
  auto view = array.unchecked<1>();
  std::vector<int> out(static_cast<std::size_t>(view.shape(0)));
  for (py::ssize_t i = 0; i < view.shape(0); ++i) out[static_cast<std::size_t>(i)] = view(i);
  return out;
}

std::vector<int> int_array_to_vector(py::array_t<int, py::array::c_style | py::array::forcecast> array) {
  auto view = array.unchecked<1>();
  std::vector<int> out(static_cast<std::size_t>(view.shape(0)));
  for (py::ssize_t i = 0; i < view.shape(0); ++i) out[static_cast<std::size_t>(i)] = view(i);
  return out;
}

py::array_t<double> vector_to_double_array(const std::vector<double>& values) {
  py::array_t<double> out(static_cast<py::ssize_t>(values.size()));
  auto view = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < view.shape(0); ++i) view(i) = values[static_cast<std::size_t>(i)];
  return out;
}

py::array_t<int> vector_to_int_array(const std::vector<int>& values) {
  py::array_t<int> out(static_cast<py::ssize_t>(values.size()));
  auto view = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < view.shape(0); ++i) view(i) = values[static_cast<std::size_t>(i)];
  return out;
}

py::array_t<int> matrix_to_int_array(const std::vector<int>& values, int rows, int cols) {
  py::array_t<int> out({rows, cols});
  auto view = out.mutable_unchecked<2>();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      view(i, j) = values[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(j)];
    }
  }
  return out;
}

py::array_t<double> matrix_to_double_array(const std::vector<double>& values, int rows, int cols) {
  py::array_t<double> out({rows, cols});
  auto view = out.mutable_unchecked<2>();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      view(i, j) = values[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(j)];
    }
  }
  return out;
}

kodama::NeighborGraph graph_from_arrays(
  py::array_t<int, py::array::c_style | py::array::forcecast> indices,
  py::array_t<float, py::array::c_style | py::array::forcecast> distances
) {
  auto idx = indices.unchecked<2>();
  auto dst = distances.unchecked<2>();
  if (idx.shape(0) != dst.shape(0) || idx.shape(1) != dst.shape(1)) {
    throw std::invalid_argument("indices and distances must have the same dimensions.");
  }
  kodama::NeighborGraph graph;
  graph.neighbors = static_cast<int>(idx.shape(1));
  graph.indices.assign(static_cast<std::size_t>(idx.shape(0)) * static_cast<std::size_t>(idx.shape(1)), 0);
  graph.distances.assign(static_cast<std::size_t>(dst.shape(0)) * static_cast<std::size_t>(dst.shape(1)), 0.0f);
  for (py::ssize_t i = 0; i < idx.shape(0); ++i) {
    for (py::ssize_t j = 0; j < idx.shape(1); ++j) {
      const std::size_t offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(idx.shape(1)) + static_cast<std::size_t>(j);
      graph.indices[offset] = idx(i, j);
      graph.distances[offset] = dst(i, j);
    }
  }
  return graph;
}

py::dict graph_to_python(const kodama::NeighborGraph& graph, int samples) {
  py::array_t<int> indices({samples, graph.neighbors});
  py::array_t<float> distances({samples, graph.neighbors});
  auto idx = indices.mutable_unchecked<2>();
  auto dst = distances.mutable_unchecked<2>();
  for (int i = 0; i < samples; ++i) {
    for (int j = 0; j < graph.neighbors; ++j) {
      const std::size_t offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(graph.neighbors) + static_cast<std::size_t>(j);
      idx(i, j) = graph.indices[offset];
      dst(i, j) = graph.distances[offset];
    }
  }
  py::dict out;
  out["indices"] = indices;
  out["distances"] = distances;
  out["neighbors"] = graph.neighbors;
  return out;
}

py::dict folds_to_python(const std::vector<kodama::FoldResult>& folds) {
  py::dict out;
  py::array_t<int> fold(static_cast<py::ssize_t>(folds.size()));
  py::array_t<int> n_train(static_cast<py::ssize_t>(folds.size()));
  py::array_t<int> n_validation(static_cast<py::ssize_t>(folds.size()));
  py::array_t<double> accuracy(static_cast<py::ssize_t>(folds.size()));
  auto f = fold.mutable_unchecked<1>();
  auto tr = n_train.mutable_unchecked<1>();
  auto va = n_validation.mutable_unchecked<1>();
  auto ac = accuracy.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(folds.size()); ++i) {
    f(i) = folds[static_cast<std::size_t>(i)].fold;
    tr(i) = folds[static_cast<std::size_t>(i)].n_train;
    va(i) = folds[static_cast<std::size_t>(i)].n_validation;
    ac(i) = folds[static_cast<std::size_t>(i)].accuracy;
  }
  out["fold"] = fold;
  out["n_train"] = n_train;
  out["n_validation"] = n_validation;
  out["accuracy"] = accuracy;
  return out;
}

py::dict confusion_to_python(const kodama::ConfusionMatrix& confusion) {
  py::dict out;
  py::array_t<int> counts({static_cast<py::ssize_t>(confusion.n_labels), static_cast<py::ssize_t>(confusion.n_labels)});
  auto view = counts.mutable_unchecked<2>();
  for (std::size_t i = 0; i < confusion.n_labels; ++i) {
    for (std::size_t j = 0; j < confusion.n_labels; ++j) {
      view(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = confusion.counts[i * confusion.n_labels + j];
    }
  }
  out["labels"] = vector_to_int_array(confusion.labels);
  out["counts"] = counts;
  return out;
}

py::dict knncv_to_python(const kodama::KNNCVResult& result) {
  py::dict out;
  out["predicted"] = vector_to_int_array(result.predicted_labels);
  out["truth"] = vector_to_int_array(result.true_labels);
  out["folds"] = vector_to_int_array(result.fold_assignments);
  out["fold_accuracy"] = folds_to_python(result.folds);
  out["accuracy"] = result.global_accuracy;
  out["confusion"] = confusion_to_python(result.confusion);
  out["runtime_seconds"] = result.runtime_seconds;
  out["peak_memory_mb"] = result.peak_memory_mb;
  out["backend"] = kodama::to_string(result.parameters.backend);
  out["metric"] = kodama::to_string(result.parameters.metric);
  out["index_type"] = kodama::to_string(result.parameters.index_type);
  out["k"] = result.parameters.k;
  return out;
}

py::dict plscv_to_python(const kodama::PLSCVResult& result) {
  py::dict out;
  out["predicted"] = vector_to_int_array(result.predicted_labels);
  out["truth"] = vector_to_int_array(result.true_labels);
  out["folds"] = vector_to_int_array(result.fold_assignments);
  out["fold_accuracy"] = folds_to_python(result.folds);
  out["accuracy_by_components"] = vector_to_double_array(result.accuracy_by_components);
  out["selected_components"] = result.selected_components;
  out["accuracy"] = result.global_accuracy;
  out["confusion"] = confusion_to_python(result.confusion);
  out["runtime_seconds"] = result.runtime_seconds;
  out["peak_memory_mb"] = result.peak_memory_mb;
  out["backend"] = kodama::to_string(result.parameters.backend);
  out["mode"] = kodama::to_string(result.parameters.mode);
  out["max_components"] = result.parameters.max_components;
  out["fixed_components"] = result.parameters.fixed_components;
  return out;
}

py::dict core_to_python(const kodama::CoreResult& result) {
  py::dict out;
  out["clbest"] = vector_to_int_array(result.clbest);
  out["clbest_dirty"] = vector_to_int_array(result.clbest_dirty);
  out["cvpredbest"] = vector_to_int_array(result.cvpredbest);
  out["accbest"] = result.accbest;
  out["scorebest"] = result.scorebest;
  out["vect_acc"] = vector_to_double_array(result.vect_acc);
  out["vect_score"] = vector_to_double_array(result.vect_score);
  out["cycles_completed"] = result.cycles_completed;
  out["success"] = result.success;
  out["runtime_seconds"] = result.runtime_seconds;
  out["peak_memory_mb"] = result.peak_memory_mb;
  return out;
}

py::dict kodama_matrix_to_python(const kodama::KODAMAMatrixResult& result, const kodama::KODAMAMatrixOptions& options) {
  py::dict out;
  out["acc"] = vector_to_double_array(result.acc);
  out["v"] = matrix_to_double_array(result.v, result.runs, result.cycles);
  out["res"] = matrix_to_int_array(result.res, result.runs, result.samples);
  out["res_constrain"] = matrix_to_int_array(result.res_constrain, result.runs, result.samples);
  out["knn"] = graph_to_python(result.knn, result.samples);
  out["base_knn"] = graph_to_python(result.base_knn, result.samples);
  out["runtime_seconds"] = result.runtime_seconds;
  out["analysis_storage"] = "float32";
  out["classifier"] = kodama::to_string(options.classifier);
  out["backend"] = kodama::to_string(options.backend);
  out["graph_feature_mode"] = kodama::to_string(options.graph_feature_mode);
  out["peak_memory_mb"] = result.peak_memory_mb;
  py::dict timing;
  timing["input_copy_seconds"] = result.input_copy_seconds;
  timing["graph_feature_seconds"] = result.graph_feature_seconds;
  timing["spatial_precompute_seconds"] = result.spatial_precompute_seconds;
  timing["graph_seconds"] = result.graph_seconds;
  timing["spatial_graph_seconds"] = result.spatial_graph_seconds;
  timing["optimization_wall_seconds"] = result.optimization_wall_seconds;
  timing["optimization_sum_seconds"] = result.optimization_sum_seconds;
  timing["dissimilarity_seconds"] = result.dissimilarity_seconds;
  timing["runtime_seconds"] = result.runtime_seconds;
  out["timing"] = timing;
  return out;
}

py::array_t<float> embedding_to_python(const kodama::EmbeddingResult& result) {
  py::array_t<float> out({result.samples, result.components});
  auto view = out.mutable_unchecked<2>();
  for (int i = 0; i < result.samples; ++i) {
    for (int j = 0; j < result.components; ++j) {
      view(i, j) = result.embedding[static_cast<std::size_t>(i) * static_cast<std::size_t>(result.components) + static_cast<std::size_t>(j)];
    }
  }
  return out;
}

py::dict graph_cluster_to_python(const kodama::GraphClusterResult& result) {
  py::dict out;
  out["membership"] = vector_to_int_array(result.membership);
  out["modularity"] = result.modularity;
  out["n_communities"] = result.n_communities;
  out["selected_run"] = result.selected_run;
  out["all_modularity"] = vector_to_double_array(result.all_modularity);
  out["n_vertices"] = result.n_vertices;
  out["n_edges"] = result.n_edges;
  out["target_clusters"] = result.target_clusters;
  out["target_gap"] = result.target_gap;
  out["target_exact"] = result.target_exact;
  out["selected_resolution"] = result.selected_resolution;
  out["runtime_seconds"] = result.runtime_seconds;
  out["backend"] = kodama::to_string(result.backend);
  out["method"] = kodama::to_string(result.method);
  return out;
}

}  // namespace

py::dict matrix(
  py::array_t<float, py::array::c_style | py::array::forcecast> data,
  py::object spatial,
  py::object W,
  py::object constrain,
  py::object fix,
  int M,
  int Tcycle,
  int ncomp,
  int landmarks,
  int splitting,
  int n_threads,
  int graph_neighbors,
  int knn_k,
  double spatial_resolution,
  bool spatial_graph_mix,
  int spatial_constraint_mode,
  const std::string& metric,
  const std::string& classifier,
  const std::string& backend,
  int seed,
  bool progress,
  bool apply_kodama_dissimilarity
) {
  auto x_view = data.unchecked<2>();
  const int n = static_cast<int>(x_view.shape(0));
  const int p = static_cast<int>(x_view.shape(1));

  kodama::KODAMAMatrixOptions options;
  options.runs = M;
  options.cycles = Tcycle;
  options.components = ncomp;
  options.landmarks = landmarks;
  options.splitting = splitting;
  options.graph_neighbors = graph_neighbors;
  options.n_threads = n_threads;
  options.spatial_resolution = spatial_resolution;
  options.spatial_graph_mix = spatial_graph_mix;
  options.spatial_constraint_mode = spatial_constraint_mode;
  options.seed = static_cast<std::uint64_t>(seed);
  options.metric = parse_metric(metric);
  options.backend = parse_backend(backend);
  options.classifier = parse_classifier(classifier);
  options.progress = progress;
  options.apply_kodama_dissimilarity = apply_kodama_dissimilarity;
  options.knn.k = knn_k;
  options.knn.hnsw_tune_k = 50;
  options.knn.hnsw_target_recall = 0.99;
  options.knn.n_threads = 1;
  options.pls.n_threads = 1;

  py::array_t<float, py::array::c_style | py::array::forcecast> spatial_array;
  if (!spatial.is_none()) {
    spatial_array = py::array_t<float, py::array::c_style | py::array::forcecast>(spatial);
    auto s = spatial_array.unchecked<2>();
    if (s.shape(0) != n) throw std::invalid_argument("spatial must have the same number of rows as data.");
    options.spatial_cols = static_cast<int>(s.shape(1));
    options.spatial.assign(static_cast<std::size_t>(s.shape(0)) * static_cast<std::size_t>(s.shape(1)), 0.0f);
    for (py::ssize_t i = 0; i < s.shape(0); ++i) {
      for (py::ssize_t j = 0; j < s.shape(1); ++j) {
        options.spatial[static_cast<std::size_t>(i) * static_cast<std::size_t>(s.shape(1)) + static_cast<std::size_t>(j)] = s(i, j);
      }
    }
  }

  kodama::MatrixView view{
    static_cast<const float*>(data.request().ptr),
    static_cast<std::size_t>(n),
    static_cast<std::size_t>(p)
  };
  kodama::KODAMAMatrixResult result = kodama::KODAMAMatrix(
    view,
    optional_int_vector(W),
    optional_int_vector(constrain),
    optional_int_vector(fix),
    options
  );

  return kodama_matrix_to_python(result, options);
}

py::dict matrix_graph(
  py::array_t<int, py::array::c_style | py::array::forcecast> indices,
  py::array_t<float, py::array::c_style | py::array::forcecast> distances,
  py::object data,
  py::object spatial,
  py::object W,
  py::object constrain,
  py::object fix,
  int M,
  int Tcycle,
  int ncomp,
  int landmarks,
  int splitting,
  int n_threads,
  int graph_neighbors,
  int knn_k,
  double spatial_resolution,
  bool spatial_graph_mix,
  int spatial_constraint_mode,
  const std::string& classifier,
  const std::string& backend,
  const std::string& graph_feature_mode,
  int graph_feature_components,
  int graph_feature_steps,
  int seed,
  bool progress,
  bool apply_kodama_dissimilarity
) {
  auto idx = indices.unchecked<2>();
  auto dst = distances.unchecked<2>();
  if (idx.shape(0) != dst.shape(0) || idx.shape(1) != dst.shape(1)) {
    throw std::invalid_argument("indices and distances must have the same dimensions.");
  }

  kodama::KODAMAMatrixOptions options;
  options.runs = M;
  options.cycles = Tcycle;
  options.components = ncomp;
  options.landmarks = landmarks;
  options.splitting = splitting;
  options.graph_neighbors = graph_neighbors;
  options.n_threads = n_threads;
  options.spatial_resolution = spatial_resolution;
  options.spatial_graph_mix = spatial_graph_mix;
  options.spatial_constraint_mode = spatial_constraint_mode;
  options.seed = static_cast<std::uint64_t>(seed);
  options.backend = parse_backend(backend);
  options.classifier = parse_classifier(classifier);
  options.progress = progress;
  options.apply_kodama_dissimilarity = apply_kodama_dissimilarity;
  options.graph_feature_mode = parse_graph_feature_mode(graph_feature_mode);
  options.graph_feature_components = graph_feature_components;
  options.graph_feature_steps = graph_feature_steps;
  options.knn.k = knn_k;
  options.knn.hnsw_tune_k = 50;
  options.knn.hnsw_target_recall = 0.99;
  options.knn.n_threads = 1;
  options.pls.n_threads = 1;

  py::array_t<float, py::array::c_style | py::array::forcecast> spatial_array;
  if (!spatial.is_none()) {
    spatial_array = py::array_t<float, py::array::c_style | py::array::forcecast>(spatial);
    auto s = spatial_array.unchecked<2>();
    if (s.shape(0) != idx.shape(0)) throw std::invalid_argument("spatial rows must match graph rows.");
    options.spatial_cols = static_cast<int>(s.shape(1));
    options.spatial.assign(static_cast<std::size_t>(s.shape(0)) * static_cast<std::size_t>(s.shape(1)), 0.0f);
    for (py::ssize_t i = 0; i < s.shape(0); ++i) {
      for (py::ssize_t j = 0; j < s.shape(1); ++j) {
        options.spatial[static_cast<std::size_t>(i) * static_cast<std::size_t>(s.shape(1)) + static_cast<std::size_t>(j)] = s(i, j);
      }
    }
  }

  const kodama::NeighborGraph graph = graph_from_arrays(indices, distances);
  const std::vector<int> labels = optional_int_vector(W);
  const std::vector<int> constraints = optional_int_vector(constrain);
  const std::vector<int> fixed = optional_int_vector(fix);
  kodama::KODAMAMatrixResult result;
  if (!data.is_none()) {
    py::array_t<float, py::array::c_style | py::array::forcecast> data_array(data);
    auto x = data_array.unchecked<2>();
    if (x.shape(0) != idx.shape(0)) throw std::invalid_argument("data rows must match graph rows.");
    kodama::MatrixView view{
      static_cast<const float*>(data_array.request().ptr),
      static_cast<std::size_t>(x.shape(0)),
      static_cast<std::size_t>(x.shape(1))
    };
    result = kodama::KODAMAMatrixFromGraphData(view, graph, labels, constraints, fixed, options);
  } else {
    result = kodama::KODAMAMatrixFromGraph(graph, static_cast<int>(idx.shape(0)), labels, constraints, fixed, options);
  }
  return kodama_matrix_to_python(result, options);
}

py::dict knncv(
  py::array_t<float, py::array::c_style | py::array::forcecast> data,
  py::array_t<int, py::array::c_style | py::array::forcecast> labels,
  py::object constrain,
  int folds,
  bool stratified,
  int seed,
  int k,
  const std::string& metric,
  const std::string& backend,
  int n_threads,
  int gpu_device
) {
  auto x_view = data.unchecked<2>();
  kodama::KNNOptions options;
  options.cv.folds = folds;
  options.cv.stratified = stratified;
  options.cv.seed = static_cast<std::uint64_t>(seed);
  options.k = k;
  options.metric = parse_metric(metric);
  options.backend = parse_backend(backend);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  return knncv_to_python(kodama::KNNCV(
    kodama::MatrixView{
      static_cast<const float*>(data.request().ptr),
      static_cast<std::size_t>(x_view.shape(0)),
      static_cast<std::size_t>(x_view.shape(1))
    },
    int_array_to_vector(labels),
    optional_int_vector(constrain),
    options
  ));
}

py::dict plsldacv(
  py::array_t<float, py::array::c_style | py::array::forcecast> data,
  py::array_t<int, py::array::c_style | py::array::forcecast> labels,
  py::object constrain,
  int folds,
  bool stratified,
  int seed,
  int ncomp,
  bool center,
  bool scale,
  const std::string& backend,
  int n_threads,
  int gpu_device
) {
  auto x_view = data.unchecked<2>();
  kodama::PLSOptions options;
  options.cv.folds = folds;
  options.cv.stratified = stratified;
  options.cv.seed = static_cast<std::uint64_t>(seed);
  options.max_components = ncomp;
  options.fixed_components = ncomp;
  options.center = center;
  options.scale = scale;
  options.backend = parse_backend(backend);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  return plscv_to_python(kodama::PLSLDACV(
    kodama::MatrixView{
      static_cast<const float*>(data.request().ptr),
      static_cast<std::size_t>(x_view.shape(0)),
      static_cast<std::size_t>(x_view.shape(1))
    },
    int_array_to_vector(labels),
    optional_int_vector(constrain),
    options
  ));
}

py::dict core_impl(
  py::array_t<float, py::array::c_style | py::array::forcecast> data,
  py::array_t<int, py::array::c_style | py::array::forcecast> labels,
  py::object constrain,
  py::object fix,
  int cycles,
  int folds,
  bool stratified,
  int seed,
  int k,
  int ncomp,
  const std::string& metric,
  const std::string& backend,
  int n_threads,
  int gpu_device,
  kodama::CoreClassifier classifier
) {
  auto x_view = data.unchecked<2>();
  kodama::CoreOptions options;
  options.cycles = cycles;
  options.classifier = classifier;
  options.seed = static_cast<std::uint64_t>(seed);
  options.knn.cv.folds = folds;
  options.knn.cv.stratified = stratified;
  options.knn.cv.seed = static_cast<std::uint64_t>(seed);
  options.knn.k = k;
  options.knn.metric = parse_metric(metric);
  options.knn.backend = parse_backend(backend);
  options.knn.n_threads = n_threads;
  options.knn.gpu_device = gpu_device;
  options.knn.hnsw_tune_k = std::max(50, k);
  options.knn.hnsw_target_recall = 0.99;
  options.pls.cv.folds = folds;
  options.pls.cv.stratified = stratified;
  options.pls.cv.seed = static_cast<std::uint64_t>(seed);
  options.pls.max_components = ncomp;
  options.pls.fixed_components = ncomp;
  options.pls.backend = parse_backend(backend);
  options.pls.n_threads = n_threads;
  options.pls.gpu_device = gpu_device;
  kodama::MatrixView view{
    static_cast<const float*>(data.request().ptr),
    static_cast<std::size_t>(x_view.shape(0)),
    static_cast<std::size_t>(x_view.shape(1))
  };
  const std::vector<int> y = int_array_to_vector(labels);
  const std::vector<int> c = optional_int_vector(constrain);
  const std::vector<int> f = optional_int_vector(fix);
  return core_to_python(classifier == kodama::CoreClassifier::KNN ?
    kodama::CoreKNN(view, y, c, f, options) :
    kodama::CorePLSLDA(view, y, c, f, options)
  );
}

py::dict core_knn(
  py::array_t<float, py::array::c_style | py::array::forcecast> data,
  py::array_t<int, py::array::c_style | py::array::forcecast> labels,
  py::object constrain,
  py::object fix,
  int cycles,
  int folds,
  bool stratified,
  int seed,
  int k,
  const std::string& metric,
  const std::string& backend,
  int n_threads,
  int gpu_device
) {
  return core_impl(data, labels, constrain, fix, cycles, folds, stratified, seed, k, 1, metric, backend, n_threads, gpu_device, kodama::CoreClassifier::KNN);
}

py::dict core_plslda(
  py::array_t<float, py::array::c_style | py::array::forcecast> data,
  py::array_t<int, py::array::c_style | py::array::forcecast> labels,
  py::object constrain,
  py::object fix,
  int cycles,
  int folds,
  bool stratified,
  int seed,
  int ncomp,
  const std::string& backend,
  int n_threads,
  int gpu_device
) {
  return core_impl(data, labels, constrain, fix, cycles, folds, stratified, seed, 30, ncomp, "euclidean", backend, n_threads, gpu_device, kodama::CoreClassifier::PLS_LDA);
}

py::dict knn_graph(
  py::array_t<float, py::array::c_style | py::array::forcecast> data,
  int k,
  const std::string& metric,
  const std::string& backend,
  int n_threads,
  int gpu_device
) {
  auto x_view = data.unchecked<2>();
  kodama::GraphClusterOptions options;
  options.k = k;
  options.metric = parse_metric(metric);
  options.backend = parse_backend(backend);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  const int n = static_cast<int>(x_view.shape(0));
  return graph_to_python(
    kodama::KODAMAKNNGraph(
      kodama::MatrixView{
        static_cast<const float*>(data.request().ptr),
        static_cast<std::size_t>(x_view.shape(0)),
        static_cast<std::size_t>(x_view.shape(1))
      },
      options
    ),
    n
  );
}

py::array_t<float> umap(
  py::array_t<int, py::array::c_style | py::array::forcecast> indices,
  py::array_t<float, py::array::c_style | py::array::forcecast> distances,
  py::object init,
  int n_neighbors,
  int n_epochs,
  double learning_rate,
  double min_dist,
  double repulsion_strength,
  int negative_sample_rate,
  int spectral_n_iter,
  int n_threads,
  int seed,
  const std::string& backend,
  int gpu_device
) {
  kodama::UMAPOptions options;
  options.n_neighbors = n_neighbors;
  options.n_epochs = n_epochs;
  options.learning_rate = learning_rate;
  options.min_dist = min_dist;
  options.repulsion_strength = repulsion_strength;
  options.negative_sample_rate = negative_sample_rate;
  options.spectral_n_iter = spectral_n_iter;
  options.n_threads = n_threads;
  options.seed = seed;
  options.gpu_device = gpu_device;
  if (!init.is_none()) {
    py::array_t<float, py::array::c_style | py::array::forcecast> init_array(init);
    auto view = init_array.unchecked<2>();
    if (view.shape(0) != indices.shape(0) || view.shape(1) != 2) throw std::invalid_argument("init must have n rows and 2 columns.");
    options.init.assign(static_cast<std::size_t>(view.shape(0)) * 2u, 0.0f);
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
      options.init[static_cast<std::size_t>(i) * 2u] = view(i, 0);
      options.init[static_cast<std::size_t>(i) * 2u + 1u] = view(i, 1);
    }
  }
  const kodama::NeighborGraph g = graph_from_arrays(indices, distances);
  const kodama::EmbeddingResult result = parse_backend(backend) == kodama::Backend::CUDA ?
    kodama::KODAMAUMAP_CUDA(g, options) :
    kodama::KODAMAUMAP_CPU(g, options);
  return embedding_to_python(result);
}

py::array_t<float> opentsne(
  py::array_t<int, py::array::c_style | py::array::forcecast> indices,
  py::array_t<float, py::array::c_style | py::array::forcecast> distances,
  py::object init,
  int n_neighbors,
  double perplexity,
  double theta,
  int early_exaggeration_iter,
  int n_iter,
  double early_exaggeration,
  double exaggeration,
  double learning_rate,
  bool learning_rate_auto,
  double initial_momentum,
  double final_momentum,
  double min_gain,
  double max_step_norm,
  int n_threads,
  int seed,
  const std::string& backend,
  int gpu_device
) {
  kodama::OpenTSNEOptions options;
  options.n_neighbors = n_neighbors;
  options.perplexity = perplexity;
  options.theta = theta;
  options.early_exaggeration_iter = early_exaggeration_iter;
  options.n_iter = n_iter;
  options.early_exaggeration = early_exaggeration;
  options.exaggeration = exaggeration;
  options.learning_rate = learning_rate;
  options.learning_rate_auto = learning_rate_auto;
  options.initial_momentum = initial_momentum;
  options.final_momentum = final_momentum;
  options.min_gain = min_gain;
  options.max_step_norm = max_step_norm;
  options.n_threads = n_threads;
  options.seed = seed;
  options.gpu_device = gpu_device;
  if (!init.is_none()) {
    py::array_t<float, py::array::c_style | py::array::forcecast> init_array(init);
    auto view = init_array.unchecked<2>();
    if (view.shape(0) != indices.shape(0) || view.shape(1) != 2) throw std::invalid_argument("init must have n rows and 2 columns.");
    options.init.assign(static_cast<std::size_t>(view.shape(0)) * 2u, 0.0f);
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
      options.init[static_cast<std::size_t>(i) * 2u] = view(i, 0);
      options.init[static_cast<std::size_t>(i) * 2u + 1u] = view(i, 1);
    }
  }
  const kodama::NeighborGraph g = graph_from_arrays(indices, distances);
  const kodama::EmbeddingResult result = parse_backend(backend) == kodama::Backend::CUDA ?
    kodama::KODAMAOpenTSNE_CUDA(g, options) :
    kodama::KODAMAOpenTSNE_CPU(g, options);
  return embedding_to_python(result);
}

py::dict graph_clustering(
  py::array_t<int, py::array::c_style | py::array::forcecast> indices,
  py::array_t<float, py::array::c_style | py::array::forcecast> distances,
  const std::string& method,
  const std::string& backend,
  const std::string& weight,
  int n_threads,
  int n_runs,
  int n_iterations,
  int random_walk_steps,
  int n_clusters,
  double resolution,
  double resolution_init,
  double resolution_delta,
  double prune,
  bool mutual,
  int seed,
  int gpu_device
) {
  kodama::GraphClusterOptions options;
  options.method = parse_graph_cluster_method(method);
  options.backend = parse_backend(backend);
  options.weight_type = parse_graph_weight_type(weight);
  options.n_threads = n_threads;
  options.n_runs = n_runs;
  options.n_iterations = n_iterations;
  options.random_walk_steps = random_walk_steps;
  options.target_clusters = n_clusters;
  options.resolution = resolution;
  options.target_resolution_init = resolution_init;
  options.target_delta = resolution_delta;
  options.prune = prune;
  options.mutual = mutual;
  options.seed = static_cast<std::uint64_t>(seed);
  options.gpu_device = gpu_device;
  const kodama::NeighborGraph g = graph_from_arrays(indices, distances);
  return graph_cluster_to_python(kodama::KODAMAGraphCluster(g, static_cast<int>(indices.shape(0)), options));
}

py::dict embedding_clustering(
  py::array_t<float, py::array::c_style | py::array::forcecast> embedding,
  const std::string& method,
  const std::string& backend,
  const std::string& graph_backend,
  const std::string& weight,
  const std::string& metric,
  int k,
  int n_threads,
  int n_runs,
  int n_iterations,
  int random_walk_steps,
  int n_clusters,
  double resolution,
  double resolution_init,
  double resolution_delta,
  double prune,
  bool mutual,
  int seed,
  int gpu_device
) {
  auto x_view = embedding.unchecked<2>();
  kodama::GraphClusterOptions graph_options;
  graph_options.k = k;
  graph_options.metric = parse_metric(metric);
  graph_options.backend = parse_backend(graph_backend);
  graph_options.n_threads = n_threads;
  graph_options.gpu_device = gpu_device;
  kodama::MatrixView view{
    static_cast<const float*>(embedding.request().ptr),
    static_cast<std::size_t>(x_view.shape(0)),
    static_cast<std::size_t>(x_view.shape(1))
  };
  const kodama::NeighborGraph g = kodama::KODAMAKNNGraph(view, graph_options);
  kodama::GraphClusterOptions cluster_options = graph_options;
  cluster_options.method = parse_graph_cluster_method(method);
  cluster_options.backend = parse_backend(backend);
  cluster_options.weight_type = parse_graph_weight_type(weight);
  cluster_options.n_runs = n_runs;
  cluster_options.n_iterations = n_iterations;
  cluster_options.random_walk_steps = random_walk_steps;
  cluster_options.target_clusters = n_clusters;
  cluster_options.resolution = resolution;
  cluster_options.target_resolution_init = resolution_init;
  cluster_options.target_delta = resolution_delta;
  cluster_options.prune = prune;
  cluster_options.mutual = mutual;
  cluster_options.seed = static_cast<std::uint64_t>(seed);
  cluster_options.gpu_device = gpu_device;
  py::dict out = graph_cluster_to_python(kodama::KODAMAEmbeddingGraphCluster(view, g, cluster_options));
  out["graph"] = graph_to_python(g, static_cast<int>(x_view.shape(0)));
  return out;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "Python bindings for kodama-cpp";
  m.def(
    "knncv",
    &knncv,
    py::arg("data"),
    py::arg("labels"),
    py::arg("constrain") = py::none(),
    py::arg("folds") = 10,
    py::arg("stratified") = true,
    py::arg("seed") = 1,
    py::arg("k") = 10,
    py::arg("metric") = "cosine",
    py::arg("backend") = "cpu",
    py::arg("n_threads") = 1,
    py::arg("gpu_device") = 0
  );
  m.def(
    "plsldacv",
    &plsldacv,
    py::arg("data"),
    py::arg("labels"),
    py::arg("constrain") = py::none(),
    py::arg("folds") = 10,
    py::arg("stratified") = true,
    py::arg("seed") = 1,
    py::arg("ncomp") = 50,
    py::arg("center") = true,
    py::arg("scale") = true,
    py::arg("backend") = "cpu",
    py::arg("n_threads") = 1,
    py::arg("gpu_device") = 0
  );
  m.def(
    "core_knn",
    &core_knn,
    py::arg("data"),
    py::arg("labels"),
    py::arg("constrain") = py::none(),
    py::arg("fix") = py::none(),
    py::arg("cycles") = 100,
    py::arg("folds") = 10,
    py::arg("stratified") = true,
    py::arg("seed") = 1,
    py::arg("k") = 30,
    py::arg("metric") = "euclidean",
    py::arg("backend") = "cpu",
    py::arg("n_threads") = 4,
    py::arg("gpu_device") = 0
  );
  m.def(
    "core_plslda",
    &core_plslda,
    py::arg("data"),
    py::arg("labels"),
    py::arg("constrain") = py::none(),
    py::arg("fix") = py::none(),
    py::arg("cycles") = 100,
    py::arg("folds") = 10,
    py::arg("stratified") = true,
    py::arg("seed") = 1,
    py::arg("ncomp") = 50,
    py::arg("backend") = "cpu",
    py::arg("n_threads") = 4,
    py::arg("gpu_device") = 0
  );
  m.def(
    "matrix",
    &matrix,
    py::arg("data"),
    py::arg("spatial") = py::none(),
    py::arg("W") = py::none(),
    py::arg("constrain") = py::none(),
    py::arg("fix") = py::none(),
    py::arg("M") = 100,
    py::arg("Tcycle") = 20,
    py::arg("ncomp") = 50,
    py::arg("landmarks") = 10000,
    py::arg("splitting") = 0,
    py::arg("n_threads") = 4,
    py::arg("graph_neighbors") = 100,
    py::arg("knn_k") = 30,
    py::arg("spatial_resolution") = 0.3,
    py::arg("spatial_graph_mix") = false,
    py::arg("spatial_constraint_mode") = 0,
    py::arg("metric") = "euclidean",
    py::arg("classifier") = "knn",
    py::arg("backend") = "cpu",
    py::arg("seed") = 1234,
    py::arg("progress") = true,
    py::arg("apply_kodama_dissimilarity") = true
  );
  m.def(
    "matrix_graph",
    &matrix_graph,
    py::arg("indices"),
    py::arg("distances"),
    py::arg("data") = py::none(),
    py::arg("spatial") = py::none(),
    py::arg("W") = py::none(),
    py::arg("constrain") = py::none(),
    py::arg("fix") = py::none(),
    py::arg("M") = 100,
    py::arg("Tcycle") = 20,
    py::arg("ncomp") = 50,
    py::arg("landmarks") = 10000,
    py::arg("splitting") = 100,
    py::arg("n_threads") = 4,
    py::arg("graph_neighbors") = 100,
    py::arg("knn_k") = 30,
    py::arg("spatial_resolution") = 0.3,
    py::arg("spatial_graph_mix") = false,
    py::arg("spatial_constraint_mode") = 0,
    py::arg("classifier") = "knn",
    py::arg("backend") = "cpu",
    py::arg("graph_feature_mode") = "laplacian_self_tuning",
    py::arg("graph_feature_components") = 0,
    py::arg("graph_feature_steps") = 3,
    py::arg("seed") = 1234,
    py::arg("progress") = true,
    py::arg("apply_kodama_dissimilarity") = true
  );
  m.def(
    "graph",
    &knn_graph,
    py::arg("data"),
    py::arg("k") = 30,
    py::arg("metric") = "euclidean",
    py::arg("backend") = "cpu",
    py::arg("n_threads") = 4,
    py::arg("gpu_device") = 0
  );
  m.def(
    "umap",
    &umap,
    py::arg("indices"),
    py::arg("distances"),
    py::arg("init") = py::none(),
    py::arg("n_neighbors") = 30,
    py::arg("n_epochs") = 200,
    py::arg("learning_rate") = 1.0,
    py::arg("min_dist") = 0.01,
    py::arg("repulsion_strength") = 1.0,
    py::arg("negative_sample_rate") = 5,
    py::arg("spectral_n_iter") = 20,
    py::arg("n_threads") = 1,
    py::arg("seed") = 1234,
    py::arg("backend") = "cpu",
    py::arg("gpu_device") = 0
  );
  m.def(
    "opentsne",
    &opentsne,
    py::arg("indices"),
    py::arg("distances"),
    py::arg("init") = py::none(),
    py::arg("n_neighbors") = 0,
    py::arg("perplexity") = 30.0,
    py::arg("theta") = 0.5,
    py::arg("early_exaggeration_iter") = 250,
    py::arg("n_iter") = 500,
    py::arg("early_exaggeration") = 12.0,
    py::arg("exaggeration") = 1.0,
    py::arg("learning_rate") = 0.0,
    py::arg("learning_rate_auto") = true,
    py::arg("initial_momentum") = 0.8,
    py::arg("final_momentum") = 0.8,
    py::arg("min_gain") = 0.01,
    py::arg("max_step_norm") = 5.0,
    py::arg("n_threads") = 1,
    py::arg("seed") = 4,
    py::arg("backend") = "cpu",
    py::arg("gpu_device") = 0
  );
  m.def(
    "graph_clustering",
    &graph_clustering,
    py::arg("indices"),
    py::arg("distances"),
    py::arg("method") = "louvain",
    py::arg("backend") = "cpu",
    py::arg("weight") = "distance",
    py::arg("n_threads") = 4,
    py::arg("n_runs") = 1,
    py::arg("n_iterations") = 10,
    py::arg("random_walk_steps") = 4,
    py::arg("n_clusters") = 0,
    py::arg("resolution") = 1.0,
    py::arg("resolution_init") = 0.0,
    py::arg("resolution_delta") = 0.2,
    py::arg("prune") = 0.0,
    py::arg("mutual") = false,
    py::arg("seed") = 1,
    py::arg("gpu_device") = 0
  );
  m.def(
    "embedding_clustering",
    &embedding_clustering,
    py::arg("embedding"),
    py::arg("method") = "louvain",
    py::arg("backend") = "cpu",
    py::arg("graph_backend") = "cpu",
    py::arg("weight") = "distance",
    py::arg("metric") = "euclidean",
    py::arg("k") = 30,
    py::arg("n_threads") = 4,
    py::arg("n_runs") = 1,
    py::arg("n_iterations") = 10,
    py::arg("random_walk_steps") = 4,
    py::arg("n_clusters") = 0,
    py::arg("resolution") = 1.0,
    py::arg("resolution_init") = 0.0,
    py::arg("resolution_delta") = 0.2,
    py::arg("prune") = 0.0,
    py::arg("mutual") = false,
    py::arg("seed") = 1,
    py::arg("gpu_device") = 0
  );
}
