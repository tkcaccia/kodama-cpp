#include <Rcpp.h>

#include "kodama/kodama.hpp"

namespace {

kodama::DistanceMetric parse_metric(const std::string& metric) {
  if (metric == "euclidean" || metric == "l2") return kodama::DistanceMetric::Euclidean;
  if (metric == "cosine") return kodama::DistanceMetric::Cosine;
  if (metric == "inner_product" || metric == "ip") return kodama::DistanceMetric::InnerProduct;
  Rcpp::stop("Unsupported metric: " + metric);
}

kodama::Backend parse_backend(const std::string& backend) {
  if (backend == "cpu") return kodama::Backend::CPU;
  if (backend == "cuda") return kodama::Backend::CUDA;
  Rcpp::stop("Unsupported backend: " + backend);
}

kodama::CoreClassifier parse_classifier(const std::string& classifier) {
  if (classifier == "knn") return kodama::CoreClassifier::KNN;
  if (classifier == "pls_lda" || classifier == "plSlda") return kodama::CoreClassifier::PLS_LDA;
  Rcpp::stop("Unsupported classifier: " + classifier);
}

kodama::GraphClusterMethod parse_graph_cluster_method(const std::string& method) {
  if (method == "louvain") return kodama::GraphClusterMethod::Louvain;
  if (method == "leiden") return kodama::GraphClusterMethod::Leiden;
  if (method == "random_walking") return kodama::GraphClusterMethod::RandomWalking;
  Rcpp::stop("Unsupported graph clustering method: " + method);
}

kodama::GraphWeightType parse_graph_weight_type(const std::string& weight) {
  if (weight == "snn") return kodama::GraphWeightType::SNN;
  if (weight == "distance") return kodama::GraphWeightType::Distance;
  if (weight == "adaptive") return kodama::GraphWeightType::Adaptive;
  if (weight == "binary") return kodama::GraphWeightType::Binary;
  Rcpp::stop("Unsupported graph weight type: " + weight);
}

std::vector<int> optional_int_vector(Rcpp::Nullable<Rcpp::IntegerVector> value) {
  if (value.isNull()) return {};
  Rcpp::IntegerVector v(value);
  std::vector<int> out(static_cast<std::size_t>(v.size()));
  for (R_xlen_t i = 0; i < v.size(); ++i) out[static_cast<std::size_t>(i)] = v[i];
  return out;
}

kodama::NeighborGraph graph_from_r(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix distances) {
  if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol()) {
    Rcpp::stop("indices and distances must have the same dimensions.");
  }
  kodama::NeighborGraph graph;
  graph.neighbors = indices.ncol();
  graph.indices.assign(static_cast<std::size_t>(indices.nrow()) * indices.ncol(), 0);
  graph.distances.assign(static_cast<std::size_t>(distances.nrow()) * distances.ncol(), 0.0f);
  for (int i = 0; i < indices.nrow(); ++i) {
    for (int j = 0; j < indices.ncol(); ++j) {
      const std::size_t offset = static_cast<std::size_t>(i) * indices.ncol() + j;
      graph.indices[offset] = indices(i, j);
      graph.distances[offset] = static_cast<float>(distances(i, j));
    }
  }
  return graph;
}

Rcpp::List graph_to_r(const kodama::NeighborGraph& graph, int samples) {
  Rcpp::IntegerMatrix indices(samples, graph.neighbors);
  Rcpp::NumericMatrix distances(samples, graph.neighbors);
  for (int i = 0; i < samples; ++i) {
    for (int j = 0; j < graph.neighbors; ++j) {
      const std::size_t offset = static_cast<std::size_t>(i) * graph.neighbors + j;
      indices(i, j) = graph.indices[offset];
      distances(i, j) = graph.distances[offset];
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("neighbors") = graph.neighbors
  );
}

Rcpp::List graph_cluster_result_to_r(const kodama::GraphClusterResult& result) {
  Rcpp::IntegerVector membership(result.membership.begin(), result.membership.end());
  Rcpp::NumericVector all_modularity(result.all_modularity.begin(), result.all_modularity.end());
  return Rcpp::List::create(
    Rcpp::Named("membership") = membership,
    Rcpp::Named("modularity") = result.modularity,
    Rcpp::Named("n_communities") = result.n_communities,
    Rcpp::Named("selected_run") = result.selected_run,
    Rcpp::Named("all_modularity") = all_modularity,
    Rcpp::Named("n_vertices") = result.n_vertices,
    Rcpp::Named("n_edges") = result.n_edges,
    Rcpp::Named("target_clusters") = result.target_clusters,
    Rcpp::Named("target_gap") = result.target_gap,
    Rcpp::Named("target_exact") = result.target_exact,
    Rcpp::Named("selected_resolution") = result.selected_resolution,
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("backend") = kodama::to_string(result.backend),
    Rcpp::Named("method") = kodama::to_string(result.method)
  );
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List kodama_matrix_cpp_temp(
  Rcpp::NumericMatrix data,
  Rcpp::Nullable<Rcpp::NumericMatrix> spatial = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> W = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> constrain = R_NilValue,
  Rcpp::Nullable<Rcpp::IntegerVector> fix = R_NilValue,
  int M = 100,
  int Tcycle = 20,
  int ncomp = 50,
  int landmarks = 10000,
  int splitting = 0,
  int n_cores = 4,
  int graph_neighbors = 100,
  int knn_k = 30,
  double spatial_resolution = 0.3,
  bool spatial_graph_mix = false,
  int spatial_constraint_mode = 0,
  std::string metric = "euclidean",
  std::string classifier = "knn",
  std::string backend = "cpu",
  int seed = 1234,
  bool progress = false,
  bool apply_kodama_dissimilarity = true
) {
  const int n = data.nrow();
  const int p = data.ncol();
  std::vector<float> x(static_cast<std::size_t>(n) * p, 0.0f);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) x[static_cast<std::size_t>(i) * p + j] = static_cast<float>(data(i, j));
  }

  kodama::KODAMAMatrixOptions options;
  options.runs = M;
  options.cycles = Tcycle;
  options.components = ncomp;
  options.landmarks = landmarks;
  options.splitting = splitting;
  options.graph_neighbors = graph_neighbors;
  options.n_threads = n_cores;
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
  if (!spatial.isNull()) {
    Rcpp::NumericMatrix s(spatial);
    if (s.nrow() != n) Rcpp::stop("spatial must have the same number of rows as data.");
    options.spatial_cols = s.ncol();
    options.spatial.assign(static_cast<std::size_t>(s.nrow()) * s.ncol(), 0.0f);
    for (int i = 0; i < s.nrow(); ++i) {
      for (int j = 0; j < s.ncol(); ++j) {
        options.spatial[static_cast<std::size_t>(i) * s.ncol() + j] = static_cast<float>(s(i, j));
      }
    }
  }

  kodama::MatrixView view{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)};
  kodama::KODAMAMatrixResult result = kodama::KODAMAMatrix(
    view,
    optional_int_vector(W),
    optional_int_vector(constrain),
    optional_int_vector(fix),
    options
  );

  Rcpp::NumericVector acc(result.acc.begin(), result.acc.end());
  Rcpp::NumericMatrix v(result.runs, result.cycles);
  for (int i = 0; i < result.runs; ++i) {
    for (int j = 0; j < result.cycles; ++j) {
      v(i, j) = result.v[static_cast<std::size_t>(i) * result.cycles + j];
    }
  }
  Rcpp::IntegerMatrix res(result.runs, result.samples);
  Rcpp::IntegerMatrix res_constrain(result.runs, result.samples);
  for (int i = 0; i < result.runs; ++i) {
    for (int j = 0; j < result.samples; ++j) {
      res(i, j) = result.res[static_cast<std::size_t>(i) * result.samples + j];
      res_constrain(i, j) = result.res_constrain[static_cast<std::size_t>(i) * result.samples + j];
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("acc") = acc,
    Rcpp::Named("v") = v,
    Rcpp::Named("res") = res,
    Rcpp::Named("knn_Rnanoflann") = graph_to_r(result.knn, result.samples),
    Rcpp::Named("base_knn") = graph_to_r(result.base_knn, result.samples),
    Rcpp::Named("data") = data,
    Rcpp::Named("res_constrain") = res_constrain,
    Rcpp::Named("n.cores") = result.n_threads,
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("timing") = Rcpp::List::create(
      Rcpp::Named("input_copy_seconds") = result.input_copy_seconds,
      Rcpp::Named("spatial_precompute_seconds") = result.spatial_precompute_seconds,
      Rcpp::Named("graph_seconds") = result.graph_seconds,
      Rcpp::Named("spatial_graph_seconds") = result.spatial_graph_seconds,
      Rcpp::Named("optimization_wall_seconds") = result.optimization_wall_seconds,
      Rcpp::Named("optimization_sum_seconds") = result.optimization_sum_seconds,
      Rcpp::Named("dissimilarity_seconds") = result.dissimilarity_seconds,
      Rcpp::Named("runtime_seconds") = result.runtime_seconds
    ),
    Rcpp::Named("peak_memory_mb") = result.peak_memory_mb
  );
}

// [[Rcpp::export]]
Rcpp::NumericMatrix kodama_umap_cuda_temp(
  Rcpp::IntegerMatrix indices,
  Rcpp::NumericMatrix distances,
  Rcpp::Nullable<Rcpp::NumericMatrix> init = R_NilValue,
  int n_neighbors = 30,
  int n_epochs = 200,
  double learning_rate = 1.0,
  double min_dist = 0.01,
  double repulsion_strength = 1.0,
  int negative_sample_rate = 5,
  int spectral_n_iter = 20,
  int seed = 1234
) {
  kodama::UMAPOptions options;
  options.n_neighbors = n_neighbors;
  options.n_epochs = n_epochs;
  options.learning_rate = learning_rate;
  options.min_dist = min_dist;
  options.repulsion_strength = repulsion_strength;
  options.negative_sample_rate = negative_sample_rate;
  options.spectral_n_iter = spectral_n_iter;
  options.seed = seed;
  if (!init.isNull()) {
    Rcpp::NumericMatrix init_matrix(init);
    if (init_matrix.nrow() != indices.nrow() || init_matrix.ncol() != 2) {
      Rcpp::stop("init must have nrow(indices) rows and 2 columns.");
    }
    options.init.assign(static_cast<std::size_t>(init_matrix.nrow()) * 2u, 0.0f);
    for (int i = 0; i < init_matrix.nrow(); ++i) {
      options.init[static_cast<std::size_t>(i) * 2u] = static_cast<float>(init_matrix(i, 0));
      options.init[static_cast<std::size_t>(i) * 2u + 1u] = static_cast<float>(init_matrix(i, 1));
    }
  }
  kodama::EmbeddingResult result = kodama::KODAMAUMAP_CUDA(
    graph_from_r(indices, distances),
    options
  );
  Rcpp::NumericMatrix out(result.samples, result.components);
  for (int i = 0; i < result.samples; ++i) {
    for (int j = 0; j < result.components; ++j) {
      out(i, j) = result.embedding[static_cast<std::size_t>(i) * result.components + j];
    }
  }
  out.attr("runtime_seconds") = result.runtime_seconds;
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix kodama_opentsne_cuda_temp(
  Rcpp::IntegerMatrix indices,
  Rcpp::NumericMatrix distances,
  Rcpp::Nullable<Rcpp::NumericMatrix> init = R_NilValue,
  int n_neighbors = 0,
  double perplexity = 15.0,
  int early_exaggeration_iter = 250,
  int n_iter = 500,
  double early_exaggeration = 12.0,
  double exaggeration = 1.0,
  double learning_rate = 0.0,
  bool learning_rate_auto = true,
  double initial_momentum = 0.8,
  double final_momentum = 0.8,
  double min_gain = 0.01,
  double max_step_norm = 5.0,
  int seed = 4
) {
  kodama::OpenTSNEOptions options;
  options.n_neighbors = n_neighbors;
  options.perplexity = perplexity;
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
  options.seed = seed;
  if (!init.isNull()) {
    Rcpp::NumericMatrix init_matrix(init);
    if (init_matrix.nrow() != indices.nrow() || init_matrix.ncol() != 2) {
      Rcpp::stop("init must have nrow(indices) rows and 2 columns.");
    }
    options.init.assign(static_cast<std::size_t>(init_matrix.nrow()) * 2u, 0.0f);
    for (int i = 0; i < init_matrix.nrow(); ++i) {
      options.init[static_cast<std::size_t>(i) * 2u] = static_cast<float>(init_matrix(i, 0));
      options.init[static_cast<std::size_t>(i) * 2u + 1u] = static_cast<float>(init_matrix(i, 1));
    }
  }
  kodama::EmbeddingResult result = kodama::KODAMAOpenTSNE_CUDA(
    graph_from_r(indices, distances),
    options
  );
  Rcpp::NumericMatrix out(result.samples, result.components);
  for (int i = 0; i < result.samples; ++i) {
    for (int j = 0; j < result.components; ++j) {
      out(i, j) = result.embedding[static_cast<std::size_t>(i) * result.components + j];
    }
  }
  out.attr("runtime_seconds") = result.runtime_seconds;
  return out;
}

// [[Rcpp::export]]
Rcpp::List kodama_knn_graph_temp(
  Rcpp::NumericMatrix data,
  int k = 30,
  std::string metric = "euclidean",
  std::string backend = "cpu",
  int n_threads = 4,
  int gpu_device = 0
) {
  const int n = data.nrow();
  const int p = data.ncol();
  std::vector<float> x(static_cast<std::size_t>(n) * p, 0.0f);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) x[static_cast<std::size_t>(i) * p + j] = static_cast<float>(data(i, j));
  }
  kodama::GraphClusterOptions options;
  options.k = k;
  options.metric = parse_metric(metric);
  options.backend = parse_backend(backend);
  options.n_threads = n_threads;
  options.gpu_device = gpu_device;
  kodama::NeighborGraph graph = kodama::KODAMAKNNGraph(kodama::MatrixView{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)}, options);
  return graph_to_r(graph, n);
}

// [[Rcpp::export]]
Rcpp::List kodama_graph_cluster_from_knn_temp(
  Rcpp::IntegerMatrix indices,
  Rcpp::NumericMatrix distances,
  std::string method = "louvain",
  std::string backend = "cpu",
  std::string weight = "distance",
  int n_threads = 4,
  int n_runs = 1,
  int n_iterations = 10,
  int random_walk_steps = 4,
  int n_clusters = 0,
  double resolution = 1.0,
  double resolution_init = 0.0,
  double resolution_delta = 0.2,
  double prune = 0.0,
  bool mutual = false,
  int seed = 1,
  int gpu_device = 0
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
  kodama::NeighborGraph graph = graph_from_r(indices, distances);
  return graph_cluster_result_to_r(kodama::KODAMAGraphCluster(graph, indices.nrow(), options));
}

// [[Rcpp::export]]
Rcpp::List kodama_embedding_cluster_temp(
  Rcpp::NumericMatrix embedding,
  std::string method = "louvain",
  std::string backend = "cpu",
  std::string graph_backend = "cpu",
  std::string weight = "distance",
  std::string metric = "euclidean",
  int k = 30,
  int n_threads = 4,
  int n_runs = 1,
  int n_iterations = 10,
  int random_walk_steps = 4,
  int n_clusters = 0,
  double resolution = 1.0,
  double resolution_init = 0.0,
  double resolution_delta = 0.2,
  double prune = 0.0,
  bool mutual = false,
  int seed = 1,
  int gpu_device = 0
) {
  const int n = embedding.nrow();
  const int p = embedding.ncol();
  std::vector<float> x(static_cast<std::size_t>(n) * p, 0.0f);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) x[static_cast<std::size_t>(i) * p + j] = static_cast<float>(embedding(i, j));
  }
  kodama::GraphClusterOptions graph_options;
  graph_options.k = k;
  graph_options.metric = parse_metric(metric);
  graph_options.backend = parse_backend(graph_backend);
  graph_options.n_threads = n_threads;
  graph_options.gpu_device = gpu_device;
  kodama::NeighborGraph graph = kodama::KODAMAKNNGraph(kodama::MatrixView{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)}, graph_options);

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
  Rcpp::List out = graph_cluster_result_to_r(kodama::KODAMAGraphCluster(graph, n, cluster_options));
  out["graph"] = graph_to_r(graph, n);
  return out;
}
