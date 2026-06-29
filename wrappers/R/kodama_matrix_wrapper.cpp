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

std::vector<int> optional_int_vector(Rcpp::Nullable<Rcpp::IntegerVector> value) {
  if (value.isNull()) return {};
  Rcpp::IntegerVector v(value);
  std::vector<int> out(static_cast<std::size_t>(v.size()));
  for (R_xlen_t i = 0; i < v.size(); ++i) out[static_cast<std::size_t>(i)] = v[i];
  return out;
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
  bool progress = false
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
  Rcpp::IntegerMatrix indices(result.samples, result.knn.neighbors);
  Rcpp::NumericMatrix distances(result.samples, result.knn.neighbors);
  for (int i = 0; i < result.samples; ++i) {
    for (int j = 0; j < result.knn.neighbors; ++j) {
      const std::size_t offset = static_cast<std::size_t>(i) * result.knn.neighbors + j;
      indices(i, j) = result.knn.indices[offset];
      distances(i, j) = result.knn.distances[offset];
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("acc") = acc,
    Rcpp::Named("v") = v,
    Rcpp::Named("res") = res,
    Rcpp::Named("knn_Rnanoflann") = Rcpp::List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("neighbors") = result.knn.neighbors
    ),
    Rcpp::Named("data") = data,
    Rcpp::Named("res_constrain") = res_constrain,
    Rcpp::Named("n.cores") = result.n_threads,
    Rcpp::Named("runtime_seconds") = result.runtime_seconds,
    Rcpp::Named("peak_memory_mb") = result.peak_memory_mb
  );
}
