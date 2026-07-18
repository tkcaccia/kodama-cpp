// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct Dataset {
  int rows = 0;
  int columns = 0;
  std::vector<float> data;
  std::vector<int> labels;
};

Dataset make_dataset(int rows, int columns, int classes, std::uint64_t seed) {
  Dataset out;
  out.rows = rows;
  out.columns = columns;
  out.data.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns));
  out.labels.resize(static_cast<std::size_t>(rows));
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> center_noise(0.0f, 0.35f);
  std::normal_distribution<float> sample_noise(0.0f, 0.55f);
  std::vector<float> centers(static_cast<std::size_t>(classes) * static_cast<std::size_t>(columns));
  for (int cls = 0; cls < classes; ++cls) {
    for (int column = 0; column < columns; ++column) {
      const float signal = column % classes == cls ? 3.0f : 0.0f;
      centers[static_cast<std::size_t>(cls) * static_cast<std::size_t>(columns) +
        static_cast<std::size_t>(column)] = signal + center_noise(rng);
    }
  }
  for (int row = 0; row < rows; ++row) {
    const int cls = row % classes;
    out.labels[static_cast<std::size_t>(row)] = cls + 1;
    for (int column = 0; column < columns; ++column) {
      out.data[static_cast<std::size_t>(row) * static_cast<std::size_t>(columns) +
        static_cast<std::size_t>(column)] =
        centers[static_cast<std::size_t>(cls) * static_cast<std::size_t>(columns) +
          static_cast<std::size_t>(column)] + sample_noise(rng);
    }
  }
  return out;
}

double graph_overlap(const kodama::NeighborGraph& reference, const kodama::NeighborGraph& observed, int rows) {
  const int k = std::min(reference.neighbors, observed.neighbors);
  double hits = 0.0;
  for (int row = 0; row < rows; ++row) {
    std::unordered_set<int> expected;
    for (int column = 0; column < k; ++column) {
      expected.insert(reference.indices[static_cast<std::size_t>(row) * reference.neighbors + column]);
    }
    for (int column = 0; column < k; ++column) {
      const int value = observed.indices[static_cast<std::size_t>(row) * observed.neighbors + column];
      if (expected.count(value) != 0) hits += 1.0;
    }
  }
  return hits / static_cast<double>(rows * k);
}

template<class Function>
double timed(Function&& function) {
  const auto start = std::chrono::steady_clock::now();
  function();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void print_row(
  const std::string& operation,
  const std::string& backend,
  double seconds,
  double quality,
  const std::string& quality_name
) {
  std::cout << operation << ',' << backend << ',' << std::fixed << std::setprecision(6)
            << seconds << ',' << quality_name << ',' << quality << '\n';
}

}  // namespace

int main() {
  std::cout << "operation,backend,seconds,quality_metric,quality\n";
  const Dataset graph_data = make_dataset(4000, 32, 10, 17);
  const kodama::MatrixView graph_view{graph_data.data.data(),
    static_cast<std::size_t>(graph_data.rows), static_cast<std::size_t>(graph_data.columns)};
  kodama::GraphClusterOptions graph_options;
  graph_options.k = 30;
  graph_options.metric = kodama::DistanceMetric::Euclidean;
  graph_options.n_threads = 4;
  kodama::NeighborGraph cpu_graph;
  graph_options.backend = kodama::Backend::CPU;
  const double cpu_graph_seconds = timed([&]() { cpu_graph = kodama::KODAMAKNNGraph_CPU(graph_view, graph_options); });
  print_row("knn_graph", "native_hnsw", cpu_graph_seconds, 1.0, "reference");

  if (!kodama::MetalAvailable()) return 0;
  kodama::NeighborGraph metal_graph;
  graph_options.backend = kodama::Backend::Metal;
  const double metal_graph_seconds = timed([&]() { metal_graph = kodama::KODAMAKNNGraph_METAL(graph_view, graph_options); });
  print_row("knn_graph", "metal_exact", metal_graph_seconds,
    graph_overlap(metal_graph, cpu_graph, graph_data.rows), "hnsw_recall_vs_exact");

  kodama::KNNOptions knn;
  knn.k = 30;
  knn.metric = kodama::DistanceMetric::Euclidean;
  knn.cv.folds = 5;
  knn.cv.seed = 19;
  knn.n_threads = 4;
  knn.backend = kodama::Backend::CPU;
  knn.index_type = kodama::KNNIndexType::NativeHNSW;
  const kodama::KNNCVResult cpu_knn = kodama::KNNCV_CPU(graph_view, graph_data.labels, {}, knn);
  print_row("knn_cv", "native_hnsw", cpu_knn.runtime_seconds, cpu_knn.global_accuracy, "accuracy");
  knn.backend = kodama::Backend::Metal;
  knn.index_type = kodama::KNNIndexType::MetalExact;
  const kodama::KNNCVResult metal_knn = kodama::KNNCV_METAL(graph_view, graph_data.labels, {}, knn);
  print_row("knn_cv", "metal_exact", metal_knn.runtime_seconds, metal_knn.global_accuracy, "accuracy");

  const Dataset pls_data = make_dataset(2500, 128, 8, 23);
  const kodama::MatrixView pls_view{pls_data.data.data(),
    static_cast<std::size_t>(pls_data.rows), static_cast<std::size_t>(pls_data.columns)};
  kodama::PLSOptions pls;
  pls.max_components = 20;
  pls.fixed_components = 20;
  pls.cv.folds = 5;
  pls.cv.seed = 29;
  pls.n_threads = 4;
  pls.backend = kodama::Backend::CPU;
  const kodama::PLSCVResult cpu_pls = kodama::PLSLDACV_CPU(pls_view, pls_data.labels, {}, pls);
  print_row("plslda_cv", "cpu_float32", cpu_pls.runtime_seconds, cpu_pls.global_accuracy, "accuracy");
  pls.backend = kodama::Backend::Metal;
  const kodama::PLSCVResult metal_pls = kodama::PLSLDACV_METAL(pls_view, pls_data.labels, {}, pls);
  print_row("plslda_cv", "metal_float32", metal_pls.runtime_seconds, metal_pls.global_accuracy, "accuracy");

  const Dataset matrix_data = make_dataset(800, 24, 6, 31);
  const kodama::MatrixView matrix_view{matrix_data.data.data(),
    static_cast<std::size_t>(matrix_data.rows), static_cast<std::size_t>(matrix_data.columns)};
  kodama::KODAMAMatrixOptions matrix;
  matrix.runs = 4;
  matrix.cycles = 4;
  matrix.landmarks = 500;
  matrix.splitting = 20;
  matrix.graph_neighbors = 40;
  matrix.components = 10;
  matrix.n_threads = 4;
  matrix.knn.k = 20;
  matrix.knn.cv.folds = 5;
  matrix.pls.cv.folds = 5;
  matrix.apply_kodama_dissimilarity = true;
  matrix.classifier = kodama::CoreClassifier::KNN;
  matrix.backend = kodama::Backend::CPU;
  const kodama::KODAMAMatrixResult cpu_matrix = kodama::KODAMAMatrix_CPU(matrix_view, {}, {}, {}, matrix);
  print_row("kodama_knn", "cpu", cpu_matrix.runtime_seconds, 0.0, "not_scored");
  matrix.backend = kodama::Backend::Metal;
  const kodama::KODAMAMatrixResult metal_matrix = kodama::KODAMAMatrix_METAL(matrix_view, {}, {}, {}, matrix);
  print_row("kodama_knn", "metal", metal_matrix.runtime_seconds, 0.0, "not_scored");

  matrix.classifier = kodama::CoreClassifier::PLS_LDA;
  matrix.backend = kodama::Backend::CPU;
  const kodama::KODAMAMatrixResult cpu_matrix_pls = kodama::KODAMAMatrix_CPU(matrix_view, {}, {}, {}, matrix);
  print_row("kodama_plslda", "cpu", cpu_matrix_pls.runtime_seconds, 0.0, "not_scored");
  matrix.backend = kodama::Backend::Metal;
  const kodama::KODAMAMatrixResult metal_matrix_pls = kodama::KODAMAMatrix_METAL(matrix_view, {}, {}, {}, matrix);
  print_row("kodama_plslda", "metal", metal_matrix_pls.runtime_seconds, 0.0, "not_scored");
  return 0;
}
