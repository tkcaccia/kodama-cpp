// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>

#include "kodama/kodama.hpp"

int main() {
  if (!kodama::MetalAvailable()) {
    std::cerr << "Apple Metal device is not visible to this process.\n";
    return 77;
  }

  constexpr int samples = 24;
  constexpr int dimensions = 4;
  std::vector<float> x(static_cast<std::size_t>(samples * dimensions), 0.0f);
  std::vector<int> labels(static_cast<std::size_t>(samples), 0);
  for (int row = 0; row < samples; ++row) {
    const int cls = row < samples / 2 ? 1 : 2;
    labels[static_cast<std::size_t>(row)] = cls;
    for (int column = 0; column < dimensions; ++column) {
      x[static_cast<std::size_t>(row * dimensions + column)] =
        static_cast<float>((cls == 1 ? -2.0 : 2.0) + 0.01 * row + 0.02 * column);
    }
  }
  kodama::KNNOptions options;
  options.backend = kodama::Backend::Metal;
  options.index_type = kodama::KNNIndexType::MetalExact;
  options.metric = kodama::DistanceMetric::Euclidean;
  options.k = 3;
  options.cv.folds = 3;
  options.cv.seed = 11;
  const kodama::KNNCVResult result = kodama::KNNCV_METAL(
    kodama::MatrixView{x.data(), samples, dimensions},
    labels,
    {},
    options
  );
  if (result.global_accuracy < 0.95 || result.parameters.backend != kodama::Backend::Metal) {
    std::cerr << "Metal KNN smoke test failed.\n";
    return 1;
  }

  options.index_type = kodama::KNNIndexType::MetalIVFFlat;
  options.ivf_nlist = 4;
  options.ivf_nprobe = 4;
  const kodama::KNNCVResult ivf_result = kodama::KNNCV_METAL(
    kodama::MatrixView{x.data(), samples, dimensions},
    labels,
    {},
    options
  );
  if (ivf_result.global_accuracy < 0.95 ||
      ivf_result.parameters.index_type != kodama::KNNIndexType::MetalIVFFlat ||
      ivf_result.parameters.ivf_nlist != 4 || ivf_result.parameters.ivf_nprobe != 4) {
    std::cerr << "Metal IVF KNN smoke test failed.\n";
    return 1;
  }

  kodama::GraphClusterOptions graph_options;
  graph_options.backend = kodama::Backend::Metal;
  graph_options.metric = kodama::DistanceMetric::Euclidean;
  graph_options.k = 5;
  const kodama::NeighborGraph graph = kodama::KODAMAKNNGraph_METAL(
    kodama::MatrixView{x.data(), samples, dimensions},
    graph_options
  );
  if (graph.neighbors != 5 || graph.indices.size() != static_cast<std::size_t>(samples * 5)) {
    std::cerr << "Metal graph smoke test failed.\n";
    return 1;
  }

  kodama::PCAOptions pca_options;
  pca_options.n_components = 3;
  pca_options.oversample = 1;
  pca_options.power_iterations = 1;
  pca_options.seed = 7;
  const kodama::PCAResult pca_cpu = kodama::PCA_CPU(
    kodama::MatrixView{x.data(), samples, dimensions}, pca_options
  );
  const kodama::PCAResult pca_metal = kodama::PCA_METAL(
    kodama::MatrixView{x.data(), samples, dimensions}, pca_options
  );
  if (pca_metal.backend != kodama::Backend::Metal ||
      pca_metal.scores.size() != static_cast<std::size_t>(samples * 3) ||
      pca_metal.loadings.size() != static_cast<std::size_t>(dimensions * 3)) {
    std::cerr << "Metal PCA dimensions or backend metadata failed.\n";
    return 1;
  }
  for (int component = 0; component < 3; ++component) {
    const float reference = std::max(1.0f, pca_cpu.singular_values[static_cast<std::size_t>(component)]);
    if (std::abs(pca_cpu.singular_values[static_cast<std::size_t>(component)] -
                 pca_metal.singular_values[static_cast<std::size_t>(component)]) / reference > 2e-3f) {
      std::cerr << "Metal PCA singular values disagree with CPU.\n";
      return 1;
    }
  }

  kodama::KODAMAMatrixOptions matrix_options;
  matrix_options.backend = kodama::Backend::Metal;
  matrix_options.runs = 2;
  matrix_options.cycles = 2;
  matrix_options.components = 2;
  matrix_options.landmarks = 18;
  matrix_options.splitting = 3;
  matrix_options.graph_neighbors = 8;
  matrix_options.n_threads = 1;
  matrix_options.knn.k = 3;
  matrix_options.knn.cv.folds = 3;
  matrix_options.pls.cv.folds = 3;
  matrix_options.apply_kodama_dissimilarity = true;

  matrix_options.classifier = kodama::CoreClassifier::KNN;
  const kodama::KODAMAMatrixResult matrix_knn = kodama::KODAMAMatrix_METAL(
    kodama::MatrixView{x.data(), samples, dimensions},
    {},
    {},
    {},
    matrix_options
  );
  if (matrix_knn.res.size() != static_cast<std::size_t>(matrix_options.runs * samples) ||
      matrix_knn.knn.indices.empty() || !std::isfinite(matrix_knn.runtime_seconds)) {
    std::cerr << "Metal KODAMA KNN smoke test failed.\n";
    return 1;
  }

  matrix_options.classifier = kodama::CoreClassifier::PLS_LDA;
  const kodama::KODAMAMatrixResult matrix_pls = kodama::KODAMAMatrix_METAL(
    kodama::MatrixView{x.data(), samples, dimensions},
    {},
    {},
    {},
    matrix_options
  );
  if (matrix_pls.res.size() != static_cast<std::size_t>(matrix_options.runs * samples) ||
      matrix_pls.knn.indices.empty() || !std::isfinite(matrix_pls.runtime_seconds)) {
    std::cerr << "Metal KODAMA PLS-LDA smoke test failed.\n";
    return 1;
  }

  matrix_options.classifier = kodama::CoreClassifier::KNN;
  const kodama::KODAMAMatrixResult matrix_graph_data = kodama::KODAMAMatrixFromGraphData(
    kodama::MatrixView{x.data(), samples, dimensions},
    graph,
    {},
    {},
    {},
    matrix_options
  );
  if (matrix_graph_data.backend != kodama::Backend::Metal ||
      matrix_graph_data.res.size() != static_cast<std::size_t>(matrix_options.runs * samples)) {
    std::cerr << "Metal graph-and-data KODAMA smoke test failed.\n";
    return 1;
  }

  matrix_options.graph_feature_components = 2;
  const kodama::KODAMAMatrixResult matrix_graph = kodama::KODAMAMatrixFromGraph(
    graph,
    samples,
    {},
    {},
    {},
    matrix_options
  );
  if (matrix_graph.backend != kodama::Backend::Metal ||
      matrix_graph.res.size() != static_cast<std::size_t>(matrix_options.runs * samples)) {
    std::cerr << "Metal graph-input KODAMA smoke test failed.\n";
    return 1;
  }

  bool rejected_metal_clustering = false;
  try {
    (void)kodama::KODAMAGraphCluster(graph, samples, graph_options);
  } catch (const std::runtime_error&) {
    rejected_metal_clustering = true;
  }
  if (!rejected_metal_clustering) {
    std::cerr << "Metal clustering silently fell back to CPU.\n";
    return 1;
  }
  std::cout << "Metal KNN and PCA smoke tests passed.\n";
  return 0;
}
