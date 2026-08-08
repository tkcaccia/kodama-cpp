// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "kodama/kodama.hpp"
#include "../src/metal_backend.hpp"
#include "../src/spatial_grid_knn.hpp"

int main() {
  if (!kodama::MetalAvailable()) {
    std::cerr << "Apple Metal device is not visible to this process.\n";
    return 77;
  }

  // Sparse landmarks must be queried against the retained IVF index. Taking
  // the induced subgraph of a fixed-k global graph leaves these rows empty.
  constexpr int sparse_rows = 8192;
  constexpr int sparse_dimensions = 6;
  constexpr int sparse_landmarks = 32;
  constexpr int sparse_k = 8;
  std::vector<float> sparse_data(
    static_cast<std::size_t>(sparse_rows * sparse_dimensions), 0.0f
  );
  for (int row = 0; row < sparse_rows; ++row) {
    for (int column = 0; column < sparse_dimensions; ++column) {
      sparse_data[static_cast<std::size_t>(row * sparse_dimensions + column)] =
        std::sin(0.013f * static_cast<float>((row + 1) * (column + 2))) +
        0.0007f * static_cast<float>(row);
    }
  }
  std::vector<int> sparse_rows_selected(static_cast<std::size_t>(sparse_landmarks));
  std::vector<float> sparse_query(
    static_cast<std::size_t>(sparse_landmarks * sparse_dimensions), 0.0f
  );
  for (int local = 0; local < sparse_landmarks; ++local) {
    const int global = (local * sparse_rows) / sparse_landmarks;
    sparse_rows_selected[static_cast<std::size_t>(local)] = global;
    std::copy_n(
      sparse_data.data() + static_cast<std::size_t>(global * sparse_dimensions),
      sparse_dimensions,
      sparse_query.data() + static_cast<std::size_t>(local * sparse_dimensions)
    );
  }
  kodama::detail::MetalIVFStats sparse_stats;
  kodama::detail::NativeMetalKODAMAGraph sparse_resident =
    kodama::detail::metal_build_resident_kodama_graph_ivf(
      sparse_data, sparse_rows, sparse_dimensions, 16,
      kodama::DistanceMetric::Euclidean, 64, 0, 1, &sparse_stats
    );
  const kodama::NeighborGraph sparse_graph =
    kodama::detail::metal_resident_landmark_knn_graph(
      sparse_resident, sparse_query, sparse_rows_selected,
      sparse_k, 0, 0.99
    );
  if (sparse_graph.neighbors != sparse_k ||
      std::any_of(sparse_graph.indices.begin(), sparse_graph.indices.end(),
                  [](int id) { return id < 0; })) {
    std::cerr << "Metal sparse-landmark IVF query returned an incomplete graph.\n";
    return 1;
  }
  const kodama::detail::NativeKNNResult sparse_exact =
    kodama::detail::metal_exact_knn_search(
      sparse_query, sparse_landmarks, sparse_query, sparse_landmarks,
      sparse_dimensions, sparse_k, kodama::DistanceMetric::Euclidean,
      [&]() {
        std::vector<int> self(static_cast<std::size_t>(sparse_landmarks));
        std::iota(self.begin(), self.end(), 0);
        return self;
      }()
    );
  std::size_t sparse_hits = 0;
  for (int row = 0; row < sparse_landmarks; ++row) {
    for (int rank = 0; rank < sparse_k; ++rank) {
      const int candidate = sparse_graph.indices[
        static_cast<std::size_t>(row * sparse_k + rank)
      ];
      const auto begin = sparse_exact.indices.begin() + row * sparse_k;
      if (std::find(begin, begin + sparse_k, candidate) != begin + sparse_k) ++sparse_hits;
    }
  }
  if (static_cast<double>(sparse_hits) /
        static_cast<double>(sparse_landmarks * sparse_k) < 0.99) {
    std::cerr << "Metal sparse-landmark IVF recall fell below 0.99.\n";
    return 1;
  }

  std::vector<int> constrained_labels(static_cast<std::size_t>(sparse_rows));
  std::vector<int> constrained_groups(static_cast<std::size_t>(sparse_rows));
  for (int row = 0; row < sparse_rows; ++row) {
    constrained_labels[static_cast<std::size_t>(row)] = row % 5 + 1;
    constrained_groups[static_cast<std::size_t>(row)] = row / 7;
  }
  std::vector<int> expected_constrained = constrained_labels;
  for (int begin = 0; begin < sparse_rows; begin += 7) {
    const int end = std::min(sparse_rows, begin + 7);
    int counts[6] = {0, 0, 0, 0, 0, 0};
    for (int row = begin; row < end; ++row) ++counts[constrained_labels[row]];
    int best = 1;
    for (int label = 2; label <= 5; ++label) {
      if (counts[label] > counts[best]) best = label;
    }
    for (int row = begin; row < end; ++row) expected_constrained[row] = best;
  }
  kodama::detail::metal_prepare_resident_results(sparse_resident, 1);
  kodama::detail::metal_store_resident_result_row(
    sparse_resident, constrained_labels, 0, 0);
  kodama::detail::metal_constrain_resident_result_row(
    sparse_resident, constrained_groups, 5, 0, 0);
  if (kodama::detail::metal_download_resident_results(sparse_resident, 1) !=
      expected_constrained) {
    std::cerr << "Metal resident constrained majority differs from CPU.\n";
    return 1;
  }

  {
    constexpr int context_rows = 257;
    constexpr int context_dimensions = 7;
    constexpr int context_clusters = 11;
    std::vector<float> context_data(
      static_cast<std::size_t>(context_rows * context_dimensions), 0.0f
    );
    for (int row = 0; row < context_rows; ++row) {
      for (int column = 0; column < context_dimensions; ++column) {
        context_data[static_cast<std::size_t>(row * context_dimensions + column)] =
          std::sin(0.031f * static_cast<float>((row + 3) * (column + 1)));
      }
    }
    auto context = kodama::detail::metal_build_kmeans_context(
      context_data, context_rows, context_dimensions, 2, context_clusters
    );
    for (int lane = 0; lane < 2; ++lane) {
      std::vector<int> order(static_cast<std::size_t>(context_rows));
      std::iota(order.begin(), order.end(), 0);
      std::mt19937_64 rng(887u + static_cast<std::uint64_t>(lane));
      std::shuffle(order.begin(), order.end(), rng);
      const std::vector<int> expected = kodama::detail::metal_kmeans_labels(
        context_data, context_rows, context_dimensions, context_clusters, order, 5
      );
      const std::vector<int> observed = kodama::detail::metal_kmeans_context_labels(
        context, lane, context_clusters, order, 5
      );
      if (observed != expected) {
        std::cerr << "Resident Metal k-means changed one-shot assignments.\n";
        return 1;
      }
    }
    if (context.input_uploads() != 1) {
      std::cerr << "Resident Metal k-means uploaded its invariant input more than once.\n";
      return 1;
    }
  }

  // The resident PLS-LDA reductions must reproduce the former host
  // calculation from the same Metal-projected score matrix.
  constexpr int stats_rows = 37;
  constexpr int stats_predictors = 9;
  constexpr int stats_components = 5;
  constexpr int stats_classes = 3;
  std::vector<float> stats_x(
    static_cast<std::size_t>(stats_rows * stats_predictors), 0.0f
  );
  std::vector<float> stats_weights(
    static_cast<std::size_t>(stats_predictors * stats_components), 0.0f
  );
  std::vector<int> stats_labels(static_cast<std::size_t>(stats_rows), 0);
  for (int row = 0; row < stats_rows; ++row) {
    stats_labels[static_cast<std::size_t>(row)] = row % stats_classes;
    for (int column = 0; column < stats_predictors; ++column) {
      stats_x[static_cast<std::size_t>(row * stats_predictors + column)] =
        0.03f * static_cast<float>((row + 1) * (column + 2)) +
        std::sin(0.07f * static_cast<float>((row + 3) * (column + 1)));
    }
  }
  for (int predictor = 0; predictor < stats_predictors; ++predictor) {
    for (int component = 0; component < stats_components; ++component) {
      stats_weights[static_cast<std::size_t>(predictor * stats_components + component)] =
        std::cos(0.11f * static_cast<float>((predictor + 1) * (component + 2)));
    }
  }
  kodama::detail::metal_set_pls_residency_epoch(0x31415926ULL);
  const std::vector<float> reference_scores = kodama::detail::metal_matrix_multiply(
    stats_x,
    stats_rows,
    stats_predictors,
    stats_weights,
    stats_predictors,
    stats_components,
    false,
    false,
    true
  );
  std::vector<float> reference_sums(
    static_cast<std::size_t>(stats_classes * stats_components), 0.0f
  );
  std::vector<float> reference_crossprod(
    static_cast<std::size_t>(stats_components * stats_components), 0.0f
  );
  for (int row = 0; row < stats_rows; ++row) {
    const int cls = stats_labels[static_cast<std::size_t>(row)];
    for (int component = 0; component < stats_components; ++component) {
      reference_sums[static_cast<std::size_t>(cls * stats_components + component)] +=
        reference_scores[static_cast<std::size_t>(row * stats_components + component)];
    }
    for (int left = 0; left < stats_components; ++left) {
      for (int right = 0; right < stats_components; ++right) {
        reference_crossprod[static_cast<std::size_t>(left * stats_components + right)] +=
          reference_scores[static_cast<std::size_t>(row * stats_components + left)] *
          reference_scores[static_cast<std::size_t>(row * stats_components + right)];
      }
    }
  }
  const kodama::detail::MetalPLSScoreStatistics resident_statistics =
    kodama::detail::metal_pls_score_statistics(
      stats_x,
      stats_rows,
      stats_predictors,
      stats_weights,
      stats_components,
      stats_labels,
      stats_classes
    );
  for (std::size_t i = 0; i < reference_sums.size(); ++i) {
    if (std::abs(reference_sums[i] - resident_statistics.class_sums[i]) > 2e-4f) {
      std::cerr << "Metal resident PLS-LDA class sums changed the calculation.\n";
      return 1;
    }
  }
  for (std::size_t i = 0; i < reference_crossprod.size(); ++i) {
    if (std::abs(reference_crossprod[i] - resident_statistics.score_crossprod[i]) > 2e-3f) {
      std::cerr << "Metal resident PLS-LDA score cross-product changed the calculation.\n";
      return 1;
    }
  }
  std::vector<float> stats_linear(
    static_cast<std::size_t>(stats_classes * stats_components), 0.0f
  );
  std::vector<float> stats_constants = {-0.2f, 0.1f, -0.05f};
  const std::vector<int> stats_class_labels = {11, 22, 33};
  for (int cls = 0; cls < stats_classes; ++cls) {
    for (int component = 0; component < stats_components; ++component) {
      stats_linear[static_cast<std::size_t>(cls * stats_components + component)] =
        0.17f * static_cast<float>(cls + 1) -
        0.09f * static_cast<float>(component + 1);
    }
  }
  std::vector<int> reference_predictions(static_cast<std::size_t>(stats_rows), 0);
  for (int row = 0; row < stats_rows; ++row) {
    int best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int cls = 0; cls < stats_classes; ++cls) {
      float score = stats_constants[static_cast<std::size_t>(cls)];
      for (int component = 0; component < stats_components; ++component) {
        score += reference_scores[static_cast<std::size_t>(row * stats_components + component)] *
          stats_linear[static_cast<std::size_t>(cls * stats_components + component)];
      }
      if (score > best_score) {
        best_score = score;
        best = cls;
      }
    }
    reference_predictions[static_cast<std::size_t>(row)] =
      stats_class_labels[static_cast<std::size_t>(best)];
  }
  const std::vector<int> resident_predictions = kodama::detail::metal_pls_lda_predict(
    stats_x,
    stats_rows,
    stats_predictors,
    stats_weights,
    stats_components,
    stats_linear,
    stats_constants,
    stats_class_labels
  );
  if (reference_predictions != resident_predictions) {
    std::cerr << "Metal resident PLS-LDA prediction changed the calculation.\n";
    return 1;
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

  constexpr int spatial_samples = 257;
  constexpr int spatial_dimensions = 2;
  constexpr int spatial_k = 17;
  std::vector<float> spatial(
    static_cast<std::size_t>(spatial_samples * spatial_dimensions), 0.0f
  );
  for (int row = 0; row < spatial_samples; ++row) {
    spatial[static_cast<std::size_t>(row * 2)] =
      std::sin(0.071f * static_cast<float>(row)) + 0.002f * static_cast<float>(row);
    spatial[static_cast<std::size_t>(row * 2 + 1)] =
      std::cos(0.053f * static_cast<float>(row)) - 0.001f * static_cast<float>(row);
  }
  const kodama::NeighborGraph spatial_cpu = kodama::detail::spatial_grid_self_knn(
    spatial.data(), spatial_samples, spatial_dimensions, spatial_k, 2, false, false
  );
  const kodama::detail::NativeKNNResult spatial_metal =
    kodama::detail::metal_spatial_grid_self_knn(
      spatial, spatial_samples, spatial_dimensions, spatial_k, false
    );
  if (spatial_cpu.indices != spatial_metal.indices) {
    std::cerr << "Metal spatial grid KNN changed exact neighbor identities.\n";
    return 1;
  }
  for (std::size_t i = 0; i < spatial_cpu.distances.size(); ++i) {
    const float metal_distance = std::sqrt(std::max(0.0f, spatial_metal.distances[i]));
    if (std::abs(spatial_cpu.distances[i] - metal_distance) > 2e-5f) {
      std::cerr << "Metal spatial grid KNN changed exact neighbor distances.\n";
      return 1;
    }
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

  kodama::KNNOptions resident_options = options;
  resident_options.backend = kodama::Backend::Metal;
  resident_options.index_type = kodama::KNNIndexType::MetalIVFFlat;
  resident_options.ivf_nlist = 4;
  resident_options.ivf_nprobe = 4;
  kodama::ResidentIVFIndex resident = kodama::BuildResidentIVFIndex(
    kodama::MatrixView{x.data(), samples, dimensions},
    resident_options
  );
  if (!resident.valid() || resident.backend() != kodama::Backend::Metal ||
      resident.rows() != samples || resident.dimensions() != dimensions ||
      resident.nlist() != 4 || resident.build_seconds() < 0.0) {
    std::cerr << "Resident Metal IVF index metadata failed.\n";
    return 1;
  }
  kodama::ResidentIVFSearchStats resident_stats;
  bool rejected_zero_k = false;
  try {
    (void)kodama::SearchResidentIVFIndexSelf(resident, 0, true);
  } catch (const std::invalid_argument&) {
    rejected_zero_k = true;
  }
  if (!rejected_zero_k) {
    std::cerr << "Resident Metal IVF accepted k=0.\n";
    return 1;
  }
  bool rejected_null_query = false;
  try {
    (void)kodama::SearchResidentIVFIndex(
      resident,
      kodama::MatrixView{static_cast<const float*>(nullptr), 1, dimensions},
      1
    );
  } catch (const std::invalid_argument&) {
    rejected_null_query = true;
  }
  if (!rejected_null_query) {
    std::cerr << "Resident Metal IVF accepted a null query pointer.\n";
    return 1;
  }
  const kodama::NeighborGraph resident_first =
    kodama::SearchResidentIVFIndexSelf(resident, 3, true, &resident_stats);
  const kodama::NeighborGraph resident_second =
    kodama::SearchResidentIVFIndexSelf(resident, 3, true);
  if (resident_stats.backend != kodama::Backend::Metal ||
      resident_stats.nlist != 4 || resident_stats.nprobe != 4 ||
      resident_first.indices != resident_second.indices ||
      resident_first.distances != resident_second.distances) {
    std::cerr << "Resident Metal IVF reuse was not deterministic.\n";
    return 1;
  }
  const kodama::NeighborGraph resident_query = kodama::SearchResidentIVFIndex(
    resident,
    kodama::MatrixView{x.data(), 5, dimensions},
    3
  );
  if (resident_query.neighbors != 3 || resident_query.indices.size() != 15 ||
      resident_query.indices.front() != 1) {
    std::cerr << "Resident Metal IVF external-query search failed.\n";
    return 1;
  }
  kodama::ResidentIVFIndex moved_resident = std::move(resident);
  if (resident.valid() || !moved_resident.valid() ||
      moved_resident.backend() != kodama::Backend::Metal ||
      moved_resident.rows() != samples || moved_resident.dimensions() != dimensions) {
    std::cerr << "Resident Metal IVF move ownership failed.\n";
    return 1;
  }
  resident = std::move(moved_resident);
  if (!resident.valid() || moved_resident.valid()) {
    std::cerr << "Resident Metal IVF move assignment failed.\n";
    return 1;
  }

  kodama::KODAMAGraphOptions kodama_graph_options;
  kodama_graph_options.neighbors = 3;
  kodama_graph_options.backend = kodama::Backend::Metal;
  kodama_graph_options.materialize_graph = true;
  kodama_graph_options.metric = kodama::DistanceMetric::Euclidean;
  kodama_graph_options.index_type = kodama::KNNIndexType::MetalIVFFlat;
  kodama_graph_options.ivf_nlist = 4;
  kodama_graph_options.ivf_nprobe = 4;
  const kodama::KODAMAGraphResult graph_result = kodama::KODAMAGraph_METAL(
    kodama::MatrixView{x.data(), samples, dimensions},
    kodama_graph_options
  );
  if (graph_result.index_type != kodama::KNNIndexType::MetalIVFFlat ||
      graph_result.ivf_nlist != 4 || graph_result.ivf_nprobe != 4 ||
      graph_result.ivf_pilot_recall < 0.99 || graph_result.knn.neighbors != 3) {
    std::cerr << "Metal KODAMA graph did not use the resident IVF path: index="
              << kodama::to_string(graph_result.index_type)
              << " nlist=" << graph_result.ivf_nlist
              << " nprobe=" << graph_result.ivf_nprobe
              << " recall=" << graph_result.ivf_pilot_recall
              << " neighbors=" << graph_result.knn.neighbors << ".\n";
    return 1;
  }
  for (int row = 0; row < samples; ++row) {
    for (int column = 0; column < resident_first.neighbors; ++column) {
      const int neighbor =
        resident_first.indices[static_cast<std::size_t>(row * resident_first.neighbors + column)];
      if (neighbor == row + 1) {
        std::cerr << "Resident Metal IVF self-exclusion failed.\n";
        return 1;
      }
    }
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

  kodama::CoreOptions graph_core_options;
  graph_core_options.classifier = kodama::CoreClassifier::KNN;
  graph_core_options.cycles = 1;
  graph_core_options.seed = 11;
  graph_core_options.knn.backend = kodama::Backend::Metal;
  graph_core_options.knn.k = 3;
  graph_core_options.knn.cv.folds = 3;
  graph_core_options.knn.cv.stratified = false;
  graph_core_options.knn.cv.seed = 11;
  const kodama::CoreResult graph_core = kodama::CoreKNNGraph_METAL(
    graph, samples, labels, {}, {}, graph_core_options
  );
  kodama::CoreOptions graph_core_cpu_options = graph_core_options;
  graph_core_cpu_options.knn.backend = kodama::Backend::CPU;
  const kodama::CoreResult graph_core_cpu = kodama::CoreKNNGraph_CPU(
    graph, samples, labels, {}, {}, graph_core_cpu_options
  );
  if (graph_core.clbest.size() != labels.size() ||
      graph_core.cycles_completed != 1 || !std::isfinite(graph_core.accbest)) {
    std::cerr << "Direct Metal graph-input KNN core failed.\n";
    return 1;
  }
  if (graph_core.clbest != graph_core_cpu.clbest ||
      graph_core.accbest != graph_core_cpu.accbest) {
    std::cerr << "Compact-code Metal KNN disagrees with CPU.\n";
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
      std::cerr << "Metal PCA singular values disagree with CPU at component "
                << component << ": cpu="
                << pca_cpu.singular_values[static_cast<std::size_t>(component)]
                << " metal="
                << pca_metal.singular_values[static_cast<std::size_t>(component)] << ".\n";
      return 1;
    }
  }

  // Reusing a host allocation must refresh general Metal matrix inputs.
  std::vector<float> cache_probe = x;
  (void)kodama::PCA_METAL(
    kodama::MatrixView{cache_probe.data(), samples, dimensions}, pca_options
  );
  for (int row = 0; row < samples; ++row) {
    for (int column = 0; column < dimensions; ++column) {
      cache_probe[static_cast<std::size_t>(row * dimensions + column)] =
        static_cast<float>(0.15 * row * row + 0.7 * column + (row % 3));
    }
  }
  const kodama::PCAResult refreshed_cpu = kodama::PCA_CPU(
    kodama::MatrixView{cache_probe.data(), samples, dimensions}, pca_options
  );
  const kodama::PCAResult refreshed_metal = kodama::PCA_METAL(
    kodama::MatrixView{cache_probe.data(), samples, dimensions}, pca_options
  );
  for (int component = 0; component < 3; ++component) {
    const float reference =
      std::max(1.0f, refreshed_cpu.singular_values[static_cast<std::size_t>(component)]);
    if (std::abs(refreshed_cpu.singular_values[static_cast<std::size_t>(component)] -
                 refreshed_metal.singular_values[static_cast<std::size_t>(component)]) /
          reference > 2e-3f) {
      std::cerr << "Metal PCA reused stale host matrix contents.\n";
      return 1;
    }
  }

  // Sequential folds may reuse host allocation addresses. Their resident
  // Metal matrices must still be distinguished from one another.
  constexpr int pls_samples = 75;
  constexpr int pls_dimensions = 32;
  std::vector<float> pls_x(
    static_cast<std::size_t>(pls_samples * pls_dimensions), 0.0f
  );
  std::vector<int> pls_labels(static_cast<std::size_t>(pls_samples), 0);
  for (int row = 0; row < pls_samples; ++row) {
    const int cls = row % 3;
    pls_labels[static_cast<std::size_t>(row)] = cls + 1;
    for (int column = 0; column < pls_dimensions; ++column) {
      pls_x[static_cast<std::size_t>(row * pls_dimensions + column)] =
        std::sin(0.013f * static_cast<float>((row + 1) * (column + 1))) +
        (column % 3 == cls ? 0.8f : -0.2f) +
        0.002f * static_cast<float>(row - column);
    }
  }
  kodama::PLSOptions pls_options;
  pls_options.backend = kodama::Backend::Metal;
  pls_options.cv.folds = 5;
  pls_options.cv.stratified = false;
  pls_options.cv.seed = 42;
  pls_options.max_components = 10;
  pls_options.fixed_components = 10;
  pls_options.center = true;
  pls_options.scale = true;
  pls_options.n_threads = 1;
  const kodama::PLSCVResult pls_serial = kodama::PLSLDACV_METAL(
    kodama::MatrixView{pls_x.data(), pls_samples, pls_dimensions},
    pls_labels,
    {},
    pls_options
  );
  pls_options.n_threads = 4;
  const kodama::PLSCVResult pls_parallel = kodama::PLSLDACV_METAL(
    kodama::MatrixView{pls_x.data(), pls_samples, pls_dimensions},
    pls_labels,
    {},
    pls_options
  );
  if (pls_serial.predicted_labels != pls_parallel.predicted_labels ||
      std::abs(pls_serial.global_accuracy - pls_parallel.global_accuracy) > 1e-12) {
    std::cerr << "Metal PLS-LDA predictions depend on fold worker count.\n";
    return 1;
  }
  kodama::PLSOptions pls_cpu_options = pls_options;
  pls_cpu_options.backend = kodama::Backend::CPU;
  pls_cpu_options.n_threads = 4;
  const kodama::PLSCVResult pls_cpu = kodama::PLSLDACV_CPU(
    kodama::MatrixView{pls_x.data(), pls_samples, pls_dimensions},
    pls_labels,
    {},
    pls_cpu_options
  );
  if (pls_cpu.predicted_labels != pls_parallel.predicted_labels ||
      std::abs(pls_cpu.global_accuracy - pls_parallel.global_accuracy) > 1e-12) {
    std::cerr << "Metal sufficient-statistics PLS-LDA disagrees with CPU.\n";
    return 1;
  }

  const int predict_train_rows = 60;
  const int predict_test_rows = pls_samples - predict_train_rows;
  const std::vector<int> predict_train_labels(
    pls_labels.begin(), pls_labels.begin() + predict_train_rows
  );
  const kodama::MatrixView predict_train{
    pls_x.data(), predict_train_rows, pls_dimensions
  };
  const kodama::MatrixView predict_test{
    pls_x.data() + static_cast<std::size_t>(predict_train_rows * pls_dimensions),
    predict_test_rows,
    pls_dimensions
  };
  const std::vector<int> predict_cpu = kodama::PLSLDAPredict_CPU(
    predict_train, predict_train_labels, predict_test, pls_cpu_options
  );
  const std::vector<int> predict_metal = kodama::PLSLDAPredict_METAL(
    predict_train, predict_train_labels, predict_test, pls_options
  );
  if (predict_cpu != predict_metal ||
      predict_metal.size() != static_cast<std::size_t>(predict_test_rows)) {
    std::cerr << "Direct Metal PLS-LDA prediction disagrees with CPU.\n";
    return 1;
  }

  // Stratified folds are label-dependent and intentionally bypass the fixed
  // fold cache. A second call must not reuse resident matrices from the first
  // call merely because temporary host allocations receive the same address.
  std::vector<int> shifted_pls_labels(static_cast<std::size_t>(pls_samples), 0);
  for (int row = 0; row < pls_samples; ++row) {
    shifted_pls_labels[static_cast<std::size_t>(row)] =
      ((row / 5 + 2 * row) % 3) + 1;
  }
  kodama::PLSOptions stratified_metal = pls_options;
  stratified_metal.cv.stratified = true;
  stratified_metal.n_threads = 1;
  (void)kodama::PLSLDACV_METAL(
    kodama::MatrixView{pls_x.data(), pls_samples, pls_dimensions},
    pls_labels,
    {},
    stratified_metal
  );
  const kodama::PLSCVResult shifted_metal = kodama::PLSLDACV_METAL(
    kodama::MatrixView{pls_x.data(), pls_samples, pls_dimensions},
    shifted_pls_labels,
    {},
    stratified_metal
  );
  kodama::PLSOptions stratified_cpu = stratified_metal;
  stratified_cpu.backend = kodama::Backend::CPU;
  const kodama::PLSCVResult shifted_cpu = kodama::PLSLDACV_CPU(
    kodama::MatrixView{pls_x.data(), pls_samples, pls_dimensions},
    shifted_pls_labels,
    {},
    stratified_cpu
  );
  if (shifted_cpu.fold_assignments != shifted_metal.fold_assignments ||
      shifted_cpu.predicted_labels != shifted_metal.predicted_labels ||
      std::abs(shifted_cpu.global_accuracy - shifted_metal.global_accuracy) > 1e-12) {
    std::cerr << "Metal stratified PLS-LDA reused stale fold matrices.\n";
    return 1;
  }

  // Match the CPU regression for a finite randomized SIMPLS power vector
  // whose float32 squared norm is not representable.
  constexpr int robust_samples = 1200;
  constexpr int robust_dimensions = 8;
  std::vector<float> robust_x(
    static_cast<std::size_t>(robust_samples * robust_dimensions), 0.0f
  );
  std::vector<int> robust_labels(static_cast<std::size_t>(robust_samples), 0);
  for (int row = 0; row < robust_samples; ++row) {
    const int cls = row % 3;
    robust_labels[static_cast<std::size_t>(row)] = cls + 1;
    for (int column = 0; column < robust_dimensions; ++column) {
      const float class_signal =
        cls == 0 ? (column == 0 ? 10000.0f : -2500.0f) :
        cls == 1 ? (column == 1 ? 10000.0f : -2500.0f) :
                   (column < 2 ? -10000.0f : 2500.0f);
      const float variation =
        700.0f * std::sin(
          0.017f * static_cast<float>((row + 1) * (column + 1))
        ) +
        350.0f * std::cos(
          0.011f * static_cast<float>((row + 3) * (column + 2))
        );
      robust_x[static_cast<std::size_t>(row * robust_dimensions + column)] =
        class_signal + variation;
    }
  }
  kodama::PLSOptions robust_pls = pls_options;
  robust_pls.n_threads = 1;
  robust_pls.cv.seed = 9;
  robust_pls.max_components = 2;
  robust_pls.fixed_components = 2;
  robust_pls.scale = false;
  const kodama::PLSCVResult robust_metal = kodama::PLSLDACV_METAL(
    kodama::MatrixView{
      robust_x.data(), robust_samples, robust_dimensions
    },
    robust_labels,
    {},
    robust_pls
  );
  if (robust_metal.selected_components != 2 ||
      robust_metal.global_accuracy <= 0.99) {
    std::cerr << "Metal robust SIMPLS norm regression failed.\n";
    return 1;
  }

  kodama::NeighborGraph corrected_cpu = graph;
  kodama::NeighborGraph corrected_metal = graph;
  constexpr int agreement_runs = 7;
  std::vector<int> run_labels(static_cast<std::size_t>(agreement_runs * samples));
  for (int run = 0; run < agreement_runs; ++run) {
    for (int sample = 0; sample < samples; ++sample) {
      run_labels[static_cast<std::size_t>(run * samples + sample)] =
        1 + ((sample + 2 * run + sample / 5) % 4);
    }
  }
  kodama::KODAMADissimilarityInPlace(
    corrected_cpu, run_labels, agreement_runs, samples, kodama::Backend::CPU, 2
  );
  kodama::KODAMADissimilarityInPlace(
    corrected_metal, run_labels, agreement_runs, samples, kodama::Backend::Metal, 1
  );
  if (corrected_cpu.indices != corrected_metal.indices ||
      corrected_cpu.distances.size() != corrected_metal.distances.size()) {
    std::cerr << "Metal KODAMA correction changed graph topology.\n";
    return 1;
  }
  for (std::size_t i = 0; i < corrected_cpu.distances.size(); ++i) {
    if (std::abs(corrected_cpu.distances[i] - corrected_metal.distances[i]) > 2e-5f) {
      std::cerr << "Metal KODAMA correction disagrees with CPU.\n";
      return 1;
    }
  }

  const std::vector<float> preprocessing_test = {
    1.5f, 2.5f, 3.5f, 4.5f,
    2.5f, 3.5f, 4.5f, 5.5f
  };
  for (const auto method : {kodama::NormalizationMethod::PQN,
                            kodama::NormalizationMethod::Sum,
                            kodama::NormalizationMethod::Median,
                            kodama::NormalizationMethod::Sqrt,
                            kodama::NormalizationMethod::None}) {
    kodama::NormalizationOptions preprocessing_options;
    preprocessing_options.method = method;
    const auto cpu = kodama::Normalization_CPU(
      kodama::MatrixView{x.data(), samples, dimensions},
      kodama::MatrixView{preprocessing_test.data(), 2, dimensions}, preprocessing_options);
    const auto metal = kodama::Normalization_METAL(
      kodama::MatrixView{x.data(), samples, dimensions},
      kodama::MatrixView{preprocessing_test.data(), 2, dimensions}, preprocessing_options);
    if (metal.backend != kodama::Backend::Metal || cpu.train.size() != metal.train.size()) {
      std::cerr << "Metal normalization metadata failed.\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu.train.size(); ++i) {
      if (std::abs(cpu.train[i] - metal.train[i]) > 3e-5f) {
        std::cerr << "Metal normalization disagrees with CPU.\n";
        return 1;
      }
    }
  }
  for (const auto method : {kodama::ScalingMethod::None,
                            kodama::ScalingMethod::Centering,
                            kodama::ScalingMethod::Autoscaling,
                            kodama::ScalingMethod::RangeScaling,
                            kodama::ScalingMethod::ParetoScaling}) {
    kodama::ScalingOptions preprocessing_options;
    preprocessing_options.method = method;
    const auto cpu = kodama::Scaling_CPU(
      kodama::MatrixView{x.data(), samples, dimensions},
      kodama::MatrixView{preprocessing_test.data(), 2, dimensions}, preprocessing_options);
    const auto metal = kodama::Scaling_METAL(
      kodama::MatrixView{x.data(), samples, dimensions},
      kodama::MatrixView{preprocessing_test.data(), 2, dimensions}, preprocessing_options);
    for (std::size_t i = 0; i < cpu.train.size(); ++i) {
      if (std::abs(cpu.train[i] - metal.train[i]) > 3e-5f) {
        std::cerr << "Metal scaling disagrees with CPU.\n";
        return 1;
      }
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
  matrix_options.n_threads = 0;
  matrix_options.knn.k = 3;
  matrix_options.knn.cv.folds = 3;
  matrix_options.pls.cv.folds = 3;
  matrix_options.apply_kodama_dissimilarity = true;
  matrix_options.compute_visual_init = true;
  matrix_options.materialize_graph = true;

  matrix_options.classifier = kodama::CoreClassifier::KNN;
  kodama::KODAMAGraphOptions lazy_metal_graph_options = kodama_graph_options;
  lazy_metal_graph_options.neighbors = matrix_options.graph_neighbors;
  lazy_metal_graph_options.materialize_graph = false;
  const kodama::KODAMAGraphResult lazy_metal_graph = kodama::KODAMAGraph_METAL(
    kodama::MatrixView{x.data(), samples, dimensions}, lazy_metal_graph_options
  );
  if (!lazy_metal_graph.handle || !lazy_metal_graph.handle->valid() ||
      !lazy_metal_graph.knn.indices.empty() || lazy_metal_graph.neighbors != 8) {
    std::cerr << "Lazy Metal KODAMA graph handle construction failed.\n";
    return 1;
  }
  kodama::KODAMAMatrixOptions lazy_matrix_options = matrix_options;
  lazy_matrix_options.materialize_graph = false;
  const kodama::KODAMAMatrixResult lazy_matrix_first = kodama::KODAMAMatrix(
    kodama::MatrixView{x.data(), samples, dimensions}, lazy_metal_graph,
    {}, {}, {}, lazy_matrix_options
  );
  const kodama::KODAMAMatrixResult lazy_matrix_second = kodama::KODAMAMatrix(
    kodama::MatrixView{x.data(), samples, dimensions}, lazy_metal_graph,
    {}, {}, {}, lazy_matrix_options
  );
  if (lazy_matrix_first.graph_builds != 0 || lazy_matrix_second.graph_builds != 0 ||
      !lazy_matrix_first.knn.indices.empty() || !lazy_matrix_second.knn.indices.empty() ||
      lazy_matrix_first.res != lazy_matrix_second.res) {
    std::size_t differing = 0;
    for (std::size_t i = 0; i < lazy_matrix_first.res.size(); ++i) {
      differing += lazy_matrix_first.res[i] != lazy_matrix_second.res[i];
    }
    std::cerr << "Reusable Metal KODAMA graph handle failed: differing labels="
              << differing << ", first acc="
              << (lazy_matrix_first.acc.empty() ? -1.0 : lazy_matrix_first.acc.front())
              << ", second acc="
              << (lazy_matrix_second.acc.empty() ? -1.0 : lazy_matrix_second.acc.front())
              << ".\n";
    return 1;
  }
  const kodama::KODAMAMatrixResult matrix_knn = kodama::KODAMAMatrix_METAL(
    kodama::MatrixView{x.data(), samples, dimensions},
    {},
    {},
    {},
    matrix_options
  );
  if (matrix_knn.res.size() != static_cast<std::size_t>(matrix_options.runs * samples) ||
      matrix_knn.knn.indices.empty() ||
      matrix_knn.graph_builds != 1 ||
      !matrix_knn.has_visual_init ||
      matrix_knn.visual_init.backend != kodama::Backend::Metal ||
      matrix_knn.visual_init.umap.size() != static_cast<std::size_t>(samples * 2) ||
      matrix_knn.visual_init.opentsne.size() != static_cast<std::size_t>(samples * 2) ||
      !matrix_knn.gpu_scheduler_enabled ||
      matrix_knn.gpu_scheduler_lanes < 1 ||
      matrix_knn.gpu_scheduler_lanes > matrix_options.runs ||
      matrix_knn.kmeans_input_uploads != 1 ||
      matrix_knn.projection_sparse_uploads !=
        static_cast<std::uint64_t>(matrix_options.runs) ||
      matrix_knn.projection_full_downloads != 0 ||
      matrix_knn.result_row_uploads != 0 ||
      matrix_knn.result_matrix_downloads != 1 ||
      !std::isfinite(matrix_knn.runtime_seconds)) {
    std::cerr << "Metal KODAMA KNN smoke test failed.\n";
    return 1;
  }

  kodama::UMAPOptions umap_options;
  umap_options.n_neighbors = 6;
  umap_options.n_epochs = 5;
  umap_options.n_threads = 1;
  umap_options.seed = 19;
  umap_options.init = matrix_knn.visual_init.umap;
  umap_options.init_source = "raw_pca";
  umap_options.init_backend = kodama::Backend::Metal;
  const kodama::EmbeddingResult metal_umap = kodama::KODAMAUMAP_METAL(
    matrix_knn.knn,
    umap_options
  );
  if (metal_umap.backend != kodama::Backend::Metal ||
      metal_umap.samples != samples || metal_umap.components != 2 ||
      metal_umap.embedding.size() != static_cast<std::size_t>(samples * 2) ||
      metal_umap.initialization != "raw_pca" ||
      metal_umap.initialization_backend != kodama::Backend::Metal ||
      metal_umap.optimizer != "metal_clean_atomic_edge_sampler" ||
      metal_umap.graph_edges == 0 || metal_umap.graph_max_weight <= 0.0f) {
    std::cerr << "Metal UMAP metadata failed.\n";
    return 1;
  }
  for (const float value : metal_umap.embedding) {
    if (!std::isfinite(value)) {
      std::cerr << "Metal UMAP produced a non-finite value.\n";
      return 1;
    }
  }

  kodama::OpenTSNEOptions tsne_options;
  tsne_options.n_neighbors = 6;
  tsne_options.perplexity = 3.0;
  tsne_options.early_exaggeration_iter = 2;
  tsne_options.n_iter = 3;
  tsne_options.n_threads = 1;
  tsne_options.seed = 19;
  tsne_options.init = matrix_knn.visual_init.opentsne;
  tsne_options.init_source = "raw_pca";
  tsne_options.init_backend = kodama::Backend::Metal;
  const kodama::EmbeddingResult metal_tsne = kodama::KODAMAOpenTSNE_METAL(
    matrix_knn.knn,
    tsne_options
  );
  if (metal_tsne.backend != kodama::Backend::Metal ||
      metal_tsne.samples != samples || metal_tsne.components != 2 ||
      metal_tsne.embedding.size() != static_cast<std::size_t>(samples * 2) ||
      metal_tsne.initialization != "raw_pca" ||
      metal_tsne.initialization_backend != kodama::Backend::Metal ||
      metal_tsne.optimizer != "metal_opentsne_fft_grid_sparse_knn_float32" ||
      metal_tsne.graph_edges != 0 || metal_tsne.graph_max_weight != 0.0f) {
    std::cerr << "Metal openTSNE metadata failed.\n";
    return 1;
  }
  for (const float value : metal_tsne.embedding) {
    if (!std::isfinite(value)) {
      std::cerr << "Metal openTSNE produced a non-finite value.\n";
      return 1;
    }
  }

  const kodama::KODAMAMatrixResult matrix_knn_repeat =
    kodama::KODAMAMatrix_METAL(
      kodama::MatrixView{x.data(), samples, dimensions},
      {},
      {},
      {},
      matrix_options
    );
  if (matrix_knn.res != matrix_knn_repeat.res ||
      matrix_knn.knn.indices != matrix_knn_repeat.knn.indices) {
    std::cerr << "Resident Metal KODAMA KNN is not repeatable.\n";
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
      matrix_pls.knn.indices.empty() || !std::isfinite(matrix_pls.runtime_seconds) ||
      !matrix_pls.gpu_auto_workers || matrix_pls.gpu_scheduler_lanes > matrix_options.runs) {
    std::cerr << "Metal KODAMA PLS-LDA smoke test failed.\n";
    return 1;
  }

  const kodama::KODAMAMatrixResult matrix_pls_repeat =
    kodama::KODAMAMatrix_METAL(
      kodama::MatrixView{x.data(), samples, dimensions},
      {},
      {},
      {},
      matrix_options
    );
  if (matrix_pls.res != matrix_pls_repeat.res ||
      matrix_pls.knn.indices != matrix_pls_repeat.knn.indices) {
    std::cerr << "Resident Metal KODAMA PLS-LDA is not repeatable.\n";
    return 1;
  }

  matrix_options.classifier = kodama::CoreClassifier::KNN;
  const kodama::KODAMAMatrixResult matrix_graph_data = kodama::KODAMAMatrixFromGraphData_METAL(
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
  const kodama::KODAMAMatrixResult matrix_graph = kodama::KODAMAMatrixFromGraph_METAL(
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
  std::cout << "Metal KNN, PLS-LDA, PCA, UMAP, openTSNE, and preprocessing tests passed.\n";
  return 0;
}
