// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include <cmath>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "kodama/kodama.hpp"
#include "../src/native_knn.hpp"
#include "../src/spatial_grid_knn.hpp"
#if defined(KODAMA_ENABLE_CUDA)
#include "../src/kodama_matrix_cuda.hpp"
#include "../src/native_cuda_backend.hpp"
#endif

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

const std::vector<std::string>& evolution_policy_ablation_names() {
  static const std::vector<std::string> names = {
    "no_prediction_guidance",
    "fixed_proposal_budget",
    "no_transition_proposal",
    "greedy_acceptance",
    "raw_cv_score",
    "no_pls_transition_coarsening",
    "no_pls_fragmentation_penalty"
  };
  return names;
}

const kodama::KODAMAStageTiming* find_timing(
  const std::vector<kodama::KODAMAStageTiming>& timings,
  const char* step
) {
  const auto it = std::find_if(
    timings.begin(), timings.end(), [step](const kodama::KODAMAStageTiming& timing) {
      return timing.step == step;
    }
  );
  return it == timings.end() ? nullptr : &*it;
}

void require_close(
  const std::vector<float>& left,
  const std::vector<float>& right,
  const float tolerance,
  const char* message
) {
  require(left.size() == right.size(), message);
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (std::isnan(left[i]) && std::isnan(right[i])) continue;
    if (std::abs(left[i] - right[i]) > tolerance) throw std::runtime_error(message);
  }
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, const char* message) {
  bool threw = false;
  try {
    callable();
  } catch (const Exception&) {
    threw = true;
  }
  require(threw, message);
}

void test_public_string_contracts() {
  const auto require_name = [](const char* actual, const char* expected) {
    require(std::string(actual) == expected, "Public enum string contract changed.");
  };

  require_name(kodama::to_string(kodama::Backend::Auto), "auto");
  require_name(kodama::to_string(kodama::Backend::CPU), "cpu");
  require_name(kodama::to_string(kodama::Backend::CUDA), "cuda");
  require_name(kodama::to_string(kodama::Backend::Metal), "metal");
  require_name(kodama::to_string(static_cast<kodama::Backend>(-1)), "unknown");

  require_name(kodama::to_string(kodama::DistanceMetric::Cosine), "cosine");
  require_name(kodama::to_string(kodama::DistanceMetric::InnerProduct), "inner_product");
  require_name(kodama::to_string(kodama::DistanceMetric::Euclidean), "euclidean");
  require_name(kodama::to_string(static_cast<kodama::DistanceMetric>(-1)), "unknown");

  require_name(kodama::to_string(kodama::KNNIndexType::PrecomputedGraph), "precomputed_graph");
  require_name(kodama::to_string(kodama::KNNIndexType::NativeHNSW), "native_hnsw");
  require_name(kodama::to_string(kodama::KNNIndexType::CudaExact), "cuda_exact");
  require_name(kodama::to_string(kodama::KNNIndexType::CudaIVFFlat), "cuda_ivf_flat");
  require_name(kodama::to_string(kodama::KNNIndexType::MetalExact), "metal_exact");
  require_name(kodama::to_string(kodama::KNNIndexType::MetalIVFFlat), "metal_ivf_flat");
  require_name(kodama::to_string(static_cast<kodama::KNNIndexType>(-1)), "unknown");

  require_name(kodama::to_string(kodama::PLSMode::PLS_DA), "pls_da");
  require_name(kodama::to_string(kodama::PLSMode::PLS_LDA), "pls_lda");
  require_name(kodama::to_string(static_cast<kodama::PLSMode>(-1)), "unknown");

  require_name(kodama::to_string(kodama::CoreClassifier::PLS_LDA), "pls_lda");
  require_name(kodama::to_string(kodama::CoreClassifier::KNN), "knn");
  require_name(kodama::to_string(static_cast<kodama::CoreClassifier>(-1)), "unknown");

  require_name(kodama::to_string(kodama::GraphWeightType::SNN), "snn");
  require_name(kodama::to_string(kodama::GraphWeightType::Distance), "distance");
  require_name(kodama::to_string(kodama::GraphWeightType::Adaptive), "adaptive");
  require_name(kodama::to_string(kodama::GraphWeightType::Binary), "binary");
  require_name(kodama::to_string(static_cast<kodama::GraphWeightType>(-1)), "unknown");

  require_name(
    kodama::to_string(kodama::GraphFeatureMode::LaplacianSelfTuning),
    "laplacian_self_tuning"
  );
  require_name(kodama::to_string(static_cast<kodama::GraphFeatureMode>(-1)), "unknown");
}

void test_public_error_contracts() {
  const std::vector<float> values = {
    0.0f, 0.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f
  };
  const kodama::MatrixView view{values.data(), 4, 2};
  const kodama::MatrixView two_rows{values.data(), 2, 2};
  const std::vector<int> labels = {1, 1, 2, 2};

  kodama::KODAMAMatrixOptions matrix_options;
  matrix_options.runs = 1;
  matrix_options.cycles = 0;
  matrix_options.landmarks = 3;
  matrix_options.graph_neighbors = 2;
  matrix_options.compute_visual_init = false;
  matrix_options.materialize_graph = false;

  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAMatrix_CPU(
      view, std::vector<int>{1}, {}, {}, matrix_options
    );
  }, "KODAMAMatrix accepted mismatched starting labels.");
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAMatrix_CPU(two_rows, {}, {}, {}, matrix_options);
  }, "KODAMAMatrix accepted fewer than three rows.");

  kodama::KODAMAMatrixOptions invalid_runs = matrix_options;
  invalid_runs.runs = 0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAMatrix_CPU(view, {}, {}, {}, invalid_runs);
  }, "KODAMAMatrix accepted zero M runs.");

  kodama::KODAMAMatrixOptions invalid_cycles = matrix_options;
  invalid_cycles.cycles = -1;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAMatrix_CPU(view, {}, {}, {}, invalid_cycles);
  }, "KODAMAMatrix accepted a negative Tcycle count.");

  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAMatrix_CPU(view, {}, std::vector<int>{1}, {}, matrix_options);
  }, "KODAMAMatrix accepted mismatched constrain metadata.");
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAMatrix_CPU(view, {}, {}, std::vector<int>{1}, matrix_options);
  }, "KODAMAMatrix accepted mismatched fixed metadata.");

  kodama::KODAMAMatrixOptions invalid_spatial = matrix_options;
  invalid_spatial.spatial.assign(8, 0.0f);
  invalid_spatial.spatial_cols = 0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAMatrix_CPU(view, {}, {}, {}, invalid_spatial);
  }, "KODAMAMatrix accepted spatial data without dimensions.");
  invalid_spatial.spatial_cols = 2;
  invalid_spatial.spatial.resize(7);
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAMatrix_CPU(view, {}, {}, {}, invalid_spatial);
  }, "KODAMAMatrix accepted mismatched spatial storage.");
  invalid_spatial.spatial.resize(8);
  invalid_spatial.spatial_resolution = 0.0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAMatrix_CPU(view, {}, {}, {}, invalid_spatial);
  }, "KODAMAMatrix accepted a non-positive spatial resolution.");

  const kodama::ResidentIVFIndex empty_index;
  require(!empty_index.valid(), "A default resident IVF index must be empty.");
  require(empty_index.backend() == kodama::Backend::CPU &&
          empty_index.metric() == kodama::DistanceMetric::Euclidean,
          "Default resident IVF backend or metric metadata is inconsistent.");
  require(empty_index.rows() == 0 && empty_index.dimensions() == 0 &&
          empty_index.nlist() == 0 && empty_index.build_seconds() == 0.0,
          "Default resident IVF metadata is inconsistent.");
  require_throws<std::invalid_argument>([&] {
    (void)kodama::SearchResidentIVFIndex(empty_index, view, 2);
  }, "Resident IVF search accepted an empty index.");
  require_throws<std::invalid_argument>([&] {
    (void)kodama::SearchResidentIVFIndexSelf(empty_index, 2);
  }, "Resident IVF self-search accepted an empty index.");
  kodama::KNNOptions cpu_ivf;
  cpu_ivf.backend = kodama::Backend::CPU;
  const kodama::MatrixView null_ivf_view{static_cast<const float*>(nullptr), 4, 2};
  require_throws<std::invalid_argument>([&] {
    (void)kodama::BuildResidentIVFIndex(null_ivf_view, cpu_ivf);
  }, "Resident IVF construction accepted a null matrix pointer.");
  require_throws<std::invalid_argument>([&] {
    (void)kodama::BuildResidentIVFIndex(view, cpu_ivf);
  }, "Resident IVF construction accepted the CPU backend.");
#if !defined(KODAMA_ENABLE_CUDA)
  kodama::KNNOptions unavailable_cuda_ivf = cpu_ivf;
  unavailable_cuda_ivf.backend = kodama::Backend::CUDA;
  require_throws<std::runtime_error>([&] {
    (void)kodama::BuildResidentIVFIndex(view, unavailable_cuda_ivf);
  }, "Resident IVF construction silently accepted unavailable CUDA.");
#endif
#if !defined(KODAMA_ENABLE_METAL)
  kodama::KNNOptions unavailable_metal_ivf = cpu_ivf;
  unavailable_metal_ivf.backend = kodama::Backend::Metal;
  require_throws<std::runtime_error>([&] {
    (void)kodama::BuildResidentIVFIndex(view, unavailable_metal_ivf);
  }, "Resident IVF construction silently accepted unavailable Metal.");
#endif

  kodama::NeighborGraph graph;
  graph.neighbors = 2;
  graph.index_base = kodama::GraphIndexBase::One;
  graph.indices = {
    2, 3,
    1, 4,
    1, 4,
    2, 3
  };
  graph.distances = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  kodama::KODAMAGraphOptions dispatched_graph_options;
  dispatched_graph_options.neighbors = 2;
  dispatched_graph_options.n_threads = 2;
  dispatched_graph_options.metric = kodama::DistanceMetric::Cosine;
  dispatched_graph_options.backend = kodama::Backend::CPU;
  dispatched_graph_options.materialize_graph = true;
  const kodama::KODAMAGraphResult dispatched_graph =
    kodama::KODAMAGraph(view, dispatched_graph_options);
  require(dispatched_graph.samples == 4 && dispatched_graph.dimensions == 2 &&
          dispatched_graph.backend == kodama::Backend::CPU &&
          dispatched_graph.knn.indices.size() == 8,
          "Generic KODAMAGraph did not dispatch to the CPU cosine path.");

  kodama::KODAMAMatrixOptions dispatched_matrix_options = matrix_options;
  dispatched_matrix_options.backend = kodama::Backend::CPU;
  const std::vector<int> orchestration_constrain = {10, 10, 20, 20};
  const kodama::KODAMAMatrixResult dispatched_matrix = kodama::KODAMAMatrix(
    view, labels, orchestration_constrain, {}, dispatched_matrix_options
  );
  require(dispatched_matrix.backend == kodama::Backend::CPU &&
          dispatched_matrix.samples == 4 && dispatched_matrix.runs == 1,
          "Generic KODAMAMatrix did not dispatch to CPU.");

  const kodama::KODAMAMatrixResult dispatched_graph_data =
    kodama::KODAMAMatrixFromGraphData(
      view, graph, labels, orchestration_constrain, {}, dispatched_matrix_options
    );
  require(dispatched_graph_data.backend == kodama::Backend::CPU &&
          dispatched_graph_data.samples == 4 &&
          dispatched_graph_data.graph_builds == 0,
          "Generic graph-and-data KODAMAMatrix did not dispatch to CPU.");
  const kodama::KODAMAMatrixResult dispatched_graph_only =
    kodama::KODAMAMatrixFromGraph(
      graph, 4, labels, orchestration_constrain, {}, dispatched_matrix_options
    );
  require(dispatched_graph_only.backend == kodama::Backend::CPU &&
          dispatched_graph_only.samples == 4 &&
          dispatched_graph_only.graph_builds == 0,
          "Generic graph-only KODAMAMatrix did not dispatch to CPU.");

  kodama::KODAMAGraphResult graph_result;
  graph_result.samples = 4;
  graph_result.dimensions = 2;
  graph_result.backend = kodama::Backend::CPU;
  graph_result.knn = graph;
  const kodama::KODAMAMatrixResult graph_data_without_init = kodama::KODAMAMatrix(
    view, graph_result, labels, orchestration_constrain, {}, dispatched_matrix_options
  );
  require(!graph_data_without_init.has_visual_init,
          "Graph-and-data KODAMAMatrix ignored compute_visual_init=false.");
  const kodama::KODAMAMatrixResult graph_only_without_init = kodama::KODAMAMatrix(
    graph_result, labels, orchestration_constrain, {}, dispatched_matrix_options
  );
  require(!graph_only_without_init.has_visual_init,
          "Graph-only KODAMAMatrix invented an unavailable raw-data initialization.");

  kodama::KODAMAGraphResult mismatched_graph = graph_result;
  mismatched_graph.samples = 3;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAMatrix(
      view, mismatched_graph, labels, orchestration_constrain, {}, dispatched_matrix_options
    );
  }, "Graph-and-data KODAMAMatrix accepted a mismatched sample count.");
  mismatched_graph = graph_result;
  mismatched_graph.dimensions = 1;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAMatrix(
      view, mismatched_graph, labels, orchestration_constrain, {}, dispatched_matrix_options
    );
  }, "Graph-and-data KODAMAMatrix accepted a mismatched feature count.");
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAMatrix(
      kodama::KODAMAGraphResult{}, labels, orchestration_constrain, {}, dispatched_matrix_options
    );
  }, "Graph-only KODAMAMatrix accepted an empty prepared graph.");

  require_throws<std::invalid_argument>([&] {
    kodama::NeighborGraph empty;
    kodama::KODAMADissimilarityInPlace(empty, labels, 1, 4);
  }, "KODAMA dissimilarity accepted an empty graph.");
  require_throws<std::invalid_argument>([&] {
    kodama::NeighborGraph inconsistent = graph;
    inconsistent.distances.pop_back();
    kodama::KODAMADissimilarityInPlace(inconsistent, labels, 1, 4);
  }, "KODAMA dissimilarity accepted inconsistent graph arrays.");
  require_throws<std::invalid_argument>([&] {
    kodama::KODAMADissimilarityInPlace(graph, std::vector<int>{1}, 1, 4);
  }, "KODAMA dissimilarity accepted a mismatched label matrix.");

  kodama::NeighborGraph malformed = graph;
  malformed.neighbors = 0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAUMAP_CPU(malformed);
  }, "UMAP accepted a graph with zero neighbors.");
  malformed = graph;
  malformed.distances.pop_back();
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAOpenTSNE_CPU(malformed);
  }, "openTSNE accepted inconsistent graph arrays.");

  kodama::UMAPOptions umap;
  umap.n_epochs = -1;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAUMAP_CPU(graph, umap);
  }, "UMAP accepted negative epochs.");
  umap = kodama::UMAPOptions();
  umap.min_dist = std::numeric_limits<double>::quiet_NaN();
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAUMAP_CPU(graph, umap);
  }, "UMAP accepted non-finite min_dist.");
  umap = kodama::UMAPOptions();
  umap.negative_sample_rate = -1;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAUMAP_CPU(graph, umap);
  }, "UMAP accepted a negative sampling rate.");
  umap = kodama::UMAPOptions();
  umap.learning_rate = 0.0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAUMAP_CPU(graph, umap);
  }, "UMAP accepted a zero learning rate.");
  umap = kodama::UMAPOptions();
  umap.repulsion_strength = 0.0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAUMAP_CPU(graph, umap);
  }, "UMAP accepted zero repulsion.");
  umap = kodama::UMAPOptions();
  umap.spectral_n_iter = 0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAUMAP_CPU(graph, umap);
  }, "UMAP accepted zero spectral iterations.");
  umap = kodama::UMAPOptions();
  umap.n_threads = 0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAUMAP_CPU(graph, umap);
  }, "UMAP accepted zero threads.");
  umap = kodama::UMAPOptions();
  umap.n_components = 3;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAUMAP_CPU(graph, umap);
  }, "CPU UMAP accepted an unsupported component count.");

  kodama::OpenTSNEOptions tsne;
  tsne.perplexity = 1.0;
  tsne.n_neighbors = 2;
  tsne.early_exaggeration_iter = 1;
  tsne.n_iter = 1;
  tsne.perplexity = 0.0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAOpenTSNE_CPU(graph, tsne);
  }, "openTSNE accepted zero perplexity.");
  tsne.perplexity = 100.0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAOpenTSNE_CPU(graph, tsne);
  }, "openTSNE accepted perplexity larger than the graph.");
  tsne.perplexity = 1.0;
  tsne.theta = -0.1;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAOpenTSNE_CPU(graph, tsne);
  }, "openTSNE accepted theta outside [0, 1].");
  tsne.theta = 0.5;
  tsne.early_exaggeration_iter = 0;
  tsne.n_iter = 0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAOpenTSNE_CPU(graph, tsne);
  }, "openTSNE accepted zero total iterations.");
  tsne.early_exaggeration_iter = 1;
  tsne.n_iter = 1;
  tsne.early_exaggeration = 0.0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAOpenTSNE_CPU(graph, tsne);
  }, "openTSNE accepted zero early exaggeration.");
  tsne.early_exaggeration = 12.0;
  tsne.learning_rate_auto = false;
  tsne.learning_rate = 0.0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAOpenTSNE_CPU(graph, tsne);
  }, "openTSNE accepted a non-automatic zero learning rate.");
  tsne.learning_rate_auto = true;
  tsne.initial_momentum = -1.0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAOpenTSNE_CPU(graph, tsne);
  }, "openTSNE accepted negative momentum.");
  tsne.initial_momentum = 0.8;
  tsne.min_gain = 0.0;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAOpenTSNE_CPU(graph, tsne);
  }, "openTSNE accepted zero min_gain.");
  tsne.min_gain = 0.01;
  tsne.n_components = 4;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::KODAMAOpenTSNE_CPU(graph, tsne);
  }, "CPU openTSNE accepted an unsupported component count.");

  kodama::CoreOptions zero_cycle_core;
  zero_cycle_core.classifier = kodama::CoreClassifier::KNN;
  zero_cycle_core.cycles = 0;
  zero_cycle_core.knn.cv.folds = 2;
  zero_cycle_core.knn.k = 1;
  const kodama::CoreResult zero_cycle = kodama::CoreKNNGraph_CPU(
    graph, 4, labels, {}, {}, zero_cycle_core
  );
  require(zero_cycle.clbest == labels && zero_cycle.cycles_completed == 0 &&
          zero_cycle.vect_acc.empty() && zero_cycle.vect_score.empty(),
          "Zero-cycle Core changed labels or reported a completed transition.");

  kodama::CoreOptions negative_cycle_core = zero_cycle_core;
  negative_cycle_core.cycles = -1;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::CoreKNNGraph_CPU(
      graph, 4, labels, {}, {}, negative_cycle_core
    );
  }, "Core accepted a negative cycle count.");

  kodama::CoreOptions fixed_core = zero_cycle_core;
  fixed_core.cycles = 5;
  fixed_core.seed = 19;
  fixed_core.evolutionary_search = true;
  fixed_core.guarded_diversity = true;
  fixed_core.auto_class_coarsening = true;
  fixed_core.many_to_one_absorption = true;
  fixed_core.adaptive_proposal_size = false;
  const std::vector<int> all_fixed(4, 1);
  const kodama::CoreResult fixed_result = kodama::CoreKNNGraph_CPU(
    graph, 4, labels, {}, all_fixed, fixed_core
  );
  require(fixed_result.clbest == labels && fixed_result.cycles_completed >= 1,
          "All-fixed Core modified an immutable label.");
  require(fixed_result.vect_acc.size() == 5 && fixed_result.vect_score.size() == 5,
          "All-fixed Core did not retain its complete trace allocation.");

  kodama::CoreOptions grouped_core = fixed_core;
  grouped_core.seed = 23;
  grouped_core.adaptive_proposal_size = true;
  const std::vector<int> grouped_labels = {1, 1, 2, 2};
  const std::vector<int> grouped_constrain = {10, 10, 20, 20};
  const kodama::CoreResult grouped_result = kodama::CoreKNNGraph_CPU(
    graph, 4, grouped_labels, grouped_constrain, {}, grouped_core
  );
  require(grouped_result.clbest[0] == grouped_result.clbest[1] &&
          grouped_result.clbest[2] == grouped_result.clbest[3],
          "Core split an atomic constrained group.");
  require(std::isfinite(grouped_result.accbest) &&
          std::isfinite(grouped_result.scorebest),
          "Core state-machine scores are not finite.");

  const kodama::CoreResult grouped_repeat = kodama::CoreKNNGraph_CPU(
    graph, 4, grouped_labels, grouped_constrain, {}, grouped_core
  );
  require(grouped_repeat.clbest == grouped_result.clbest &&
          grouped_repeat.cvpredbest == grouped_result.cvpredbest &&
          grouped_repeat.vect_acc == grouped_result.vect_acc &&
          grouped_repeat.vect_score == grouped_result.vect_score &&
          grouped_repeat.cycles_completed == grouped_result.cycles_completed,
          "Core stochastic evolution is not reproducible for a fixed seed.");

  const auto policy_fields = [](const kodama::EvolutionPolicy& policy) {
    return std::array<bool, 8>{
      policy.prediction_guidance,
      policy.adaptive_proposal_size,
      policy.transition_proposal,
      policy.stochastic_acceptance,
      policy.diversity_multiplier,
      policy.pls_transition_coarsening,
      policy.pls_fragmentation_penalty,
      policy.reject_single_class
    };
  };
  const kodama::EvolutionPolicy full_policy =
    kodama::EvolutionPolicy::from_name("full");
  require(policy_fields(full_policy) == policy_fields(kodama::EvolutionPolicy::standard()),
          "Explicit full evolution policy differs from the standard policy.");
  for (const std::string& name : evolution_policy_ablation_names()) {
    const auto fields = policy_fields(kodama::EvolutionPolicy::from_name(name));
    const auto full_fields = policy_fields(full_policy);
    int differences = 0;
    for (std::size_t i = 0; i < fields.size(); ++i) {
      differences += fields[i] != full_fields[i] ? 1 : 0;
    }
    require(differences == 1,
            "An evolution ablation does not change exactly one policy field.");
  }
  require_throws<std::invalid_argument>([] {
    (void)kodama::EvolutionPolicy::from_name("not_a_policy");
  }, "Unknown evolution policy name was accepted.");

  kodama::CoreOptions explicit_full_core = grouped_core;
  explicit_full_core.evolution = kodama::EvolutionPolicy::from_name("full");
  const kodama::CoreResult explicit_full_result = kodama::CoreKNNGraph_CPU(
    graph, 4, grouped_labels, grouped_constrain, {}, explicit_full_core
  );
  require(explicit_full_result.clbest == grouped_result.clbest &&
          explicit_full_result.cvpredbest == grouped_result.cvpredbest &&
          explicit_full_result.vect_acc == grouped_result.vect_acc &&
          explicit_full_result.vect_score == grouped_result.vect_score,
          "Explicit full policy differs from the ordinary default call.");
  require(explicit_full_result.cv_evaluations == explicit_full_core.cycles + 1,
          "Core did not perform exactly one initial CV plus one CV per cycle.");

  for (const std::string& name : evolution_policy_ablation_names()) {
    kodama::CoreOptions ablation = grouped_core;
    ablation.evolution = kodama::EvolutionPolicy::from_name(name);
    const kodama::CoreResult ablated = kodama::CoreKNNGraph_CPU(
      graph, 4, grouped_labels, grouped_constrain, {}, ablation
    );
    require(ablated.fold_assignments == grouped_result.fold_assignments,
            "Evolution policy changed fixed fold assignments.");
    require(ablated.cv_evaluations == ablation.cycles + 1,
            "Evolution ablation did not perform one CV evaluation per cycle.");
  }

  for (const std::string& name : {
         std::string("no_pls_transition_coarsening"),
         std::string("no_pls_fragmentation_penalty")}) {
    kodama::CoreOptions knn_specificity = grouped_core;
    knn_specificity.evolution = kodama::EvolutionPolicy::from_name(name);
    const kodama::CoreResult specificity_result = kodama::CoreKNNGraph_CPU(
      graph, 4, grouped_labels, grouped_constrain, {}, knn_specificity
    );
    require(specificity_result.clbest == grouped_result.clbest &&
            specificity_result.vect_acc == grouped_result.vect_acc &&
            specificity_result.vect_score == grouped_result.vect_score,
            "A PLS-specific evolution ablation changed KNN evolution.");
  }
  for (int cycle = 0; cycle < grouped_result.cycles_completed; ++cycle) {
    require(grouped_result.vect_acc[static_cast<std::size_t>(cycle)] >= 0.0 &&
            grouped_result.vect_score[static_cast<std::size_t>(cycle)] >= 0.0,
            "Core completed trace contains a sentinel value.");
    if (cycle > 0) {
      require(grouped_result.vect_acc[static_cast<std::size_t>(cycle)] >=
                grouped_result.vect_acc[static_cast<std::size_t>(cycle - 1)] &&
              grouped_result.vect_score[static_cast<std::size_t>(cycle)] >=
                grouped_result.vect_score[static_cast<std::size_t>(cycle - 1)],
              "Core best-state trace is not monotone.");
    }
  }

  kodama::CoreOptions guarded_single_class = fixed_core;
  guarded_single_class.cycles = 5;
  guarded_single_class.auto_class_coarsening = false;
  guarded_single_class.many_to_one_absorption = false;
  const std::vector<int> single_class(4, 7);
  const kodama::CoreResult single_class_result = kodama::CoreKNNGraph_CPU(
    graph, 4, single_class, {}, {}, guarded_single_class
  );
  require(!single_class_result.success &&
          single_class_result.cycles_completed == guarded_single_class.cycles,
          "The single-class guard did not retain the complete experimental trace.");
  require(std::isinf(single_class_result.scorebest) &&
          single_class_result.scorebest < 0.0,
          "The single-class state was not explicitly rejected.");
  require(single_class_result.cv_evaluations == guarded_single_class.cycles + 1,
          "The guarded single-class run did not perform one CV per cycle.");
  kodama::CoreOptions raw_single_class = guarded_single_class;
  raw_single_class.evolution = kodama::EvolutionPolicy::from_name("raw_cv_score");
  const kodama::CoreResult raw_single_class_result = kodama::CoreKNNGraph_CPU(
    graph, 4, single_class, {}, {}, raw_single_class
  );
  require(std::isinf(raw_single_class_result.scorebest) &&
          raw_single_class_result.scorebest < 0.0,
          "raw_cv_score disabled the independent single-class rejection guard.");

  const std::vector<int> fragmented_labels = {1, 2, 3, 4};
  for (const bool absorption : {false, true}) {
    kodama::CoreOptions merge_guard = zero_cycle_core;
    merge_guard.cycles = 30;
    merge_guard.seed = absorption ? 41 : 37;
    merge_guard.evolutionary_search = true;
    merge_guard.guarded_diversity = true;
    merge_guard.auto_class_coarsening = !absorption;
    merge_guard.many_to_one_absorption = absorption;
    const kodama::CoreResult merged = kodama::CoreKNNGraph_CPU(
      graph, 4, fragmented_labels, {}, {}, merge_guard
    );
    require(std::set<int>(merged.clbest.begin(), merged.clbest.end()).size() >= 2,
            "Core merge evolution collapsed to one active class.");
    require(std::isfinite(merged.accbest) && std::isfinite(merged.scorebest),
            "Core merge evolution produced a non-finite objective.");
  }

  kodama::CoreOptions shaken = fixed_core;
  shaken.shake = true;
  shaken.cycles = 2;
  const kodama::CoreResult shaken_result = kodama::CoreKNNGraph_CPU(
    graph, 4, labels, {}, all_fixed, shaken
  );
  require(std::isfinite(shaken_result.accbest) &&
          std::isfinite(shaken_result.scorebest) &&
          shaken_result.vect_acc.front() >= 0.0,
          "Shake initialization did not replace its sentinel best state.");

  kodama::PLSOptions predict_options;
  predict_options.max_components = 2;
  require_throws<std::invalid_argument>([&] {
    (void)kodama::PLSLDAPredict_CPU(view, std::vector<int>{1}, view, predict_options);
  }, "PLS-LDA prediction accepted mismatched training labels.");
  const kodama::MatrixView mismatched_test{values.data(), 2, 1};
  require_throws<std::invalid_argument>([&] {
    (void)kodama::PLSLDAPredict_CPU(view, labels, mismatched_test, predict_options);
  }, "PLS-LDA prediction accepted mismatched train/test columns.");
  const std::vector<int> one_class_labels(4, 9);
  require(kodama::PLSLDAPredict_CPU(
            view, one_class_labels, view, predict_options
          ) == one_class_labels,
          "Single-class PLS-LDA prediction did not return the only class.");

#if !defined(KODAMA_ENABLE_CUDA)
  kodama::KNNOptions cuda_knn;
  cuda_knn.backend = kodama::Backend::CUDA;
  require_throws<std::runtime_error>([&] {
    (void)kodama::KNNCV_CUDA(view, labels, {}, cuda_knn);
  }, "KNNCV_CUDA did not reject a non-CUDA build.");
  kodama::KODAMAGraphOptions cuda_graph;
  cuda_graph.backend = kodama::Backend::CUDA;
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAGraph_CUDA(view, cuda_graph);
  }, "KODAMAGraph_CUDA did not reject a non-CUDA build.");
  kodama::KODAMAMatrixOptions cuda_matrix = matrix_options;
  cuda_matrix.backend = kodama::Backend::CUDA;
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAMatrix_CUDA(view, labels, {}, {}, cuda_matrix);
  }, "KODAMAMatrix_CUDA did not reject a non-CUDA build.");
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAGraph(view, cuda_graph);
  }, "Generic KODAMAGraph did not reject unavailable CUDA.");
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAMatrix(view, labels, {}, {}, cuda_matrix);
  }, "Generic KODAMAMatrix did not reject unavailable CUDA.");
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAMatrixFromGraphData_CUDA(
      view, graph, labels, {}, {}, cuda_matrix
    );
  }, "KODAMAMatrixFromGraphData_CUDA did not reject a non-CUDA build.");
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAMatrixFromGraphData(
      view, graph, labels, {}, {}, cuda_matrix
    );
  }, "Generic graph-and-data KODAMAMatrix did not reject unavailable CUDA.");
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAMatrixFromGraph_CUDA(
      graph, 4, labels, {}, {}, cuda_matrix
    );
  }, "KODAMAMatrixFromGraph_CUDA did not reject a non-CUDA build.");
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAMatrixFromGraph(
      graph, 4, labels, {}, {}, cuda_matrix
    );
  }, "Generic graph-only KODAMAMatrix did not reject unavailable CUDA.");
  require_throws<std::runtime_error>([&] {
    kodama::NeighborGraph candidate = graph;
    kodama::KODAMADissimilarityInPlace(
      candidate, labels, 1, 4, kodama::Backend::CUDA
    );
  }, "KODAMA dissimilarity did not reject unavailable CUDA.");
#endif

#if !defined(KODAMA_ENABLE_METAL)
  kodama::KNNOptions metal_knn;
  metal_knn.backend = kodama::Backend::Metal;
  require_throws<std::runtime_error>([&] {
    (void)kodama::KNNCV_METAL(view, labels, {}, metal_knn);
  }, "KNNCV_METAL did not reject a non-Metal build.");
  kodama::KODAMAGraphOptions metal_graph;
  metal_graph.backend = kodama::Backend::Metal;
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAGraph_METAL(view, metal_graph);
  }, "KODAMAGraph_METAL did not reject a non-Metal build.");
  kodama::KODAMAMatrixOptions metal_matrix = matrix_options;
  metal_matrix.backend = kodama::Backend::Metal;
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAMatrix_METAL(view, labels, {}, {}, metal_matrix);
  }, "KODAMAMatrix_METAL did not reject a non-Metal build.");
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAGraph(view, metal_graph);
  }, "Generic KODAMAGraph did not reject unavailable Metal.");
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAMatrix(view, labels, {}, {}, metal_matrix);
  }, "Generic KODAMAMatrix did not reject unavailable Metal.");
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAMatrixFromGraphData_METAL(
      view, graph, labels, {}, {}, metal_matrix
    );
  }, "KODAMAMatrixFromGraphData_METAL did not reject a non-Metal build.");
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAMatrixFromGraphData(
      view, graph, labels, {}, {}, metal_matrix
    );
  }, "Generic graph-and-data KODAMAMatrix did not reject unavailable Metal.");
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAMatrixFromGraph_METAL(
      graph, 4, labels, {}, {}, metal_matrix
    );
  }, "KODAMAMatrixFromGraph_METAL did not reject a non-Metal build.");
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAMatrixFromGraph(
      graph, 4, labels, {}, {}, metal_matrix
    );
  }, "Generic graph-only KODAMAMatrix did not reject unavailable Metal.");
  require_throws<std::runtime_error>([&] {
    kodama::NeighborGraph candidate = graph;
    kodama::KODAMADissimilarityInPlace(
      candidate, labels, 1, 4, kodama::Backend::Metal
    );
  }, "KODAMA dissimilarity did not reject unavailable Metal.");
#endif
}

void test_preprocessing() {
  const std::vector<float> train = {
    1.0f, 2.0f, 3.0f,
    2.0f, 4.0f, 8.0f,
    4.0f, 1.0f, 5.0f,
    3.0f, 6.0f, 9.0f
  };
  const std::vector<float> test = {2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 1.0f};
  const kodama::MatrixView train_view{train.data(), 4, 3};
  const kodama::MatrixView test_view{test.data(), 2, 3};
  const std::vector<kodama::NormalizationMethod> normalization_methods = {
    kodama::NormalizationMethod::PQN,
    kodama::NormalizationMethod::Sum,
    kodama::NormalizationMethod::Median,
    kodama::NormalizationMethod::Sqrt,
    kodama::NormalizationMethod::None
  };
  const std::vector<std::vector<float>> expected_train_coefficients = {
    {6.0f, 12.9230769231f, 10.0f, 18.0f},
    {6.0f, 14.0f, 10.0f, 18.0f},
    {2.0f, 4.0f, 4.0f, 6.0f},
    {3.7416573868f, 9.1651513899f, 6.4807406984f, 11.2249721603f},
    {1.0f, 1.0f, 1.0f, 1.0f}
  };
  for (std::size_t method_index = 0; method_index < normalization_methods.size(); ++method_index) {
    const auto method = normalization_methods[method_index];
    kodama::NormalizationOptions one;
    one.method = method;
    one.n_threads = 1;
    kodama::NormalizationOptions four = one;
    four.n_threads = 4;
    const auto reference = kodama::Normalization_CPU(train_view, test_view, one);
    const auto parallel = kodama::Normalization_CPU(train_view, test_view, four);
    require(reference.backend == kodama::Backend::CPU, "Normalization CPU metadata mismatch.");
    require_close(reference.train, parallel.train, 0.0f, "Normalization changed across CPU thread counts.");
    require_close(reference.test, parallel.test, 0.0f, "Test normalization changed across CPU thread counts.");
    require_close(reference.train_coefficients, parallel.train_coefficients, 0.0f,
      "Normalization coefficients changed across CPU thread counts.");
    require_close(reference.train_coefficients, expected_train_coefficients[method_index], 2e-5f,
      "Normalization coefficients disagree with the KODAMA reference formulas.");
    for (std::size_t row = 0; row < 4; ++row) {
      for (std::size_t column = 0; column < 3; ++column) {
        require(std::abs(reference.train[row * 3 + column] *
          reference.train_coefficients[row] - train[row * 3 + column]) < 2e-6f,
          "Training normalization does not reconstruct the input.");
      }
    }
    if (method == kodama::NormalizationMethod::PQN) {
      require(reference.reference.size() == 3, "PQN reference size mismatch.");
    }
#ifdef KODAMA_ENABLE_CUDA
    kodama::NormalizationOptions cuda_options = one;
    cuda_options.backend = kodama::Backend::CUDA;
    const auto cuda = kodama::Normalization_CUDA(train_view, test_view, cuda_options);
    require(cuda.backend == kodama::Backend::CUDA, "Normalization CUDA metadata mismatch.");
    require_close(reference.train, cuda.train, 2e-6f, "Normalization CUDA disagrees with CPU.");
    require_close(reference.test, cuda.test, 2e-6f, "Test normalization CUDA disagrees with CPU.");
    require_close(reference.train_coefficients, cuda.train_coefficients, 2e-6f,
      "Normalization CUDA coefficients disagree with CPU.");
#endif
  }

  const std::vector<kodama::ScalingMethod> scaling_methods = {
    kodama::ScalingMethod::None,
    kodama::ScalingMethod::Centering,
    kodama::ScalingMethod::Autoscaling,
    kodama::ScalingMethod::RangeScaling,
    kodama::ScalingMethod::ParetoScaling
  };
  const std::vector<std::vector<float>> expected_scales = {
    {1.0f, 1.0f, 1.0f},
    {1.0f, 1.0f, 1.0f},
    {1.2909944487f, 2.2173557826f, 2.7537852736f},
    {3.0f, 5.0f, 6.0f},
    {1.1362193665f, 1.4890788369f, 1.6594533057f}
  };
  for (std::size_t method_index = 0; method_index < scaling_methods.size(); ++method_index) {
    const auto method = scaling_methods[method_index];
    kodama::ScalingOptions one;
    one.method = method;
    one.n_threads = 1;
    kodama::ScalingOptions four = one;
    four.n_threads = 4;
    const auto reference = kodama::Scaling_CPU(train_view, test_view, one);
    const auto parallel = kodama::Scaling_CPU(train_view, test_view, four);
    require_close(reference.train, parallel.train, 0.0f, "Scaling changed across CPU thread counts.");
    require_close(reference.test, parallel.test, 0.0f, "Test scaling changed across CPU thread counts.");
    require_close(reference.center, parallel.center, 0.0f, "Scaling centers changed across CPU thread counts.");
    require_close(reference.scale, parallel.scale, 0.0f, "Scaling factors changed across CPU thread counts.");
    require_close(reference.scale, expected_scales[method_index], 2e-5f,
      "Scaling factors disagree with the KODAMA reference formulas.");
    for (std::size_t row = 0; row < 4; ++row) {
      for (std::size_t column = 0; column < 3; ++column) {
        require(std::abs(reference.train[row * 3 + column] * reference.scale[column] +
          reference.center[column] - train[row * 3 + column]) < 2e-6f,
          "Training scaling does not reconstruct the input.");
      }
    }
#ifdef KODAMA_ENABLE_CUDA
    kodama::ScalingOptions cuda_options = one;
    cuda_options.backend = kodama::Backend::CUDA;
    const auto cuda = kodama::Scaling_CUDA(train_view, test_view, cuda_options);
    require(cuda.backend == kodama::Backend::CUDA, "Scaling CUDA metadata mismatch.");
    require_close(reference.train, cuda.train, 2e-6f, "Scaling CUDA disagrees with CPU.");
    require_close(reference.test, cuda.test, 2e-6f, "Test scaling CUDA disagrees with CPU.");
    require_close(reference.center, cuda.center, 2e-6f, "Scaling CUDA centers disagree with CPU.");
    require_close(reference.scale, cuda.scale, 2e-6f, "Scaling CUDA factors disagree with CPU.");
#endif
  }

  const std::vector<float> signed_train = {-2.0f, 1.0f, 2.0f, 2.0f};
  const std::vector<float> signed_test = {-2.0f, 1.0f};
  kodama::NormalizationOptions sum_options;
  sum_options.method = kodama::NormalizationMethod::Sum;
  const auto signed_result = kodama::Normalization_CPU(
    kodama::MatrixView{signed_train.data(), 2, 2},
    kodama::MatrixView{signed_test.data(), 1, 2}, sum_options
  );
  require(signed_result.train_coefficients[0] == -1.0f,
    "Training sum normalization no longer follows the KODAMA signed-sum rule.");
  require(signed_result.test_coefficients[0] == 3.0f,
    "Test sum normalization no longer follows the KODAMA absolute-sum rule.");

  const float missing = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> missing_train = {1.0f, missing, 3.0f, 2.0f, 4.0f, 6.0f};
  const std::vector<float> missing_test = {1.0f, missing, 3.0f};
  for (const auto method : {kodama::NormalizationMethod::Sum,
                            kodama::NormalizationMethod::Median,
                            kodama::NormalizationMethod::Sqrt}) {
    kodama::NormalizationOptions options;
    options.method = method;
    const auto result = kodama::Normalization_CPU(
      kodama::MatrixView{missing_train.data(), 2, 3},
      kodama::MatrixView{missing_test.data(), 1, 3}, options);
    require(std::isnan(result.train_coefficients[0]),
      "Training normalization no longer propagates missing values.");
    require(std::isfinite(result.test_coefficients[0]),
      "Test normalization no longer removes missing values.");
  }
}

double exact_graph_recall(
  const kodama::NeighborGraph& graph,
  const std::vector<float>& data,
  int rows,
  int columns
) {
  const int k = graph.neighbors;
  double hits = 0.0;
  std::vector<std::pair<float, int>> exact;
  exact.reserve(static_cast<std::size_t>(rows - 1));
  for (int row = 0; row < rows; ++row) {
    exact.clear();
    const float* query =
      data.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(columns);
    for (int candidate = 0; candidate < rows; ++candidate) {
      if (candidate == row) continue;
      const float* point =
        data.data() + static_cast<std::size_t>(candidate) * static_cast<std::size_t>(columns);
      float distance = 0.0f;
      for (int column = 0; column < columns; ++column) {
        const float delta = query[column] - point[column];
        distance += delta * delta;
      }
      exact.emplace_back(distance, candidate + 1);
    }
    std::partial_sort(
      exact.begin(),
      exact.begin() + k,
      exact.end()
    );
    for (int rank = 0; rank < k; ++rank) {
      const int observed =
        graph.indices[static_cast<std::size_t>(row) * static_cast<std::size_t>(k) +
                      static_cast<std::size_t>(rank)];
      for (int expected_rank = 0; expected_rank < k; ++expected_rank) {
        if (observed == exact[static_cast<std::size_t>(expected_rank)].second) {
          hits += 1.0;
          break;
        }
      }
    }
  }
  return hits / static_cast<double>(rows * k);
}

void check_parallel_hnsw_graph() {
  constexpr int rows = 600;
  constexpr int columns = 24;
  constexpr int neighbors = 20;
  std::vector<float> data(static_cast<std::size_t>(rows) * columns);
  std::mt19937 rng(107);
  std::normal_distribution<float> noise(0.0f, 1.0f);
  for (float& value : data) value = noise(rng);

  const kodama::MatrixView view{
    data.data(),
    static_cast<std::size_t>(rows),
    static_cast<std::size_t>(columns)
  };
  kodama::GraphClusterOptions options;
  options.k = neighbors;
  options.metric = kodama::DistanceMetric::Euclidean;
  options.n_threads = 1;
  const kodama::NeighborGraph serial = kodama::KODAMAKNNGraph_CPU(view, options);
  options.n_threads = 4;
  const kodama::NeighborGraph parallel = kodama::KODAMAKNNGraph_CPU(view, options);

  for (const kodama::NeighborGraph* graph : {&serial, &parallel}) {
    require(graph->neighbors == neighbors, "Native HNSW neighbor count mismatch.");
    require(
      graph->indices.size() == static_cast<std::size_t>(rows * neighbors),
      "Native HNSW index size mismatch."
    );
    require(
      graph->distances.size() == graph->indices.size(),
      "Native HNSW distance size mismatch."
    );
    for (int row = 0; row < rows; ++row) {
      float previous = -1.0f;
      for (int rank = 0; rank < neighbors; ++rank) {
        const std::size_t offset =
          static_cast<std::size_t>(row) * neighbors + static_cast<std::size_t>(rank);
        require(
          graph->indices[offset] >= 1 && graph->indices[offset] <= rows,
          "Native HNSW returned an invalid index."
        );
        require(
          graph->indices[offset] != row + 1,
          "Native HNSW returned a self-neighbor."
        );
        require(
          std::isfinite(graph->distances[offset]),
          "Native HNSW returned a non-finite distance."
        );
        require(
          graph->distances[offset] + 1e-6f >= previous,
          "Native HNSW distances are not ordered."
        );
        previous = graph->distances[offset];
      }
    }
  }
  require(
    exact_graph_recall(serial, data, rows, columns) >= 0.99,
    "Single-core native HNSW recall fell below 0.99."
  );
  require(
    exact_graph_recall(parallel, data, rows, columns) >= 0.99,
    "Parallel native HNSW recall fell below 0.99."
  );
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

void check_graph_cluster_contracts() {
  kodama::NeighborGraph graph;
  graph.neighbors = 2;
  graph.index_base = kodama::GraphIndexBase::One;
  graph.indices = {
    2, 3, 1, 3, 1, 2,
    5, 6, 4, 6, 4, 5
  };
  graph.distances = {
    0.10f, 0.20f, 0.10f, 0.15f, 0.20f, 0.15f,
    0.11f, 0.21f, 0.11f, 0.16f, 0.21f, 0.16f
  };

  kodama::GraphClusterOptions options;
  options.backend = kodama::Backend::CPU;
  options.n_iterations = 8;
  options.random_walk_steps = 2;
  options.n_threads = 2;
  for (const kodama::GraphWeightType weight_type : {
         kodama::GraphWeightType::SNN,
         kodama::GraphWeightType::Distance,
         kodama::GraphWeightType::Adaptive,
         kodama::GraphWeightType::Binary
       }) {
    options.weight_type = weight_type;
    const kodama::GraphClusterResult result =
      kodama::KODAMAGraphCluster_CPU(graph, 6, options);
    require(result.membership.size() == 6, "Graph clustering membership size mismatch.");
    require(result.n_vertices == 6, "Graph clustering vertex count mismatch.");
    require(result.n_edges > 0, "Graph clustering discarded every valid edge.");
    require(result.n_communities == 2, "Graph clustering did not preserve disconnected triangles.");
    require(result.backend == kodama::Backend::CPU, "Graph clustering backend metadata mismatch.");
    require(std::isfinite(result.modularity), "Graph clustering modularity is not finite.");
  }

  kodama::NeighborGraph zero_based = graph;
  zero_based.index_base = kodama::GraphIndexBase::Auto;
  for (int& index : zero_based.indices) --index;
  options.weight_type = kodama::GraphWeightType::Distance;
  options.mutual = true;
  const kodama::GraphClusterResult zero_result =
    kodama::KODAMAGraphCluster(zero_based, 6, options);
  require(zero_result.n_communities == 2, "Auto zero-based graph detection failed.");

  options.mutual = false;
  options.weight_type = kodama::GraphWeightType::Binary;
  options.prune = 1.0;
  const kodama::GraphClusterResult pruned =
    kodama::KODAMAGraphCluster_CPU(graph, 6, options);
  require(pruned.n_communities == 6, "Graph pruning did not isolate all vertices.");
  require(pruned.n_edges == 0, "Graph pruning retained an edge above its threshold.");

  const std::vector<float> embedding_values = {
    0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 0.1f,
    5.0f, 5.0f, 5.1f, 5.0f, 5.0f, 5.1f
  };
  const kodama::MatrixView embedding{embedding_values.data(), 6, 2};
  options.prune = 0.0;
  options.target_clusters = 2;
  const kodama::GraphClusterResult embedded =
    kodama::KODAMAEmbeddingGraphCluster(embedding, graph, options);
  require(embedded.target_exact && embedded.n_communities == 2,
          "Embedding graph clustering did not satisfy the exact target.");

  options.target_clusters = 0;
  options.k = 2;
  const kodama::GraphClusterResult built =
    kodama::KODAMAEmbeddingCluster(embedding, options);
  require(built.membership.size() == 6 && built.n_vertices == 6,
          "Embedding clustering did not build and cluster its graph.");

  bool rejected = false;
  try {
    kodama::NeighborGraph malformed = graph;
    malformed.indices.pop_back();
    (void)kodama::KODAMAGraphCluster_CPU(malformed, 6, options);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "Malformed graph dimensions were accepted.");

  rejected = false;
  try {
    options.target_clusters = 7;
    (void)kodama::KODAMAGraphCluster_CPU(graph, 6, options);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "An impossible target cluster count was accepted.");

  rejected = false;
  try {
    options.target_clusters = 0;
    options.backend = kodama::Backend::Metal;
    (void)kodama::KODAMAGraphCluster(graph, 6, options);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "CPU-only clustering silently accepted an accelerator backend.");

  kodama::ResidentIVFIndex empty_index;
  require(!empty_index.valid() && empty_index.rows() == 0 && empty_index.dimensions() == 0,
          "Default resident IVF index metadata is invalid.");
  rejected = false;
  try {
    (void)kodama::SearchResidentIVFIndex(empty_index, embedding, 2);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "Search accepted an empty resident IVF index.");

  rejected = false;
  try {
    kodama::KNNOptions resident_options;
    resident_options.backend = kodama::Backend::CPU;
    (void)kodama::BuildResidentIVFIndex(embedding, resident_options);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "Resident IVF construction silently accepted the CPU backend.");
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

  kodama::GraphClusterOptions wide_options = options;
  wide_options.k = 100;
  const kodama::NeighborGraph wide_cpu = kodama::KODAMAKNNGraph_CPU(view, wide_options);
  wide_options.backend = kodama::Backend::CUDA;
  const kodama::NeighborGraph wide_cuda = kodama::KODAMAKNNGraph_CUDA(view, wide_options);
  require(wide_cuda.indices == wide_cpu.indices,
          "CUDA 2D spatial grid k=100 indices differ from CPU.");
  require(wide_cuda.distances.size() == wide_cpu.distances.size(),
          "CUDA 2D spatial grid k=100 distance size mismatch.");
  for (std::size_t i = 0; i < wide_cpu.distances.size(); ++i) {
    require(std::abs(wide_cuda.distances[i] - wide_cpu.distances[i]) < 1e-5f,
            "CUDA 2D spatial grid k=100 distances differ from CPU.");
  }

  constexpr int n3 = 10000;
  std::vector<float> x3(static_cast<std::size_t>(n3) * 3);
  for (int row = 0; row < n3; ++row) {
    x3[static_cast<std::size_t>(row) * 3] = static_cast<float>(row % 22);
    x3[static_cast<std::size_t>(row) * 3 + 1] =
      static_cast<float>((row / 22) % 22);
    x3[static_cast<std::size_t>(row) * 3 + 2] = static_cast<float>(row / 484);
  }
  const kodama::MatrixView view3{x3.data(), n3, 3};
  wide_options.backend = kodama::Backend::CPU;
  const kodama::NeighborGraph wide_cpu3 = kodama::KODAMAKNNGraph_CPU(view3, wide_options);
  wide_options.backend = kodama::Backend::CUDA;
  const kodama::NeighborGraph wide_cuda3 = kodama::KODAMAKNNGraph_CUDA(view3, wide_options);
  require(wide_cuda3.indices == wide_cpu3.indices,
          "CUDA 3D spatial grid k=100 indices differ from CPU.");
  require(wide_cuda3.distances.size() == wide_cpu3.distances.size(),
          "CUDA 3D spatial grid k=100 distance size mismatch.");
  for (std::size_t i = 0; i < wide_cpu3.distances.size(); ++i) {
    require(std::abs(wide_cuda3.distances[i] - wide_cpu3.distances[i]) < 1e-5f,
            "CUDA 3D spatial grid k=100 distances differ from CPU.");
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

void check_spatial_grid_query_nearest() {
  const std::vector<float> base_2d = {
    0.0f, 0.0f,
    2.0f, 0.0f,
    0.0f, 3.0f,
    5.0f, 5.0f,
    -3.0f, 2.0f
  };
  const std::vector<float> query_2d = {
    1.0f, 0.0f,
    9.0f, 9.0f,
    -4.0f, 2.0f
  };
  const kodama::NeighborGraph graph_2d = kodama::detail::spatial_grid_query_nearest(
    base_2d.data(),
    5,
    query_2d.data(),
    3,
    2,
    2
  );
  require(graph_2d.indices == std::vector<int>({0, 3, 4}),
          "Spatial grid 2D base/query nearest indices mismatch.");
  require(std::abs(graph_2d.distances[0] - 1.0f) < 1e-6f &&
          std::abs(graph_2d.distances[1] - std::sqrt(32.0f)) < 1e-6f &&
          std::abs(graph_2d.distances[2] - 1.0f) < 1e-6f,
          "Spatial grid 2D base/query nearest distances mismatch.");

  const std::vector<float> base_3d = {
    0.0f, 0.0f, 0.0f,
    2.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 4.0f,
    5.0f, 5.0f, 5.0f
  };
  const std::vector<float> query_3d = {
    1.0f, 0.0f, 0.0f,
    7.0f, 6.0f, 5.0f
  };
  const kodama::NeighborGraph graph_3d = kodama::detail::spatial_grid_query_nearest(
    base_3d.data(),
    4,
    query_3d.data(),
    2,
    3,
    2
  );
  require(graph_3d.indices == std::vector<int>({0, 3}),
          "Spatial grid 3D base/query nearest indices mismatch.");
  require(std::abs(graph_3d.distances[0] - 1.0f) < 1e-6f &&
          std::abs(graph_3d.distances[1] - std::sqrt(5.0f)) < 1e-6f,
          "Spatial grid 3D base/query nearest distances mismatch.");
}

kodama::NeighborGraph reference_kodama_dissimilarity(
  kodama::NeighborGraph graph,
  const std::vector<int>& labels,
  int runs,
  int samples
) {
  for (int row = 0; row < samples; ++row) {
    std::vector<std::pair<float, int>> corrected(
      static_cast<std::size_t>(graph.neighbors)
    );
    const std::size_t row_offset =
      static_cast<std::size_t>(row) * static_cast<std::size_t>(graph.neighbors);
    for (int rank = 0; rank < graph.neighbors; ++rank) {
      const std::size_t offset = row_offset + static_cast<std::size_t>(rank);
      const int neighbor_one_based = graph.indices[offset];
      const int neighbor = neighbor_one_based - 1;
      float distance = graph.distances[offset];
      if (
        neighbor < 0 || neighbor >= samples ||
        !std::isfinite(distance)
      ) {
        corrected[static_cast<std::size_t>(rank)] = {
          std::numeric_limits<float>::infinity(),
          neighbor_one_based
        };
        continue;
      }
      int same = 0;
      int valid = 0;
      for (int run = 0; run < runs; ++run) {
        const std::size_t base =
          static_cast<std::size_t>(run) * static_cast<std::size_t>(samples);
        const int lhs = labels[base + static_cast<std::size_t>(row)];
        const int rhs = labels[base + static_cast<std::size_t>(neighbor)];
        if (lhs == 0 || rhs == 0) continue;
        ++valid;
        if (lhs == rhs) ++same;
      }
      if (same == 0 || valid == 0) {
        distance = std::numeric_limits<float>::infinity();
      } else {
        const double agreement =
          static_cast<double>(same) / static_cast<double>(valid);
        distance = static_cast<float>(
          (1.0 + static_cast<double>(distance)) /
          (agreement * agreement)
        );
      }
      corrected[static_cast<std::size_t>(rank)] = {
        distance,
        neighbor_one_based
      };
    }
    std::stable_sort(
      corrected.begin(),
      corrected.end(),
      [](const auto& left, const auto& right) {
        if (left.first != right.first) return left.first < right.first;
        return left.second < right.second;
      }
    );
    for (int rank = 0; rank < graph.neighbors; ++rank) {
      const std::size_t offset = row_offset + static_cast<std::size_t>(rank);
      graph.distances[offset] = corrected[static_cast<std::size_t>(rank)].first;
      graph.indices[offset] = corrected[static_cast<std::size_t>(rank)].second;
    }
  }
  return graph;
}

void check_pca_cpu() {
  constexpr int n = 80;
  constexpr int p = 5;
  std::vector<double> values(static_cast<std::size_t>(n) * p);
  std::vector<float> values_float(values.size());
  for (int row = 0; row < n; ++row) {
    const double a = std::sin(0.17 * row);
    const double b = std::cos(0.11 * row);
    for (int col = 0; col < p; ++col) {
      const double value = (col + 1.0) * a + (p - col) * 0.35 * b +
        0.12 * std::sin((col + 2.0) * 0.07 * row);
      values[static_cast<std::size_t>(row) * p + col] = value;
      values_float[static_cast<std::size_t>(row) * p + col] = static_cast<float>(value);
    }
  }

  kodama::PCAOptions options;
  options.n_components = 3;
  options.oversample = 2;
  options.power_iterations = 2;
  options.n_threads = 2;
  options.seed = 19;
  const kodama::PCAResult result = kodama::PCA_CPU(
    kodama::MatrixView{values.data(), n, p}, options
  );
  require(result.backend == kodama::Backend::CPU, "PCA CPU backend metadata mismatch.");
  require(result.samples == n && result.variables == p && result.components == 3,
          "PCA CPU dimensions mismatch.");
  require(result.scores.size() == static_cast<std::size_t>(n * 3),
          "PCA CPU score size mismatch.");
  require(result.loadings.size() == static_cast<std::size_t>(p * 3),
          "PCA CPU loading size mismatch.");
  require(result.center.size() == p && result.scale.size() == p,
          "PCA CPU preprocessing metadata mismatch.");
  for (const float value : result.scores) require(std::isfinite(value), "PCA CPU score is non-finite.");
  for (const float value : result.loadings) require(std::isfinite(value), "PCA CPU loading is non-finite.");
  for (int component = 0; component < result.components; ++component) {
    double mean = 0.0;
    for (int row = 0; row < n; ++row) {
      mean += result.scores[static_cast<std::size_t>(row) * result.components + component];
    }
    require(std::abs(mean / n) < 2e-4, "Centered PCA scores do not have zero mean.");
    for (int other = 0; other < result.components; ++other) {
      double dot = 0.0;
      for (int variable = 0; variable < p; ++variable) {
        dot += result.loadings[static_cast<std::size_t>(variable) * result.components + component] *
               result.loadings[static_cast<std::size_t>(variable) * result.components + other];
      }
      require(std::abs(dot - (component == other ? 1.0 : 0.0)) < 2e-3,
              "PCA loadings are not orthonormal.");
    }
  }
  require(result.singular_values[0] >= result.singular_values[1] &&
          result.singular_values[1] >= result.singular_values[2],
          "PCA singular values are not sorted.");
  require(result.cumulative_variance_explained.back() <= 1.001f,
          "PCA cumulative explained variance exceeds one.");

  const kodama::PCAResult repeat = kodama::PCA_CPU(
    kodama::MatrixView{values.data(), n, p}, options
  );
  require(result.scores == repeat.scores && result.loadings == repeat.loadings,
          "PCA CPU is not deterministic for a fixed seed.");
  const kodama::PCAResult float_result = kodama::PCA_CPU(
    kodama::MatrixView{values_float.data(), n, p}, options
  );
  for (int component = 0; component < result.components; ++component) {
    const float reference = std::max(1.0f, result.singular_values[static_cast<std::size_t>(component)]);
    require(
      std::abs(result.singular_values[static_cast<std::size_t>(component)] -
               float_result.singular_values[static_cast<std::size_t>(component)]) / reference < 2e-4f,
      "PCA float32 and float64 input paths disagree."
    );
  }
}

}  // namespace

int main() {
  require(kodama::CoreOptions().knn.k == 30,
          "CoreKNN production default should use k=30.");
  require(kodama::KODAMAMatrixOptions().knn.k == 30,
          "KODAMA matrix production default should use knn.k=30.");
  require(kodama::KNNOptions().k == 10,
          "Standalone KNNCV default should remain k=10.");
  test_public_string_contracts();
  test_public_error_contracts();
  test_preprocessing();
  check_parallel_hnsw_graph();
  check_graph_cluster_contracts();
  check_spatial_grid_graph();
  check_spatial_grid_query_nearest();
  check_pca_cpu();

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

  kodama::PLSOptions parallel_pls = pls;
  parallel_pls.n_threads = 4;
  kodama::PLSCVResult parallel_flres = kodama::PLSLDACV_CPU(
    fview, d.y, d.constrain, parallel_pls
  );
  require(
    parallel_flres.predicted_labels == flres.predicted_labels,
    "Streamed CPU PLS-LDA predictions depend on fold worker count."
  );
  require(
    parallel_flres.selected_components == flres.selected_components,
    "Streamed CPU PLS-LDA component count depends on fold worker count."
  );

  // A large supervised cross-product can leave the randomized SIMPLS power
  // vector finite while its float32 squared norm overflows. The norm reduction
  // must not truncate an otherwise valid fit to the majority fallback.
  constexpr int robust_n = 1200;
  constexpr int robust_p = 8;
  std::vector<float> robust_x(
    static_cast<std::size_t>(robust_n * robust_p), 0.0f
  );
  std::vector<int> robust_y(static_cast<std::size_t>(robust_n), 0);
  for (int row = 0; row < robust_n; ++row) {
    const int cls = row % 3;
    robust_y[static_cast<std::size_t>(row)] = cls + 1;
    for (int column = 0; column < robust_p; ++column) {
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
      robust_x[static_cast<std::size_t>(row * robust_p + column)] =
        class_signal + variation;
    }
  }
  kodama::PLSOptions robust_pls = pls;
  robust_pls.cv.folds = 5;
  robust_pls.cv.seed = 9;
  robust_pls.max_components = 2;
  robust_pls.fixed_components = 2;
  robust_pls.center = true;
  robust_pls.scale = false;
  const kodama::PLSCVResult robust_lres = kodama::PLSLDACV_CPU(
    kodama::MatrixView{robust_x.data(), robust_n, robust_p},
    robust_y,
    {},
    robust_pls
  );
  require(
    robust_lres.selected_components == 2,
    "CPU SIMPLS lost a valid component after a large finite power refresh."
  );
  require(
    robust_lres.global_accuracy > 0.99,
    "CPU robust SIMPLS norm regression accuracy unexpectedly low."
  );

  kodama::PLSCVResult fdispatch_lres = kodama::PLSLDACV(fview, d.y, d.constrain, pls);
  check_pls_result(fdispatch_lres, d.y, d.constrain, 4);
  require(fdispatch_lres.global_accuracy > 0.60, "Float32 generic PLS-LDA accuracy unexpectedly low.");

  // Multiple labels can still have zero supervised covariance, for example
  // when every predictor is constant. Such a fold must receive a majority
  // prediction instead of aborting an entire KODAMA run.
  const int degenerate_n = 40;
  const int degenerate_p = 3;
  std::vector<float> degenerate_x(
    static_cast<std::size_t>(degenerate_n * degenerate_p),
    1.0f
  );
  std::vector<int> degenerate_y(static_cast<std::size_t>(degenerate_n), 0);
  std::vector<int> degenerate_constrain(static_cast<std::size_t>(degenerate_n), 0);
  for (int row = 0; row < degenerate_n; ++row) {
    degenerate_y[static_cast<std::size_t>(row)] = row < 30 ? 10 : 20;
    degenerate_constrain[static_cast<std::size_t>(row)] = row;
  }
  kodama::MatrixView degenerate_view{
    degenerate_x.data(),
    static_cast<std::size_t>(degenerate_n),
    static_cast<std::size_t>(degenerate_p)
  };
  kodama::PLSOptions degenerate_pls = pls;
  degenerate_pls.cv.folds = 5;
  degenerate_pls.fixed_components = 0;
  kodama::PLSCVResult degenerate_lres = kodama::PLSLDACV_CPU(
    degenerate_view,
    degenerate_y,
    degenerate_constrain,
    degenerate_pls
  );
  require(
    degenerate_lres.predicted_labels.size() == degenerate_y.size(),
    "Degenerate PLS-LDA prediction size mismatch."
  );
  require(
    std::isfinite(degenerate_lres.global_accuracy),
    "Degenerate PLS-LDA accuracy must be finite."
  );
  require(
    degenerate_lres.selected_components == 1,
    "Degenerate PLS-LDA should report the majority fallback as one component."
  );
  require(
    std::all_of(
      degenerate_lres.predicted_labels.begin(),
      degenerate_lres.predicted_labels.end(),
      [](int label) { return label == 10; }
    ),
    "Degenerate PLS-LDA did not use the training-fold majority class."
  );

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

  kodama::NeighborGraph empty_graph;
  empty_graph.neighbors = 1;
  empty_graph.indices.assign(3, -1);
  empty_graph.distances.assign(3, std::numeric_limits<float>::infinity());
  kodama::CoreOptions graph_fallback_options;
  graph_fallback_options.cycles = 0;
  graph_fallback_options.knn.k = 1;
  graph_fallback_options.knn.cv.folds = 3;
  graph_fallback_options.knn.cv.stratified = false;
  graph_fallback_options.knn.cv.seed = 7;
  const std::vector<int> graph_fallback_labels{1, 2, 2};
  const kodama::CoreResult graph_fallback = kodama::CoreKNNGraph_CPU(
    empty_graph,
    3,
    graph_fallback_labels,
    std::vector<int>(),
    std::vector<int>(),
    graph_fallback_options
  );
  require(
    graph_fallback.cvpredbest == std::vector<int>({2, 1, 1}),
    "Graph KNN fallback used validation labels instead of the training-fold majority."
  );
  require(graph_fallback.runtime_seconds > 0.0, "Graph KNN inclusive runtime was not recorded.");
  require(
    std::string(kodama::to_string(kodama::KNNIndexType::PrecomputedGraph)) == "precomputed_graph",
    "Precomputed graph index provenance is missing."
  );

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
  km_options.compute_visual_init = true;
  km_options.materialize_graph = true;
  km_options.knn.k = 10;
  km_options.knn.n_threads = 1;
  kodama::KODAMAGraphOptions prepared_graph_options;
  prepared_graph_options.neighbors = 45;
  prepared_graph_options.n_threads = km_options.n_threads;
  prepared_graph_options.seed = km_options.seed;
  prepared_graph_options.metric = km_options.metric;
  prepared_graph_options.backend = kodama::Backend::CPU;
  prepared_graph_options.materialize_graph = true;
  const kodama::KODAMAGraphResult prepared_graph =
    kodama::KODAMAGraph_CPU(fview, prepared_graph_options);
  require(prepared_graph.samples == d.n, "KODAMAGraph sample count mismatch.");
  require(prepared_graph.dimensions == d.p, "KODAMAGraph dimension count mismatch.");
  require(prepared_graph.graph_builds == 1, "KODAMAGraph should build exactly one graph.");
  require(
    prepared_graph.index_type == kodama::KNNIndexType::NativeHNSW,
    "CPU KODAMAGraph should report native HNSW."
  );
  require(prepared_graph.knn.neighbors == 45, "KODAMAGraph neighbor count mismatch.");
  require(prepared_graph.knn.indices.front() >= 1, "KODAMAGraph indices should be one-based.");
  require(
    prepared_graph.visual_init.umap.size() == static_cast<std::size_t>(d.n * 2) &&
      prepared_graph.visual_init.opentsne.size() == static_cast<std::size_t>(d.n * 2),
    "KODAMAGraph did not retain both PCA initializations."
  );
  kodama::KODAMAGraphOptions lazy_graph_options = prepared_graph_options;
  lazy_graph_options.materialize_graph = false;
  const kodama::KODAMAGraphResult lazy_graph =
    kodama::KODAMAGraph_CPU(fview, lazy_graph_options);
  require(lazy_graph.handle && lazy_graph.handle->valid() && lazy_graph.knn.indices.empty(),
          "Handle-backed KODAMAGraph unexpectedly materialized host arrays.");
  require(lazy_graph.neighbors == prepared_graph.knn.neighbors,
          "Handle-backed KODAMAGraph lost neighbor metadata.");
  const kodama::NeighborGraph lazy_materialized =
    kodama::KODAMAGraphMaterialize(lazy_graph);
  const kodama::NeighborGraph lazy_materialized_again =
    kodama::KODAMAGraphMaterialize(lazy_graph);
  require(lazy_materialized.indices == lazy_materialized_again.indices &&
          lazy_materialized.distances == lazy_materialized_again.distances &&
          lazy_materialized.neighbors == prepared_graph.knn.neighbors,
          "Explicit graph materialization is not stable.");
  std::vector<float> spatial(static_cast<std::size_t>(d.n) * 2u, 0.0f);
  for (int i = 0; i < d.n; ++i) {
    spatial[static_cast<std::size_t>(i) * 2u] = d.x[static_cast<std::size_t>(i) * d.p];
    spatial[static_cast<std::size_t>(i) * 2u + 1u] =
      d.x[static_cast<std::size_t>(i) * d.p + 1u];
  }
  const kodama::MatrixView spatial_view{
    spatial.data(), static_cast<std::size_t>(d.n), 2u
  };
  const kodama::KODAMAGraphResult spatial_prepared_graph =
    kodama::KODAMAGraph_CPU(fview, spatial_view, prepared_graph_options);
  require(
    spatial_prepared_graph.spatial_graph_builds == 1 &&
      spatial_prepared_graph.spatial_dimensions == 2 &&
      spatial_prepared_graph.spatial_jitter.size() == 2u &&
      spatial_prepared_graph.spatial_knn.neighbors == 45,
    "KODAMAGraph did not retain reusable spatial graph state."
  );
  kodama::KODAMAMatrixOptions spatial_prepared_options = km_options;
  spatial_prepared_options.spatial = spatial;
  spatial_prepared_options.spatial_cols = 2;
  spatial_prepared_options.spatial_resolution = 0.4;
  const kodama::KODAMAMatrixResult spatial_prepared_result = kodama::KODAMAMatrix(
    fview,
    spatial_prepared_graph,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    spatial_prepared_options
  );
  require(
    spatial_prepared_result.graph_builds == 0 &&
      spatial_prepared_result.spatial_graph_builds == 0 &&
      spatial_prepared_result.spatial_precompute_seconds < 0.01,
    "KODAMAMatrix rebuilt prepared spatial graph state."
  );
  kodama::KODAMAMatrixResult km_res = kodama::KODAMAMatrix_CPU(fview, std::vector<int>(), std::vector<int>(), fixed, km_options);
  kodama::KODAMAMatrixOptions explicit_full_matrix_options = km_options;
  explicit_full_matrix_options.evolution =
    kodama::EvolutionPolicy::from_name("full");
  const kodama::KODAMAMatrixResult default_policy_matrix =
    kodama::KODAMAMatrix(
      fview, prepared_graph, std::vector<int>(), std::vector<int>(), fixed,
      km_options
    );
  const kodama::KODAMAMatrixResult explicit_full_matrix =
    kodama::KODAMAMatrix(
      fview, prepared_graph, std::vector<int>(), std::vector<int>(), fixed,
      explicit_full_matrix_options
    );
  require(explicit_full_matrix.acc == default_policy_matrix.acc &&
          explicit_full_matrix.v == default_policy_matrix.v &&
          explicit_full_matrix.res == default_policy_matrix.res,
          "Explicit full KODAMA.matrix policy differs from the default call.");
  for (const std::string& name : evolution_policy_ablation_names()) {
    kodama::KODAMAMatrixOptions ablation_options = km_options;
    ablation_options.evolution = kodama::EvolutionPolicy::from_name(name);
    const kodama::KODAMAMatrixResult ablation_result =
      kodama::KODAMAMatrix(
        fview, prepared_graph, std::vector<int>(), std::vector<int>(), fixed,
        ablation_options
      );
    require(ablation_result.run_diagnostics.size() ==
              default_policy_matrix.run_diagnostics.size(),
            "KODAMA.matrix ablation run diagnostics size mismatch.");
    require(ablation_result.cycle_diagnostics.size() ==
              static_cast<std::size_t>(km_options.runs * km_options.cycles),
            "KODAMA.matrix ablation cycle diagnostics size mismatch.");
    for (std::size_t run = 0; run < default_policy_matrix.run_diagnostics.size(); ++run) {
      const auto& baseline = default_policy_matrix.run_diagnostics[run];
      const auto& ablated = ablation_result.run_diagnostics[run];
      require(ablated.landmark_rows_hash == baseline.landmark_rows_hash &&
              ablated.initial_labels_hash == baseline.initial_labels_hash &&
              ablated.fold_assignments_hash == baseline.fold_assignments_hash,
              "Evolution policy changed landmarks, initial labels, or folds.");
      require(ablated.cv_evaluations == km_options.cycles + 1,
              "KODAMA.matrix ablation failed the Tcycle + 1 CV invariant.");
      if (name == "no_transition_proposal") {
        require(ablated.transition_attempted == 0 &&
                ablated.many_to_one_attempted == 0 &&
                ablated.pls_coarsening_attempted == 0,
                "no_transition_proposal attempted a transition move.");
      }
    }
  }
  const kodama::KODAMAMatrixResult km_prepared_data_res = kodama::KODAMAMatrix(
    fview,
    prepared_graph,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    km_options
  );
  require(km_prepared_data_res.graph_builds == 0, "Prepared KODAMA graph was rebuilt.");
  require(km_prepared_data_res.has_visual_init, "Prepared PCA initialization was not reused.");
  require(
    km_prepared_data_res.res.size() == km_res.res.size(),
    "Prepared-graph-plus-data KODAMA returned an invalid label matrix."
  );
  require(
    std::all_of(km_prepared_data_res.acc.begin(), km_prepared_data_res.acc.end(),
                [](double value) { return std::isfinite(value); }),
    "Prepared-graph-plus-data KNN KODAMA returned non-finite accuracy."
  );
  const kodama::KODAMAMatrixResult km_prepared_only_res = kodama::KODAMAMatrix(
    prepared_graph,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    km_options
  );
  require(km_prepared_only_res.graph_builds == 0, "Graph-only KODAMA rebuilt its input graph.");
  require(km_prepared_only_res.has_visual_init, "Graph-only KODAMA lost the prepared PCA initialization.");
  require(
    km_prepared_only_res.res.size() == static_cast<std::size_t>(km_options.runs * d.n),
    "Graph-only KODAMA result label size mismatch."
  );
  require(km_res.runs == km_options.runs, "KODAMAMatrix run count mismatch.");
  require(km_res.samples == d.n, "KODAMAMatrix sample count mismatch.");
  require(km_res.cycles == km_options.cycles, "KODAMAMatrix cycle count mismatch.");
  require(km_res.acc.size() == static_cast<std::size_t>(km_options.runs), "KODAMAMatrix acc size mismatch.");
  require(km_res.v.size() == static_cast<std::size_t>(km_options.runs * km_options.cycles), "KODAMAMatrix trace size mismatch.");
  require(km_res.res.size() == static_cast<std::size_t>(km_options.runs * d.n), "KODAMAMatrix result label size mismatch.");
  require(km_res.res_constrain_rows == 1, "Nonspatial KODAMAMatrix should retain one shared constraint row.");
  require(km_res.res_constrain.size() == static_cast<std::size_t>(d.n), "KODAMAMatrix constrain size mismatch.");
  require(km_res.graph_builds == 1, "KODAMAMatrix should build the global graph exactly once for all M runs.");
  require(km_res.has_visual_init, "KODAMAMatrix did not retain the requested visualization initialization.");
  require(km_res.visual_init.umap.size() == static_cast<std::size_t>(d.n * 2), "KODAMAMatrix UMAP initialization size mismatch.");
  require(km_res.visual_init.opentsne.size() == static_cast<std::size_t>(d.n * 2), "KODAMAMatrix openTSNE initialization size mismatch.");
  require(km_res.effective_landmarks == km_options.landmarks, "KODAMAMatrix effective landmark count mismatch.");
  require(km_res.landmark_seconds.size() == static_cast<std::size_t>(km_options.runs), "KODAMAMatrix landmark timing size mismatch.");
  require(km_res.coarse_partition_seconds.size() == static_cast<std::size_t>(km_options.runs), "KODAMAMatrix coarse-partition timing size mismatch.");
  require(km_res.landmark_sampling_seconds.size() == static_cast<std::size_t>(km_options.runs), "KODAMAMatrix landmark-sampling timing size mismatch.");
  require(km_res.constraint_seconds.size() == static_cast<std::size_t>(km_options.runs), "KODAMAMatrix constraint timing size mismatch.");
  require(km_res.landmark_prepare_seconds.size() == static_cast<std::size_t>(km_options.runs), "KODAMAMatrix landmark-preparation timing size mismatch.");
  require(km_res.landmark_initialization_seconds.size() == static_cast<std::size_t>(km_options.runs), "KODAMAMatrix landmark-initialization timing size mismatch.");
  require(km_res.landmark_graph_seconds.size() == static_cast<std::size_t>(km_options.runs), "KODAMAMatrix landmark-graph timing size mismatch.");
  require(km_res.core_evolution_seconds.size() == static_cast<std::size_t>(km_options.runs), "KODAMAMatrix core-evolution timing size mismatch.");
  require(km_res.projection_seconds.size() == static_cast<std::size_t>(km_options.runs), "KODAMAMatrix projection timing size mismatch.");
  require(km_res.shared_landmark_partition_used,
          "Nonspatial KODAMAMatrix did not use the shared landmark partition.");
  require(km_res.shared_landmark_partition_strata == km_options.splitting,
          "Shared landmark partition stratum count mismatch.");
  require(km_res.shared_landmark_partition_seconds >= 0.0,
          "Shared landmark partition timing is invalid.");
  require(std::all_of(
            km_res.coarse_partition_seconds.begin(),
            km_res.coarse_partition_seconds.end(),
            [](double seconds) { return seconds == 0.0; }),
          "Shared landmark partition unexpectedly repeated coarse k-means per M run.");
  require(find_timing(km_res.timings, "shared_landmark_partition") != nullptr,
          "KODAMAMatrix did not report shared-partition timing.");
  require(find_timing(km_res.timings, "core_evolution") != nullptr,
          "KODAMAMatrix did not report accumulated evolution timing.");
  const kodama::KODAMAStageTiming* matrix_total =
    find_timing(km_res.timings, "total");
  require(matrix_total != nullptr &&
            std::abs(matrix_total->wall_seconds - km_res.runtime_seconds) < 1e-9,
          "KODAMAMatrix total timing differs from runtime.");
  require(km_res.landmark_grid_bins == std::vector<int>(static_cast<std::size_t>(km_options.runs), 0), "Nonspatial KODAMAMatrix unexpectedly used a spatial grid.");
  for (int run = 0; run < km_options.runs; ++run) {
    require(km_res.landmark_occupied_strata[static_cast<std::size_t>(run)] == km_options.splitting, "Nonspatial landmark stratum count mismatch.");
    require(km_res.landmark_represented_strata[static_cast<std::size_t>(run)] == km_options.splitting, "Nonspatial landmark sampling omitted a well-supported stratum.");
  }
  require(km_res.knn.neighbors > 0, "KODAMAMatrix HNSW neighbor count was not recorded.");
  require(km_res.knn.indices.size() == static_cast<std::size_t>(d.n * km_res.knn.neighbors), "KODAMAMatrix neighbor index size mismatch.");
  require(km_res.knn.distances.size() == km_res.knn.indices.size(), "KODAMAMatrix neighbor distance size mismatch.");
  require(km_res.knn.indices.front() >= 1, "KODAMAMatrix neighbor indices should be one-based for R compatibility.");
  require(
    std::all_of(km_res.knn.indices.begin(), km_res.knn.indices.end(), [&](const int index) {
      return index >= 1 && index <= d.n;
    }),
    "KODAMAMatrix returned a neighbor index outside the public one-based range."
  );
  require(km_res.knn.index_base == kodama::GraphIndexBase::One, "KODAMAMatrix graph index-base metadata is incorrect.");
  require(km_res.knn_is_kodama_corrected, "KODAMAMatrix did not mark its corrected graph.");
  require(
    km_res.graph_storage_bytes >=
      km_res.knn.indices.size() * sizeof(int) +
      km_res.knn.distances.size() * sizeof(float),
    "KODAMAMatrix graph storage accounting mismatch."
  );

  kodama::KODAMAMatrixOptions km_labels_only_options = km_options;
  km_labels_only_options.materialize_graph = false;
  km_labels_only_options.compute_visual_init = false;
  const kodama::KODAMAMatrixResult km_labels_only = kodama::KODAMAMatrix(
    fview,
    prepared_graph,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    km_labels_only_options
  );
  require(km_labels_only.knn.indices.empty() && km_labels_only.knn.distances.empty(),
          "Labels-only KODAMAMatrix unexpectedly materialized graph matrices.");
  require(km_labels_only.graph_storage_bytes == 0,
          "Labels-only KODAMAMatrix reported host graph storage.");
  require(!km_labels_only.knn_is_kodama_corrected &&
          km_labels_only.dissimilarity_seconds == 0.0,
          "Labels-only KODAMAMatrix performed an unobservable graph correction.");
  require(km_labels_only.res == km_prepared_data_res.res,
          "Disabling graph materialization changed optimized labels.");

  kodama::KODAMAMatrixOptions km_base_options = km_options;
  km_base_options.apply_kodama_dissimilarity = false;
  km_base_options.compute_visual_init = false;
  kodama::KODAMAMatrixResult km_base_res = kodama::KODAMAMatrix(
    fview,
    prepared_graph,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    km_base_options
  );
  require(
    km_base_res.res == km_prepared_data_res.res,
    "Deferring graph correction changed the optimized KODAMA labels."
  );
  require(
    !km_base_res.knn_is_kodama_corrected,
    "Deferred KODAMA graph correction was incorrectly marked as complete."
  );
  const kodama::NeighborGraph lazy_reference = reference_kodama_dissimilarity(
    km_base_res.knn,
    km_base_res.res,
    km_base_res.runs,
    km_base_res.samples
  );
  const int* lazy_indices_storage = km_base_res.knn.indices.data();
  const float* lazy_distances_storage = km_base_res.knn.distances.data();
  kodama::KODAMADissimilarityInPlace(
    km_base_res.knn,
    km_base_res.res,
    km_base_res.runs,
    km_base_res.samples,
    kodama::Backend::CPU,
    km_base_options.n_threads
  );
  require(
    lazy_indices_storage == km_base_res.knn.indices.data() &&
      lazy_distances_storage == km_base_res.knn.distances.data(),
    "Lazy KODAMA correction reallocated graph storage."
  );
  require(
    km_base_res.knn.indices == lazy_reference.indices &&
      km_base_res.knn.distances == lazy_reference.distances,
    "Lazy in-place KODAMA correction differs from the mathematical reference."
  );

  kodama::KODAMAMatrixOptions km_spatial_options = km_options;
  km_spatial_options.runs = 1;
  km_spatial_options.cycles = 1;
  km_spatial_options.landmarks = 20;
  km_spatial_options.n_threads = 1;
  km_spatial_options.knn.n_threads = 1;
  km_spatial_options.spatial_cols = 2;
  km_spatial_options.spatial_resolution = 0.2;
  km_spatial_options.spatial.resize(static_cast<std::size_t>(d.n) * 2);
  for (int row = 0; row < d.n; ++row) {
    km_spatial_options.spatial[static_cast<std::size_t>(row) * 2] = static_cast<float>(row % 15);
    km_spatial_options.spatial[static_cast<std::size_t>(row) * 2 + 1] = static_cast<float>(row / 15);
  }
  kodama::KODAMAMatrixResult km_spatial_res = kodama::KODAMAMatrix_CPU(
    fview,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    km_spatial_options
  );
  require(km_spatial_res.effective_landmarks == 20, "Spatial KODAMAMatrix effective landmark count mismatch.");
  require(km_spatial_res.landmark_grid_bins == std::vector<int>{5}, "Spatial landmark grid did not derive the expected bin count.");
  require(km_spatial_res.landmark_occupied_strata == std::vector<int>{25}, "Spatial landmark grid occupied-cell count mismatch.");
  require(km_spatial_res.landmark_represented_strata == std::vector<int>{20}, "Spatial quota sampling did not omit exactly the unsupported cells.");
  kodama::KODAMAMatrixResult km_spatial_repeat = kodama::KODAMAMatrix_CPU(
    fview,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    km_spatial_options
  );
  require(km_spatial_repeat.res == km_spatial_res.res, "Spatial landmark sampling is not repeatable for a fixed seed.");

  kodama::UMAPOptions umap_options;
  require(
    umap_options.graph_mode == kodama::UMAPGraphMode::Fuzzy,
    "Fuzzy UMAP graph weighting should be the C++ default."
  );
  umap_options.n_neighbors = std::min(12, km_res.knn.neighbors);
  umap_options.n_epochs = 5;
  umap_options.n_threads = 2;
  umap_options.seed = 9;
  kodama::EmbeddingResult umap_res = kodama::KODAMAUMAP_CPU(km_res.knn, umap_options);
  require(umap_res.samples == d.n, "CPU UMAP sample count mismatch.");
  require(umap_res.components == 2, "CPU UMAP component count mismatch.");
  require(umap_res.embedding.size() == static_cast<std::size_t>(d.n * 2), "CPU UMAP embedding size mismatch.");
  require(umap_res.initialization == "graph_spectral", "CPU UMAP graph fallback was not reported.");
  require(umap_res.optimizer == "csr_epoch_schedule", "CPU UMAP optimizer metadata mismatch.");
  require(umap_res.graph_edges > 0 && umap_res.graph_max_weight > 0.0f,
          "CPU UMAP graph diagnostics are missing.");
  for (float value : umap_res.embedding) require(std::isfinite(value), "CPU UMAP produced a non-finite value.");

  // The public binary-graph route is inherited from fastEmbedR and remains a
  // distinct contract from the default fuzzy graph. Duplicate and self edges
  // must be compacted before optimization.
  constexpr int binary_samples = 6;
  kodama::NeighborGraph binary_graph;
  binary_graph.neighbors = 4;
  binary_graph.index_base = kodama::GraphIndexBase::Zero;
  binary_graph.indices.resize(24);
  binary_graph.distances.resize(24, 1.0f);
  for (int row = 0; row < binary_samples; ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) * 4u;
    binary_graph.indices[offset] = (row + 1) % binary_samples;
    binary_graph.indices[offset + 1u] = (row + 1) % binary_samples;
    binary_graph.indices[offset + 2u] =
      (row + binary_samples - 1) % binary_samples;
    binary_graph.indices[offset + 3u] = row;
  }
  kodama::UMAPOptions binary_umap_options;
  binary_umap_options.graph_mode = kodama::UMAPGraphMode::Binary;
  binary_umap_options.n_neighbors = 4;
  binary_umap_options.n_epochs = 4;
  binary_umap_options.n_threads = 2;
  binary_umap_options.min_dist = 0.3;
  binary_umap_options.seed = 314;
  binary_umap_options.init_source = "test_explicit";
  binary_umap_options.init_backend = kodama::Backend::CPU;
  binary_umap_options.init.resize(12);
  for (int row = 0; row < binary_samples; ++row) {
    binary_umap_options.init[static_cast<std::size_t>(row) * 2u] =
      static_cast<float>(row) * 0.01f;
    binary_umap_options.init[static_cast<std::size_t>(row) * 2u + 1u] =
      static_cast<float>(row % 2) * 0.01f;
  }
  const kodama::EmbeddingResult binary_umap =
    kodama::KODAMAUMAP_CPU(binary_graph, binary_umap_options);
  const kodama::EmbeddingResult binary_umap_replay =
    kodama::KODAMAUMAP_CPU(binary_graph, binary_umap_options);
  require(binary_umap.graph_edges == 12 && binary_umap.graph_max_weight == 1.0f,
          "Binary UMAP did not compact the cyclic graph to the expected edges.");
  require(binary_umap.initialization == "test_explicit" &&
          binary_umap.initialization_backend == kodama::Backend::CPU,
          "Binary UMAP explicit initialization metadata mismatch.");
  require(binary_umap.embedding == binary_umap_replay.embedding,
          "Binary UMAP fixed-seed replay changed coordinates.");
  for (float value : binary_umap.embedding) {
    require(std::isfinite(value), "Binary UMAP produced a non-finite value.");
  }
  kodama::NeighborGraph self_only_graph;
  self_only_graph.neighbors = 1;
  self_only_graph.index_base = kodama::GraphIndexBase::Zero;
  self_only_graph.indices = {0, 1, 2};
  self_only_graph.distances = {0.0f, 0.0f, 0.0f};
  kodama::UMAPOptions self_only_options = binary_umap_options;
  self_only_options.n_neighbors = 1;
  self_only_options.init.clear();
  require_throws<std::runtime_error>([&] {
    (void)kodama::KODAMAUMAP_CPU(self_only_graph, self_only_options);
  }, "Binary UMAP accepted a graph containing only self edges.");

  kodama::VisualizationInitOptions visual_init_options;
  visual_init_options.n_components = 2;
  visual_init_options.n_threads = 2;
  visual_init_options.seed = 9;
  visual_init_options.backend = kodama::Backend::CPU;
  const kodama::VisualizationInitResult visual_init =
    kodama::KODAMAVisualizationPCAInit(fview, visual_init_options);
  require(visual_init.samples == d.n, "Visualization PCA initialization sample count mismatch.");
  require(visual_init.components == 2, "Visualization PCA initialization component count mismatch.");
  require(visual_init.backend == kodama::Backend::CPU, "Visualization PCA initialization backend mismatch.");
  require(visual_init.umap.size() == static_cast<std::size_t>(d.n * 2), "UMAP PCA initialization size mismatch.");
  require(visual_init.opentsne.size() == static_cast<std::size_t>(d.n * 2), "openTSNE PCA initialization size mismatch.");
  float max_tsne_sd = 0.0f;
  for (int component = 0; component < 2; ++component) {
    float mean = 0.0f;
    for (int row = 0; row < d.n; ++row) {
      mean += visual_init.opentsne[static_cast<std::size_t>(row) * 2u + component];
    }
    mean /= static_cast<float>(d.n);
    require(std::abs(mean) < 1e-6f, "openTSNE PCA initialization is not centered.");
    float sum_squares = 0.0f;
    for (int row = 0; row < d.n; ++row) {
      const float value =
        visual_init.opentsne[static_cast<std::size_t>(row) * 2u + component] - mean;
      sum_squares += value * value;
    }
    max_tsne_sd = std::max(
      max_tsne_sd,
      std::sqrt(sum_squares / static_cast<float>(d.n - 1))
    );
  }
  require(std::abs(max_tsne_sd - 1.0e-4f) < 1.0e-6f, "openTSNE PCA initialization scale mismatch.");

  const kodama::EmbeddingResult raw_umap_res =
    kodama::KODAMAUMAP_CPU(km_res.knn, fview, umap_options);
  require(raw_umap_res.initialization == "raw_pca", "Raw-data UMAP initialization was not used.");
  require(raw_umap_res.initialization_backend == kodama::Backend::CPU, "Raw-data UMAP initialization backend mismatch.");

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
  require(tsne_res.initialization == "random", "CPU openTSNE graph fallback was not reported.");
  require(tsne_res.optimizer == "opentsne_exact_sparse_knn_float32" &&
          tsne_res.graph_edges == 0 && tsne_res.graph_max_weight == 0.0f,
          "CPU openTSNE optimizer diagnostics mismatch.");
  for (float value : tsne_res.embedding) require(std::isfinite(value), "CPU openTSNE produced a non-finite value.");
  const kodama::EmbeddingResult raw_tsne_res =
    kodama::KODAMAOpenTSNE_CPU(km_res.knn, fview, tsne_options);
  require(raw_tsne_res.initialization == "raw_pca", "Raw-data openTSNE initialization was not used.");
  require(raw_tsne_res.initialization_backend == kodama::Backend::CPU, "Raw-data openTSNE initialization backend mismatch.");

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
  km_graph_options.graph_neighbors = std::min(20, km_base_res.knn.neighbors);
  km_graph_options.knn.k = 5;
  km_graph_options.apply_kodama_dissimilarity = true;
  kodama::KODAMAMatrixResult km_graph_knn_res = kodama::KODAMAMatrixFromGraph_CPU(
    km_base_res.knn,
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
    km_base_res.knn,
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
    km_base_res.knn,
    d.n,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    km_graph_pls_options
  );
  require(km_graph_pls_res.samples == d.n, "Graph-input PLS-LDA sample count mismatch.");
  require(km_graph_pls_res.res.size() == static_cast<std::size_t>(d.n), "Graph-input PLS-LDA result size mismatch.");
  require(km_graph_pls_res.graph_feature_seconds >= 0.0, "Graph-input PLS-LDA feature timing missing.");

  std::vector<float> graph_laplacian_features = kodama::KODAMAGraphFeatures_CPU(km_base_res.knn, d.n, [&]() {
    kodama::KODAMAMatrixOptions opts = km_graph_pls_options;
    opts.graph_feature_components = 4;
    opts.graph_feature_steps = 4;
    return opts;
  }());
  require(graph_laplacian_features.size() == static_cast<std::size_t>(d.n * 4), "Self-tuning graph feature size mismatch.");
  for (float value : graph_laplacian_features) require(std::isfinite(value), "Self-tuning graph features contain non-finite values.");

  {
    constexpr int rows = 4096;
    constexpr int dimensions = 5;
    constexpr int landmarks = 24;
    constexpr int k = 6;
    std::vector<float> data(static_cast<std::size_t>(rows * dimensions));
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < dimensions; ++column) {
        data[static_cast<std::size_t>(row * dimensions + column)] =
          std::sin(0.019f * static_cast<float>((row + 1) * (column + 2))) +
          0.0005f * static_cast<float>(row);
      }
    }
    std::vector<int> selected(static_cast<std::size_t>(landmarks));
    std::vector<int> allowed(static_cast<std::size_t>(rows), -1);
    std::vector<float> query(static_cast<std::size_t>(landmarks * dimensions));
    for (int local = 0; local < landmarks; ++local) {
      const int global = (local * rows) / landmarks;
      selected[static_cast<std::size_t>(local)] = global;
      allowed[static_cast<std::size_t>(global)] = local;
      std::copy_n(data.data() + static_cast<std::size_t>(global * dimensions),
                  dimensions,
                  query.data() + static_cast<std::size_t>(local * dimensions));
    }
    kodama::detail::NativeHNSWIndex index = kodama::detail::native_build_hnsw_index(
      data, rows, dimensions, kodama::DistanceMetric::Euclidean,
      kodama::detail::NativeHNSWParameters{24, 200, 150}, 4
    );
    const kodama::detail::NativeKNNResult filtered =
      kodama::detail::native_hnsw_index_filtered_search(
        index, query, landmarks, k, 4, selected, allowed
      );
    require(filtered.neighbors == k &&
            std::none_of(filtered.indices.begin(), filtered.indices.end(),
                         [](int id) { return id < 0; }),
            "CPU sparse-landmark HNSW query returned an incomplete graph.");
  }

#if defined(KODAMA_ENABLE_CUDA)
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
    auto context = kodama::detail::native_cuda_build_kmeans_context(
      context_data, context_rows, context_dimensions, 2, context_clusters, 0
    );
    for (int lane = 0; lane < 2; ++lane) {
      const std::uint64_t seed = 721u + static_cast<std::uint64_t>(lane);
      const std::vector<int> expected = kodama::detail::native_cuda_kmeans_labels(
        context_data, context_rows, context_dimensions, context_clusters, 5, seed, 0
      );
      const std::vector<int> observed = kodama::detail::native_cuda_kmeans_context_labels(
        context, lane, context_clusters, 5, seed
      );
      require(observed == expected,
              "Resident CUDA k-means changed assignments relative to the one-shot path.");
    }
    require(context.input_uploads() == 1,
            "Resident CUDA k-means uploaded its invariant input more than once.");
  }

  {
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
    std::vector<int> selected(static_cast<std::size_t>(sparse_landmarks));
    std::vector<float> query(
      static_cast<std::size_t>(sparse_landmarks * sparse_dimensions), 0.0f
    );
    for (int local = 0; local < sparse_landmarks; ++local) {
      const int global = (local * sparse_rows) / sparse_landmarks;
      selected[static_cast<std::size_t>(local)] = global;
      std::copy_n(
        sparse_data.data() + static_cast<std::size_t>(global * sparse_dimensions),
        sparse_dimensions,
        query.data() + static_cast<std::size_t>(local * sparse_dimensions)
      );
    }
    kodama::detail::NativeCudaIVFStats sparse_stats;
    kodama::detail::CudaResidentKODAMAGraph resident =
      kodama::detail::make_cuda_resident_kodama_graph_ivf(
        sparse_data, sparse_rows, sparse_dimensions, 16,
        kodama::DistanceMetric::Euclidean, 64, 0, 0, 1, &sparse_stats
      );
    const kodama::NeighborGraph filtered =
      kodama::detail::cuda_resident_landmark_knn_graph(
        resident, query, selected, sparse_k, 0, 0.99
      );
    require(filtered.neighbors == sparse_k,
            "CUDA sparse-landmark graph has the wrong width.");
    require(std::none_of(filtered.indices.begin(), filtered.indices.end(),
                         [](int id) { return id < 0; }),
            "CUDA sparse-landmark IVF query returned an incomplete graph.");
    std::vector<int> local_self(static_cast<std::size_t>(sparse_landmarks));
    std::iota(local_self.begin(), local_self.end(), 0);
    const kodama::detail::NativeKNNResult exact =
      kodama::detail::native_cuda_exact_knn_search(
        query, sparse_landmarks, query, sparse_landmarks, sparse_dimensions,
        sparse_k, kodama::DistanceMetric::Euclidean, 0, local_self
      );
    std::size_t hits = 0;
    for (int row = 0; row < sparse_landmarks; ++row) {
      for (int rank = 0; rank < sparse_k; ++rank) {
        const int candidate = filtered.indices[
          static_cast<std::size_t>(row * sparse_k + rank)
        ];
        const auto begin = exact.indices.begin() + row * sparse_k;
        if (std::find(begin, begin + sparse_k, candidate) != begin + sparse_k) ++hits;
      }
    }
    require(static_cast<double>(hits) /
              static_cast<double>(sparse_landmarks * sparse_k) >= 0.99,
            "CUDA sparse-landmark IVF recall fell below 0.99.");

    std::vector<int> constrained_labels(static_cast<std::size_t>(sparse_rows));
    std::vector<int> constrained_groups(static_cast<std::size_t>(sparse_rows));
    for (int row = 0; row < sparse_rows; ++row) {
      constrained_labels[static_cast<std::size_t>(row)] = row % 5 + 1;
      constrained_groups[static_cast<std::size_t>(row)] = row / 7;
    }
    std::vector<int> expected = constrained_labels;
    for (int begin = 0; begin < sparse_rows; begin += 7) {
      const int end = std::min(sparse_rows, begin + 7);
      int counts[6] = {0, 0, 0, 0, 0, 0};
      for (int row = begin; row < end; ++row) ++counts[constrained_labels[row]];
      int best = 1;
      for (int label = 2; label <= 5; ++label) {
        if (counts[label] > counts[best]) best = label;
      }
      for (int row = begin; row < end; ++row) expected[row] = best;
    }
    kodama::detail::cuda_resident_prepare_results(resident, 1);
    kodama::detail::cuda_resident_store_result_row(resident, constrained_labels, 0, 0);
    kodama::detail::cuda_resident_constrain_result_row(
      resident, constrained_groups, 5, 0, 0);
    require(kodama::detail::cuda_resident_download_results(resident, 1) == expected,
            "CUDA resident constrained majority differs from the CPU rule.");
  }

  kodama::NeighborGraph corrected_cpu = prepared_graph.knn;
  kodama::NeighborGraph corrected_cuda = prepared_graph.knn;
  kodama::KODAMADissimilarityInPlace(
    corrected_cpu,
    km_base_res.res,
    km_base_res.runs,
    km_base_res.samples,
    kodama::Backend::CPU,
    km_base_options.n_threads
  );
  kodama::KODAMADissimilarityInPlace(
    corrected_cuda,
    km_base_res.res,
    km_base_res.runs,
    km_base_res.samples,
    kodama::Backend::CUDA,
    1
  );
  require(corrected_cpu.indices == corrected_cuda.indices,
          "CUDA KODAMA correction changed graph topology.");
  require(corrected_cpu.distances.size() == corrected_cuda.distances.size(),
          "CUDA KODAMA correction changed graph size.");
  float correction_max_difference = 0.0f;
  for (std::size_t i = 0; i < corrected_cpu.distances.size(); ++i) {
    correction_max_difference = std::max(
      correction_max_difference,
      std::abs(corrected_cpu.distances[i] - corrected_cuda.distances[i])
    );
  }
  if (!(correction_max_difference < 2e-5f)) {
    throw std::runtime_error(
      "CUDA KODAMA correction disagrees with CPU; max absolute difference=" +
      std::to_string(correction_max_difference)
    );
  }

  kodama::PCAOptions cuda_pca_options;
  cuda_pca_options.n_components = 4;
  cuda_pca_options.oversample = 2;
  cuda_pca_options.power_iterations = 1;
  cuda_pca_options.seed = 13;
  const kodama::PCAResult cpu_pca_reference = kodama::PCA_CPU(fview, cuda_pca_options);
  const kodama::PCAResult cuda_pca = kodama::PCA_CUDA(fview, cuda_pca_options);
  require(cuda_pca.backend == kodama::Backend::CUDA, "CUDA PCA backend metadata mismatch.");
  require(cuda_pca.scores.size() == static_cast<std::size_t>(d.n * 4),
          "CUDA PCA score size mismatch.");
  for (int component = 0; component < 4; ++component) {
    const float reference = std::max(
      1.0f, cpu_pca_reference.singular_values[static_cast<std::size_t>(component)]
    );
    require(
      std::abs(cuda_pca.singular_values[static_cast<std::size_t>(component)] -
               cpu_pca_reference.singular_values[static_cast<std::size_t>(component)]) / reference < 3e-3f,
      "CUDA PCA singular values disagree with CPU."
    );
  }

  kodama::VisualizationInitOptions cuda_visual_init_options;
  cuda_visual_init_options.n_components = 2;
  cuda_visual_init_options.n_threads = 1;
  cuda_visual_init_options.seed = 13;
  cuda_visual_init_options.backend = kodama::Backend::CUDA;
  const kodama::VisualizationInitResult cuda_visual_init =
    kodama::KODAMAVisualizationPCAInit(fview, cuda_visual_init_options);
  require(cuda_visual_init.backend == kodama::Backend::CUDA,
          "CUDA visualization initialization backend metadata mismatch.");
  require(cuda_visual_init.umap.size() == static_cast<std::size_t>(d.n * 2) &&
          cuda_visual_init.opentsne.size() == static_cast<std::size_t>(d.n * 2),
          "CUDA visualization initialization size mismatch.");

  const kodama::EmbeddingResult cuda_umap = kodama::KODAMAUMAP_CUDA(
    km_res.knn, fview, umap_options
  );
  require(cuda_umap.backend == kodama::Backend::CUDA &&
          cuda_umap.embedding.size() == static_cast<std::size_t>(d.n * 2),
          "CUDA UMAP smoke test failed.");
  require(cuda_umap.initialization == "raw_pca" &&
          cuda_umap.initialization_backend == kodama::Backend::CUDA,
          "CUDA raw-data UMAP initialization metadata mismatch.");
  require(cuda_umap.optimizer == "cuda_atomic_coo_epoch_schedule" &&
          cuda_umap.graph_edges > 0 && cuda_umap.graph_max_weight > 0.0f,
          "CUDA UMAP optimizer diagnostics mismatch.");
  for (const float value : cuda_umap.embedding) {
    require(std::isfinite(value), "CUDA UMAP produced a non-finite value.");
  }
  const kodama::EmbeddingResult cuda_tsne = kodama::KODAMAOpenTSNE_CUDA(
    km_res.knn, fview, tsne_options
  );
  require(cuda_tsne.backend == kodama::Backend::CUDA &&
          cuda_tsne.embedding.size() == static_cast<std::size_t>(d.n * 2),
          "CUDA openTSNE smoke test failed.");
  require(cuda_tsne.initialization == "raw_pca" &&
          cuda_tsne.initialization_backend == kodama::Backend::CUDA,
          "CUDA raw-data openTSNE initialization metadata mismatch.");
  require(cuda_tsne.optimizer == "cuda_opentsne_fft_grid_sparse_knn_float32" &&
          cuda_tsne.graph_edges == 0 && cuda_tsne.graph_max_weight == 0.0f,
          "CUDA openTSNE optimizer diagnostics mismatch.");
  for (const float value : cuda_tsne.embedding) {
    require(std::isfinite(value), "CUDA openTSNE produced a non-finite value.");
  }

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

  std::vector<float> predict_train_x;
  std::vector<float> predict_test_x;
  std::vector<int> predict_train_labels;
  predict_train_x.reserve(static_cast<std::size_t>(120 * d.p));
  predict_test_x.reserve(static_cast<std::size_t>(30 * d.p));
  predict_train_labels.reserve(120);
  for (int cls = 0; cls < 3; ++cls) {
    for (int within = 0; within < 50; ++within) {
      const int row = cls * 50 + within;
      std::vector<float>& destination = within < 40 ? predict_train_x : predict_test_x;
      destination.insert(
        destination.end(),
        xf.begin() + static_cast<std::ptrdiff_t>(row * d.p),
        xf.begin() + static_cast<std::ptrdiff_t>((row + 1) * d.p)
      );
      if (within < 40) predict_train_labels.push_back(d.y[static_cast<std::size_t>(row)]);
    }
  }
  const kodama::MatrixView predict_train_view{
    predict_train_x.data(), 120, static_cast<std::size_t>(d.p)
  };
  const kodama::MatrixView predict_test_view{
    predict_test_x.data(), 30, static_cast<std::size_t>(d.p)
  };
  kodama::PLSOptions predict_cpu_options = cuda_pls;
  predict_cpu_options.backend = kodama::Backend::CPU;
  const std::vector<int> predict_cpu = kodama::PLSLDAPredict_CPU(
    predict_train_view, predict_train_labels, predict_test_view, predict_cpu_options
  );
  const std::vector<int> predict_cuda = kodama::PLSLDAPredict_CUDA(
    predict_train_view, predict_train_labels, predict_test_view, cuda_pls
  );
  require(predict_cuda == predict_cpu, "Direct CUDA PLS-LDA prediction disagrees with CPU.");
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
  const kodama::CoreResult cuda_graph_core_kres = kodama::CoreKNNGraph_CUDA(
    km_base_res.knn, d.n, noisy, d.constrain, fixed, cuda_core_knn
  );
  require(cuda_graph_core_kres.clbest.size() == noisy.size(),
          "Direct CUDA graph-input KNN core size mismatch.");
  require(cuda_graph_core_kres.cycles_completed >= 1,
          "Direct CUDA graph-input KNN core did not run any cycles.");

  kodama::KNNOptions resident_cuda_options = knn;
  resident_cuda_options.backend = kodama::Backend::CUDA;
  resident_cuda_options.index_type = kodama::KNNIndexType::CudaIVFFlat;
  resident_cuda_options.metric = kodama::DistanceMetric::Euclidean;
  resident_cuda_options.ivf_nlist = 8;
  resident_cuda_options.ivf_nprobe = 8;
  kodama::ResidentIVFIndex resident_cuda =
    kodama::BuildResidentIVFIndex(fview, resident_cuda_options);
  require(resident_cuda.valid(), "Resident CUDA IVF index is invalid.");
  require(resident_cuda.backend() == kodama::Backend::CUDA,
          "Resident CUDA IVF index reported the wrong backend.");
  require(resident_cuda.rows() == d.n && resident_cuda.dimensions() == d.p,
          "Resident CUDA IVF index dimensions are incorrect.");
  require(resident_cuda.nlist() == 8, "Resident CUDA IVF nlist mismatch.");
  std::vector<float> nlist_regression_data(400 * 4);
  for (int row = 0; row < 400; ++row) {
    for (int column = 0; column < 4; ++column) {
      nlist_regression_data[static_cast<std::size_t>(row * 4 + column)] =
        static_cast<float>((row * 17 + column * 29) % 101) / 101.0f;
    }
  }
  kodama::KNNOptions nlist_regression_options = resident_cuda_options;
  nlist_regression_options.ivf_nlist = 300;
  nlist_regression_options.ivf_nprobe = 32;
  kodama::ResidentIVFIndex nlist_regression_index =
    kodama::BuildResidentIVFIndex(
      kodama::MatrixView{nlist_regression_data.data(), 400, 4},
      nlist_regression_options
    );
  require(nlist_regression_index.nlist() == 300,
          "CUDA IVF nlist was incorrectly capped by the nprobe limit.");
  kodama::ResidentIVFSearchStats resident_cuda_stats;
  const kodama::NeighborGraph resident_cuda_first =
    kodama::SearchResidentIVFIndexSelf(
      resident_cuda,
      3,
      true,
      &resident_cuda_stats
    );
  const kodama::NeighborGraph resident_cuda_second =
    kodama::SearchResidentIVFIndexSelf(resident_cuda, 3, true);
  require(resident_cuda_stats.backend == kodama::Backend::CUDA,
          "Resident CUDA IVF search reported the wrong backend.");
  require(resident_cuda_stats.nlist == 8 && resident_cuda_stats.nprobe == 8,
          "Resident CUDA IVF search parameters mismatch.");
  require(resident_cuda_first.indices == resident_cuda_second.indices,
          "Resident CUDA IVF index reuse changed neighbor indices.");
  require(resident_cuda_first.distances == resident_cuda_second.distances,
          "Resident CUDA IVF index reuse changed neighbor distances.");
  const kodama::NeighborGraph resident_cuda_query =
    kodama::SearchResidentIVFIndex(
      resident_cuda,
      kodama::MatrixView{
        xf.data(),
        5,
        static_cast<std::size_t>(d.p)
      },
      3
    );
  require(resident_cuda_query.neighbors == 3 &&
          resident_cuda_query.indices.size() == 15,
          "Resident CUDA IVF external-query shape mismatch.");
  require(resident_cuda_query.indices.front() == 1,
          "Resident CUDA IVF external query did not retain the nearest row.");
  for (int row = 0; row < d.n; ++row) {
    for (int column = 0; column < resident_cuda_first.neighbors; ++column) {
      const int neighbor = resident_cuda_first.indices[
        static_cast<std::size_t>(row * resident_cuda_first.neighbors + column)
      ];
      require(neighbor != row + 1, "Resident CUDA IVF self-exclusion failed.");
    }
  }

  kodama::KODAMAGraphOptions direct_cuda_graph_options;
  direct_cuda_graph_options.neighbors = 3;
  direct_cuda_graph_options.backend = kodama::Backend::CUDA;
  direct_cuda_graph_options.materialize_graph = true;
  direct_cuda_graph_options.metric = kodama::DistanceMetric::Euclidean;
  direct_cuda_graph_options.index_type = kodama::KNNIndexType::CudaIVFFlat;
  direct_cuda_graph_options.ivf_nlist = 8;
  direct_cuda_graph_options.ivf_nprobe = 8;
  const kodama::KODAMAGraphResult direct_cuda_graph = kodama::KODAMAGraph_CUDA(
    fview, direct_cuda_graph_options
  );
  require(direct_cuda_graph.knn.neighbors == 3,
          "Direct CUDA KODAMAGraph neighbor count mismatch.");
  require(direct_cuda_graph.index_type == kodama::KNNIndexType::CudaIVFFlat,
          "Direct CUDA KODAMAGraph did not retain IVF provenance.");

  kodama::KODAMAGraphOptions lazy_cuda_graph_options = direct_cuda_graph_options;
  lazy_cuda_graph_options.materialize_graph = false;
  const kodama::KODAMAGraphResult lazy_cuda_graph = kodama::KODAMAGraph_CUDA(
    fview, lazy_cuda_graph_options
  );
  require(lazy_cuda_graph.handle && lazy_cuda_graph.handle->valid() &&
          lazy_cuda_graph.knn.indices.empty(),
          "Lazy CUDA KODAMAGraph materialized host arrays.");

  kodama::KODAMAMatrixOptions cuda_km_options = km_options;
  cuda_km_options.backend = kodama::Backend::CUDA;
  cuda_km_options.runs = 1;
  cuda_km_options.cycles = 1;
  cuda_km_options.n_threads = 1;
  cuda_km_options.knn.backend = kodama::Backend::CUDA;
  cuda_km_options.knn.index_type = kodama::KNNIndexType::CudaExact;
  kodama::KODAMAMatrixOptions lazy_cuda_matrix_options = cuda_km_options;
  lazy_cuda_matrix_options.materialize_graph = false;
  const kodama::KODAMAMatrixResult lazy_cuda_first = kodama::KODAMAMatrix(
    fview, lazy_cuda_graph, {}, {}, fixed, lazy_cuda_matrix_options
  );
  const kodama::KODAMAMatrixResult lazy_cuda_second = kodama::KODAMAMatrix(
    fview, lazy_cuda_graph, {}, {}, fixed, lazy_cuda_matrix_options
  );
  require(lazy_cuda_first.graph_builds == 0 && lazy_cuda_second.graph_builds == 0 &&
          lazy_cuda_first.knn.indices.empty() && lazy_cuda_second.knn.indices.empty() &&
          lazy_cuda_first.res == lazy_cuda_second.res,
          "Reusable CUDA KODAMAGraph handle failed.");
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
  require(cuda_km_res.effective_landmarks == cuda_km_options.landmarks, "CUDA KODAMAMatrix effective landmark count mismatch.");
  require(cuda_km_res.landmark_occupied_strata == std::vector<int>{cuda_km_options.splitting}, "CUDA nonspatial landmark stratum count mismatch.");
  require(cuda_km_res.landmark_represented_strata == std::vector<int>{cuda_km_options.splitting}, "CUDA nonspatial landmark coverage mismatch.");
  require(cuda_km_res.kmeans_input_uploads == 1,
          "CUDA KODAMAMatrix did not reuse one resident coarse-k-means input.");
  require(cuda_km_res.projection_sparse_uploads == 1 &&
          cuda_km_res.projection_full_downloads == 0 &&
          cuda_km_res.result_row_uploads == 0 &&
          cuda_km_res.result_matrix_downloads == 1,
          "CUDA KODAMAMatrix did not retain projected labels through dissimilarity.");

  const kodama::KODAMAMatrixResult cuda_graph_data_matrix =
    kodama::KODAMAMatrixFromGraphData_CUDA(
      fview,
      km_base_res.knn,
      std::vector<int>(),
      std::vector<int>(),
      fixed,
      cuda_km_options
    );
  require(cuda_graph_data_matrix.backend == kodama::Backend::CUDA,
          "Direct CUDA graph-and-data KODAMAMatrix backend mismatch.");
  require(cuda_graph_data_matrix.res.size() == static_cast<std::size_t>(d.n),
          "Direct CUDA graph-and-data KODAMAMatrix result size mismatch.");
  const kodama::KODAMAMatrixResult cuda_graph_matrix =
    kodama::KODAMAMatrixFromGraph_CUDA(
      km_base_res.knn,
      d.n,
      std::vector<int>(),
      std::vector<int>(),
      fixed,
      cuda_km_options
    );
  require(cuda_graph_matrix.backend == kodama::Backend::CUDA,
          "Direct CUDA graph-input KODAMAMatrix backend mismatch.");
  require(cuda_graph_matrix.res.size() == static_cast<std::size_t>(d.n),
          "Direct CUDA graph-input KODAMAMatrix result size mismatch.");

  const kodama::KODAMAMatrixResult cuda_km_repeat =
    kodama::KODAMAMatrix_CUDA(
      fview,
      std::vector<int>(),
      std::vector<int>(),
      fixed,
      cuda_km_options
    );
  require(
    cuda_km_res.res == cuda_km_repeat.res,
    "Resident CUDA KODAMA KNN is not repeatable."
  );
  require(
    cuda_km_res.knn.indices == cuda_km_repeat.knn.indices,
    "Resident CUDA KODAMA KNN graph ordering is not repeatable."
  );
  require(
    cuda_km_res.knn.distances == cuda_km_repeat.knn.distances,
    "Resident CUDA KODAMA KNN graph distances are not repeatable."
  );

  kodama::KODAMAMatrixOptions cuda_cosine_ivf_options = cuda_km_options;
  cuda_cosine_ivf_options.metric = kodama::DistanceMetric::Cosine;
  cuda_cosine_ivf_options.knn.metric = kodama::DistanceMetric::Cosine;
  cuda_cosine_ivf_options.knn.index_type = kodama::KNNIndexType::CudaIVFFlat;
  cuda_cosine_ivf_options.knn.ivf_nlist = 8;
  cuda_cosine_ivf_options.knn.ivf_nprobe = 8;
  cuda_cosine_ivf_options.apply_kodama_dissimilarity = false;
  const kodama::KODAMAMatrixResult cuda_cosine_ivf =
    kodama::KODAMAMatrix_CUDA(
      fview,
      std::vector<int>(),
      std::vector<int>(),
      fixed,
      cuda_cosine_ivf_options
    );
  require(cuda_cosine_ivf.graph_index_type == kodama::KNNIndexType::CudaIVFFlat,
          "CUDA KODAMAMatrix did not select resident IVF-Flat explicitly.");
  for (int row = 0; row < d.n; ++row) {
    float row_norm = 0.0f;
    for (int dimension = 0; dimension < d.p; ++dimension) {
      const float value = xf[static_cast<std::size_t>(row * d.p + dimension)];
      row_norm += value * value;
    }
    row_norm = std::sqrt(row_norm);
    for (int rank = 0; rank < cuda_cosine_ivf.knn.neighbors; ++rank) {
      const std::size_t offset = static_cast<std::size_t>(
        row * cuda_cosine_ivf.knn.neighbors + rank
      );
      const int neighbor = cuda_cosine_ivf.knn.indices[offset] - 1;
      float neighbor_norm = 0.0f;
      float dot = 0.0f;
      for (int dimension = 0; dimension < d.p; ++dimension) {
        const float left = xf[static_cast<std::size_t>(row * d.p + dimension)];
        const float right = xf[static_cast<std::size_t>(neighbor * d.p + dimension)];
        dot += left * right;
        neighbor_norm += right * right;
      }
      neighbor_norm = std::sqrt(neighbor_norm);
      const float expected = row_norm > 0.0f && neighbor_norm > 0.0f ?
        1.0f - dot / (row_norm * neighbor_norm) : 1.0f;
      require(std::fabs(cuda_cosine_ivf.knn.distances[offset] - expected) < 1.0e-4f,
              "Resident CUDA IVF graph did not preserve cosine distances.");
    }
  }

  kodama::KODAMAMatrixOptions cuda_pls_matrix_options = cuda_km_options;
  cuda_pls_matrix_options.classifier = kodama::CoreClassifier::PLS_LDA;
  cuda_pls_matrix_options.components = 3;
  const kodama::KODAMAMatrixResult cuda_pls_matrix =
    kodama::KODAMAMatrix_CUDA(
      fview,
      std::vector<int>(),
      std::vector<int>(),
      fixed,
      cuda_pls_matrix_options
    );
  const kodama::KODAMAMatrixResult cuda_pls_matrix_repeat =
    kodama::KODAMAMatrix_CUDA(
      fview,
      std::vector<int>(),
      std::vector<int>(),
      fixed,
      cuda_pls_matrix_options
    );
  require(
    cuda_pls_matrix.res == cuda_pls_matrix_repeat.res,
    "Resident CUDA KODAMA PLS-LDA is not repeatable."
  );
  require(
    cuda_pls_matrix.knn.indices == cuda_pls_matrix_repeat.knn.indices,
    "Resident CUDA KODAMA PLS-LDA graph ordering is not repeatable."
  );

  kodama::KODAMAMatrixOptions cuda_spatial_options = km_spatial_options;
  cuda_spatial_options.backend = kodama::Backend::CUDA;
  cuda_spatial_options.knn.backend = kodama::Backend::CUDA;
  cuda_spatial_options.knn.index_type = kodama::KNNIndexType::CudaExact;
  kodama::KODAMAMatrixResult cuda_spatial_res = kodama::KODAMAMatrix_CUDA(
    fview,
    std::vector<int>(),
    std::vector<int>(),
    fixed,
    cuda_spatial_options
  );
  require(cuda_spatial_res.effective_landmarks == km_spatial_res.effective_landmarks, "CUDA spatial effective landmark count differs from CPU.");
  require(cuda_spatial_res.landmark_grid_bins == km_spatial_res.landmark_grid_bins, "CUDA spatial landmark grid differs from CPU.");
  require(cuda_spatial_res.landmark_occupied_strata == km_spatial_res.landmark_occupied_strata, "CUDA spatial occupied-cell count differs from CPU.");
  require(cuda_spatial_res.landmark_represented_strata == km_spatial_res.landmark_represented_strata, "CUDA spatial represented-cell count differs from CPU.");
#endif

#if defined(KODAMA_ENABLE_METAL)
  if (kodama::MetalAvailable()) {
    kodama::KNNOptions metal_knn = knn;
    metal_knn.backend = kodama::Backend::Metal;
    metal_knn.index_type = kodama::KNNIndexType::MetalExact;
    kodama::KNNCVResult metal_kres =
        kodama::KNNCV_METAL(fview, d.y, d.constrain, metal_knn);
    require(metal_kres.parameters.backend == kodama::Backend::Metal,
            "Metal KNNCV did not report Metal backend.");
    require(metal_kres.parameters.index_type ==
                kodama::KNNIndexType::MetalExact,
            "Metal KNNCV did not report exact Metal search.");
    check_constrained_folds(d.constrain, metal_kres.fold_assignments);
    require(metal_kres.global_accuracy > 0.95,
            "Metal KNNCV accuracy unexpectedly low.");

    kodama::PLSOptions metal_pls = pls;
    metal_pls.backend = kodama::Backend::Metal;
    kodama::PLSCVResult metal_lres =
        kodama::PLSLDACV_METAL(fview, d.y, d.constrain, metal_pls);
    require(metal_lres.parameters.backend == kodama::Backend::Metal,
            "Metal PLS-LDA did not report Metal backend.");
    check_pls_result(metal_lres, d.y, d.constrain, 4);
    require(metal_lres.global_accuracy > 0.60,
            "Metal PLS-LDA accuracy unexpectedly low.");
    require(std::abs(metal_lres.global_accuracy - flres.global_accuracy) < 0.10,
            "Metal PLS-LDA diverged from CPU accuracy.");

    kodama::CoreOptions metal_core_knn = core_knn;
    metal_core_knn.knn.backend = kodama::Backend::Metal;
    metal_core_knn.knn.index_type = kodama::KNNIndexType::MetalExact;
    kodama::CoreResult metal_core_kres =
        kodama::CoreKNN_METAL(fview, noisy, d.constrain, fixed, metal_core_knn);
    require(metal_core_kres.clbest.size() == noisy.size(),
            "Metal Core KNN clbest size mismatch.");
    require(metal_core_kres.cycles_completed >= 1,
            "Metal Core KNN did not run any cycles.");

    kodama::CoreOptions metal_core_pls = core_pls;
    metal_core_pls.pls.backend = kodama::Backend::Metal;
    kodama::CoreResult metal_core_lres = kodama::CorePLSLDA_METAL(
        fview, noisy, d.constrain, fixed, metal_core_pls);
    require(metal_core_lres.clbest.size() == noisy.size(),
            "Metal Core PLS-LDA clbest size mismatch.");
    require(metal_core_lres.cycles_completed >= 1,
            "Metal Core PLS-LDA did not run any cycles.");
  } else {
    std::cout << "Metal runtime checks skipped: no Apple Metal device is "
                 "available.\n";
  }
#endif

  std::cout << "All kodama-cpp CV tests passed.\n";
  return 0;
}
