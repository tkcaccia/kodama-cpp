#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/MetricType.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(KODAMA_ENABLE_CUDA)
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#endif

namespace kodama {
namespace {

struct CVPrediction {
  std::vector<int> predicted;
  double runtime_seconds = 0.0;
  double peak_memory_mb = 0.0;
};

struct Neighbor {
  double score = 0.0;
  int label = 0;
};

struct HNSWParameters {
  int m = 32;
  int ef_construction = 200;
  int ef_search = 150;
  int tune_k = 50;
  double target_recall = 0.99;
};

class OmpThreadScope {
 public:
  explicit OmpThreadScope(int n_threads) {
#ifdef _OPENMP
    previous_ = omp_get_max_threads();
    if (n_threads > 0) {
      omp_set_num_threads(std::max(1, n_threads));
    }
#else
    (void)n_threads;
#endif
  }

  ~OmpThreadScope() {
#ifdef _OPENMP
    if (previous_ > 0) {
      omp_set_num_threads(previous_);
    }
#endif
  }

 private:
#ifdef _OPENMP
  int previous_ = 0;
#endif
};

int clamp_int(int value, int fallback, int lo, int hi) {
  const int x = value > 0 ? value : fallback;
  return std::min(hi, std::max(lo, x));
}

double hnsw_target_recall(double target_recall) {
  if (!std::isfinite(target_recall)) return 0.99;
  if (target_recall <= 0.925) return 0.90;
  if (target_recall <= 0.975) return 0.95;
  return 0.99;
}

HNSWParameters tune_hnsw_parameters(int n, int p, int k, const KNNOptions& options) {
  HNSWParameters out;
  out.tune_k = options.hnsw_tune_k > 0 ? options.hnsw_tune_k : 50;
  out.target_recall = hnsw_target_recall(options.hnsw_target_recall);
  const int tune_k = std::max(1, out.tune_k);
  const bool high_dim = p >= 256;
  const bool large_n = n >= 50000;
  const bool very_large_high_dim = large_n && high_dim;
  const bool small_k = tune_k <= 15;
  const bool large_k = tune_k >= 100;

  if (small_k) {
    out.m = 32;
    out.ef_construction = 160;
    out.ef_search = std::max(120, 4 * tune_k);
  } else if ((very_large_high_dim && large_k) || (large_k || high_dim)) {
    out.m = 48;
    out.ef_construction = 240;
    out.ef_search = std::max(220, 3 * tune_k);
  } else {
    out.m = 32;
    out.ef_construction = 200;
    out.ef_search = std::max(150, 3 * tune_k);
  }

  out.m = clamp_int(options.hnsw_m, out.m, 1, 256);
  out.m = std::min(out.m, std::max(1, n > 1 ? n - 1 : 1));
  out.ef_construction = clamp_int(options.hnsw_ef_construction, out.ef_construction, out.m, 4096);
  out.ef_search = clamp_int(options.hnsw_ef_search, out.ef_search, std::max(k, tune_k), 4096);
  return out;
}

int majority_vote(const std::vector<Neighbor>& neighbors) {
  if (neighbors.empty()) return 0;
  std::map<int, std::pair<int, double>> votes;
  for (const auto& nb : neighbors) {
    auto& vote = votes[nb.label];
    vote.first += 1;
    vote.second += nb.score;
  }
  int best_label = votes.begin()->first;
  int best_count = votes.begin()->second.first;
  double best_score = votes.begin()->second.second;
  for (const auto& kv : votes) {
    if (kv.second.first > best_count ||
        (kv.second.first == best_count && kv.second.second > best_score) ||
        (kv.second.first == best_count && kv.second.second == best_score && kv.first < best_label)) {
      best_label = kv.first;
      best_count = kv.second.first;
      best_score = kv.second.second;
    }
  }
  return best_label;
}

std::vector<float> make_faiss_matrix(
  MatrixView x,
  const std::vector<int>& rows,
  DistanceMetric metric
) {
  std::vector<float> out(rows.size() * x.cols, 0.0f);
  if (metric != DistanceMetric::Cosine) {
    if (x.value_type == MatrixValueType::Float32) {
      const float* data = static_cast<const float*>(x.data);
      for (std::size_t r = 0; r < rows.size(); ++r) {
        const std::size_t src = static_cast<std::size_t>(rows[r]) * x.cols;
        std::copy_n(data + src, x.cols, out.data() + r * x.cols);
      }
    } else {
      for (std::size_t r = 0; r < rows.size(); ++r) {
        const int row = rows[r];
        for (std::size_t j = 0; j < x.cols; ++j) {
          out[r * x.cols + j] = x.value_float(static_cast<std::size_t>(row), j);
        }
      }
    }
    return out;
  }

  for (std::size_t r = 0; r < rows.size(); ++r) {
    long double ss = 0.0;
    const int row = rows[r];
    for (std::size_t j = 0; j < x.cols; ++j) {
      const float v = x.value_float(static_cast<std::size_t>(row), j);
      ss += static_cast<long double>(v) * static_cast<long double>(v);
    }
    const double n = std::sqrt(static_cast<double>(ss));
    const double scale = metric == DistanceMetric::Cosine && n > 0.0 && std::isfinite(n) ? 1.0 / n : 1.0;
    for (std::size_t j = 0; j < x.cols; ++j) {
      out[r * x.cols + j] = static_cast<float>(static_cast<double>(x.value_float(static_cast<std::size_t>(row), j)) * scale);
    }
  }
  return out;
}

std::vector<float> make_faiss_matrix(MatrixView x, DistanceMetric metric) {
  std::vector<float> out(x.rows * x.cols, 0.0f);
  if (metric != DistanceMetric::Cosine) {
    if (x.value_type == MatrixValueType::Float32) {
      const float* data = static_cast<const float*>(x.data);
      std::copy_n(data, x.rows * x.cols, out.data());
    } else {
      for (std::size_t r = 0; r < x.rows; ++r) {
        for (std::size_t j = 0; j < x.cols; ++j) {
          out[r * x.cols + j] = x.value_float(r, j);
        }
      }
    }
    return out;
  }

  for (std::size_t r = 0; r < x.rows; ++r) {
    long double ss = 0.0;
    for (std::size_t j = 0; j < x.cols; ++j) {
      const float v = x.value_float(r, j);
      ss += static_cast<long double>(v) * static_cast<long double>(v);
    }
    const double n = std::sqrt(static_cast<double>(ss));
    const double scale = metric == DistanceMetric::Cosine && n > 0.0 && std::isfinite(n) ? 1.0 / n : 1.0;
    for (std::size_t j = 0; j < x.cols; ++j) {
      out[r * x.cols + j] = static_cast<float>(static_cast<double>(x.value_float(r, j)) * scale);
    }
  }
  return out;
}

std::vector<float> gather_faiss_rows(
  const std::vector<float>& all_x,
  std::size_t cols,
  const std::vector<int>& rows
) {
  std::vector<float> out(rows.size() * cols, 0.0f);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const int row = rows[r];
    std::copy_n(all_x.data() + static_cast<std::size_t>(row) * cols, cols, out.data() + r * cols);
  }
  return out;
}

void gather_faiss_rows(
  const std::vector<float>& all_x,
  std::size_t cols,
  const std::vector<int>& rows,
  std::vector<float>& out
) {
  out.resize(rows.size() * cols);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const int row = rows[r];
    std::copy_n(all_x.data() + static_cast<std::size_t>(row) * cols, cols, out.data() + r * cols);
  }
}

std::vector<int> normalized_constrain(const std::vector<int>& constrain, std::size_t n) {
  if (!constrain.empty()) return constrain;
  std::vector<int> out(n);
  std::iota(out.begin(), out.end(), 0);
  return out;
}

std::vector<int> normalized_fixed(const std::vector<int>& fixed, std::size_t n) {
  if (fixed.empty()) return std::vector<int>(n, 0);
  if (fixed.size() != n) throw std::invalid_argument("fixed size must be zero or match number of rows.");
  return fixed;
}

struct PrecomputedKNNFold {
  int fold = 0;
  std::vector<int> validation;
  std::vector<int> neighbor_rows;
  std::vector<float> scores;
  int k = 0;
};

struct PrecomputedKNN {
  std::vector<int> folds;
  std::vector<PrecomputedKNNFold> fold_data;
  KNNParametersUsed parameters;
};

struct KNNPredictionScratch {
  std::unordered_map<int, int> label_to_code;
  std::vector<int> label_values;
  std::vector<int> label_codes;
  std::vector<int> vote_counts;
  std::vector<int> fallback_counts;
  std::vector<double> vote_scores;
  std::vector<int> touched;
  bool label_map_initialized = false;
};

void initialize_knn_label_map(KNNPredictionScratch& scratch, const std::vector<int>& labels) {
  scratch.label_to_code.clear();
  scratch.label_values.clear();
  scratch.label_values.reserve(labels.size());
  scratch.label_to_code.reserve(labels.size());
  for (int label : labels) {
    if (scratch.label_to_code.find(label) != scratch.label_to_code.end()) continue;
    const int code = static_cast<int>(scratch.label_values.size());
    scratch.label_to_code.emplace(label, code);
    scratch.label_values.push_back(label);
  }
  scratch.label_map_initialized = true;
}

KNNParametersUsed resolve_core_knn_parameters(const KNNOptions& options) {
  KNNParametersUsed used;
  used.backend = Backend::CPU;
  used.index_type = options.index_type == KNNIndexType::FaissIVFFlat ?
    KNNIndexType::FaissIVFFlat : KNNIndexType::FaissHNSWFlat;
  used.metric = options.metric;
  used.k = options.k;
  used.ivf_nlist = options.ivf_nlist;
  used.ivf_nprobe = options.ivf_nprobe;
  used.hnsw_m = options.hnsw_m;
  used.hnsw_ef_construction = options.hnsw_ef_construction;
  used.hnsw_ef_search = options.hnsw_ef_search;
  used.hnsw_tune_k = options.hnsw_tune_k > 0 ? options.hnsw_tune_k : 50;
  used.hnsw_target_recall = hnsw_target_recall(options.hnsw_target_recall);
  used.gpu_device = options.gpu_device;
  used.n_threads = options.n_threads;
  return used;
}

PrecomputedKNN precompute_knn_cv_cpu(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
  if (options.k < 1) throw std::invalid_argument("KNNOptions::k must be positive.");
  if (options.backend == Backend::CUDA) {
    throw std::invalid_argument("CoreKNN precomputed-neighbor mode currently supports CPU FAISS KNN.");
  }

  PrecomputedKNN precomputed;
  precomputed.parameters = resolve_core_knn_parameters(options);
  precomputed.folds = detail::make_folds(labels, constrain, options.cv);
  const std::vector<int> fold_ids = detail::sorted_unique_folds(precomputed.folds);
  const std::vector<float> all_x = make_faiss_matrix(x, precomputed.parameters.metric);
  std::vector<float> train_x;
  std::vector<float> val_x;
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(precomputed.folds, fold, true);
    const std::vector<int> train = detail::indices_where_fold(precomputed.folds, fold, false);
    if (train.empty()) throw std::runtime_error("KNN fold has no training samples.");

    const int k = std::min(precomputed.parameters.k, static_cast<int>(train.size()));
    gather_faiss_rows(all_x, x.cols, train, train_x);
    gather_faiss_rows(all_x, x.cols, validation, val_x);
    const int d = static_cast<int>(x.cols);
    std::vector<faiss::idx_t> idx(validation.size() * static_cast<std::size_t>(k), -1);
    std::vector<float> scores(validation.size() * static_cast<std::size_t>(k), -std::numeric_limits<float>::infinity());
    if (precomputed.parameters.index_type == KNNIndexType::FaissIVFFlat) {
      if (precomputed.parameters.metric == DistanceMetric::Euclidean) {
        throw std::invalid_argument("FAISS IVF CPU KNN currently supports cosine/inner product; use FAISS HNSW for euclidean.");
      }
      int nlist = precomputed.parameters.ivf_nlist > 0 ? precomputed.parameters.ivf_nlist :
        std::max(1, static_cast<int>(std::sqrt(static_cast<double>(train.size() / 4 + 1))));
      if (precomputed.parameters.ivf_nlist <= 0 && train.size() < 1000) {
        nlist = std::max(1, static_cast<int>(train.size() / 40));
      }
      nlist = std::min(nlist, static_cast<int>(train.size()));
      faiss::IndexFlatIP quantizer(d);
      faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_INNER_PRODUCT);
      index.nprobe = precomputed.parameters.ivf_nprobe > 0 ? precomputed.parameters.ivf_nprobe :
        std::max<std::size_t>(1, std::min<std::size_t>(static_cast<std::size_t>(nlist), static_cast<std::size_t>(std::sqrt(static_cast<double>(nlist))) + 1));
      index.nprobe = std::min<std::size_t>(index.nprobe, static_cast<std::size_t>(nlist));
      index.train(train.size(), train_x.data());
      index.add(train.size(), train_x.data());
      OmpThreadScope threads(precomputed.parameters.n_threads);
      index.search(validation.size(), val_x.data(), k, scores.data(), idx.data());
    } else {
      const HNSWParameters hnsw = tune_hnsw_parameters(static_cast<int>(train.size()), d, k, options);
      precomputed.parameters.hnsw_m = hnsw.m;
      precomputed.parameters.hnsw_ef_construction = hnsw.ef_construction;
      precomputed.parameters.hnsw_ef_search = hnsw.ef_search;
      precomputed.parameters.hnsw_tune_k = hnsw.tune_k;
      precomputed.parameters.hnsw_target_recall = hnsw.target_recall;
      const faiss::MetricType faiss_metric = precomputed.parameters.metric == DistanceMetric::Euclidean ?
        faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
      faiss::IndexHNSWFlat index(d, hnsw.m, faiss_metric);
      index.hnsw.efConstruction = hnsw.ef_construction;
      index.hnsw.efSearch = hnsw.ef_search;
      index.add(train.size(), train_x.data());
      OmpThreadScope threads(precomputed.parameters.n_threads);
      index.search(validation.size(), val_x.data(), k, scores.data(), idx.data());
      if (precomputed.parameters.metric == DistanceMetric::Euclidean) {
        for (float& score : scores) score = -score;
      }
    }

    PrecomputedKNNFold fold_data;
    fold_data.fold = fold;
    fold_data.validation = validation;
    fold_data.scores = std::move(scores);
    fold_data.k = k;
    fold_data.neighbor_rows.assign(validation.size() * static_cast<std::size_t>(k), -1);
    for (std::size_t qi = 0; qi < validation.size(); ++qi) {
      for (int j = 0; j < k; ++j) {
        const auto id = idx[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)];
        if (id >= 0) {
          fold_data.neighbor_rows[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)] =
            train[static_cast<std::size_t>(id)];
        }
      }
    }
    precomputed.fold_data.push_back(std::move(fold_data));
  }
  return precomputed;
}

NeighborGraph normalize_graph_indices(const NeighborGraph& graph, int samples) {
  if (samples < 2) throw std::invalid_argument("NeighborGraph samples must be at least 2.");
  if (graph.neighbors <= 0) throw std::invalid_argument("NeighborGraph.neighbors must be positive.");
  const std::size_t expected = static_cast<std::size_t>(samples) * static_cast<std::size_t>(graph.neighbors);
  if (graph.indices.size() != expected || graph.distances.size() != expected) {
    throw std::invalid_argument("NeighborGraph indices/distances size must equal samples * neighbors.");
  }

  int min_index = std::numeric_limits<int>::max();
  int max_index = std::numeric_limits<int>::min();
  for (int value : graph.indices) {
    if (value < 0) continue;
    min_index = std::min(min_index, value);
    max_index = std::max(max_index, value);
  }
  const bool one_based = min_index >= 1 && max_index <= samples;

  NeighborGraph out;
  out.neighbors = graph.neighbors;
  out.indices.resize(graph.indices.size(), -1);
  out.distances.resize(graph.distances.size(), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < graph.indices.size(); ++i) {
    int id = graph.indices[i];
    if (one_based && id > 0) --id;
    if (id < 0 || id >= samples) {
      out.indices[i] = -1;
      out.distances[i] = std::numeric_limits<float>::infinity();
      continue;
    }
    out.indices[i] = id;
    const float d = graph.distances[i];
    out.distances[i] = std::isfinite(d) && d >= 0.0f ? d : std::numeric_limits<float>::infinity();
  }
  return out;
}

PrecomputedKNN precompute_knn_cv_graph(
  const NeighborGraph& input_graph,
  int samples,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
  if (options.k < 1) throw std::invalid_argument("KNNOptions::k must be positive.");
  if (labels.size() != static_cast<std::size_t>(samples)) {
    throw std::invalid_argument("labels size must match NeighborGraph samples.");
  }
  if (!constrain.empty() && constrain.size() != labels.size()) {
    throw std::invalid_argument("constrain size must be zero or match NeighborGraph samples.");
  }

  detail::Timer timer;
  const NeighborGraph graph = normalize_graph_indices(input_graph, samples);
  PrecomputedKNN precomputed;
  precomputed.parameters = resolve_core_knn_parameters(options);
  precomputed.parameters.index_type = KNNIndexType::FaissHNSWFlat;
  precomputed.folds = detail::make_folds(labels, constrain, options.cv);
  const std::vector<int> fold_ids = detail::sorted_unique_folds(precomputed.folds);
  const int k = std::max(1, std::min(options.k, graph.neighbors));

  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(precomputed.folds, fold, true);
    if (validation.empty()) continue;
    PrecomputedKNNFold fold_data;
    fold_data.fold = fold;
    fold_data.validation = validation;
    fold_data.k = k;
    fold_data.neighbor_rows.assign(validation.size() * static_cast<std::size_t>(k), -1);
    fold_data.scores.assign(validation.size() * static_cast<std::size_t>(k), -std::numeric_limits<float>::infinity());

    for (std::size_t qi = 0; qi < validation.size(); ++qi) {
      const int row = validation[qi];
      int out_col = 0;
      const std::size_t in_base = static_cast<std::size_t>(row) * static_cast<std::size_t>(graph.neighbors);
      const std::size_t out_base = qi * static_cast<std::size_t>(k);
      for (int j = 0; j < graph.neighbors && out_col < k; ++j) {
        const std::size_t in_offset = in_base + static_cast<std::size_t>(j);
        const int nb = graph.indices[in_offset];
        if (nb < 0 || nb >= samples || nb == row) continue;
        if (precomputed.folds[static_cast<std::size_t>(nb)] == fold) continue;
        const float distance = graph.distances[in_offset];
        if (!std::isfinite(distance)) continue;
        const std::size_t out_offset = out_base + static_cast<std::size_t>(out_col);
        fold_data.neighbor_rows[out_offset] = nb;
        const float d = std::max(0.0f, distance);
        if (precomputed.parameters.metric == DistanceMetric::Euclidean) {
          fold_data.scores[out_offset] = -(d * d);
        } else {
          fold_data.scores[out_offset] = 1.0f - d;
        }
        ++out_col;
      }
    }
    precomputed.fold_data.push_back(std::move(fold_data));
  }

  precomputed.parameters.k = k;
  (void)timer;
  return precomputed;
}

#if defined(KODAMA_ENABLE_CUDA)
PrecomputedKNN precompute_knn_cv_cuda(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
  if (options.k < 1) throw std::invalid_argument("KNNOptions::k must be positive.");

  PrecomputedKNN precomputed;
  precomputed.parameters = resolve_core_knn_parameters(options);
  precomputed.parameters.backend = Backend::CUDA;
  precomputed.parameters.index_type = KNNIndexType::FaissIVFFlat;
  precomputed.folds = detail::make_folds(labels, constrain, options.cv);
  const std::vector<int> fold_ids = detail::sorted_unique_folds(precomputed.folds);
  const std::vector<float> all_x = make_faiss_matrix(x, precomputed.parameters.metric);
  std::vector<float> train_x;
  std::vector<float> val_x;
  faiss::gpu::StandardGpuResources resources;

  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(precomputed.folds, fold, true);
    const std::vector<int> train = detail::indices_where_fold(precomputed.folds, fold, false);
    if (train.empty()) throw std::runtime_error("KNN fold has no training samples.");

    const int k = std::min(precomputed.parameters.k, static_cast<int>(train.size()));
    gather_faiss_rows(all_x, x.cols, train, train_x);
    gather_faiss_rows(all_x, x.cols, validation, val_x);
    const int d = static_cast<int>(x.cols);
    int nlist = precomputed.parameters.ivf_nlist > 0 ? precomputed.parameters.ivf_nlist :
      std::max(1, static_cast<int>(std::sqrt(static_cast<double>(train.size()))));
    nlist = std::min(nlist, static_cast<int>(train.size()));
    int nprobe = precomputed.parameters.ivf_nprobe > 0 ? precomputed.parameters.ivf_nprobe :
      std::max(1, static_cast<int>(std::sqrt(static_cast<double>(nlist))) + 1);
    nprobe = std::min(nprobe, nlist);

    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = precomputed.parameters.gpu_device;
    const faiss::MetricType faiss_metric = precomputed.parameters.metric == DistanceMetric::Euclidean ?
      faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
    faiss::gpu::GpuIndexIVFFlat index(&resources, d, nlist, faiss_metric, config);
    index.train(train.size(), train_x.data());
    index.add(train.size(), train_x.data());

    std::vector<faiss::idx_t> idx(validation.size() * static_cast<std::size_t>(k), -1);
    std::vector<float> scores(validation.size() * static_cast<std::size_t>(k), -std::numeric_limits<float>::infinity());
    faiss::SearchParametersIVF search_params;
    search_params.nprobe = static_cast<std::size_t>(nprobe);
    index.search(validation.size(), val_x.data(), k, scores.data(), idx.data(), &search_params);
    if (precomputed.parameters.metric == DistanceMetric::Euclidean) {
      for (float& score : scores) score = -score;
    }

    PrecomputedKNNFold fold_data;
    fold_data.fold = fold;
    fold_data.validation = validation;
    fold_data.scores = std::move(scores);
    fold_data.k = k;
    fold_data.neighbor_rows.assign(validation.size() * static_cast<std::size_t>(k), -1);
    for (std::size_t qi = 0; qi < validation.size(); ++qi) {
      for (int j = 0; j < k; ++j) {
        const auto id = idx[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)];
        if (id >= 0) {
          fold_data.neighbor_rows[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)] =
            train[static_cast<std::size_t>(id)];
        }
      }
    }
    precomputed.fold_data.push_back(std::move(fold_data));
  }
  return precomputed;
}
#endif

CVPrediction predict_precomputed_knn(
  const PrecomputedKNN& precomputed,
  const std::vector<int>& labels,
  KNNPredictionScratch& scratch
) {
  CVPrediction out;
  out.predicted.assign(labels.size(), 0);
  if (!scratch.label_map_initialized) {
    initialize_knn_label_map(scratch, labels);
  }
  scratch.label_codes.resize(labels.size());
  for (;;) {
    bool rebuilt = false;
    for (std::size_t i = 0; i < labels.size(); ++i) {
      const int label = labels[i];
      auto it = scratch.label_to_code.find(label);
      if (it == scratch.label_to_code.end()) {
        initialize_knn_label_map(scratch, labels);
        rebuilt = true;
        break;
      }
      scratch.label_codes[i] = it->second;
    }
    if (!rebuilt) break;
  }

  if (scratch.vote_counts.size() < scratch.label_values.size()) {
    scratch.vote_counts.resize(scratch.label_values.size(), 0);
  }
  if (scratch.fallback_counts.size() < scratch.label_values.size()) {
    scratch.fallback_counts.resize(scratch.label_values.size(), 0);
  }
  if (scratch.vote_scores.size() < scratch.label_values.size()) {
    scratch.vote_scores.resize(scratch.label_values.size(), 0.0);
  }
  std::fill(scratch.fallback_counts.begin(), scratch.fallback_counts.end(), 0);
  for (int code : scratch.label_codes) {
    scratch.fallback_counts[static_cast<std::size_t>(code)]++;
  }
  int fallback_label = scratch.label_values.empty() ? 0 : scratch.label_values.front();
  int fallback_count = -1;
  for (std::size_t code = 0; code < scratch.label_values.size(); ++code) {
    const int label = scratch.label_values[code];
    const int count = scratch.fallback_counts[code];
    if (count > fallback_count || (count == fallback_count && label < fallback_label)) {
      fallback_label = label;
      fallback_count = count;
    }
    scratch.fallback_counts[code] = 0;
  }
  scratch.touched.reserve(static_cast<std::size_t>(precomputed.parameters.k));
  for (const PrecomputedKNNFold& fold : precomputed.fold_data) {
    const std::size_t k = static_cast<std::size_t>(fold.k);
    for (std::size_t qi = 0; qi < fold.validation.size(); ++qi) {
      scratch.touched.clear();
      const std::size_t row_offset = qi * k;
      const int* neighbor_rows = fold.neighbor_rows.data() + row_offset;
      const float* neighbor_scores = fold.scores.data() + row_offset;
      for (std::size_t j = 0; j < k; ++j) {
        const int row = neighbor_rows[j];
        if (row < 0) continue;
        const int code = scratch.label_codes[static_cast<std::size_t>(row)];
        if (scratch.vote_counts[static_cast<std::size_t>(code)] == 0) {
          scratch.touched.push_back(code);
        }
        scratch.vote_counts[static_cast<std::size_t>(code)]++;
        scratch.vote_scores[static_cast<std::size_t>(code)] += static_cast<double>(neighbor_scores[j]);
      }

      int best_label = 0;
      int best_count = -1;
      double best_score = -std::numeric_limits<double>::infinity();
      if (scratch.touched.empty()) {
        out.predicted[static_cast<std::size_t>(fold.validation[qi])] = fallback_label;
        continue;
      }
      for (int code : scratch.touched) {
        const std::size_t idx = static_cast<std::size_t>(code);
        const int label = scratch.label_values[idx];
        if (scratch.vote_counts[idx] > best_count ||
            (scratch.vote_counts[idx] == best_count && scratch.vote_scores[idx] > best_score) ||
            (scratch.vote_counts[idx] == best_count && scratch.vote_scores[idx] == best_score && label < best_label)) {
          best_label = label;
          best_count = scratch.vote_counts[idx];
          best_score = scratch.vote_scores[idx];
        }
        scratch.vote_counts[idx] = 0;
        scratch.vote_scores[idx] = 0.0;
      }
      out.predicted[static_cast<std::size_t>(fold.validation[qi])] = best_label;
    }
  }
  return out;
}

double core_objective_score(
  const std::vector<int>& labels,
  const std::vector<int>& predictions,
  double accuracy,
  const CoreOptions& options
) {
  if (!options.guarded_diversity &&
      !options.auto_class_coarsening) {
    return accuracy;
  }

  if (labels.empty()) return accuracy;
  std::unordered_map<int, int> counts;
  counts.reserve(labels.size());
  for (int label : labels) counts[label]++;
  const int n_classes = static_cast<int>(counts.size());
  if (n_classes < 1) return accuracy;

  const double n = static_cast<double>(labels.size());
  double score = accuracy;
  if (options.guarded_diversity) {
    double same_label_probability = 0.0;
    for (const auto& kv : counts) {
      const double p = static_cast<double>(kv.second) / n;
      same_label_probability += p * p;
    }
    const double different_label_probability = std::max(0.0, 1.0 - same_label_probability);
    score *= std::sqrt(different_label_probability);
  }

  if (options.auto_class_coarsening && n_classes > 1) {
    double entropy = 0.0;
    for (const auto& kv : counts) {
      const double p = static_cast<double>(kv.second) / n;
      if (p > 0.0) entropy -= p * std::log(p);
    }
    const double max_sample_entropy = n > 1.0 ? std::log(n) : 0.0;
    const double label_code_cost = max_sample_entropy > 0.0
      ? std::clamp(entropy / max_sample_entropy, 0.0, 1.0)
      : 0.0;
    const double keff = std::exp(entropy);
    const double fragmentation = std::max(0.0, std::log(static_cast<double>(n_classes) / std::max(1.0, keff)));
    const double fragmentation_cost = fragmentation / (1.0 + fragmentation);
    const double parsimony_cost = std::max(label_code_cost, fragmentation_cost);
    score -= (1.0 - accuracy) * parsimony_cost;
  }

  return score;
}

bool propose_auto_class_coarsening(
  std::vector<int>& labels,
  const std::vector<int>& previous_predictions,
  const std::vector<int>& fixed_flags,
  const CoreOptions& options,
  std::mt19937_64& rng
) {
  if (!options.auto_class_coarsening) return false;
  if (labels.size() != previous_predictions.size()) return false;
  if (!fixed_flags.empty() && fixed_flags.size() != labels.size()) {
    throw std::invalid_argument("fixed size must be zero or match number of rows.");
  }

  std::map<int, int> class_sizes;
  std::map<int, int> movable_sizes;
  std::map<int, std::map<int, int>> transitions;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const int label = labels[i];
    class_sizes[label]++;
    transitions[label][previous_predictions[i]]++;
    if (fixed_flags.empty() || fixed_flags[i] != 1) movable_sizes[label]++;
  }

  const int n_classes = static_cast<int>(class_sizes.size());
  if (n_classes <= 2) return false;

  double class_entropy = 0.0;
  const double n = static_cast<double>(labels.size());
  for (const auto& kv : class_sizes) {
    const double p = static_cast<double>(kv.second) / n;
    if (p > 0.0) class_entropy -= p * std::log(p);
  }
  const double keff = std::exp(class_entropy);
  const double fragmentation = std::max(0.0, std::log(static_cast<double>(n_classes) / std::max(1.0, keff)));

  double weighted_transition_entropy = 0.0;
  struct SourceCandidate {
    int source = 0;
    int destination = 0;
    double score = 0.0;
  };
  std::vector<SourceCandidate> candidates;

  for (const auto& kv : class_sizes) {
    const int source = kv.first;
    const int source_size = kv.second;
    const int movable = movable_sizes[source];
    if (movable <= 0) continue;

    int destination = source;
    int destination_size = source_size;
    int best_count = 0;
    double transition_entropy = 0.0;
    const auto& row = transitions[source];
    for (const auto& dst : row) {
      const double p = static_cast<double>(dst.second) / static_cast<double>(source_size);
      if (p > 0.0) transition_entropy -= p * std::log(p);
      const auto class_it = class_sizes.find(dst.first);
      if (dst.first != source && class_it != class_sizes.end() &&
          (dst.second > best_count ||
           (dst.second == best_count && class_it->second > destination_size) ||
           (dst.second == best_count && class_it->second == destination_size && dst.first < destination))) {
        destination = dst.first;
        destination_size = class_it->second;
        best_count = dst.second;
      }
    }

    weighted_transition_entropy += (static_cast<double>(source_size) / n) * transition_entropy;
    if (destination == source || best_count == 0) continue;

    const double stay = static_cast<double>(row.count(source) ? row.at(source) : 0) / static_cast<double>(source_size);
    const double smallness = 1.0 / std::sqrt(static_cast<double>(movable));
    const double instability = 1.0 - stay;
    candidates.push_back(SourceCandidate{source, destination, smallness * instability});
  }
  if (candidates.empty()) return false;

  const double temperature = fragmentation / (fragmentation + weighted_transition_entropy + 1.0);
  const double redundant = std::max(0.0, static_cast<double>(n_classes) - keff);
  int merge_budget = static_cast<int>(std::ceil(temperature * redundant));
  if (merge_budget < 1 && fragmentation > 0.0) merge_budget = 1;
  merge_budget = std::min(merge_budget, n_classes - 2);
  if (merge_budget <= 0) return false;

  std::shuffle(candidates.begin(), candidates.end(), rng);
  std::stable_sort(candidates.begin(), candidates.end(), [](const SourceCandidate& a, const SourceCandidate& b) {
    return a.score > b.score;
  });

  std::unordered_map<int, int> remap;
  remap.reserve(class_sizes.size());
  for (const auto& kv : class_sizes) remap[kv.first] = kv.first;

  int merged = 0;
  for (const SourceCandidate& candidate : candidates) {
    if (merged >= merge_budget) break;
    if (remap[candidate.source] != candidate.source) continue;
    int destination = candidate.destination;
    while (remap[destination] != destination) destination = remap[destination];
    if (destination == candidate.source) continue;
    remap[candidate.source] = destination;
    merged++;
  }
  if (merged == 0) return false;

  bool changed = false;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (!fixed_flags.empty() && fixed_flags[i] == 1) continue;
    int destination = labels[i];
    while (remap[destination] != destination) destination = remap[destination];
    if (destination != labels[i]) {
      labels[i] = destination;
      changed = true;
    }
  }
  return changed;
}

bool propose_many_to_one_absorption(
  std::vector<int>& labels,
  const std::vector<int>& previous_predictions,
  const std::vector<int>& fixed_flags,
  std::mt19937_64& rng
) {
  if (labels.size() != previous_predictions.size()) return false;
  if (!fixed_flags.empty() && fixed_flags.size() != labels.size()) {
    throw std::invalid_argument("fixed size must be zero or match number of rows.");
  }

  std::map<int, int> class_sizes;
  std::map<int, int> movable_sizes;
  std::map<int, std::map<int, int>> transitions;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const int source = labels[i];
    class_sizes[source]++;
    transitions[source][previous_predictions[i]]++;
    if (fixed_flags.empty() || fixed_flags[i] != 1) movable_sizes[source]++;
  }

  const int n_classes = static_cast<int>(class_sizes.size());
  if (n_classes <= 2) return false;
  const double n = static_cast<double>(labels.size());

  struct SourceMove {
    int source = 0;
    double surplus = 0.0;
  };
  struct TargetMove {
    int target = 0;
    double weight = 0.0;
    std::vector<SourceMove> sources;
  };

  std::map<int, TargetMove> by_target;
  for (const auto& kv : class_sizes) {
    const int source = kv.first;
    const int source_size = kv.second;
    if (movable_sizes[source] <= 0) continue;

    const auto& row = transitions[source];
    const int stay = row.count(source) ? row.at(source) : 0;
    int best_target = source;
    int best_count = 0;
    int best_target_size = source_size;
    double best_surplus = 0.0;
    for (const auto& dst : row) {
      if (dst.first == source) continue;
      const auto dst_size_it = class_sizes.find(dst.first);
      if (dst_size_it == class_sizes.end()) continue;
      const double expected = static_cast<double>(source_size) *
        static_cast<double>(dst_size_it->second) / std::max(1.0, n);
      const double surplus = static_cast<double>(dst.second) - expected;
      if (surplus > best_surplus ||
          (surplus == best_surplus && dst.second > best_count) ||
          (surplus == best_surplus && dst.second == best_count && dst_size_it->second > best_target_size) ||
          (surplus == best_surplus && dst.second == best_count && dst_size_it->second == best_target_size && dst.first < best_target)) {
        best_target = dst.first;
        best_target_size = dst_size_it->second;
        best_count = dst.second;
        best_surplus = surplus;
      }
    }
    if (best_target == source || best_surplus <= 0.0 || best_count <= stay) continue;

    TargetMove& move = by_target[best_target];
    move.target = best_target;
    move.weight += best_surplus;
    move.sources.push_back(SourceMove{source, best_surplus});
  }
  if (by_target.empty()) return false;

  std::vector<TargetMove> multi_source_moves;
  std::vector<TargetMove> fallback_moves;
  for (auto& kv : by_target) {
    std::sort(kv.second.sources.begin(), kv.second.sources.end(), [](const SourceMove& a, const SourceMove& b) {
      if (a.surplus != b.surplus) return a.surplus > b.surplus;
      return a.source < b.source;
    });
    if (kv.second.sources.size() > 1) {
      multi_source_moves.push_back(std::move(kv.second));
    } else {
      fallback_moves.push_back(std::move(kv.second));
    }
  }
  std::vector<TargetMove>& moves = multi_source_moves.empty() ? fallback_moves : multi_source_moves;
  if (moves.empty()) return false;

  std::vector<double> weights;
  weights.reserve(moves.size());
  for (const TargetMove& move : moves) weights.push_back(std::max(move.weight, std::numeric_limits<double>::min()));
  std::discrete_distribution<int> target_dist(weights.begin(), weights.end());
  const TargetMove& selected = moves[static_cast<std::size_t>(target_dist(rng))];

  std::unordered_map<int, int> remap;
  remap.reserve(selected.sources.size());
  for (const SourceMove& source : selected.sources) {
    if (source.source != selected.target) remap[source.source] = selected.target;
  }
  if (remap.empty()) return false;

  std::vector<int> proposed = labels;
  bool changed = false;
  for (std::size_t i = 0; i < proposed.size(); ++i) {
    if (!fixed_flags.empty() && fixed_flags[i] == 1) continue;
    const auto it = remap.find(proposed[i]);
    if (it == remap.end()) continue;
    proposed[i] = it->second;
    changed = true;
  }
  if (!changed) return false;
  std::unordered_map<int, int> proposed_counts;
  proposed_counts.reserve(proposed.size());
  for (int label : proposed) proposed_counts[label]++;
  const int after_classes = static_cast<int>(proposed_counts.size());
  if (after_classes <= 1 || after_classes >= n_classes) return false;

  labels.swap(proposed);
  return true;
}

int sample_proposal_group_count(
  int group_count,
  int cycle,
  int cycles,
  bool adaptive,
  std::mt19937_64& rng
) {
  if (group_count <= 1) return std::max(0, group_count);
  if (!adaptive || cycles <= 1) {
    std::uniform_int_distribution<int> n_dist(1, group_count);
    return n_dist(rng);
  }

  const double progress = std::clamp(
    static_cast<double>(cycle + 1) / static_cast<double>(cycles + 1),
    0.0,
    1.0
  );
  const double smooth = progress * progress * (3.0 - 2.0 * progress);
  const double temperature = std::clamp(1.0 - smooth, 0.0, 1.0);
  const int max_count = std::max(
    1,
    1 + static_cast<int>(std::floor(static_cast<double>(group_count - 1) * temperature))
  );
  std::uniform_int_distribution<int> n_dist(1, max_count);
  return n_dist(rng);
}

void apply_group_relabel(
  std::vector<int>& labels,
  const std::vector<int>& eligible,
  int replacement
) {
  for (int idx : eligible) {
    int& label = labels[static_cast<std::size_t>(idx)];
    if (label == replacement) continue;
    label = replacement;
  }
}

template <class Predictor>
CoreResult maximize_core(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options,
  Predictor&& predictor
) {
  detail::validate_inputs(x, initial_clbest, constrain);
  if (options.cycles < 0) throw std::invalid_argument("Core cycles must be non-negative.");

  detail::Timer timer;
  CoreResult result;
  result.clbest = initial_clbest;
  result.clbest_dirty = initial_clbest;
  result.vect_acc.assign(static_cast<std::size_t>(options.cycles), -1.0);
  result.vect_score.assign(static_cast<std::size_t>(options.cycles), -1.0);

  const std::vector<int> group_id = normalized_constrain(constrain, x.rows);
  const std::vector<int> fixed_flags = normalized_fixed(fixed, x.rows);

  std::vector<int> groups;
  std::vector<const std::vector<int>*> group_member_refs;
  std::vector<std::vector<int>> group_members_storage;
  std::map<int, std::vector<int>> group_members;
  if (constrain.empty()) {
    group_members_storage.resize(x.rows);
    groups.reserve(x.rows);
    group_member_refs.reserve(x.rows);
    for (std::size_t i = 0; i < x.rows; ++i) {
      group_members_storage[i].push_back(static_cast<int>(i));
      groups.push_back(static_cast<int>(i));
      group_member_refs.push_back(&group_members_storage[i]);
    }
  } else {
    for (std::size_t i = 0; i < group_id.size(); ++i) {
      group_members[group_id[i]].push_back(static_cast<int>(i));
    }
    groups.reserve(group_members.size());
    group_member_refs.reserve(group_members.size());
    for (const auto& kv : group_members) {
      groups.push_back(static_cast<int>(groups.size()));
      group_member_refs.push_back(&kv.second);
    }
  }
  std::mt19937_64 rng(options.seed);
  CVPrediction best_cv = predictor(result.clbest);
  result.cvpredbest = best_cv.predicted;
  result.accbest = options.shake ? 0.0 : detail::accuracy(result.clbest, result.cvpredbest);
  result.scorebest = options.shake
    ? -std::numeric_limits<double>::infinity()
    : core_objective_score(result.clbest, result.cvpredbest, result.accbest, options);
  result.runtime_seconds += best_cv.runtime_seconds;
  result.peak_memory_mb = std::max(result.peak_memory_mb, best_cv.peak_memory_mb);

  std::vector<int> current_cl = result.clbest;
  std::vector<int> current_cvpred = result.cvpredbest;
  double current_acc = result.accbest;
  double current_score = result.scorebest;

  std::vector<int> sampled_groups;
  std::vector<int> eligible;
  std::vector<int> candidate_labels;
  std::vector<int> candidate_counts;
  sampled_groups.reserve(groups.size());
  eligible.reserve(x.rows);
  candidate_labels.reserve(x.rows);
  candidate_counts.reserve(x.rows);

  for (int cycle = 0; cycle < options.cycles && !result.success; ++cycle) {
    std::vector<int> cl = options.evolutionary_search ? current_cl : result.clbest;
    std::vector<int> cl_dirty = cl;
    const std::vector<int>& proposal_predictions = options.evolutionary_search ? current_cvpred : result.cvpredbest;

    if (!groups.empty()) {
      const int n_to_sample = sample_proposal_group_count(
        static_cast<int>(groups.size()),
        cycle,
        options.cycles,
        options.adaptive_proposal_size,
        rng
      );
      sampled_groups = groups;
      std::shuffle(sampled_groups.begin(), sampled_groups.end(), rng);
      sampled_groups.resize(static_cast<std::size_t>(n_to_sample));

      for (int group : sampled_groups) {
        const std::vector<int>& members = *group_member_refs[static_cast<std::size_t>(group)];
        eligible.clear();
        for (int idx : members) {
          if (fixed_flags[static_cast<std::size_t>(idx)] != 1) eligible.push_back(idx);
        }
        if (eligible.empty()) continue;

        candidate_labels.clear();
        candidate_counts.clear();
        for (int idx : eligible) {
          candidate_labels.push_back(proposal_predictions[static_cast<std::size_t>(idx)]);
        }
        std::sort(candidate_labels.begin(), candidate_labels.end());
        for (std::size_t i = 0; i < candidate_labels.size();) {
          std::size_t j = i + 1;
          while (j < candidate_labels.size() && candidate_labels[j] == candidate_labels[i]) ++j;
          candidate_counts.push_back(static_cast<int>(j - i));
          i = j;
        }
        candidate_labels.erase(std::unique(candidate_labels.begin(), candidate_labels.end()), candidate_labels.end());
        if (candidate_labels.empty()) continue;

        std::discrete_distribution<int> label_dist(candidate_counts.begin(), candidate_counts.end());
        const int replacement = candidate_labels[static_cast<std::size_t>(label_dist(rng))];
        apply_group_relabel(cl, eligible, replacement);
      }
    }

    propose_auto_class_coarsening(cl, proposal_predictions, fixed_flags, options, rng);
    if (options.many_to_one_absorption) {
      propose_many_to_one_absorption(cl, proposal_predictions, fixed_flags, rng);
    }

    CVPrediction cv = predictor(cl);
    const double acc = detail::accuracy(cl, cv.predicted);
    const double score = core_objective_score(cl, cv.predicted, acc, options);
    result.runtime_seconds += cv.runtime_seconds;
    result.peak_memory_mb = std::max(result.peak_memory_mb, cv.peak_memory_mb);

    if (score > result.scorebest) {
      result.cvpredbest = cv.predicted;
      result.clbest = cl;
      result.clbest_dirty = cl_dirty;
      result.accbest = acc;
      result.scorebest = score;
    }

    if (options.evolutionary_search) {
      const double cooling = 1.0 - static_cast<double>(cycle + 1) /
        static_cast<double>(std::max(1, options.cycles));
      const double temperature = std::max(1.0e-9, 0.10 * std::max(0.0, 1.0 - current_acc) * cooling);
      bool accept_current = score >= current_score;
      if (!accept_current && temperature > 1.0e-9) {
        std::uniform_real_distribution<double> accept_dist(0.0, 1.0);
        accept_current = accept_dist(rng) < std::exp((score - current_score) / temperature);
      }
      if (accept_current) {
        current_cl = cl;
        current_cvpred = cv.predicted;
        current_acc = acc;
        current_score = score;
      }
    }

    result.vect_acc[static_cast<std::size_t>(cycle)] = result.accbest;
    result.vect_score[static_cast<std::size_t>(cycle)] = result.scorebest;
    result.cycles_completed = cycle + 1;
    if (acc == 1.0 && (!options.guarded_diversity || score >= result.scorebest)) result.success = true;
  }

  result.runtime_seconds += timer.seconds();
  result.peak_memory_mb = std::max(result.peak_memory_mb, detail::peak_memory_mb());
  return result;
}

}  // namespace

namespace {

PLSOptions to_plslda_cv_options(const CorePLSLDAOptions& options, Backend backend) {
  PLSOptions out;
  out.cv = options.cv;
  out.max_components = options.max_components;
  out.fixed_components = options.fixed_components;
  out.center = options.center;
  out.scale = options.scale;
  out.backend = backend;
  out.gpu_device = options.gpu_device;
  out.n_threads = options.n_threads;
  return out;
}

}  // namespace

CoreResult CorePLSLDA_CPU(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  CoreOptions pls_options = options;
  pls_options.classifier = CoreClassifier::PLS_LDA;
  pls_options.pls.backend = Backend::CPU;
  const PLSOptions cv_options = to_plslda_cv_options(pls_options.pls, Backend::CPU);
  return maximize_core(x, initial_clbest, constrain, fixed, pls_options, [&](const std::vector<int>& labels) {
    PLSCVResult cv = PLSLDACV_CPU(x, labels, constrain, cv_options);
    return CVPrediction{cv.predicted_labels, cv.runtime_seconds, cv.peak_memory_mb};
  });
}

CoreResult CoreKNN_CPU(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  CoreOptions knn_options = options;
  knn_options.classifier = CoreClassifier::KNN;
  knn_options.knn.backend = Backend::CPU;
  detail::validate_inputs(x, initial_clbest, constrain);
  const PrecomputedKNN precomputed = precompute_knn_cv_cpu(x, initial_clbest, constrain, knn_options.knn);
  KNNPredictionScratch scratch;
  initialize_knn_label_map(scratch, initial_clbest);
  return maximize_core(x, initial_clbest, constrain, fixed, knn_options, [&](const std::vector<int>& labels) {
    return predict_precomputed_knn(precomputed, labels, scratch);
  });
}

CoreResult CoreKNNGraph_CPU(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  if (samples < 2) throw std::invalid_argument("CoreKNNGraph requires at least two samples.");
  if (initial_clbest.size() != static_cast<std::size_t>(samples)) {
    throw std::invalid_argument("initial labels size must match NeighborGraph samples.");
  }
  if (!constrain.empty() && constrain.size() != initial_clbest.size()) {
    throw std::invalid_argument("constrain size must be zero or match NeighborGraph samples.");
  }
  if (!fixed.empty() && fixed.size() != initial_clbest.size()) {
    throw std::invalid_argument("fixed size must be zero or match NeighborGraph samples.");
  }

  CoreOptions knn_options = options;
  knn_options.classifier = CoreClassifier::KNN;
  knn_options.knn.backend = Backend::CPU;
  const PrecomputedKNN precomputed = precompute_knn_cv_graph(graph, samples, initial_clbest, constrain, knn_options.knn);
  KNNPredictionScratch scratch;
  initialize_knn_label_map(scratch, initial_clbest);
  std::vector<float> dummy(static_cast<std::size_t>(samples), 0.0f);
  MatrixView dummy_view{dummy.data(), static_cast<std::size_t>(samples), static_cast<std::size_t>(1)};
  return maximize_core(dummy_view, initial_clbest, constrain, fixed, knn_options, [&](const std::vector<int>& labels) {
    return predict_precomputed_knn(precomputed, labels, scratch);
  });
}

CoreResult CorePLSLDA_CUDA(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  CoreOptions pls_options = options;
  pls_options.classifier = CoreClassifier::PLS_LDA;
  pls_options.pls.backend = Backend::CUDA;
  const PLSOptions cv_options = to_plslda_cv_options(pls_options.pls, Backend::CUDA);
  return maximize_core(x, initial_clbest, constrain, fixed, pls_options, [&](const std::vector<int>& labels) {
    PLSCVResult cv = PLSLDACV_CUDA(x, labels, constrain, cv_options);
    return CVPrediction{cv.predicted_labels, cv.runtime_seconds, cv.peak_memory_mb};
  });
#else
  (void)x;
  (void)initial_clbest;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("CorePLSLDA_CUDA requires a CUDA build.");
#endif
}

CoreResult CoreKNN_CUDA(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  CoreOptions knn_options = options;
  knn_options.classifier = CoreClassifier::KNN;
  knn_options.knn.backend = Backend::CUDA;
  detail::validate_inputs(x, initial_clbest, constrain);
  const PrecomputedKNN precomputed = precompute_knn_cv_cuda(x, initial_clbest, constrain, knn_options.knn);
  KNNPredictionScratch scratch;
  initialize_knn_label_map(scratch, initial_clbest);
  return maximize_core(x, initial_clbest, constrain, fixed, knn_options, [&](const std::vector<int>& labels) {
    return predict_precomputed_knn(precomputed, labels, scratch);
  });
#else
  (void)x;
  (void)initial_clbest;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("CoreKNN_CUDA requires a CUDA/FAISS GPU build.");
#endif
}

CoreResult CorePLSLDA(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  if (options.pls.backend == Backend::CUDA) return CorePLSLDA_CUDA(x, initial_clbest, constrain, fixed, options);
  return CorePLSLDA_CPU(x, initial_clbest, constrain, fixed, options);
}

CoreResult CoreKNN(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  if (options.knn.backend == Backend::CUDA) return CoreKNN_CUDA(x, initial_clbest, constrain, fixed, options);
  return CoreKNN_CPU(x, initial_clbest, constrain, fixed, options);
}

CoreResult core_cpp(
  MatrixView x,
  const std::vector<int>& initial_clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  if (options.classifier == CoreClassifier::KNN) return CoreKNN(x, initial_clbest, constrain, fixed, options);
  return CorePLSLDA(x, initial_clbest, constrain, fixed, options);
}

CoreResult Core(
  MatrixView x,
  const std::vector<int>& clbest,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const CoreOptions& options
) {
  return core_cpp(x, clbest, constrain, fixed, options);
}

}  // namespace kodama
