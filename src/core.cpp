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
          out[r * x.cols + j] = static_cast<float>(x(static_cast<std::size_t>(row), j));
        }
      }
    }
    return out;
  }

  for (std::size_t r = 0; r < rows.size(); ++r) {
    long double ss = 0.0;
    const int row = rows[r];
    for (std::size_t j = 0; j < x.cols; ++j) {
      const double v = x(static_cast<std::size_t>(row), j);
      ss += static_cast<long double>(v) * static_cast<long double>(v);
    }
    const double n = std::sqrt(static_cast<double>(ss));
    const double scale = metric == DistanceMetric::Cosine && n > 0.0 && std::isfinite(n) ? 1.0 / n : 1.0;
    for (std::size_t j = 0; j < x.cols; ++j) {
      out[r * x.cols + j] = static_cast<float>(x(static_cast<std::size_t>(row), j) * scale);
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
          out[r * x.cols + j] = static_cast<float>(x(r, j));
        }
      }
    }
    return out;
  }

  for (std::size_t r = 0; r < x.rows; ++r) {
    long double ss = 0.0;
    for (std::size_t j = 0; j < x.cols; ++j) {
      const double v = x(r, j);
      ss += static_cast<long double>(v) * static_cast<long double>(v);
    }
    const double n = std::sqrt(static_cast<double>(ss));
    const double scale = metric == DistanceMetric::Cosine && n > 0.0 && std::isfinite(n) ? 1.0 / n : 1.0;
    for (std::size_t j = 0; j < x.cols; ++j) {
      out[r * x.cols + j] = static_cast<float>(x(r, j) * scale);
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
  std::vector<double> vote_scores;
  std::vector<int> touched;
};

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
  scratch.label_to_code.clear();
  scratch.label_to_code.reserve(labels.size());
  scratch.label_values.clear();
  scratch.label_values.reserve(labels.size());
  scratch.label_codes.resize(labels.size());
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const int label = labels[i];
    auto it = scratch.label_to_code.find(label);
    if (it == scratch.label_to_code.end()) {
      const int code = static_cast<int>(scratch.label_values.size());
      it = scratch.label_to_code.emplace(label, code).first;
      scratch.label_values.push_back(label);
    }
    scratch.label_codes[i] = it->second;
  }

  if (scratch.vote_counts.size() < scratch.label_values.size()) {
    scratch.vote_counts.resize(scratch.label_values.size(), 0);
  }
  if (scratch.vote_scores.size() < scratch.label_values.size()) {
    scratch.vote_scores.resize(scratch.label_values.size(), 0.0);
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
  double accuracy,
  const CoreOptions& options
) {
  if (!options.auto_class_coarsening &&
      (options.target_classes <= 0 || options.class_count_penalty == 0.0) &&
      options.imbalance_penalty == 0.0) {
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
  if (options.target_classes > 0 && options.class_count_penalty != 0.0) {
    const double observed = static_cast<double>(n_classes);
    const double target = static_cast<double>(options.target_classes);
    score -= options.class_count_penalty * std::abs(std::log(observed / target));
  }

  if (options.imbalance_penalty != 0.0) {
    double entropy = 0.0;
    for (const auto& kv : counts) {
      const double p = static_cast<double>(kv.second) / n;
      if (p > 0.0) entropy -= p * std::log(p);
    }
    const double max_entropy = n_classes > 1 ? std::log(static_cast<double>(n_classes)) : 0.0;
    const double entropy_loss = max_entropy > 0.0 ? 1.0 - entropy / max_entropy : 1.0;
    score -= options.imbalance_penalty * entropy_loss;
  }

  if (options.auto_class_coarsening && n_classes > 1) {
    double entropy = 0.0;
    for (const auto& kv : counts) {
      const double p = static_cast<double>(kv.second) / n;
      if (p > 0.0) entropy -= p * std::log(p);
    }
    const double keff = std::exp(entropy);
    const double fragmentation = std::max(0.0, std::log(static_cast<double>(n_classes) / std::max(1.0, keff)));
    score -= (1.0 - accuracy) * fragmentation / (1.0 + fragmentation);
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
    : core_objective_score(result.clbest, result.accbest, options);
  result.runtime_seconds += best_cv.runtime_seconds;
  result.peak_memory_mb = std::max(result.peak_memory_mb, best_cv.peak_memory_mb);

  std::vector<int> sampled_groups;
  std::vector<int> eligible;
  std::vector<int> candidate_labels;
  std::vector<int> candidate_counts;
  sampled_groups.reserve(groups.size());
  eligible.reserve(x.rows);
  candidate_labels.reserve(x.rows);
  candidate_counts.reserve(x.rows);

  for (int cycle = 0; cycle < options.cycles && !result.success; ++cycle) {
    std::vector<int> cl = result.clbest;
    std::vector<int> cl_dirty = cl;

    if (!groups.empty()) {
      std::uniform_int_distribution<int> n_dist(1, static_cast<int>(groups.size()));
      const int n_to_sample = n_dist(rng);
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
        for (int idx : eligible) candidate_labels.push_back(result.cvpredbest[static_cast<std::size_t>(idx)]);
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
        for (int idx : eligible) cl[static_cast<std::size_t>(idx)] = replacement;
      }
    }

    propose_auto_class_coarsening(cl, result.cvpredbest, fixed_flags, options, rng);

    CVPrediction cv = predictor(cl);
    const double acc = detail::accuracy(cl, cv.predicted);
    const double score = core_objective_score(cl, acc, options);
    result.runtime_seconds += cv.runtime_seconds;
    result.peak_memory_mb = std::max(result.peak_memory_mb, cv.peak_memory_mb);

    if (score > result.scorebest) {
      result.cvpredbest = std::move(cv.predicted);
      result.clbest = std::move(cl);
      result.clbest_dirty = std::move(cl_dirty);
      result.accbest = acc;
      result.scorebest = score;
    }

    result.vect_acc[static_cast<std::size_t>(cycle)] = result.accbest;
    result.vect_score[static_cast<std::size_t>(cycle)] = result.scorebest;
    result.cycles_completed = cycle + 1;
    if (acc == 1.0) result.success = true;
  }

  result.runtime_seconds += timer.seconds();
  result.peak_memory_mb = std::max(result.peak_memory_mb, detail::peak_memory_mb());
  return result;
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
  return maximize_core(x, initial_clbest, constrain, fixed, pls_options, [&](const std::vector<int>& labels) {
    PLSCVResult cv = PLSLDACV_CPU(x, labels, constrain, pls_options.pls);
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
  return maximize_core(x, initial_clbest, constrain, fixed, knn_options, [&](const std::vector<int>& labels) {
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
  return maximize_core(x, initial_clbest, constrain, fixed, pls_options, [&](const std::vector<int>& labels) {
    PLSCVResult cv = PLSLDACV_CUDA(x, labels, constrain, pls_options.pls);
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
