#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/MetricType.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(KODAMA_ENABLE_CUDA)
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#endif

namespace kodama {

namespace {

struct Neighbor {
  double score;
  int label;
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

bool better_neighbor(const Neighbor& a, const Neighbor& b) {
  if (a.score != b.score) return a.score > b.score;
  return a.label < b.label;
}

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

HNSWParameters tune_hnsw_parameters(int n, int p, int k, DistanceMetric metric, const KNNOptions& options) {
  HNSWParameters out;
  out.tune_k = options.hnsw_tune_k > 0 ? options.hnsw_tune_k : 50;
  out.target_recall = hnsw_target_recall(options.hnsw_target_recall);

  const int tune_k = std::max(1, out.tune_k);
  const bool high_dim = p >= 256;
  const bool large_n = n >= 50000;
  const bool very_large_high_dim = large_n && high_dim;
  const bool small_k = tune_k <= 15;
  const bool large_k = tune_k >= 100;
  const bool non_euclidean = metric == DistanceMetric::Cosine || metric == DistanceMetric::InnerProduct;

  if (very_large_high_dim && large_k) {
    out.m = 48;
    out.ef_construction = 240;
    out.ef_search = std::max(220, 3 * tune_k);
  } else if (non_euclidean && small_k) {
    out.m = 32;
    out.ef_construction = 160;
    out.ef_search = std::max(120, 4 * tune_k);
  } else if (non_euclidean && (large_k || high_dim)) {
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

std::vector<float> make_faiss_matrix(
  MatrixView x,
  const std::vector<int>& rows,
  DistanceMetric metric
) {
  std::vector<float> out(rows.size() * x.cols, 0.0f);
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

int majority_vote(const std::vector<Neighbor>& neighbors) {
  std::map<int, std::pair<int, double>> votes;
  for (const auto& nb : neighbors) {
    auto& v = votes[nb.label];
    v.first += 1;
    v.second += nb.score;
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

std::vector<int> knn_predict_faiss_ivf_flat_cpu(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& train,
  const std::vector<int>& validation,
  int k,
  DistanceMetric metric,
  int requested_nlist,
  int requested_nprobe,
  int n_threads
) {
  if (train.empty()) throw std::runtime_error("KNN fold has no training samples.");
  if (k < 1) throw std::invalid_argument("k must be positive.");
  if (metric == DistanceMetric::Euclidean) {
    throw std::invalid_argument("FAISS IVF CPU KNN currently supports cosine/inner product; use FAISS HNSW for euclidean.");
  }
  k = std::min(k, static_cast<int>(train.size()));
  const std::vector<float> train_x = make_faiss_matrix(x, train, metric);
  const std::vector<float> val_x = make_faiss_matrix(x, validation, metric);
  const int d = static_cast<int>(x.cols);
  int nlist = requested_nlist > 0 ? requested_nlist :
    std::max(1, static_cast<int>(std::sqrt(static_cast<double>(train.size() / 4 + 1))));
  if (requested_nlist <= 0 && train.size() < 1000) {
    nlist = std::max(1, static_cast<int>(train.size() / 40));
  }
  nlist = std::min(nlist, static_cast<int>(train.size()));
  faiss::IndexFlatIP quantizer(d);
  faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_INNER_PRODUCT);
  index.nprobe = requested_nprobe > 0 ? requested_nprobe :
    std::max<std::size_t>(1, std::min<std::size_t>(static_cast<std::size_t>(nlist), static_cast<std::size_t>(std::sqrt(static_cast<double>(nlist))) + 1));
  index.nprobe = std::min<std::size_t>(index.nprobe, static_cast<std::size_t>(nlist));
  index.train(train.size(), train_x.data());
  index.add(train.size(), train_x.data());

  std::vector<faiss::idx_t> idx(validation.size() * static_cast<std::size_t>(k), -1);
  std::vector<float> scores(validation.size() * static_cast<std::size_t>(k), -std::numeric_limits<float>::infinity());
  OmpThreadScope threads(n_threads);
  index.search(validation.size(), val_x.data(), k, scores.data(), idx.data());

  std::vector<int> pred(validation.size(), 0);
  for (std::size_t qi = 0; qi < validation.size(); ++qi) {
    std::vector<Neighbor> best;
    best.reserve(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      const auto id = idx[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)];
      if (id < 0) continue;
      best.push_back(Neighbor{
        static_cast<double>(scores[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)]),
        labels[static_cast<std::size_t>(train[static_cast<std::size_t>(id)])]
      });
    }
    pred[qi] = best.empty() ? labels[static_cast<std::size_t>(train.front())] : majority_vote(best);
  }
  return pred;
}

std::vector<int> knn_predict_faiss_hnsw_flat_cpu(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& train,
  const std::vector<int>& validation,
  int k,
  DistanceMetric metric,
  const KNNOptions& options,
  HNSWParameters* used_params
) {
  if (train.empty()) throw std::runtime_error("KNN fold has no training samples.");
  if (k < 1) throw std::invalid_argument("k must be positive.");
  k = std::min(k, static_cast<int>(train.size()));
  const std::vector<float> train_x = make_faiss_matrix(x, train, metric);
  const std::vector<float> val_x = make_faiss_matrix(x, validation, metric);
  const int d = static_cast<int>(x.cols);
  HNSWParameters params = tune_hnsw_parameters(static_cast<int>(train.size()), d, k, metric, options);
  if (used_params != nullptr) *used_params = params;

  const faiss::MetricType faiss_metric = metric == DistanceMetric::Euclidean ?
    faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
  faiss::IndexHNSWFlat index(d, params.m, faiss_metric);
  index.hnsw.efConstruction = params.ef_construction;
  index.hnsw.efSearch = params.ef_search;
  index.add(train.size(), train_x.data());

  std::vector<faiss::idx_t> idx(validation.size() * static_cast<std::size_t>(k), -1);
  std::vector<float> scores(validation.size() * static_cast<std::size_t>(k), -std::numeric_limits<float>::infinity());
  OmpThreadScope threads(options.n_threads);
  index.search(validation.size(), val_x.data(), k, scores.data(), idx.data());
  if (metric == DistanceMetric::Euclidean) {
    for (float& score : scores) score = -score;
  }

  std::vector<int> pred(validation.size(), 0);
  for (std::size_t qi = 0; qi < validation.size(); ++qi) {
    std::vector<Neighbor> best;
    best.reserve(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      const auto id = idx[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)];
      if (id < 0) continue;
      best.push_back(Neighbor{
        static_cast<double>(scores[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)]),
        labels[static_cast<std::size_t>(train[static_cast<std::size_t>(id)])]
      });
    }
    pred[qi] = best.empty() ? labels[static_cast<std::size_t>(train.front())] : majority_vote(best);
  }
  return pred;
}

#if defined(KODAMA_ENABLE_CUDA)
std::vector<int> knn_predict_faiss_ivf_flat_cuda(
  faiss::gpu::StandardGpuResources& resources,
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& train,
  const std::vector<int>& validation,
  int k,
  DistanceMetric metric,
  int gpu_device,
  int nlist,
  int nprobe
) {
  if (train.empty()) throw std::runtime_error("KNN fold has no training samples.");
  if (k < 1) throw std::invalid_argument("k must be positive.");
  k = std::min(k, static_cast<int>(train.size()));
  const std::vector<float> train_x = make_faiss_matrix(x, train, metric);
  const std::vector<float> val_x = make_faiss_matrix(x, validation, metric);
  const int d = static_cast<int>(x.cols);
  nlist = nlist > 0 ? nlist : std::max(1, static_cast<int>(std::sqrt(static_cast<double>(train.size()))));
  nlist = std::min(nlist, static_cast<int>(train.size()));
  nprobe = nprobe > 0 ? nprobe : std::max(1, static_cast<int>(std::sqrt(static_cast<double>(nlist))) + 1);
  nprobe = std::min(nprobe, nlist);

  faiss::gpu::GpuIndexIVFFlatConfig config;
  config.device = gpu_device;
  const faiss::MetricType faiss_metric = metric == DistanceMetric::Euclidean ?
    faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
  faiss::gpu::GpuIndexIVFFlat index(&resources, d, nlist, faiss_metric, config);
  index.train(train.size(), train_x.data());
  index.add(train.size(), train_x.data());

  std::vector<faiss::idx_t> idx(validation.size() * static_cast<std::size_t>(k), -1);
  std::vector<float> scores(validation.size() * static_cast<std::size_t>(k), -std::numeric_limits<float>::infinity());
  faiss::SearchParametersIVF search_params;
  search_params.nprobe = static_cast<std::size_t>(nprobe);
  index.search(validation.size(), val_x.data(), k, scores.data(), idx.data(), &search_params);
  if (metric == DistanceMetric::Euclidean) {
    for (float& score : scores) score = -score;
  }

  std::vector<int> pred(validation.size(), 0);
  for (std::size_t qi = 0; qi < validation.size(); ++qi) {
    std::vector<Neighbor> best;
    best.reserve(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      const auto id = idx[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)];
      if (id < 0) continue;
      best.push_back(Neighbor{
        static_cast<double>(scores[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)]),
        labels[static_cast<std::size_t>(train[static_cast<std::size_t>(id)])]
      });
    }
    pred[qi] = majority_vote(best);
  }
  return pred;
}
#endif

KNNParametersUsed resolve_knn_parameters(const KNNOptions& options) {
  KNNParametersUsed used;
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

  if (options.backend == Backend::CUDA || options.index_type == KNNIndexType::CuvsIVFFlat) {
#if defined(KODAMA_ENABLE_CUDA)
    used.backend = Backend::CUDA;
    used.index_type = options.index_type;
#else
    throw std::runtime_error("CUDA KNNCV backend was requested but this build was not compiled with KODAMA_ENABLE_CUDA.");
#endif
  } else {
    used.backend = Backend::CPU;
    used.index_type = options.index_type == KNNIndexType::FaissIVFFlat ?
      KNNIndexType::FaissIVFFlat : KNNIndexType::FaissHNSWFlat;
  }
  return used;
}

}  // namespace

KNNCVResult KNNCV(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
  if (options.backend == Backend::CUDA) return KNNCV_CUDA(x, labels, constrain, options);
  return KNNCV_CPU(x, labels, constrain, options);
}

KNNCVResult KNNCV_CPU(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
  detail::validate_inputs(x, labels, constrain);
  if (options.k < 1) throw std::invalid_argument("KNNOptions::k must be positive.");
  detail::Timer timer;
  KNNCVResult result;
  result.true_labels = labels;
  result.predicted_labels.assign(labels.size(), 0);
  result.fold_assignments = detail::make_folds(labels, constrain, options.cv);
  result.parameters = resolve_knn_parameters(options);

  const std::vector<int> fold_ids = detail::sorted_unique_folds(result.fold_assignments);
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(result.fold_assignments, fold, true);
    const std::vector<int> train = detail::indices_where_fold(result.fold_assignments, fold, false);
    std::vector<int> fold_pred;
    if (result.parameters.index_type == KNNIndexType::FaissIVFFlat) {
      fold_pred = knn_predict_faiss_ivf_flat_cpu(
        x,
        labels,
        train,
        validation,
        result.parameters.k,
        result.parameters.metric,
        result.parameters.ivf_nlist,
        result.parameters.ivf_nprobe,
        result.parameters.n_threads
      );
    } else {
      HNSWParameters hnsw_params;
      fold_pred = knn_predict_faiss_hnsw_flat_cpu(
        x,
        labels,
        train,
        validation,
        result.parameters.k,
        result.parameters.metric,
        options,
        &hnsw_params
      );
      result.parameters.hnsw_m = hnsw_params.m;
      result.parameters.hnsw_ef_construction = hnsw_params.ef_construction;
      result.parameters.hnsw_ef_search = hnsw_params.ef_search;
      result.parameters.hnsw_tune_k = hnsw_params.tune_k;
      result.parameters.hnsw_target_recall = hnsw_params.target_recall;
    }
    for (std::size_t i = 0; i < validation.size(); ++i) {
      result.predicted_labels[static_cast<std::size_t>(validation[i])] = fold_pred[i];
    }
    result.folds.push_back(FoldResult{
      fold,
      static_cast<int>(train.size()),
      static_cast<int>(validation.size()),
      detail::accuracy_on_indices(labels, result.predicted_labels, validation)
    });
  }

  result.global_accuracy = detail::accuracy(labels, result.predicted_labels);
  result.confusion = detail::make_confusion(labels, result.predicted_labels);
  result.runtime_seconds = timer.seconds();
  result.peak_memory_mb = detail::peak_memory_mb();
  return result;
}

KNNCVResult KNNCV_CUDA(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const KNNOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  KNNOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  detail::validate_inputs(x, labels, constrain);
  if (cuda_options.k < 1) throw std::invalid_argument("KNNOptions::k must be positive.");
  detail::Timer timer;
  KNNCVResult result;
  result.true_labels = labels;
  result.predicted_labels.assign(labels.size(), 0);
  result.fold_assignments = detail::make_folds(labels, constrain, cuda_options.cv);
  result.parameters = resolve_knn_parameters(cuda_options);
  result.parameters.index_type = KNNIndexType::FaissIVFFlat;

  const std::vector<int> fold_ids = detail::sorted_unique_folds(result.fold_assignments);
  faiss::gpu::StandardGpuResources resources;
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(result.fold_assignments, fold, true);
    const std::vector<int> train = detail::indices_where_fold(result.fold_assignments, fold, false);
    std::vector<int> fold_pred = knn_predict_faiss_ivf_flat_cuda(
      resources,
      x,
      labels,
      train,
      validation,
      result.parameters.k,
      result.parameters.metric,
      result.parameters.gpu_device,
      result.parameters.ivf_nlist,
      result.parameters.ivf_nprobe
    );
    for (std::size_t i = 0; i < validation.size(); ++i) {
      result.predicted_labels[static_cast<std::size_t>(validation[i])] = fold_pred[i];
    }
    result.folds.push_back(FoldResult{
      fold,
      static_cast<int>(train.size()),
      static_cast<int>(validation.size()),
      detail::accuracy_on_indices(labels, result.predicted_labels, validation)
    });
  }

  result.global_accuracy = detail::accuracy(labels, result.predicted_labels);
  result.confusion = detail::make_confusion(labels, result.predicted_labels);
  result.runtime_seconds = timer.seconds();
  result.peak_memory_mb = detail::peak_memory_mb();
  return result;
#else
  (void)x;
  (void)labels;
  (void)constrain;
  (void)options;
  throw std::runtime_error("KNNCV_CUDA requires a CUDA/FAISS GPU build.");
#endif
}

}  // namespace kodama
