#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/MetricType.h>

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

bool better_neighbor(const Neighbor& a, const Neighbor& b) {
  if (a.score != b.score) return a.score > b.score;
  return a.label < b.label;
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
  int requested_nprobe
) {
  if (train.empty()) throw std::runtime_error("KNN fold has no training samples.");
  if (k < 1) throw std::invalid_argument("k must be positive.");
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
    std::sort(best.begin(), best.end(), better_neighbor);
    pred[qi] = majority_vote(best);
  }
  return pred;
}

#if defined(KODAMA_ENABLE_CUDA)
std::vector<int> knn_predict_faiss_ivf_flat_cuda(
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

  faiss::gpu::StandardGpuResources resources;
  faiss::gpu::GpuIndexIVFFlatConfig config;
  config.device = gpu_device;
  faiss::gpu::GpuIndexIVFFlat index(&resources, d, nlist, faiss::METRIC_INNER_PRODUCT, config);
  index.setNumProbes(nprobe);
  index.train(train.size(), train_x.data());
  index.add(train.size(), train_x.data());

  std::vector<faiss::idx_t> idx(validation.size() * static_cast<std::size_t>(k), -1);
  std::vector<float> scores(validation.size() * static_cast<std::size_t>(k), -std::numeric_limits<float>::infinity());
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
    std::sort(best.begin(), best.end(), better_neighbor);
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
    used.index_type = KNNIndexType::FaissIVFFlat;
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
    fold_pred = knn_predict_faiss_ivf_flat_cpu(
      x,
      labels,
      train,
      validation,
      result.parameters.k,
      result.parameters.metric,
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
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(result.fold_assignments, fold, true);
    const std::vector<int> train = detail::indices_where_fold(result.fold_assignments, fold, false);
    std::vector<int> fold_pred = knn_predict_faiss_ivf_flat_cuda(
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
