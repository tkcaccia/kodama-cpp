#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

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

std::vector<double> row_norms(MatrixView x, const std::vector<int>& rows) {
  std::vector<double> norms(rows.size(), 1.0);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    long double ss = 0.0;
    const int row = rows[r];
    for (std::size_t j = 0; j < x.cols; ++j) {
      const double v = x(static_cast<std::size_t>(row), j);
      ss += static_cast<long double>(v) * static_cast<long double>(v);
    }
    const double n = std::sqrt(static_cast<double>(ss));
    norms[r] = n > 0.0 && std::isfinite(n) ? n : 1.0;
  }
  return norms;
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

std::vector<int> knn_predict_cpu_exact(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& train,
  const std::vector<int>& validation,
  int k,
  DistanceMetric metric
) {
  if (train.empty()) throw std::runtime_error("KNN fold has no training samples.");
  if (k < 1) throw std::invalid_argument("k must be positive.");
  k = std::min(k, static_cast<int>(train.size()));
  const std::vector<double> train_norm = row_norms(x, train);
  const std::vector<double> val_norm = row_norms(x, validation);
  std::vector<int> pred(validation.size(), 0);

  for (std::size_t qi = 0; qi < validation.size(); ++qi) {
    const int qrow = validation[qi];
    std::vector<Neighbor> best;
    best.reserve(static_cast<std::size_t>(k + 1));
    for (std::size_t ti = 0; ti < train.size(); ++ti) {
      const int trow = train[ti];
      long double dot = 0.0;
      for (std::size_t j = 0; j < x.cols; ++j) {
        dot += static_cast<long double>(x(static_cast<std::size_t>(qrow), j)) *
               static_cast<long double>(x(static_cast<std::size_t>(trow), j));
      }
      double score = static_cast<double>(dot);
      if (metric == DistanceMetric::Cosine) {
        score /= val_norm[qi] * train_norm[ti];
      }
      best.push_back(Neighbor{score, labels[static_cast<std::size_t>(trow)]});
      if (static_cast<int>(best.size()) > k) {
        std::nth_element(best.begin(), best.begin() + k, best.end(), better_neighbor);
        best.resize(static_cast<std::size_t>(k));
      }
    }
    std::sort(best.begin(), best.end(), better_neighbor);
    pred[qi] = majority_vote(best);
  }
  return pred;
}

KNNParametersUsed resolve_knn_parameters(const KNNOptions& options) {
  KNNParametersUsed used;
  used.metric = options.metric;
  used.k = options.k;
  used.ivf_nlist = options.ivf_nlist;
  used.ivf_nprobe = options.ivf_nprobe;
  used.gpu_device = options.gpu_device;
  used.n_threads = options.n_threads;

  if (options.backend == Backend::CUDA || options.index_type == KNNIndexType::CuvsIVFFlat) {
#if defined(KODAMA_ENABLE_FAISS) || defined(KODAMA_ENABLE_CUVS)
    used.backend = Backend::CUDA;
    used.index_type = options.index_type == KNNIndexType::Auto ? KNNIndexType::FaissIVFFlat : options.index_type;
#else
    throw std::runtime_error("CUDA/FAISS/cuVS KNN backend was requested but this build has no FAISS/cuVS support.");
#endif
  } else {
    used.backend = Backend::CPU;
    used.index_type = options.index_type == KNNIndexType::Auto ? KNNIndexType::Exact : options.index_type;
    if (used.index_type != KNNIndexType::Exact && used.index_type != KNNIndexType::FaissFlat && used.index_type != KNNIndexType::FaissIVFFlat) {
      used.index_type = KNNIndexType::Exact;
    }
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
    if (result.parameters.backend == Backend::CPU) {
      fold_pred = knn_predict_cpu_exact(x, labels, train, validation, result.parameters.k, result.parameters.metric);
    } else {
      throw std::runtime_error("FAISS/cuVS CUDA KNNCV backend is declared but not linked in this build.");
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

}  // namespace kodama

