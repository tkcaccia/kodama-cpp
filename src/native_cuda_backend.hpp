// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

#include "native_knn.hpp"

namespace kodama::detail {

struct NativeCudaIVFStats {
  int nlist = 0;
  int nprobe = 0;
  double pilot_recall = 0.0;
};

bool native_cuda_backend_available(int device = 0);

NativeKNNResult native_cuda_exact_knn_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  int device,
  const std::vector<int>& query_train_indices = {}
);

NativeKNNResult native_cuda_ivf_knn_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  int requested_nlist,
  int requested_nprobe,
  double target_recall,
  int device,
  const std::vector<int>& query_train_indices = {},
  NativeCudaIVFStats* stats = nullptr
);

std::vector<int> native_cuda_kmeans_labels(
  const std::vector<float>& data,
  int rows,
  int dimensions,
  int clusters,
  int max_iterations,
  std::uint64_t seed,
  int device
);

}  // namespace kodama::detail
