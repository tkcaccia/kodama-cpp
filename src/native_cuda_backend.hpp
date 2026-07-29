// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "native_knn.hpp"

namespace kodama::detail {

struct NativeCudaIVFStats {
  int nlist = 0;
  int nprobe = 0;
  double pilot_recall = 0.0;
};

class NativeCudaIVFIndex {
 public:
  NativeCudaIVFIndex();
  ~NativeCudaIVFIndex();
  NativeCudaIVFIndex(NativeCudaIVFIndex&&) noexcept;
  NativeCudaIVFIndex& operator=(NativeCudaIVFIndex&&) noexcept;

  NativeCudaIVFIndex(const NativeCudaIVFIndex&) = delete;
  NativeCudaIVFIndex& operator=(const NativeCudaIVFIndex&) = delete;

  bool valid() const noexcept;
  int rows() const noexcept;
  int dimensions() const noexcept;
  int nlist() const noexcept;
  int device() const noexcept;
  DistanceMetric metric() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit NativeCudaIVFIndex(std::unique_ptr<Impl> impl);

  friend NativeCudaIVFIndex native_cuda_build_ivf_index(
    const std::vector<float>&,
    int,
    int,
    DistanceMetric,
    int,
    int
  );
  friend NativeKNNResult native_cuda_ivf_index_search(
    const NativeCudaIVFIndex&,
    const std::vector<float>&,
    int,
    int,
    int,
    double,
    const std::vector<int>&,
    NativeCudaIVFStats*
  );
  friend NativeKNNResult native_cuda_ivf_index_self_search(
    const NativeCudaIVFIndex&,
    int,
    int,
    double,
    const std::vector<int>&,
    NativeCudaIVFStats*
  );
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

NativeCudaIVFIndex native_cuda_build_ivf_index(
  const std::vector<float>& train,
  int train_rows,
  int dimensions,
  DistanceMetric metric,
  int requested_nlist,
  int device
);

NativeKNNResult native_cuda_ivf_index_search(
  const NativeCudaIVFIndex& index,
  const std::vector<float>& query,
  int query_rows,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices = {},
  NativeCudaIVFStats* stats = nullptr
);

NativeKNNResult native_cuda_ivf_index_self_search(
  const NativeCudaIVFIndex& index,
  int k,
  int requested_nprobe,
  double target_recall,
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
