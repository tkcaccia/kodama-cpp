// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "native_knn.hpp"

namespace kodama::detail {

bool metal_backend_available();
void metal_set_pls_residency_epoch(std::uint64_t epoch);

NativeKNNResult metal_exact_knn_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  const std::vector<int>& query_train_indices = {}
);

struct MetalIVFStats {
  int nlist = 0;
  int nprobe = 0;
  double pilot_recall = 0.0;
};

class NativeMetalIVFIndex {
 public:
  NativeMetalIVFIndex();
  ~NativeMetalIVFIndex();
  NativeMetalIVFIndex(NativeMetalIVFIndex&&) noexcept;
  NativeMetalIVFIndex& operator=(NativeMetalIVFIndex&&) noexcept;

  NativeMetalIVFIndex(const NativeMetalIVFIndex&) = delete;
  NativeMetalIVFIndex& operator=(const NativeMetalIVFIndex&) = delete;

  bool valid() const noexcept;
  int rows() const noexcept;
  int dimensions() const noexcept;
  int nlist() const noexcept;
  DistanceMetric metric() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit NativeMetalIVFIndex(std::unique_ptr<Impl> impl);

  friend NativeMetalIVFIndex metal_build_ivf_index(
    const std::vector<float>&,
    int,
    int,
    DistanceMetric,
    int
  );
  friend NativeKNNResult metal_ivf_index_search(
    const NativeMetalIVFIndex&,
    const std::vector<float>&,
    int,
    int,
    int,
    double,
    const std::vector<int>&,
    MetalIVFStats*
  );
  friend NativeKNNResult metal_ivf_index_self_search(
    const NativeMetalIVFIndex&,
    int,
    int,
    double,
    const std::vector<int>&,
    MetalIVFStats*
  );
};

class NativeMetalKNNVoteGraph {
 public:
  NativeMetalKNNVoteGraph();
  ~NativeMetalKNNVoteGraph();
  NativeMetalKNNVoteGraph(NativeMetalKNNVoteGraph&&) noexcept;
  NativeMetalKNNVoteGraph& operator=(NativeMetalKNNVoteGraph&&) noexcept;

  NativeMetalKNNVoteGraph(const NativeMetalKNNVoteGraph&) = delete;
  NativeMetalKNNVoteGraph& operator=(const NativeMetalKNNVoteGraph&) = delete;

  bool valid() const noexcept;
  int samples() const noexcept;
  int neighbors() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit NativeMetalKNNVoteGraph(std::unique_ptr<Impl> impl);

  friend NativeMetalKNNVoteGraph metal_build_knn_vote_graph(
    const std::vector<int>&,
    const std::vector<float>&,
    int,
    int
  );
  friend std::vector<int> metal_knn_vote_predict(
    const NativeMetalKNNVoteGraph&,
    const std::vector<int>&,
    int
  );
};

class NativeMetalKODAMAGraph {
 public:
  NativeMetalKODAMAGraph();
  ~NativeMetalKODAMAGraph();
  NativeMetalKODAMAGraph(NativeMetalKODAMAGraph&&) noexcept;
  NativeMetalKODAMAGraph& operator=(NativeMetalKODAMAGraph&&) noexcept;

  NativeMetalKODAMAGraph(const NativeMetalKODAMAGraph&) = delete;
  NativeMetalKODAMAGraph& operator=(const NativeMetalKODAMAGraph&) = delete;

  bool valid() const noexcept;
  int samples() const noexcept;
  int neighbors() const noexcept;
  int lanes() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit NativeMetalKODAMAGraph(std::unique_ptr<Impl> impl);

  friend NativeMetalKODAMAGraph metal_build_resident_kodama_graph(
    const NeighborGraph&,
    int,
    int
  );
  friend std::vector<int> metal_project_landmark_labels(
    const NativeMetalKODAMAGraph&,
    const std::vector<char>&,
    const std::vector<int>&,
    int,
    int,
    int
  );
  friend void metal_apply_resident_kodama_dissimilarity(
    NativeMetalKODAMAGraph&,
    const std::vector<int>&,
    int,
    bool,
    bool
  );
  friend NeighborGraph metal_download_resident_kodama_graph(
    const NativeMetalKODAMAGraph&
  );
};

NativeKNNResult metal_ivf_knn_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  int requested_nlist,
  int requested_nprobe,
  const std::vector<int>& query_train_indices = {},
  MetalIVFStats* stats = nullptr
);

NativeMetalIVFIndex metal_build_ivf_index(
  const std::vector<float>& train,
  int train_rows,
  int dimensions,
  DistanceMetric metric,
  int requested_nlist
);

NativeKNNResult metal_ivf_index_search(
  const NativeMetalIVFIndex& index,
  const std::vector<float>& query,
  int query_rows,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices = {},
  MetalIVFStats* stats = nullptr
);

NativeKNNResult metal_ivf_index_self_search(
  const NativeMetalIVFIndex& index,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices = {},
  MetalIVFStats* stats = nullptr
);

std::vector<int> metal_kmeans_labels(
  const std::vector<float>& data,
  int rows,
  int dimensions,
  int clusters,
  const std::vector<int>& initial_point_indices,
  int max_iterations
);

NativeMetalKNNVoteGraph metal_build_knn_vote_graph(
  const std::vector<int>& neighbor_rows,
  const std::vector<float>& scores,
  int samples,
  int neighbors
);

std::vector<int> metal_knn_vote_predict(
  const NativeMetalKNNVoteGraph& graph,
  const std::vector<int>& labels,
  int fallback_label
);

NativeMetalKODAMAGraph metal_build_resident_kodama_graph(
  const NeighborGraph& graph,
  int samples,
  int lanes
);

std::vector<int> metal_project_landmark_labels(
  const NativeMetalKODAMAGraph& graph,
  const std::vector<char>& is_landmark,
  const std::vector<int>& labels,
  int projection_k,
  int fallback_label,
  int lane
);

void metal_apply_resident_kodama_dissimilarity(
  NativeMetalKODAMAGraph& graph,
  const std::vector<int>& res,
  int runs,
  bool input_one_based_indices = false,
  bool output_one_based_indices = false
);

NeighborGraph metal_download_resident_kodama_graph(
  const NativeMetalKODAMAGraph& graph
);

std::vector<float> metal_matrix_multiply(
  const std::vector<float>& left,
  int left_rows,
  int left_cols,
  const std::vector<float>& right,
  int right_rows,
  int right_cols,
  bool transpose_left = false,
  bool transpose_right = false
);

struct MetalSIMPLSResult {
  int predictors = 0;
  int responses = 0;
  int components = 0;
  std::vector<float> weights;
  std::vector<float> y_weights;
};

MetalSIMPLSResult metal_simpls_fit(
  const std::vector<float>& x,
  int rows,
  int predictors,
  const std::vector<float>& cross_product,
  int responses,
  int max_components
);

}  // namespace kodama::detail
