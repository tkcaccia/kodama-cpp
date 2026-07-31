// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "kodama/kodama.hpp"

namespace kodama::detail {

class CudaResidentKODAMAGraph {
 public:
  CudaResidentKODAMAGraph();
  ~CudaResidentKODAMAGraph();
  CudaResidentKODAMAGraph(CudaResidentKODAMAGraph&&) noexcept;
  CudaResidentKODAMAGraph& operator=(CudaResidentKODAMAGraph&&) noexcept;

  CudaResidentKODAMAGraph(const CudaResidentKODAMAGraph&) = delete;
  CudaResidentKODAMAGraph& operator=(const CudaResidentKODAMAGraph&) = delete;

  bool valid() const noexcept;
  int samples() const noexcept;
  int neighbors() const noexcept;
  int lanes() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit CudaResidentKODAMAGraph(std::unique_ptr<Impl> impl);

  friend CudaResidentKODAMAGraph make_cuda_resident_kodama_graph(
    const NeighborGraph&,
    int,
    int,
    int
  );
  friend std::vector<int> cuda_resident_project_landmark_labels(
    const CudaResidentKODAMAGraph&,
    const std::vector<char>&,
    const std::vector<int>&,
    int,
    int,
    int
  );
  friend void cuda_resident_apply_kodama_dissimilarity(
    CudaResidentKODAMAGraph&,
    const std::vector<int>&,
    int,
    bool,
    bool
  );
  friend NeighborGraph download_cuda_resident_kodama_graph(
    const CudaResidentKODAMAGraph&
  );
};

CudaResidentKODAMAGraph make_cuda_resident_kodama_graph(
  const NeighborGraph& graph,
  int samples,
  int gpu_device,
  int lanes
);

std::vector<int> cuda_resident_project_landmark_labels(
  const CudaResidentKODAMAGraph& graph,
  const std::vector<char>& is_landmark,
  const std::vector<int>& labels,
  int projection_k,
  int fallback_label,
  int lane
);

void cuda_resident_apply_kodama_dissimilarity(
  CudaResidentKODAMAGraph& graph,
  const std::vector<int>& res,
  int runs,
  bool input_one_based_indices = false,
  bool output_one_based_indices = false
);

NeighborGraph download_cuda_resident_kodama_graph(
  const CudaResidentKODAMAGraph& graph
);

void apply_kodama_dissimilarity_cuda(
  NeighborGraph& graph,
  const std::vector<int>& res,
  int runs,
  int samples,
  int gpu_device,
  bool input_one_based_indices = false,
  bool output_one_based_indices = false
);

NeighborGraph spatial_grid_self_knn_cuda(
  const std::vector<float>& data,
  int n,
  int dims,
  int neighbors,
  int gpu_device,
  bool one_based_indices = false,
  bool include_self = false
);

}  // namespace kodama::detail
