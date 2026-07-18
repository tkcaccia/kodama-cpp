// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "kodama/kodama.hpp"

namespace kodama::detail {

void apply_kodama_dissimilarity_cuda(
  NeighborGraph& graph,
  const std::vector<int>& res,
  int runs,
  int samples,
  int gpu_device,
  bool one_based_indices = false
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
