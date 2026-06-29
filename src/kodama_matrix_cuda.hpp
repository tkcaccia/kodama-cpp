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

}  // namespace kodama::detail
