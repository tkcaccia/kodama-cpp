#pragma once

#include <cstddef>
#include <vector>

#include "kodama/kodama.hpp"

namespace kodama::detail {

struct NativeHNSWParameters {
  int m = 16;
  int ef_construction = 80;
  int ef_search = 64;
};

struct NativeKNNResult {
  int queries = 0;
  int neighbors = 0;
  std::vector<int> indices;
  // Internal smaller-is-better values: squared L2 or negative inner product.
  std::vector<float> distances;
};

std::vector<float> prepare_native_matrix(
  MatrixView x,
  const std::vector<int>& rows,
  DistanceMetric metric
);

std::vector<float> prepare_native_matrix(MatrixView x, DistanceMetric metric);

NativeKNNResult native_hnsw_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  const NativeHNSWParameters& parameters,
  int n_threads,
  const std::vector<int>& query_train_indices = {}
);

float native_knn_score(float internal_distance, DistanceMetric metric);
float native_knn_output_distance(float internal_distance, DistanceMetric metric);

}  // namespace kodama::detail
