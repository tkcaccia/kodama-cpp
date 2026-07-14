#include "metal_backend.hpp"

#include <stdexcept>

namespace kodama::detail {

bool metal_backend_available() {
  return false;
}

NativeKNNResult metal_exact_knn_search(
  const std::vector<float>&,
  int,
  const std::vector<float>&,
  int,
  int,
  int,
  DistanceMetric,
  const std::vector<int>&
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

NativeKNNResult metal_ivf_knn_search(
  const std::vector<float>&,
  int,
  const std::vector<float>&,
  int,
  int,
  int,
  DistanceMetric,
  int,
  int,
  const std::vector<int>&,
  MetalIVFStats*
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

std::vector<int> metal_kmeans_labels(
  const std::vector<float>&,
  int,
  int,
  int,
  const std::vector<int>&,
  int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

std::vector<float> metal_matrix_multiply(
  const std::vector<float>&,
  int,
  int,
  const std::vector<float>&,
  int,
  int,
  bool,
  bool
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

MetalSIMPLSResult metal_simpls_fit(
  const std::vector<float>&,
  int,
  int,
  const std::vector<float>&,
  int,
  int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

}  // namespace kodama::detail
