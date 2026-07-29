// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "metal_backend.hpp"

#include <stdexcept>
#include <utility>

namespace kodama::detail {

struct NativeMetalIVFIndex::Impl {};

NativeMetalIVFIndex::NativeMetalIVFIndex() = default;
NativeMetalIVFIndex::~NativeMetalIVFIndex() = default;
NativeMetalIVFIndex::NativeMetalIVFIndex(NativeMetalIVFIndex&&) noexcept = default;
NativeMetalIVFIndex& NativeMetalIVFIndex::operator=(NativeMetalIVFIndex&&) noexcept = default;
NativeMetalIVFIndex::NativeMetalIVFIndex(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool NativeMetalIVFIndex::valid() const noexcept { return false; }
int NativeMetalIVFIndex::rows() const noexcept { return 0; }
int NativeMetalIVFIndex::dimensions() const noexcept { return 0; }
int NativeMetalIVFIndex::nlist() const noexcept { return 0; }
DistanceMetric NativeMetalIVFIndex::metric() const noexcept {
  return DistanceMetric::Euclidean;
}

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

NativeMetalIVFIndex metal_build_ivf_index(
  const std::vector<float>&,
  int,
  int,
  DistanceMetric,
  int
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

NativeKNNResult metal_ivf_index_search(
  const NativeMetalIVFIndex&,
  const std::vector<float>&,
  int,
  int,
  int,
  double,
  const std::vector<int>&,
  MetalIVFStats*
) {
  throw std::runtime_error("The Metal backend is not available in this build.");
}

NativeKNNResult metal_ivf_index_self_search(
  const NativeMetalIVFIndex&,
  int,
  int,
  double,
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
