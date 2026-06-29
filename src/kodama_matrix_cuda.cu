#include "kodama_matrix_cuda.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace kodama::detail {
namespace {

constexpr int kMaxGraphNeighbors = 1024;
constexpr float kCudaInfinity = INFINITY;

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
T* device_alloc_copy(const std::vector<T>& host, const char* what) {
  if (host.empty()) return nullptr;
  T* device = nullptr;
  check_cuda(cudaMalloc(&device, host.size() * sizeof(T)), what);
  check_cuda(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), what);
  return device;
}

__global__ void kodama_dissimilarity_kernel(
  int* indices,
  float* distances,
  const int* res,
  int runs,
  int samples,
  int neighbors,
  bool one_based_indices
) {
  const int row_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_id >= samples) return;

  float row_dist[kMaxGraphNeighbors];
  int row_idx[kMaxGraphNeighbors];
  const int row_offset = row_id * neighbors;

  for (int j = 0; j < neighbors; ++j) {
    const int offset = row_offset + j;
    const int neighbor = indices[offset];
    float distance = distances[offset];
    if (neighbor < 0 || neighbor >= samples || !isfinite(distance)) {
      row_dist[j] = kCudaInfinity;
      row_idx[j] = neighbor;
      continue;
    }

    int same = 0;
    int valid = 0;
    for (int run = 0; run < runs; ++run) {
      const int base = run * samples;
      const int lhs = res[base + row_id];
      const int rhs = res[base + neighbor];
      if (lhs == 0 || rhs == 0) continue;
      ++valid;
      if (lhs == rhs) ++same;
    }
    if (same == 0 || valid == 0) {
      distance = kCudaInfinity;
    } else {
      const double agreement = static_cast<double>(same) / static_cast<double>(valid);
      distance = static_cast<float>((1.0 + static_cast<double>(distance)) / (agreement * agreement));
    }
    row_dist[j] = distance;
    row_idx[j] = neighbor;
  }

  for (int i = 1; i < neighbors; ++i) {
    const float dist = row_dist[i];
    const int idx = row_idx[i];
    int j = i - 1;
    while (j >= 0 && (row_dist[j] > dist || (row_dist[j] == dist && row_idx[j] > idx))) {
      row_dist[j + 1] = row_dist[j];
      row_idx[j + 1] = row_idx[j];
      --j;
    }
    row_dist[j + 1] = dist;
    row_idx[j + 1] = idx;
  }

  for (int j = 0; j < neighbors; ++j) {
    const int offset = row_offset + j;
    distances[offset] = row_dist[j];
    indices[offset] = row_idx[j] >= 0 && one_based_indices ? row_idx[j] + 1 : row_idx[j];
  }
}

}  // namespace

void apply_kodama_dissimilarity_cuda(
  NeighborGraph& graph,
  const std::vector<int>& res,
  int runs,
  int samples,
  int gpu_device,
  bool one_based_indices
) {
  if (runs <= 0 || samples <= 0 || graph.neighbors <= 0) return;
  if (graph.neighbors > kMaxGraphNeighbors) {
    throw std::invalid_argument("CUDA KODAMA dissimilarity supports at most 1024 graph neighbors.");
  }
  if (static_cast<int>(res.size()) != runs * samples) {
    throw std::invalid_argument("CUDA KODAMA dissimilarity label matrix size mismatch.");
  }

  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice");
  int* device_indices = device_alloc_copy(graph.indices, "copy KODAMA graph indices");
  float* device_distances = device_alloc_copy(graph.distances, "copy KODAMA graph distances");
  int* device_res = device_alloc_copy(res, "copy KODAMA result labels");

  const int threads = 128;
  const int blocks = (samples + threads - 1) / threads;
  kodama_dissimilarity_kernel<<<blocks, threads>>>(
    device_indices,
    device_distances,
    device_res,
    runs,
    samples,
    graph.neighbors,
    one_based_indices
  );
  check_cuda(cudaGetLastError(), "launch KODAMA dissimilarity kernel");
  check_cuda(cudaDeviceSynchronize(), "run KODAMA dissimilarity kernel");

  check_cuda(
    cudaMemcpy(graph.indices.data(), device_indices, graph.indices.size() * sizeof(int), cudaMemcpyDeviceToHost),
    "copy KODAMA graph indices"
  );
  check_cuda(
    cudaMemcpy(graph.distances.data(), device_distances, graph.distances.size() * sizeof(float), cudaMemcpyDeviceToHost),
    "copy KODAMA graph distances"
  );

  cudaFree(device_indices);
  cudaFree(device_distances);
  cudaFree(device_res);
}

}  // namespace kodama::detail
