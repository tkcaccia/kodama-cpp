// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama_matrix_cuda.hpp"
#include "native_cuda_backend.hpp"
#include "spatial_grid_knn.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <climits>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace kodama::detail {
namespace {

constexpr int kMaxGraphNeighbors = 1024;
constexpr int kMaxGridNeighbors = 256;
constexpr float kCudaInfinity = INFINITY;

__global__ void convert_native_graph_distances_kernel(
  float* distances,
  std::size_t items,
  int metric
) {
  const std::size_t index =
    static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= items) return;
  const float value = distances[index];
  if (metric == 0) {
    distances[index] = sqrtf(fmaxf(0.0f, value));
  } else if (metric == 1) {
    distances[index] = fmaxf(0.0f, 1.0f + value);
  } else {
    distances[index] = 1.0f + value;
  }
}

struct CudaSpatialGridParams {
  int n;
  int dims;
  int k;
  int nonself_k;
  int bins;
  int include_self;
  int one_based;
  int sort_width;
  float min_x;
  float min_y;
  float min_z;
  float cell_x;
  float cell_y;
  float cell_z;
};

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

__global__ void scatter_landmark_labels_kernel(
  const int* landmark_rows,
  const int* landmark_input_labels,
  int* landmark_epoch,
  int* labels,
  int landmarks,
  int epoch,
  int samples
) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= landmarks) return;
  const int row = landmark_rows[i];
  if (row < 0 || row >= samples) return;
  labels[row] = landmark_input_labels[i];
  __threadfence();
  landmark_epoch[row] = epoch;
}

__global__ void project_landmark_labels_kernel(
  const int* graph_indices,
  const int* landmark_epoch,
  const int* labels,
  int* projected,
  int samples,
  int neighbors,
  int projection_k,
  int fallback_label,
  int epoch
) {
  const int query = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (query >= samples) return;
  if (landmark_epoch[query] == epoch) {
    projected[query] = labels[query];
    return;
  }

  const std::size_t base =
    static_cast<std::size_t>(query) * static_cast<std::size_t>(neighbors);
  int cutoff = neighbors;
  int accepted = 0;
  for (int rank = 0; rank < neighbors; ++rank) {
    const int neighbor = graph_indices[base + static_cast<std::size_t>(rank)];
    if (neighbor < 0 || neighbor >= samples || landmark_epoch[neighbor] != epoch) continue;
    ++accepted;
    if (accepted == projection_k) {
      cutoff = rank + 1;
      break;
    }
  }
  if (accepted == 0) {
    projected[query] = fallback_label;
    return;
  }

  int best_label = fallback_label;
  int best_count = -1;
  for (int rank = 0; rank < cutoff; ++rank) {
    const int neighbor = graph_indices[base + static_cast<std::size_t>(rank)];
    if (neighbor < 0 || neighbor >= samples || landmark_epoch[neighbor] != epoch) continue;
    const int candidate = labels[neighbor];
    bool seen = false;
    for (int prior = 0; prior < rank; ++prior) {
      const int prior_neighbor =
        graph_indices[base + static_cast<std::size_t>(prior)];
      if (prior_neighbor >= 0 && prior_neighbor < samples &&
          landmark_epoch[prior_neighbor] == epoch &&
          labels[prior_neighbor] == candidate) {
        seen = true;
        break;
      }
    }
    if (seen) continue;
    int count = 0;
    for (int other = rank; other < cutoff; ++other) {
      const int other_neighbor =
        graph_indices[base + static_cast<std::size_t>(other)];
      if (other_neighbor >= 0 && other_neighbor < samples &&
          landmark_epoch[other_neighbor] == epoch &&
          labels[other_neighbor] == candidate) {
        ++count;
      }
    }
    if (count > best_count ||
        (count == best_count && candidate < best_label)) {
      best_label = candidate;
      best_count = count;
    }
  }
  projected[query] = best_count < 0 ? fallback_label : best_label;
}

__global__ void count_constrained_labels_kernel(
  const int* labels,
  const int* groups,
  int* counts,
  int samples,
  int label_width
) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= samples) return;
  const int label = labels[row];
  if (label >= 0 && label < label_width) {
    atomicAdd(counts + groups[row] * label_width + label, 1);
  }
}

__global__ void select_constrained_majority_kernel(
  const int* counts,
  int* group_labels,
  int groups,
  int label_width
) {
  const int group = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (group >= groups) return;
  int best_label = 0;
  int best_count = -1;
  const int base = group * label_width;
  for (int label = 0; label < label_width; ++label) {
    const int count = counts[base + label];
    if (count > best_count) {
      best_count = count;
      best_label = label;
    }
  }
  group_labels[group] = best_label;
}

__global__ void apply_constrained_majority_kernel(
  int* labels,
  const int* groups,
  const int* group_labels,
  int samples
) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < samples) labels[row] = group_labels[groups[row]];
}

__device__ bool kodama_pair_greater(float lhs_dist, int lhs_idx, float rhs_dist, int rhs_idx) {
  return lhs_dist > rhs_dist || (lhs_dist == rhs_dist && lhs_idx > rhs_idx);
}

__device__ int grid_coord_device(float value, float min_value, float cell_size, int bins) {
  int out = static_cast<int>((value - min_value) / cell_size);
  if (out < 0) out = 0;
  if (out >= bins) out = bins - 1;
  return out;
}

__device__ int grid_cell_2d_device(int ix, int iy, int bins) {
  return iy * bins + ix;
}

__device__ int grid_cell_3d_device(int ix, int iy, int iz, int bins) {
  return (iz * bins + iy) * bins + ix;
}

__device__ float grid_lower_outside_device(
  float x,
  float y,
  float z,
  const CudaSpatialGridParams params,
  int x0,
  int x1,
  int y0,
  int y1,
  int z0,
  int z1
) {
  float best = kCudaInfinity;
  if (x0 > 0) {
    const float border = params.min_x + static_cast<float>(x0) * params.cell_x;
    const float dx = fmaxf(0.0f, x - border);
    best = fminf(best, dx * dx);
  }
  if (x1 + 1 < params.bins) {
    const float border = params.min_x + static_cast<float>(x1 + 1) * params.cell_x;
    const float dx = fmaxf(0.0f, border - x);
    best = fminf(best, dx * dx);
  }
  if (y0 > 0) {
    const float border = params.min_y + static_cast<float>(y0) * params.cell_y;
    const float dy = fmaxf(0.0f, y - border);
    best = fminf(best, dy * dy);
  }
  if (y1 + 1 < params.bins) {
    const float border = params.min_y + static_cast<float>(y1 + 1) * params.cell_y;
    const float dy = fmaxf(0.0f, border - y);
    best = fminf(best, dy * dy);
  }
  if (params.dims == 3) {
    if (z0 > 0) {
      const float border = params.min_z + static_cast<float>(z0) * params.cell_z;
      const float dz = fmaxf(0.0f, z - border);
      best = fminf(best, dz * dz);
    }
    if (z1 + 1 < params.bins) {
      const float border = params.min_z + static_cast<float>(z1 + 1) * params.cell_z;
      const float dz = fmaxf(0.0f, border - z);
      best = fminf(best, dz * dz);
    }
  }
  return best;
}

__device__ void grid_sort_candidates_warp(
  float* distances,
  int* indices,
  int width,
  int lane
) {
  for (int sequence = 2; sequence <= width; sequence <<= 1) {
    for (int stride = sequence >> 1; stride > 0; stride >>= 1) {
      for (int left = lane; left < width; left += warpSize) {
        const int right = left ^ stride;
        if (right <= left) continue;
        const bool ascending = (left & sequence) == 0;
        const bool greater = kodama_pair_greater(
          distances[left], indices[left], distances[right], indices[right]
        );
        if (greater == ascending) {
          const float distance = distances[left];
          const int index = indices[left];
          distances[left] = distances[right];
          indices[left] = indices[right];
          distances[right] = distance;
          indices[right] = index;
        }
      }
      __syncwarp();
    }
  }
}

__device__ void merge_grid_cell_candidates_warp(
  const float* data,
  const int* offsets,
  const int* rows,
  const CudaSpatialGridParams params,
  int query,
  int ix,
  int iy,
  int iz,
  float* best_dist,
  int* best_idx,
  int lane
) {
  if (ix < 0 || iy < 0 || ix >= params.bins || iy >= params.bins) return;
  if (params.dims == 3 && (iz < 0 || iz >= params.bins)) return;
  const int cell = params.dims == 3 ?
    grid_cell_3d_device(ix, iy, iz, params.bins) :
    grid_cell_2d_device(ix, iy, params.bins);
  const int start = offsets[cell];
  const int end = offsets[cell + 1];
  const std::size_t qbase = static_cast<std::size_t>(query) * params.dims;
  const float qx = data[qbase];
  const float qy = data[qbase + 1];
  const float qz = params.dims == 3 ? data[qbase + 2] : 0.0f;
  const int batch_capacity = params.sort_width - params.nonself_k;
  for (int batch = start; batch < end; batch += batch_capacity) {
    const int batch_size = min(batch_capacity, end - batch);
    for (int slot = lane; slot < batch_capacity; slot += warpSize) {
      const int destination = params.nonself_k + slot;
      if (slot >= batch_size) {
        best_dist[destination] = kCudaInfinity;
        best_idx[destination] = INT_MAX;
        continue;
      }
      const int candidate = rows[batch + slot];
      if (candidate == query) {
        best_dist[destination] = kCudaInfinity;
        best_idx[destination] = INT_MAX;
        continue;
      }
      const std::size_t cbase = static_cast<std::size_t>(candidate) * params.dims;
      const float dx = qx - data[cbase];
      const float dy = qy - data[cbase + 1];
      float distance = dx * dx + dy * dy;
      if (params.dims == 3) {
        const float dz = qz - data[cbase + 2];
        distance += dz * dz;
      }
      best_dist[destination] = distance;
      best_idx[destination] = candidate;
    }
    __syncwarp();
    grid_sort_candidates_warp(
      best_dist,
      best_idx,
      params.sort_width,
      lane
    );
  }
}

__global__ void spatial_grid_self_knn_kernel(
  const float* data,
  const int* offsets,
  const int* rows,
  int* out_idx,
  float* out_dist,
  CudaSpatialGridParams params
) {
  const int lane = static_cast<int>(threadIdx.x & (warpSize - 1));
  const int warp = static_cast<int>(threadIdx.x / warpSize);
  const int warps_per_block = static_cast<int>(blockDim.x / warpSize);
  const int q = static_cast<int>(blockIdx.x) * warps_per_block + warp;
  if (q >= params.n) return;

  extern __shared__ unsigned char shared_storage[];
  float* all_dist = reinterpret_cast<float*>(shared_storage);
  int* all_idx = reinterpret_cast<int*>(
    all_dist + static_cast<std::size_t>(warps_per_block) * params.sort_width
  );
  float* best_dist = all_dist + static_cast<std::size_t>(warp) * params.sort_width;
  int* best_idx = all_idx + static_cast<std::size_t>(warp) * params.sort_width;
  for (int j = lane; j < params.sort_width; j += warpSize) {
    best_dist[j] = kCudaInfinity;
    best_idx[j] = INT_MAX;
  }
  __syncwarp();

  const std::size_t qbase = static_cast<std::size_t>(q) * params.dims;
  const float qx = data[qbase];
  const float qy = data[qbase + 1];
  const float qz = params.dims == 3 ? data[qbase + 2] : 0.0f;
  const int cx = grid_coord_device(qx, params.min_x, params.cell_x, params.bins);
  const int cy = grid_coord_device(qy, params.min_y, params.cell_y, params.bins);
  const int cz = params.dims == 3 ? grid_coord_device(qz, params.min_z, params.cell_z, params.bins) : 0;

  if (params.nonself_k > 0) {
    for (int radius = 0; radius <= params.bins; ++radius) {
      const int raw_x0 = cx - radius;
      const int raw_x1 = cx + radius;
      const int raw_y0 = cy - radius;
      const int raw_y1 = cy + radius;
      const int raw_z0 = cz - radius;
      const int raw_z1 = cz + radius;
      const int x0 = max(0, raw_x0);
      const int x1 = min(params.bins - 1, raw_x1);
      const int y0 = max(0, raw_y0);
      const int y1 = min(params.bins - 1, raw_y1);
      const int z0 = params.dims == 3 ? max(0, raw_z0) : 0;
      const int z1 = params.dims == 3 ? min(params.bins - 1, raw_z1) : 0;

      if (params.dims == 2) {
        if (radius == 0) {
          merge_grid_cell_candidates_warp(data, offsets, rows, params, q, cx, cy, 0, best_dist, best_idx, lane);
        } else {
          for (int ix = raw_x0; ix <= raw_x1; ++ix) {
            if (raw_y0 >= 0 && raw_y0 < params.bins) {
              merge_grid_cell_candidates_warp(data, offsets, rows, params, q, ix, raw_y0, 0, best_dist, best_idx, lane);
            }
            if (raw_y1 != raw_y0 && raw_y1 >= 0 && raw_y1 < params.bins) {
              merge_grid_cell_candidates_warp(data, offsets, rows, params, q, ix, raw_y1, 0, best_dist, best_idx, lane);
            }
          }
          for (int iy = raw_y0 + 1; iy <= raw_y1 - 1; ++iy) {
            if (raw_x0 >= 0 && raw_x0 < params.bins) {
              merge_grid_cell_candidates_warp(data, offsets, rows, params, q, raw_x0, iy, 0, best_dist, best_idx, lane);
            }
            if (raw_x1 != raw_x0 && raw_x1 >= 0 && raw_x1 < params.bins) {
              merge_grid_cell_candidates_warp(data, offsets, rows, params, q, raw_x1, iy, 0, best_dist, best_idx, lane);
            }
          }
        }
      } else {
        if (radius == 0) {
          merge_grid_cell_candidates_warp(data, offsets, rows, params, q, cx, cy, cz, best_dist, best_idx, lane);
        } else {
          for (int iz = raw_z0; iz <= raw_z1; ++iz) {
            if (iz < 0 || iz >= params.bins) continue;
            for (int iy = raw_y0; iy <= raw_y1; ++iy) {
              if (iy < 0 || iy >= params.bins) continue;
              for (int ix = raw_x0; ix <= raw_x1; ++ix) {
                if (ix < 0 || ix >= params.bins) continue;
                if (ix != raw_x0 && ix != raw_x1 && iy != raw_y0 && iy != raw_y1 && iz != raw_z0 && iz != raw_z1) continue;
                merge_grid_cell_candidates_warp(data, offsets, rows, params, q, ix, iy, iz, best_dist, best_idx, lane);
              }
            }
          }
        }
      }
      if (best_idx[params.nonself_k - 1] != INT_MAX) {
        const float lower = grid_lower_outside_device(qx, qy, qz, params, x0, x1, y0, y1, z0, z1);
        if (lower > best_dist[params.nonself_k - 1]) break;
      }
    }
  }

  if (params.include_self && lane == 0) {
    const std::size_t offset = static_cast<std::size_t>(q) * params.k;
    out_idx[offset] = q + (params.one_based ? 1 : 0);
    out_dist[offset] = 0.0f;
  }
  const int first_nonself = params.include_self ? 1 : 0;
  for (int j = lane; j < params.nonself_k; j += warpSize) {
    const int out_col = first_nonself + j;
    const std::size_t offset = static_cast<std::size_t>(q) * params.k + static_cast<std::size_t>(out_col);
    if (best_idx[j] == INT_MAX) {
      out_idx[offset] = -1;
      out_dist[offset] = kCudaInfinity;
    } else {
      out_idx[offset] = best_idx[j] + (params.one_based ? 1 : 0);
      out_dist[offset] = sqrtf(fmaxf(best_dist[j], 0.0f));
    }
  }
}

__global__ void kodama_dissimilarity_shared_kernel(
  int* indices,
  float* distances,
  const int* res,
  int runs,
  int samples,
  int neighbors,
  int sort_width,
  bool input_one_based_indices,
  bool output_one_based_indices
) {
  const int row_id = static_cast<int>(blockIdx.x);
  if (row_id >= samples) return;

  extern __shared__ unsigned char shared_raw[];
  float* row_dist = reinterpret_cast<float*>(shared_raw);
  int* row_idx = reinterpret_cast<int*>(row_dist + sort_width);

  const int tid = static_cast<int>(threadIdx.x);
  const int row_offset = row_id * neighbors;

  if (tid < neighbors) {
    const int offset = row_offset + tid;
    const int stored_neighbor = indices[offset];
    const int neighbor =
      stored_neighbor >= 0 && input_one_based_indices ?
        stored_neighbor - 1 :
        stored_neighbor;
    float distance = distances[offset];
    if (neighbor < 0 || neighbor >= samples || !isfinite(distance)) {
      row_dist[tid] = kCudaInfinity;
      row_idx[tid] = neighbor;
    } else {
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
      row_dist[tid] = distance;
      row_idx[tid] = neighbor;
    }
  } else if (tid < sort_width) {
    row_dist[tid] = kCudaInfinity;
    row_idx[tid] = INT_MAX;
  }
  __syncthreads();

  for (int width = 2; width <= sort_width; width <<= 1) {
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
      const int other = tid ^ stride;
      if (other > tid && other < sort_width) {
        const bool ascending = (tid & width) == 0;
        const float self_dist = row_dist[tid];
        const int self_idx = row_idx[tid];
        const float other_dist = row_dist[other];
        const int other_idx = row_idx[other];
        const bool swap_pair =
          ascending ?
            kodama_pair_greater(self_dist, self_idx, other_dist, other_idx) :
            kodama_pair_greater(other_dist, other_idx, self_dist, self_idx);
        if (swap_pair) {
          row_dist[tid] = other_dist;
          row_idx[tid] = other_idx;
          row_dist[other] = self_dist;
          row_idx[other] = self_idx;
        }
      }
      __syncthreads();
    }
  }

  if (tid < neighbors) {
    const int offset = row_offset + tid;
    distances[offset] = row_dist[tid];
    indices[offset] =
      row_idx[tid] >= 0 && output_one_based_indices ?
        row_idx[tid] + 1 :
        row_idx[tid];
  }
}

}  // namespace

struct CudaResidentKODAMAGraph::Impl {
  struct Lane {
    int* landmark_epoch = nullptr;
    int* labels = nullptr;
    int* landmark_rows = nullptr;
    int* landmark_input_labels = nullptr;
    std::size_t landmark_capacity = 0;
    int* constrain = nullptr;
    int* constrain_counts = nullptr;
    int* constrain_labels = nullptr;
    std::size_t constrain_count_capacity = 0;
    std::size_t constrain_group_capacity = 0;
    cudaStream_t stream = nullptr;
  };

  int samples = 0;
  int neighbors = 0;
  int device = 0;
  int* indices = nullptr;
  float* distances = nullptr;
  int* base_indices = nullptr;
  float* base_distances = nullptr;
  int* result_labels = nullptr;
  std::size_t result_capacity = 0;
  int result_runs = 0;
  std::unique_ptr<NativeCudaIVFIndex> ivf_index;
  DistanceMetric metric = DistanceMetric::Euclidean;
  int dimensions = 0;
  std::vector<Lane> lane_buffers;

  ~Impl() {
    cudaSetDevice(device);
    for (Lane& lane : lane_buffers) {
      if (lane.stream != nullptr) cudaStreamSynchronize(lane.stream);
      cudaFree(lane.landmark_epoch);
      cudaFree(lane.labels);
      cudaFree(lane.landmark_rows);
      cudaFree(lane.landmark_input_labels);
      cudaFree(lane.constrain);
      cudaFree(lane.constrain_counts);
      cudaFree(lane.constrain_labels);
      if (lane.stream != nullptr) cudaStreamDestroy(lane.stream);
    }
    cudaFree(indices);
    cudaFree(distances);
    cudaFree(base_indices);
    cudaFree(base_distances);
    cudaFree(result_labels);
  }
};

CudaResidentKODAMAGraph::CudaResidentKODAMAGraph() = default;
CudaResidentKODAMAGraph::~CudaResidentKODAMAGraph() = default;
CudaResidentKODAMAGraph::CudaResidentKODAMAGraph(
  CudaResidentKODAMAGraph&&
) noexcept = default;
CudaResidentKODAMAGraph& CudaResidentKODAMAGraph::operator=(
  CudaResidentKODAMAGraph&&
) noexcept = default;
CudaResidentKODAMAGraph::CudaResidentKODAMAGraph(
  std::unique_ptr<Impl> impl
) : impl_(std::move(impl)) {}

bool CudaResidentKODAMAGraph::valid() const noexcept {
  return impl_ != nullptr;
}
int CudaResidentKODAMAGraph::samples() const noexcept {
  return impl_ == nullptr ? 0 : impl_->samples;
}
int CudaResidentKODAMAGraph::neighbors() const noexcept {
  return impl_ == nullptr ? 0 : impl_->neighbors;
}
int CudaResidentKODAMAGraph::lanes() const noexcept {
  return impl_ == nullptr ? 0 : static_cast<int>(impl_->lane_buffers.size());
}
bool CudaResidentKODAMAGraph::has_landmark_index() const noexcept {
  return impl_ != nullptr && impl_->ivf_index != nullptr;
}

CudaResidentKODAMAGraph make_cuda_resident_kodama_graph(
  const NeighborGraph& graph,
  int samples,
  int gpu_device,
  int lanes
) {
  const std::size_t expected =
    static_cast<std::size_t>(samples) *
    static_cast<std::size_t>(graph.neighbors);
  if (samples < 1 || graph.neighbors < 1 ||
      graph.indices.size() != expected ||
      graph.distances.size() != expected) {
    throw std::invalid_argument("Invalid CUDA resident KODAMA graph.");
  }
  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice resident KODAMA graph");
  auto impl = std::make_unique<CudaResidentKODAMAGraph::Impl>();
  impl->samples = samples;
  impl->neighbors = graph.neighbors;
  impl->device = gpu_device;
  impl->indices = device_alloc_copy(
    graph.indices,
    "copy resident KODAMA graph indices"
  );
  impl->distances = device_alloc_copy(
    graph.distances,
    "copy resident KODAMA graph distances"
  );
  impl->base_indices = device_alloc_copy(
    graph.indices, "copy immutable resident KODAMA graph indices");
  impl->base_distances = device_alloc_copy(
    graph.distances, "copy immutable resident KODAMA graph distances");
  impl->lane_buffers.resize(static_cast<std::size_t>(std::max(1, lanes)));
  for (CudaResidentKODAMAGraph::Impl::Lane& lane : impl->lane_buffers) {
    check_cuda(
      cudaStreamCreateWithFlags(&lane.stream, cudaStreamNonBlocking),
      "create resident KODAMA graph stream"
    );
    check_cuda(
      cudaMalloc(&lane.landmark_epoch, static_cast<std::size_t>(samples) * sizeof(int)),
      "allocate resident KODAMA landmark epochs"
    );
    check_cuda(
      cudaMalloc(&lane.labels, static_cast<std::size_t>(samples) * sizeof(int)),
      "allocate resident KODAMA labels"
    );
    check_cuda(cudaMalloc(&lane.constrain, static_cast<std::size_t>(samples) * sizeof(int)),
               "allocate resident KODAMA constraints");
    check_cuda(cudaMemset(lane.landmark_epoch, 0, static_cast<std::size_t>(samples) * sizeof(int)),
               "clear resident KODAMA landmark epochs");
  }
  return CudaResidentKODAMAGraph(std::move(impl));
}

CudaResidentKODAMAGraph make_cuda_resident_kodama_graph_ivf(
  const std::vector<float>& data,
  int samples,
  int dimensions,
  int neighbors,
  DistanceMetric metric,
  int requested_nlist,
  int requested_nprobe,
  int gpu_device,
  int lanes,
  NativeCudaIVFStats* stats
) {
  if (samples < 2 || dimensions < 1 || neighbors < 1 || neighbors >= samples ||
      data.size() != static_cast<std::size_t>(samples) * dimensions) {
    throw std::invalid_argument("Invalid CUDA resident IVF KODAMA graph input.");
  }
  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice resident IVF KODAMA graph");
  auto impl = std::make_unique<CudaResidentKODAMAGraph::Impl>();
  impl->samples = samples;
  impl->neighbors = neighbors;
  impl->device = gpu_device;
  impl->metric = metric;
  impl->dimensions = dimensions;
  const std::size_t items = static_cast<std::size_t>(samples) * neighbors;
  check_cuda(cudaMalloc(&impl->indices, items * sizeof(int)), "allocate resident IVF graph indices");
  check_cuda(cudaMalloc(&impl->distances, items * sizeof(float)), "allocate resident IVF graph distances");
  check_cuda(cudaMalloc(&impl->base_indices, items * sizeof(int)), "allocate immutable resident IVF graph indices");
  check_cuda(cudaMalloc(&impl->base_distances, items * sizeof(float)), "allocate immutable resident IVF graph distances");

  NativeCudaIVFIndex index = native_cuda_build_ivf_index(
    data,
    samples,
    dimensions,
    metric,
    requested_nlist,
    gpu_device
  );
  std::vector<int> exclusions(static_cast<std::size_t>(samples));
  std::iota(exclusions.begin(), exclusions.end(), 0);
  native_cuda_ivf_index_self_search_device(
    index,
    neighbors,
    requested_nprobe,
    0.99,
    exclusions,
    impl->indices,
    impl->distances,
    stats
  );
  impl->ivf_index = std::make_unique<NativeCudaIVFIndex>(std::move(index));
  const int metric_code = metric == DistanceMetric::Euclidean ? 0 :
    (metric == DistanceMetric::Cosine ? 1 : 2);
  const int threads = 256;
  const int blocks = static_cast<int>((items + threads - 1) / threads);
  convert_native_graph_distances_kernel<<<blocks, threads>>>(
    impl->distances,
    items,
    metric_code
  );
  check_cuda(cudaGetLastError(), "convert resident IVF graph distances");
  check_cuda(cudaDeviceSynchronize(), "finish resident IVF KODAMA graph");
  check_cuda(cudaMemcpy(impl->base_indices, impl->indices, items * sizeof(int),
                        cudaMemcpyDeviceToDevice), "retain immutable resident IVF graph indices");
  check_cuda(cudaMemcpy(impl->base_distances, impl->distances, items * sizeof(float),
                        cudaMemcpyDeviceToDevice), "retain immutable resident IVF graph distances");

  impl->lane_buffers.resize(static_cast<std::size_t>(std::max(1, lanes)));
  for (CudaResidentKODAMAGraph::Impl::Lane& lane : impl->lane_buffers) {
    check_cuda(cudaStreamCreateWithFlags(&lane.stream, cudaStreamNonBlocking), "create resident IVF graph stream");
    check_cuda(cudaMalloc(&lane.landmark_epoch, static_cast<std::size_t>(samples) * sizeof(int)), "allocate resident IVF landmark epochs");
    check_cuda(cudaMalloc(&lane.labels, static_cast<std::size_t>(samples) * sizeof(int)), "allocate resident IVF labels");
    check_cuda(cudaMalloc(&lane.constrain, static_cast<std::size_t>(samples) * sizeof(int)), "allocate resident IVF constraints");
    check_cuda(cudaMemset(lane.landmark_epoch, 0, static_cast<std::size_t>(samples) * sizeof(int)), "clear resident IVF landmark epochs");
  }
  return CudaResidentKODAMAGraph(std::move(impl));
}

NeighborGraph cuda_resident_landmark_knn_graph(
  const CudaResidentKODAMAGraph& graph,
  const std::vector<float>& landmark_data,
  const std::vector<int>& landmark_rows,
  int k,
  int requested_nprobe,
  double target_recall
) {
  if (!graph.valid() || graph.impl_->ivf_index == nullptr) {
    throw std::invalid_argument("CUDA resident KODAMA graph has no reusable IVF index.");
  }
  const CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  if (landmark_rows.size() < 2 ||
      landmark_data.size() != landmark_rows.size() * static_cast<std::size_t>(impl.dimensions)) {
    throw std::invalid_argument("Invalid CUDA resident landmark query matrix.");
  }
  std::vector<int> allowed_local_ids(static_cast<std::size_t>(impl.samples), -1);
  for (std::size_t local = 0; local < landmark_rows.size(); ++local) {
    const int global = landmark_rows[local];
    if (global < 0 || global >= impl.samples) {
      throw std::out_of_range("CUDA landmark row is outside the resident index.");
    }
    allowed_local_ids[static_cast<std::size_t>(global)] = static_cast<int>(local);
  }
  std::vector<float> prepared_landmarks = landmark_data;
  if (impl.metric == DistanceMetric::Cosine) {
    for (std::size_t row = 0; row < landmark_rows.size(); ++row) {
      float norm2 = 0.0f;
      float* values = prepared_landmarks.data() + row * static_cast<std::size_t>(impl.dimensions);
      for (int d = 0; d < impl.dimensions; ++d) norm2 += values[d] * values[d];
      if (norm2 > 0.0f && std::isfinite(norm2)) {
        const float scale = 1.0f / std::sqrt(norm2);
        for (int d = 0; d < impl.dimensions; ++d) values[d] *= scale;
      }
    }
  }
  NativeCudaIVFStats stats;
  NativeKNNResult result = native_cuda_ivf_index_filtered_search(
    *impl.ivf_index,
    prepared_landmarks,
    static_cast<int>(landmark_rows.size()),
    std::min(k, static_cast<int>(landmark_rows.size()) - 1),
    requested_nprobe,
    target_recall,
    landmark_rows,
    allowed_local_ids,
    &stats
  );
  NeighborGraph output;
  output.neighbors = result.neighbors;
  output.index_base = GraphIndexBase::Zero;
  output.indices = std::move(result.indices);
  output.distances.resize(result.distances.size());
  for (std::size_t i = 0; i < result.distances.size(); ++i) {
    output.distances[i] = native_knn_output_distance(result.distances[i], impl.metric);
  }
  return output;
}

void cuda_resident_prepare_results(
  CudaResidentKODAMAGraph& graph,
  int runs,
  int lanes
) {
  if (!graph.valid() || runs < 1 || lanes < 1) {
    throw std::invalid_argument("Invalid CUDA resident KODAMA result matrix.");
  }
  CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  check_cuda(cudaSetDevice(impl.device), "cudaSetDevice resident KODAMA results");
  const std::size_t prior_lanes = impl.lane_buffers.size();
  if (prior_lanes < static_cast<std::size_t>(lanes)) {
    impl.lane_buffers.resize(static_cast<std::size_t>(lanes));
    for (std::size_t lane_id = prior_lanes; lane_id < impl.lane_buffers.size(); ++lane_id) {
      CudaResidentKODAMAGraph::Impl::Lane& lane = impl.lane_buffers[lane_id];
      check_cuda(
        cudaStreamCreateWithFlags(&lane.stream, cudaStreamNonBlocking),
        "create reusable resident KODAMA graph stream"
      );
      check_cuda(
        cudaMalloc(
          &lane.landmark_epoch,
          static_cast<std::size_t>(impl.samples) * sizeof(int)
        ),
        "allocate reusable resident KODAMA landmark epochs"
      );
      check_cuda(
        cudaMalloc(&lane.labels, static_cast<std::size_t>(impl.samples) * sizeof(int)),
        "allocate reusable resident KODAMA labels"
      );
      check_cuda(
        cudaMalloc(&lane.constrain, static_cast<std::size_t>(impl.samples) * sizeof(int)),
        "allocate reusable resident KODAMA constraints"
      );
    }
  }
  const std::size_t required = static_cast<std::size_t>(runs) * impl.samples;
  if (impl.result_capacity < required) {
    cudaFree(impl.result_labels);
    impl.result_labels = nullptr;
    check_cuda(cudaMalloc(&impl.result_labels, required * sizeof(int)),
               "allocate resident KODAMA result labels");
    impl.result_capacity = required;
  }
  for (CudaResidentKODAMAGraph::Impl::Lane& lane : impl.lane_buffers) {
    check_cuda(
      cudaMemset(
        lane.landmark_epoch,
        0,
        static_cast<std::size_t>(impl.samples) * sizeof(int)
      ),
      "reset resident KODAMA landmark epochs"
    );
  }
  impl.result_runs = runs;
}

void cuda_resident_project_landmark_labels_to_result(
  CudaResidentKODAMAGraph& graph,
  const std::vector<int>& landmark_rows,
  const std::vector<int>& landmark_labels,
  int projection_k,
  int fallback_label,
  int run,
  int lane
) {
  if (!graph.valid()) {
    throw std::invalid_argument("CUDA resident KODAMA graph is empty.");
  }
  CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  if (landmark_rows.empty() || landmark_rows.size() != landmark_labels.size() ||
      run < 0 || run >= impl.result_runs || impl.result_labels == nullptr) {
    throw std::invalid_argument(
      "CUDA resident KODAMA projection input size mismatch."
    );
  }
  check_cuda(cudaSetDevice(impl.device), "cudaSetDevice resident KODAMA projection");
  const int lane_id =
    std::max(0, lane) % static_cast<int>(impl.lane_buffers.size());
  CudaResidentKODAMAGraph::Impl::Lane& workspace =
    impl.lane_buffers[static_cast<std::size_t>(lane_id)];
  if (workspace.landmark_capacity < landmark_rows.size()) {
    cudaFree(workspace.landmark_rows);
    cudaFree(workspace.landmark_input_labels);
    check_cuda(cudaMalloc(&workspace.landmark_rows, landmark_rows.size() * sizeof(int)),
               "allocate sparse resident KODAMA landmark rows");
    check_cuda(cudaMalloc(&workspace.landmark_input_labels, landmark_rows.size() * sizeof(int)),
               "allocate sparse resident KODAMA landmark labels");
    workspace.landmark_capacity = landmark_rows.size();
  }
  check_cuda(
    cudaMemcpyAsync(
      workspace.landmark_rows,
      landmark_rows.data(),
      landmark_rows.size() * sizeof(int),
      cudaMemcpyHostToDevice,
      workspace.stream
    ),
    "upload sparse resident KODAMA landmark rows"
  );
  check_cuda(
    cudaMemcpyAsync(
      workspace.landmark_input_labels,
      landmark_labels.data(),
      landmark_labels.size() * sizeof(int),
      cudaMemcpyHostToDevice,
      workspace.stream
    ),
    "upload sparse resident KODAMA landmark labels"
  );
  const int threads = 128;
  const int epoch = run + 1;
  const int landmark_blocks =
    (static_cast<int>(landmark_rows.size()) + threads - 1) / threads;
  scatter_landmark_labels_kernel<<<landmark_blocks, threads, 0, workspace.stream>>>(
    workspace.landmark_rows,
    workspace.landmark_input_labels,
    workspace.landmark_epoch,
    workspace.labels,
    static_cast<int>(landmark_rows.size()),
    epoch,
    impl.samples
  );
  check_cuda(cudaGetLastError(), "launch sparse resident KODAMA landmark scatter");
  const int blocks = (impl.samples + threads - 1) / threads;
  project_landmark_labels_kernel<<<blocks, threads, 0, workspace.stream>>>(
    impl.indices,
    workspace.landmark_epoch,
    workspace.labels,
    impl.result_labels + static_cast<std::size_t>(run) * impl.samples,
    impl.samples,
    impl.neighbors,
    std::max(1, std::min(projection_k, impl.neighbors)),
    fallback_label,
    epoch
  );
  check_cuda(cudaGetLastError(), "launch resident KODAMA label projection");
  check_cuda(
    cudaStreamSynchronize(workspace.stream),
    "synchronize resident KODAMA label projection"
  );
}

void cuda_resident_store_result_row(
  CudaResidentKODAMAGraph& graph,
  const std::vector<int>& labels,
  int run,
  int lane
) {
  if (!graph.valid() || labels.size() != static_cast<std::size_t>(graph.impl_->samples) ||
      run < 0 || run >= graph.impl_->result_runs) {
    throw std::invalid_argument("CUDA resident KODAMA result row size mismatch.");
  }
  CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  auto& workspace = impl.lane_buffers[static_cast<std::size_t>(
    std::max(0, lane) % static_cast<int>(impl.lane_buffers.size()))];
  check_cuda(cudaMemcpyAsync(
    impl.result_labels + static_cast<std::size_t>(run) * impl.samples,
    labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice, workspace.stream),
    "upload resident KODAMA result row");
  check_cuda(cudaStreamSynchronize(workspace.stream), "synchronize resident KODAMA result row");
}

void cuda_resident_constrain_result_row(
  CudaResidentKODAMAGraph& graph,
  const std::vector<int>& constrain,
  int max_label,
  int run,
  int lane
) {
  if (!graph.valid() || constrain.size() != static_cast<std::size_t>(graph.impl_->samples) ||
      max_label < 0 || run < 0 || run >= graph.impl_->result_runs) {
    throw std::invalid_argument("CUDA resident KODAMA constraint input mismatch.");
  }
  CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  auto& workspace = impl.lane_buffers[static_cast<std::size_t>(
    std::max(0, lane) % static_cast<int>(impl.lane_buffers.size()))];
  std::unordered_map<int, int> group_map;
  group_map.reserve(constrain.size() / 4 + 1);
  std::vector<int> compact(constrain.size());
  for (std::size_t i = 0; i < constrain.size(); ++i) {
    const auto inserted = group_map.emplace(
      constrain[i], static_cast<int>(group_map.size()));
    compact[i] = inserted.first->second;
  }
  const int groups = static_cast<int>(group_map.size());
  const int label_width = max_label + 1;
  const std::size_t count_items = static_cast<std::size_t>(groups) * label_width;
  if (workspace.constrain_count_capacity < count_items) {
    cudaFree(workspace.constrain_counts);
    check_cuda(cudaMalloc(&workspace.constrain_counts, count_items * sizeof(int)),
               "allocate resident KODAMA constraint counts");
    workspace.constrain_count_capacity = count_items;
  }
  if (workspace.constrain_group_capacity < static_cast<std::size_t>(groups)) {
    cudaFree(workspace.constrain_labels);
    check_cuda(cudaMalloc(&workspace.constrain_labels,
                          static_cast<std::size_t>(groups) * sizeof(int)),
               "allocate resident KODAMA constraint labels");
    workspace.constrain_group_capacity = static_cast<std::size_t>(groups);
  }
  check_cuda(cudaMemcpyAsync(workspace.constrain, compact.data(), compact.size() * sizeof(int),
                             cudaMemcpyHostToDevice, workspace.stream),
             "upload resident KODAMA compact constraints");
  check_cuda(cudaMemsetAsync(workspace.constrain_counts, 0,
                             count_items * sizeof(int), workspace.stream),
             "clear resident KODAMA constraint counts");
  const int threads = 256;
  int* row = impl.result_labels + static_cast<std::size_t>(run) * impl.samples;
  count_constrained_labels_kernel<<<(impl.samples + threads - 1) / threads,
                                     threads, 0, workspace.stream>>>(
    row, workspace.constrain, workspace.constrain_counts,
    impl.samples, label_width);
  select_constrained_majority_kernel<<<(groups + threads - 1) / threads,
                                       threads, 0, workspace.stream>>>(
    workspace.constrain_counts, workspace.constrain_labels,
    groups, label_width);
  apply_constrained_majority_kernel<<<(impl.samples + threads - 1) / threads,
                                      threads, 0, workspace.stream>>>(
    row, workspace.constrain, workspace.constrain_labels, impl.samples);
  check_cuda(cudaGetLastError(), "launch resident KODAMA constrained majority");
  check_cuda(cudaStreamSynchronize(workspace.stream),
             "synchronize resident KODAMA constrained majority");
}

std::vector<int> cuda_resident_download_results(
  const CudaResidentKODAMAGraph& graph,
  int runs
) {
  if (!graph.valid() || runs != graph.impl_->result_runs || graph.impl_->result_labels == nullptr) {
    throw std::invalid_argument("CUDA resident KODAMA results are unavailable.");
  }
  CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  for (auto& lane : impl.lane_buffers) check_cuda(cudaStreamSynchronize(lane.stream), "finish resident KODAMA results");
  std::vector<int> out(static_cast<std::size_t>(runs) * impl.samples);
  check_cuda(cudaMemcpy(out.data(), impl.result_labels, out.size() * sizeof(int), cudaMemcpyDeviceToHost),
             "download resident KODAMA results");
  return out;
}

std::vector<int> cuda_resident_download_result_row(
  const CudaResidentKODAMAGraph& graph,
  int run,
  int lane
) {
  if (!graph.valid() || run < 0 || run >= graph.impl_->result_runs) {
    throw std::invalid_argument("CUDA resident KODAMA result row is unavailable.");
  }
  CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  auto& workspace = impl.lane_buffers[static_cast<std::size_t>(
    std::max(0, lane) % static_cast<int>(impl.lane_buffers.size()))];
  std::vector<int> out(static_cast<std::size_t>(impl.samples));
  check_cuda(cudaMemcpyAsync(
    out.data(), impl.result_labels + static_cast<std::size_t>(run) * impl.samples,
    out.size() * sizeof(int), cudaMemcpyDeviceToHost, workspace.stream),
    "download resident KODAMA result row");
  check_cuda(cudaStreamSynchronize(workspace.stream), "synchronize resident KODAMA result row download");
  return out;
}

void cuda_resident_apply_kodama_dissimilarity(
  CudaResidentKODAMAGraph& graph,
  int runs,
  bool input_one_based_indices,
  bool output_one_based_indices
) {
  if (!graph.valid()) {
    throw std::invalid_argument("CUDA resident KODAMA graph is empty.");
  }
  CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  if (runs < 1 || runs != impl.result_runs || impl.result_labels == nullptr) {
    throw std::invalid_argument(
      "CUDA resident KODAMA dissimilarity label size mismatch."
    );
  }
  if (impl.neighbors > kMaxGraphNeighbors) {
    throw std::invalid_argument(
      "CUDA KODAMA dissimilarity supports at most 1024 graph neighbors."
    );
  }
  check_cuda(cudaSetDevice(impl.device), "cudaSetDevice resident KODAMA dissimilarity");
  for (auto& lane : impl.lane_buffers) {
    check_cuda(cudaStreamSynchronize(lane.stream), "finish resident KODAMA result rows");
  }
  int sort_width = 1;
  while (sort_width < impl.neighbors) sort_width <<= 1;
  const std::size_t shared_bytes =
    static_cast<std::size_t>(sort_width) * (sizeof(float) + sizeof(int));
  kodama_dissimilarity_shared_kernel<<<impl.samples, sort_width, shared_bytes>>>(
    impl.indices,
    impl.distances,
    impl.result_labels,
    runs,
    impl.samples,
    impl.neighbors,
    sort_width,
    input_one_based_indices,
    output_one_based_indices
  );
  check_cuda(cudaGetLastError(), "launch resident KODAMA dissimilarity");
  check_cuda(
    cudaDeviceSynchronize(),
    "synchronize resident KODAMA dissimilarity"
  );
}

NeighborGraph download_cuda_resident_kodama_graph(
  const CudaResidentKODAMAGraph& graph
) {
  if (!graph.valid()) {
    throw std::invalid_argument("CUDA resident KODAMA graph is empty.");
  }
  const CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  check_cuda(cudaSetDevice(impl.device), "cudaSetDevice download resident KODAMA graph");
  NeighborGraph out;
  out.neighbors = impl.neighbors;
  const std::size_t items =
    static_cast<std::size_t>(impl.samples) *
    static_cast<std::size_t>(impl.neighbors);
  out.indices.resize(items);
  out.distances.resize(items);
  check_cuda(
    cudaMemcpy(
      out.indices.data(),
      impl.indices,
      items * sizeof(int),
      cudaMemcpyDeviceToHost
    ),
    "download resident KODAMA graph indices"
  );
  check_cuda(
    cudaMemcpy(
      out.distances.data(),
      impl.distances,
      items * sizeof(float),
      cudaMemcpyDeviceToHost
    ),
    "download resident KODAMA graph distances"
  );
  return out;
}

void cuda_resident_replace_graph(
  CudaResidentKODAMAGraph& graph,
  const NeighborGraph& replacement
) {
  if (!graph.valid()) throw std::invalid_argument("CUDA resident graph is empty.");
  CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  const std::size_t items = static_cast<std::size_t>(impl.samples) * impl.neighbors;
  if (replacement.neighbors != impl.neighbors || replacement.indices.size() != items ||
      replacement.distances.size() != items) {
    throw std::invalid_argument("CUDA resident replacement graph dimensions mismatch.");
  }
  check_cuda(cudaSetDevice(impl.device), "cudaSetDevice resident graph replacement");
  check_cuda(cudaMemcpy(impl.indices, replacement.indices.data(), items * sizeof(int),
                        cudaMemcpyHostToDevice), "replace resident graph indices");
  check_cuda(cudaMemcpy(impl.distances, replacement.distances.data(), items * sizeof(float),
                        cudaMemcpyHostToDevice), "replace resident graph distances");
  check_cuda(cudaMemcpy(impl.base_indices, impl.indices, items * sizeof(int),
                        cudaMemcpyDeviceToDevice), "retain replacement graph indices");
  check_cuda(cudaMemcpy(impl.base_distances, impl.distances, items * sizeof(float),
                        cudaMemcpyDeviceToDevice), "retain replacement graph distances");
}

void cuda_resident_reset_graph(CudaResidentKODAMAGraph& graph) {
  if (!graph.valid()) throw std::invalid_argument("CUDA resident graph is empty.");
  CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  const std::size_t items = static_cast<std::size_t>(impl.samples) * impl.neighbors;
  check_cuda(cudaSetDevice(impl.device), "cudaSetDevice resident graph reset");
  check_cuda(cudaMemcpy(impl.indices, impl.base_indices, items * sizeof(int),
                        cudaMemcpyDeviceToDevice), "reset resident graph indices");
  check_cuda(cudaMemcpy(impl.distances, impl.base_distances, items * sizeof(float),
                        cudaMemcpyDeviceToDevice), "reset resident graph distances");
}

void apply_kodama_dissimilarity_cuda(
  NeighborGraph& graph,
  const std::vector<int>& res,
  int runs,
  int samples,
  int gpu_device,
  bool input_one_based_indices,
  bool output_one_based_indices
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

  int sort_width = 1;
  while (sort_width < graph.neighbors) sort_width <<= 1;
  const int threads = sort_width;
  const int blocks = samples;
  const std::size_t shared_bytes =
    static_cast<std::size_t>(sort_width) * (sizeof(float) + sizeof(int));
  kodama_dissimilarity_shared_kernel<<<blocks, threads, shared_bytes>>>(
    device_indices,
    device_distances,
    device_res,
    runs,
    samples,
    graph.neighbors,
    sort_width,
    input_one_based_indices,
    output_one_based_indices
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

NeighborGraph spatial_grid_self_knn_cuda(
  const std::vector<float>& data,
  int n,
  int dims,
  int neighbors,
  int gpu_device,
  bool one_based_indices,
  bool include_self
) {
  if (n < 2 || (dims != 2 && dims != 3)) {
    throw std::invalid_argument("CUDA spatial grid KNN supports only 2D/3D matrices with at least two rows.");
  }
  if (static_cast<std::size_t>(n) * static_cast<std::size_t>(dims) != data.size()) {
    throw std::invalid_argument("CUDA spatial grid KNN data size mismatch.");
  }
  const int k = include_self ?
    std::max(1, std::min(neighbors, n)) :
    std::max(1, std::min(neighbors, n - 1));
  const int nonself_k = include_self ? std::max(0, k - 1) : k;
  if (k > kMaxGridNeighbors) {
    throw std::invalid_argument("CUDA spatial grid KNN supports at most 256 neighbors.");
  }

  const int bins = spatial_grid_bins_per_dim(n, std::max(1, nonself_k), dims);
  const SpatialGridIndex grid = build_spatial_grid_index(data.data(), n, dims, bins);

  NeighborGraph graph;
  graph.neighbors = k;
  graph.indices.assign(static_cast<std::size_t>(n) * k, -1);
  graph.distances.assign(static_cast<std::size_t>(n) * k, std::numeric_limits<float>::infinity());

  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice");
  float* device_data = nullptr;
  int* device_offsets = nullptr;
  int* device_rows = nullptr;
  int* device_indices = nullptr;
  float* device_distances = nullptr;

  try {
    device_data = device_alloc_copy(data, "copy CUDA spatial grid data");
    device_offsets = device_alloc_copy(grid.offsets, "copy CUDA spatial grid offsets");
    device_rows = device_alloc_copy(grid.rows, "copy CUDA spatial grid rows");
    check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&device_indices), graph.indices.size() * sizeof(int)),
      "allocate CUDA spatial grid indices"
    );
    check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&device_distances), graph.distances.size() * sizeof(float)),
      "allocate CUDA spatial grid distances"
    );

    CudaSpatialGridParams params;
    params.n = n;
    params.dims = dims;
    params.k = k;
    params.nonself_k = nonself_k;
    params.bins = grid.bins;
    params.include_self = include_self ? 1 : 0;
    params.one_based = one_based_indices ? 1 : 0;
    params.sort_width = 1;
    while (params.sort_width < 2 * std::max(1, nonself_k)) {
      params.sort_width <<= 1;
    }
    params.min_x = grid.min_x;
    params.min_y = grid.min_y;
    params.min_z = grid.min_z;
    params.cell_x = grid.cell_x;
    params.cell_y = grid.cell_y;
    params.cell_z = grid.cell_z;

    constexpr int threads = 128;
    constexpr int warps_per_block = threads / 32;
    const int blocks = (n + warps_per_block - 1) / warps_per_block;
    const std::size_t shared_bytes =
      static_cast<std::size_t>(warps_per_block) * params.sort_width *
      (sizeof(float) + sizeof(int));
    spatial_grid_self_knn_kernel<<<blocks, threads, shared_bytes>>>(
      device_data,
      device_offsets,
      device_rows,
      device_indices,
      device_distances,
      params
    );
    check_cuda(cudaGetLastError(), "launch CUDA spatial grid KNN kernel");
    check_cuda(cudaDeviceSynchronize(), "run CUDA spatial grid KNN kernel");

    check_cuda(
      cudaMemcpy(graph.indices.data(), device_indices, graph.indices.size() * sizeof(int), cudaMemcpyDeviceToHost),
      "copy CUDA spatial grid indices"
    );
    check_cuda(
      cudaMemcpy(graph.distances.data(), device_distances, graph.distances.size() * sizeof(float), cudaMemcpyDeviceToHost),
      "copy CUDA spatial grid distances"
    );
  } catch (...) {
    cudaFree(device_data);
    cudaFree(device_offsets);
    cudaFree(device_rows);
    cudaFree(device_indices);
    cudaFree(device_distances);
    throw;
  }

  cudaFree(device_data);
  cudaFree(device_offsets);
  cudaFree(device_rows);
  cudaFree(device_indices);
  cudaFree(device_distances);
  return graph;
}

}  // namespace kodama::detail
