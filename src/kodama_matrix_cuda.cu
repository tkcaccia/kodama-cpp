// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama_matrix_cuda.hpp"
#include "spatial_grid_knn.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <climits>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace kodama::detail {
namespace {

constexpr int kMaxGraphNeighbors = 1024;
constexpr int kMaxGridNeighbors = 256;
constexpr float kCudaInfinity = INFINITY;

struct CudaSpatialGridParams {
  int n;
  int dims;
  int k;
  int nonself_k;
  int bins;
  int include_self;
  int one_based;
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

__global__ void project_landmark_labels_kernel(
  const int* graph_indices,
  const char* is_landmark,
  const int* labels,
  int* projected,
  int samples,
  int neighbors,
  int projection_k,
  int fallback_label
) {
  const int query = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (query >= samples) return;
  if (is_landmark[query] != 0) {
    projected[query] = labels[query];
    return;
  }

  const std::size_t base =
    static_cast<std::size_t>(query) * static_cast<std::size_t>(neighbors);
  int cutoff = neighbors;
  int accepted = 0;
  for (int rank = 0; rank < neighbors; ++rank) {
    const int neighbor = graph_indices[base + static_cast<std::size_t>(rank)];
    if (neighbor < 0 || neighbor >= samples || is_landmark[neighbor] == 0) continue;
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
    if (neighbor < 0 || neighbor >= samples || is_landmark[neighbor] == 0) continue;
    const int candidate = labels[neighbor];
    bool seen = false;
    for (int prior = 0; prior < rank; ++prior) {
      const int prior_neighbor =
        graph_indices[base + static_cast<std::size_t>(prior)];
      if (prior_neighbor >= 0 && prior_neighbor < samples &&
          is_landmark[prior_neighbor] != 0 &&
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
          is_landmark[other_neighbor] != 0 &&
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

__device__ bool kodama_pair_greater(float lhs_dist, int lhs_idx, float rhs_dist, int rhs_idx) {
  return lhs_dist > rhs_dist || (lhs_dist == rhs_dist && lhs_idx > rhs_idx);
}

__device__ void grid_insert_candidate_device(float dist, int idx, float* best_dist, int* best_idx, int k) {
  if (k <= 0) return;
  if (dist > best_dist[k - 1] || (dist == best_dist[k - 1] && idx >= best_idx[k - 1])) return;
  int pos = k - 1;
  while (pos > 0 && (dist < best_dist[pos - 1] || (dist == best_dist[pos - 1] && idx < best_idx[pos - 1]))) {
    best_dist[pos] = best_dist[pos - 1];
    best_idx[pos] = best_idx[pos - 1];
    --pos;
  }
  best_dist[pos] = dist;
  best_idx[pos] = idx;
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

__device__ void add_grid_cell_candidates_device(
  const float* data,
  const int* offsets,
  const int* rows,
  const CudaSpatialGridParams params,
  int query,
  int ix,
  int iy,
  int iz,
  float* best_dist,
  int* best_idx
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
  for (int pos = start; pos < end; ++pos) {
    const int candidate = rows[pos];
    if (candidate == query) continue;
    const std::size_t cbase = static_cast<std::size_t>(candidate) * params.dims;
    const float dx = qx - data[cbase];
    const float dy = qy - data[cbase + 1];
    float dist = dx * dx + dy * dy;
    if (params.dims == 3) {
      const float dz = qz - data[cbase + 2];
      dist += dz * dz;
    }
    grid_insert_candidate_device(dist, candidate, best_dist, best_idx, params.nonself_k);
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
  const int q = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (q >= params.n) return;

  float best_dist[kMaxGridNeighbors];
  int best_idx[kMaxGridNeighbors];
  for (int j = 0; j < params.nonself_k; ++j) {
    best_dist[j] = kCudaInfinity;
    best_idx[j] = INT_MAX;
  }

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
          add_grid_cell_candidates_device(data, offsets, rows, params, q, cx, cy, 0, best_dist, best_idx);
        } else {
          for (int ix = raw_x0; ix <= raw_x1; ++ix) {
            if (raw_y0 >= 0 && raw_y0 < params.bins) {
              add_grid_cell_candidates_device(data, offsets, rows, params, q, ix, raw_y0, 0, best_dist, best_idx);
            }
            if (raw_y1 != raw_y0 && raw_y1 >= 0 && raw_y1 < params.bins) {
              add_grid_cell_candidates_device(data, offsets, rows, params, q, ix, raw_y1, 0, best_dist, best_idx);
            }
          }
          for (int iy = raw_y0 + 1; iy <= raw_y1 - 1; ++iy) {
            if (raw_x0 >= 0 && raw_x0 < params.bins) {
              add_grid_cell_candidates_device(data, offsets, rows, params, q, raw_x0, iy, 0, best_dist, best_idx);
            }
            if (raw_x1 != raw_x0 && raw_x1 >= 0 && raw_x1 < params.bins) {
              add_grid_cell_candidates_device(data, offsets, rows, params, q, raw_x1, iy, 0, best_dist, best_idx);
            }
          }
        }
      } else {
        if (radius == 0) {
          add_grid_cell_candidates_device(data, offsets, rows, params, q, cx, cy, cz, best_dist, best_idx);
        } else {
          for (int iz = raw_z0; iz <= raw_z1; ++iz) {
            if (iz < 0 || iz >= params.bins) continue;
            for (int iy = raw_y0; iy <= raw_y1; ++iy) {
              if (iy < 0 || iy >= params.bins) continue;
              for (int ix = raw_x0; ix <= raw_x1; ++ix) {
                if (ix < 0 || ix >= params.bins) continue;
                if (ix != raw_x0 && ix != raw_x1 && iy != raw_y0 && iy != raw_y1 && iz != raw_z0 && iz != raw_z1) continue;
                add_grid_cell_candidates_device(data, offsets, rows, params, q, ix, iy, iz, best_dist, best_idx);
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

  int out_col = 0;
  if (params.include_self) {
    const std::size_t offset = static_cast<std::size_t>(q) * params.k;
    out_idx[offset] = q + (params.one_based ? 1 : 0);
    out_dist[offset] = 0.0f;
    out_col = 1;
  }
  for (int j = 0; out_col < params.k && j < params.nonself_k; ++j, ++out_col) {
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
    char* is_landmark = nullptr;
    int* labels = nullptr;
    int* projected = nullptr;
    cudaStream_t stream = nullptr;
  };

  int samples = 0;
  int neighbors = 0;
  int device = 0;
  int* indices = nullptr;
  float* distances = nullptr;
  int* result_labels = nullptr;
  std::size_t result_capacity = 0;
  std::vector<Lane> lane_buffers;

  ~Impl() {
    cudaSetDevice(device);
    for (Lane& lane : lane_buffers) {
      if (lane.stream != nullptr) cudaStreamSynchronize(lane.stream);
      cudaFree(lane.is_landmark);
      cudaFree(lane.labels);
      cudaFree(lane.projected);
      if (lane.stream != nullptr) cudaStreamDestroy(lane.stream);
    }
    cudaFree(indices);
    cudaFree(distances);
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
  impl->lane_buffers.resize(static_cast<std::size_t>(std::max(1, lanes)));
  for (CudaResidentKODAMAGraph::Impl::Lane& lane : impl->lane_buffers) {
    check_cuda(
      cudaStreamCreateWithFlags(&lane.stream, cudaStreamNonBlocking),
      "create resident KODAMA graph stream"
    );
    check_cuda(
      cudaMalloc(&lane.is_landmark, static_cast<std::size_t>(samples) * sizeof(char)),
      "allocate resident KODAMA landmark mask"
    );
    check_cuda(
      cudaMalloc(&lane.labels, static_cast<std::size_t>(samples) * sizeof(int)),
      "allocate resident KODAMA labels"
    );
    check_cuda(
      cudaMalloc(&lane.projected, static_cast<std::size_t>(samples) * sizeof(int)),
      "allocate resident KODAMA projected labels"
    );
  }
  return CudaResidentKODAMAGraph(std::move(impl));
}

std::vector<int> cuda_resident_project_landmark_labels(
  const CudaResidentKODAMAGraph& graph,
  const std::vector<char>& is_landmark,
  const std::vector<int>& labels,
  int projection_k,
  int fallback_label,
  int lane
) {
  if (!graph.valid()) {
    throw std::invalid_argument("CUDA resident KODAMA graph is empty.");
  }
  CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  if (is_landmark.size() != static_cast<std::size_t>(impl.samples) ||
      labels.size() != static_cast<std::size_t>(impl.samples)) {
    throw std::invalid_argument(
      "CUDA resident KODAMA projection input size mismatch."
    );
  }
  check_cuda(cudaSetDevice(impl.device), "cudaSetDevice resident KODAMA projection");
  const int lane_id =
    std::max(0, lane) % static_cast<int>(impl.lane_buffers.size());
  CudaResidentKODAMAGraph::Impl::Lane& workspace =
    impl.lane_buffers[static_cast<std::size_t>(lane_id)];
  check_cuda(
    cudaMemcpyAsync(
      workspace.is_landmark,
      is_landmark.data(),
      is_landmark.size() * sizeof(char),
      cudaMemcpyHostToDevice,
      workspace.stream
    ),
    "upload resident KODAMA landmark mask"
  );
  check_cuda(
    cudaMemcpyAsync(
      workspace.labels,
      labels.data(),
      labels.size() * sizeof(int),
      cudaMemcpyHostToDevice,
      workspace.stream
    ),
    "upload resident KODAMA landmark labels"
  );
  const int threads = 128;
  const int blocks = (impl.samples + threads - 1) / threads;
  project_landmark_labels_kernel<<<blocks, threads, 0, workspace.stream>>>(
    impl.indices,
    workspace.is_landmark,
    workspace.labels,
    workspace.projected,
    impl.samples,
    impl.neighbors,
    std::max(1, std::min(projection_k, impl.neighbors)),
    fallback_label
  );
  check_cuda(cudaGetLastError(), "launch resident KODAMA label projection");
  std::vector<int> projected(static_cast<std::size_t>(impl.samples));
  check_cuda(
    cudaMemcpyAsync(
      projected.data(),
      workspace.projected,
      projected.size() * sizeof(int),
      cudaMemcpyDeviceToHost,
      workspace.stream
    ),
    "download resident KODAMA projected labels"
  );
  check_cuda(
    cudaStreamSynchronize(workspace.stream),
    "synchronize resident KODAMA label projection"
  );
  return projected;
}

void cuda_resident_apply_kodama_dissimilarity(
  CudaResidentKODAMAGraph& graph,
  const std::vector<int>& res,
  int runs,
  bool input_one_based_indices,
  bool output_one_based_indices
) {
  if (!graph.valid()) {
    throw std::invalid_argument("CUDA resident KODAMA graph is empty.");
  }
  CudaResidentKODAMAGraph::Impl& impl = *graph.impl_;
  if (runs < 1 ||
      res.size() !=
        static_cast<std::size_t>(runs) *
        static_cast<std::size_t>(impl.samples)) {
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
  if (impl.result_capacity < res.size()) {
    cudaFree(impl.result_labels);
    impl.result_labels = nullptr;
    check_cuda(
      cudaMalloc(&impl.result_labels, res.size() * sizeof(int)),
      "allocate resident KODAMA result labels"
    );
    impl.result_capacity = res.size();
  }
  check_cuda(
    cudaMemcpy(
      impl.result_labels,
      res.data(),
      res.size() * sizeof(int),
      cudaMemcpyHostToDevice
    ),
    "upload resident KODAMA result labels"
  );
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
    params.min_x = grid.min_x;
    params.min_y = grid.min_y;
    params.min_z = grid.min_z;
    params.cell_x = grid.cell_x;
    params.cell_y = grid.cell_y;
    params.cell_z = grid.cell_z;

    constexpr int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    spatial_grid_self_knn_kernel<<<blocks, threads>>>(
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
