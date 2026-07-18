/*
 * SPDX-FileCopyrightText: Meta Platforms, Inc. and affiliates
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Package-owned float32 CUDA nearest-neighbor and k-means primitives.
 *
 * The exact/IVF-Flat organization is informed by FAISS 1.14.3 (MIT), RAPIDS
 * cuVS (Apache-2.0), and the native Metal implementation in fastEmbedR. The
 * k-means initialization, seed convention, Lloyd iteration semantics, and
 * empty-cluster repair are adapted from FAISS 1.14.3 clustering and random
 * utilities. No FAISS, cuVS, RAFT, or RMM binary is linked, and no cuVS source
 * is included. This standalone adaptation is distributed under the MIT
 * License; retain the FAISS notice and license with redistributed derivatives.
 */

#include "native_cuda_backend.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kodama::detail {
namespace {

constexpr int kCudaThreads = 128;
constexpr int kCudaWarps = kCudaThreads / 32;
constexpr int kMaximumCudaK = 256;
constexpr int kMaximumCudaProbe = 256;
constexpr int kMaximumCudaLists = 4096;
constexpr int kCudaProjectionDimension = 128;
// Each independent M lane reuses this bounded exact-assignment workspace.
constexpr std::size_t kKMeansScoreWorkspaceBytes = 256ull * 1024ull * 1024ull;

void cuda_check(cudaError_t status, const char* context) {
  if (status == cudaSuccess) return;
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

void cublas_check(cublasStatus_t status, const char* context) {
  if (status == CUBLAS_STATUS_SUCCESS) return;
  throw std::runtime_error(std::string(context) + " failed with cuBLAS status " + std::to_string(status));
}

class CudaDeviceScope {
 public:
  explicit CudaDeviceScope(int device) {
    cuda_check(cudaGetDevice(&previous_), "cudaGetDevice");
    cuda_check(cudaSetDevice(device), "cudaSetDevice");
  }

  ~CudaDeviceScope() {
    cudaSetDevice(previous_);
  }

  CudaDeviceScope(const CudaDeviceScope&) = delete;
  CudaDeviceScope& operator=(const CudaDeviceScope&) = delete;

 private:
  int previous_ = 0;
};

template <typename T>
class CudaBuffer {
 public:
  CudaBuffer() = default;
  explicit CudaBuffer(std::size_t items) { allocate(items); }
  ~CudaBuffer() {
    if (data_ != nullptr) cudaFree(data_);
  }

  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  CudaBuffer(CudaBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  CudaBuffer& operator=(CudaBuffer&& other) noexcept {
    if (this == &other) return *this;
    if (data_ != nullptr) cudaFree(data_);
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  void allocate(std::size_t items) {
    if (data_ != nullptr) cudaFree(data_);
    data_ = nullptr;
    size_ = 0;
    if (items == 0) return;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), items * sizeof(T)), "cudaMalloc");
    size_ = items;
  }

  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return size_; }

  void copy_from_host(const T* source, std::size_t items) {
    if (items > size_) throw std::invalid_argument("CUDA buffer host upload exceeds allocation.");
    if (items == 0) return;
    cuda_check(cudaMemcpy(data_, source, items * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy host to device");
  }

  void copy_to_host(T* destination, std::size_t items) const {
    if (items > size_) throw std::invalid_argument("CUDA buffer host download exceeds allocation.");
    if (items == 0) return;
    cuda_check(cudaMemcpy(destination, data_, items * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy device to host");
  }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

class CublasHandle {
 public:
  CublasHandle() {
    cublas_check(cublasCreate(&handle_), "cublasCreate");
    cublas_check(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
  }

  ~CublasHandle() {
    if (handle_ != nullptr) cublasDestroy(handle_);
  }

  cublasHandle_t get() const { return handle_; }

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

 private:
  cublasHandle_t handle_ = nullptr;
};

__device__ bool better_pair(float distance_a, int id_a, float distance_b, int id_b) {
  return distance_a < distance_b || (distance_a == distance_b && id_a < id_b);
}

__device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ void insert_sorted(
  float* distances,
  int* ids,
  int count,
  float distance,
  int id
) {
  if (count < 1 || !better_pair(distance, id, distances[count - 1], ids[count - 1])) return;
  int position = count - 1;
  while (position > 0 && better_pair(distance, id, distances[position - 1], ids[position - 1])) {
    distances[position] = distances[position - 1];
    ids[position] = ids[position - 1];
    --position;
  }
  distances[position] = distance;
  ids[position] = id;
}

__global__ void exact_topk_kernel(
  const float* train,
  const float* query,
  const int* excluded_train_id,
  int* output_ids,
  float* output_distances,
  int train_rows,
  int query_rows,
  int dimensions,
  int k,
  int metric
) {
  const int query_id = static_cast<int>(blockIdx.x);
  if (query_id >= query_rows) return;
  extern __shared__ unsigned char shared_bytes[];
  float* local_distances = reinterpret_cast<float*>(shared_bytes);
  int* local_ids = reinterpret_cast<int*>(local_distances + kCudaWarps * k);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int lane = thread_id & 31;
  const int warp = thread_id >> 5;
  for (int position = thread_id; position < kCudaWarps * k; position += blockDim.x) {
    local_distances[position] = CUDART_INF_F;
    local_ids[position] = -1;
  }
  __syncthreads();

  const float* query_row = query + static_cast<std::size_t>(query_id) * dimensions;
  const int excluded = excluded_train_id[query_id];
  float* warp_distances = local_distances + warp * k;
  int* warp_ids = local_ids + warp * k;
  for (int candidate = warp; candidate < train_rows; candidate += kCudaWarps) {
    if (candidate == excluded) continue;
    const float* train_row = train + static_cast<std::size_t>(candidate) * dimensions;
    float partial = 0.0f;
    if (metric == 0) {
      for (int dimension = lane; dimension < dimensions; dimension += 32) {
        const float delta = query_row[dimension] - train_row[dimension];
        partial = fmaf(delta, delta, partial);
      }
    } else {
      for (int dimension = lane; dimension < dimensions; dimension += 32) {
        partial = fmaf(query_row[dimension], train_row[dimension], partial);
      }
      partial = -partial;
    }
    const float distance = warp_sum(partial);
    if (lane == 0) insert_sorted(warp_distances, warp_ids, k, distance, candidate);
  }
  __syncthreads();

  if (thread_id == 0) {
    int heads[kCudaWarps] = {0, 0, 0, 0};
    for (int rank = 0; rank < k; ++rank) {
      float best_distance = CUDART_INF_F;
      int best_id = -1;
      int best_warp = 0;
      for (int current_warp = 0; current_warp < kCudaWarps; ++current_warp) {
        const int head = heads[current_warp];
        if (head >= k) continue;
        const int position = current_warp * k + head;
        if (better_pair(local_distances[position], local_ids[position], best_distance, best_id)) {
          best_distance = local_distances[position];
          best_id = local_ids[position];
          best_warp = current_warp;
        }
      }
      const std::size_t output = static_cast<std::size_t>(query_id) * k + rank;
      output_distances[output] = best_distance;
      output_ids[output] = best_id;
      ++heads[best_warp];
    }
  }
}

__global__ void signed_hash_project_kernel(
  const float* data,
  const std::uint32_t* feature_offsets,
  const std::uint32_t* feature_ids,
  const std::int8_t* feature_signs,
  float* projected,
  int rows,
  int dimensions,
  int projected_dimensions
) {
  const std::size_t output_id = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(rows) * projected_dimensions;
  if (output_id >= total) return;
  const int row = static_cast<int>(output_id / projected_dimensions);
  const int bucket = static_cast<int>(output_id - static_cast<std::size_t>(row) * projected_dimensions);
  float value = 0.0f;
  const std::uint32_t begin = feature_offsets[bucket];
  const std::uint32_t end = feature_offsets[bucket + 1];
  for (std::uint32_t position = begin; position < end; ++position) {
    const int dimension = static_cast<int>(feature_ids[position]);
    if (dimension < dimensions) {
      value = fmaf(static_cast<float>(feature_signs[position]),
                   data[static_cast<std::size_t>(row) * dimensions + dimension], value);
    }
  }
  projected[output_id] = value;
}

__global__ void row_norms_kernel(const float* data, float* norms, int rows, int dimensions) {
  const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const float* values = data + static_cast<std::size_t>(row) * dimensions;
  float norm = 0.0f;
  for (int dimension = 0; dimension < dimensions; ++dimension) {
    norm = fmaf(values[dimension], values[dimension], norm);
  }
  norms[row] = norm;
}

__global__ void gather_centroids_kernel(
  const float* data,
  const int* initial_indices,
  float* centroids,
  int clusters,
  int dimensions
) {
  const std::size_t id = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(clusters) * dimensions;
  if (id >= total) return;
  const int cluster = static_cast<int>(id / dimensions);
  const int dimension = static_cast<int>(id - static_cast<std::size_t>(cluster) * dimensions);
  centroids[id] = data[static_cast<std::size_t>(initial_indices[cluster]) * dimensions + dimension];
}

__global__ void assign_kmeans_kernel(
  const float* dot_products,
  const float* data_norms,
  const float* centroid_norms,
  int* assignments,
  unsigned int* changed,
  int first_row,
  int batch_rows,
  int clusters
) {
  const int batch_row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (batch_row >= batch_rows) return;
  const int row = first_row + batch_row;
  float best_distance = CUDART_INF_F;
  int best_cluster = -1;
  const float* row_dot = dot_products + static_cast<std::size_t>(batch_row) * clusters;
  for (int cluster = 0; cluster < clusters; ++cluster) {
    const float distance = fmaxf(0.0f, data_norms[row] + centroid_norms[cluster] - 2.0f * row_dot[cluster]);
    if (better_pair(distance, cluster, best_distance, best_cluster)) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }
  if (assignments[row] != best_cluster) {
    assignments[row] = best_cluster;
    atomicAdd(changed, 1u);
  }
}

__global__ void accumulate_centroids_kernel(
  const float* data,
  const int* assignments,
  float* sums,
  unsigned int* counts,
  int rows,
  int dimensions,
  int clusters
) {
  const std::size_t id = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(rows) * dimensions;
  if (id >= total) return;
  const int row = static_cast<int>(id / dimensions);
  const int dimension = static_cast<int>(id - static_cast<std::size_t>(row) * dimensions);
  const int cluster = assignments[row];
  if (cluster < 0 || cluster >= clusters) return;
  atomicAdd(sums + static_cast<std::size_t>(cluster) * dimensions + dimension, data[id]);
  if (dimension == 0) atomicAdd(counts + cluster, 1u);
}

__global__ void finalize_centroids_kernel(
  const float* data,
  const float* sums,
  const unsigned int* counts,
  const int* initial_indices,
  float* centroids,
  int rows,
  int dimensions,
  int clusters,
  int iteration
) {
  const std::size_t id = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(clusters) * dimensions;
  if (id >= total) return;
  const int cluster = static_cast<int>(id / dimensions);
  const int dimension = static_cast<int>(id - static_cast<std::size_t>(cluster) * dimensions);
  const unsigned int count = counts[cluster];
  if (count > 0) {
    centroids[id] = sums[id] / static_cast<float>(count);
  } else {
    const int replacement = initial_indices[(cluster + iteration) % rows];
    centroids[id] = data[static_cast<std::size_t>(replacement) * dimensions + dimension];
  }
}

__global__ void ivf_topk_kernel(
  const float* train,
  const float* query,
  const float* projected_query,
  const float* centroids,
  const std::uint32_t* list_offsets,
  const int* list_ids,
  const int* excluded_train_id,
  int* output_ids,
  float* output_distances,
  int train_rows,
  int query_rows,
  int dimensions,
  int projected_dimensions,
  int nlist,
  int nprobe,
  int k,
  int metric
) {
  const int query_id = static_cast<int>(blockIdx.x);
  if (query_id >= query_rows) return;
  extern __shared__ unsigned char shared_bytes[];
  float* coarse_distances = reinterpret_cast<float*>(shared_bytes);
  int* coarse_ids = reinterpret_cast<int*>(coarse_distances + kCudaWarps * nprobe);
  int* probes = coarse_ids + kCudaWarps * nprobe;
  float* local_distances = reinterpret_cast<float*>(probes + nprobe);
  int* local_ids = reinterpret_cast<int*>(local_distances + kCudaWarps * k);

  const int thread_id = static_cast<int>(threadIdx.x);
  const int lane = thread_id & 31;
  const int warp = thread_id >> 5;
  for (int position = thread_id; position < kCudaWarps * nprobe; position += blockDim.x) {
    coarse_distances[position] = CUDART_INF_F;
    coarse_ids[position] = -1;
  }
  for (int position = thread_id; position < kCudaWarps * k; position += blockDim.x) {
    local_distances[position] = CUDART_INF_F;
    local_ids[position] = -1;
  }
  __syncthreads();

  const float* projected_row = projected_query + static_cast<std::size_t>(query_id) * projected_dimensions;
  float* warp_coarse_distances = coarse_distances + warp * nprobe;
  int* warp_coarse_ids = coarse_ids + warp * nprobe;
  for (int centroid_id = warp; centroid_id < nlist; centroid_id += kCudaWarps) {
    const float* centroid = centroids + static_cast<std::size_t>(centroid_id) * projected_dimensions;
    float partial = 0.0f;
    for (int dimension = lane; dimension < projected_dimensions; dimension += 32) {
      const float delta = projected_row[dimension] - centroid[dimension];
      partial = fmaf(delta, delta, partial);
    }
    const float distance = warp_sum(partial);
    if (lane == 0) insert_sorted(warp_coarse_distances, warp_coarse_ids, nprobe, distance, centroid_id);
  }
  __syncthreads();

  if (thread_id == 0) {
    int heads[kCudaWarps] = {0, 0, 0, 0};
    for (int rank = 0; rank < nprobe; ++rank) {
      float best_distance = CUDART_INF_F;
      int best_id = -1;
      int best_warp = 0;
      for (int current_warp = 0; current_warp < kCudaWarps; ++current_warp) {
        const int head = heads[current_warp];
        if (head >= nprobe) continue;
        const int position = current_warp * nprobe + head;
        if (better_pair(coarse_distances[position], coarse_ids[position], best_distance, best_id)) {
          best_distance = coarse_distances[position];
          best_id = coarse_ids[position];
          best_warp = current_warp;
        }
      }
      probes[rank] = best_id;
      ++heads[best_warp];
    }
  }
  __syncthreads();

  const float* query_row = query + static_cast<std::size_t>(query_id) * dimensions;
  const int excluded = excluded_train_id[query_id];
  float* warp_local_distances = local_distances + warp * k;
  int* warp_local_ids = local_ids + warp * k;
  for (int probe = 0; probe < nprobe; ++probe) {
    const int list = probes[probe];
    if (list < 0 || list >= nlist) continue;
    const std::uint32_t begin = list_offsets[list];
    const std::uint32_t end = list_offsets[list + 1];
    for (std::uint32_t position = begin + static_cast<std::uint32_t>(warp);
         position < end;
         position += static_cast<std::uint32_t>(kCudaWarps)) {
      const int candidate = list_ids[position];
      if (candidate < 0 || candidate >= train_rows || candidate == excluded) continue;
      const float* train_row = train + static_cast<std::size_t>(candidate) * dimensions;
      float partial = 0.0f;
      if (metric == 0) {
        for (int dimension = lane; dimension < dimensions; dimension += 32) {
          const float delta = query_row[dimension] - train_row[dimension];
          partial = fmaf(delta, delta, partial);
        }
      } else {
        for (int dimension = lane; dimension < dimensions; dimension += 32) {
          partial = fmaf(query_row[dimension], train_row[dimension], partial);
        }
        partial = -partial;
      }
      const float distance = warp_sum(partial);
      if (lane == 0) insert_sorted(warp_local_distances, warp_local_ids, k, distance, candidate);
    }
  }
  __syncthreads();

  if (thread_id == 0) {
    int heads[kCudaWarps] = {0, 0, 0, 0};
    for (int rank = 0; rank < k; ++rank) {
      float best_distance = CUDART_INF_F;
      int best_id = -1;
      int best_warp = 0;
      for (int current_warp = 0; current_warp < kCudaWarps; ++current_warp) {
        const int head = heads[current_warp];
        if (head >= k) continue;
        const int position = current_warp * k + head;
        if (better_pair(local_distances[position], local_ids[position], best_distance, best_id)) {
          best_distance = local_distances[position];
          best_id = local_ids[position];
          best_warp = current_warp;
        }
      }
      const std::size_t output = static_cast<std::size_t>(query_id) * k + rank;
      output_distances[output] = best_distance;
      output_ids[output] = best_id;
      ++heads[best_warp];
    }
  }
}

int blocks_for(std::size_t items, int threads = 256) {
  return static_cast<int>((items + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads));
}

int kmeans_score_batch_rows(int rows, int clusters) {
  const std::size_t workspace_items = kKMeansScoreWorkspaceBytes / sizeof(float);
  const std::size_t rows_per_batch = std::max<std::size_t>(
    1,
    workspace_items / static_cast<std::size_t>(clusters)
  );
  return static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(rows), rows_per_batch));
}

void assign_kmeans_rows(
  cublasHandle_t handle,
  const float* device_data,
  const float* device_centroids,
  const float* device_data_norms,
  const float* device_centroid_norms,
  float* device_scores,
  int* device_assignments,
  unsigned int* device_changed,
  int rows,
  int dimensions,
  int clusters,
  int batch_rows
) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cuda_check(cudaMemset(device_changed, 0, sizeof(unsigned int)), "cudaMemset k-means changed");
  for (int first_row = 0; first_row < rows; first_row += batch_rows) {
    const int current_rows = std::min(batch_rows, rows - first_row);
    cublas_check(
      cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        clusters,
        current_rows,
        dimensions,
        &alpha,
        device_centroids,
        dimensions,
        device_data + static_cast<std::size_t>(first_row) * dimensions,
        dimensions,
        &beta,
        device_scores,
        clusters
      ),
      "native CUDA k-means batched matrix product"
    );
    assign_kmeans_kernel<<<blocks_for(static_cast<std::size_t>(current_rows)), 256>>>(
      device_scores,
      device_data_norms,
      device_centroid_norms,
      device_assignments,
      device_changed,
      first_row,
      current_rows,
      clusters
    );
    cuda_check(cudaGetLastError(), "native CUDA k-means batched assignment");
  }
}

struct KMeansOutput {
  std::vector<float> centroids;
  std::vector<int> assignments;
};

KMeansOutput run_kmeans_device(
  const float* device_data,
  int rows,
  int dimensions,
  int clusters,
  int max_iterations,
  std::uint64_t seed
) {
  if (rows < 1 || dimensions < 1 || clusters < 1 || clusters > rows) {
    throw std::invalid_argument("Invalid native CUDA k-means dimensions.");
  }
  clusters = std::min(clusters, rows);
  max_iterations = std::max(1, max_iterations);
  const std::size_t data_items = static_cast<std::size_t>(rows) * dimensions;
  const std::size_t centroid_items = static_cast<std::size_t>(clusters) * dimensions;
  const int score_batch_rows = kmeans_score_batch_rows(rows, clusters);
  const std::size_t score_items = static_cast<std::size_t>(score_batch_rows) * clusters;

  std::vector<int> initial_indices(static_cast<std::size_t>(rows));
  std::iota(initial_indices.begin(), initial_indices.end(), 0);
  std::mt19937_64 generator(seed);
  std::shuffle(initial_indices.begin(), initial_indices.end(), generator);

  CudaBuffer<int> device_initial_indices(initial_indices.size());
  CudaBuffer<float> device_centroids(centroid_items);
  CudaBuffer<int> device_assignments(static_cast<std::size_t>(rows));
  CudaBuffer<float> device_data_norms(static_cast<std::size_t>(rows));
  CudaBuffer<float> device_centroid_norms(static_cast<std::size_t>(clusters));
  CudaBuffer<float> device_scores(score_items);
  CudaBuffer<float> device_sums(centroid_items);
  CudaBuffer<unsigned int> device_counts(static_cast<std::size_t>(clusters));
  CudaBuffer<unsigned int> device_changed(1);
  device_initial_indices.copy_from_host(initial_indices.data(), initial_indices.size());
  cuda_check(cudaMemset(device_assignments.get(), 0xff, static_cast<std::size_t>(rows) * sizeof(int)),
             "cudaMemset k-means assignments");
  gather_centroids_kernel<<<blocks_for(centroid_items), 256>>>(
    device_data,
    device_initial_indices.get(),
    device_centroids.get(),
    clusters,
    dimensions
  );
  cuda_check(cudaGetLastError(), "native CUDA k-means centroid initialization");
  row_norms_kernel<<<blocks_for(static_cast<std::size_t>(rows)), 256>>>(
    device_data,
    device_data_norms.get(),
    rows,
    dimensions
  );
  cuda_check(cudaGetLastError(), "native CUDA k-means row norms");

  CublasHandle handle;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    row_norms_kernel<<<blocks_for(static_cast<std::size_t>(clusters)), 256>>>(
      device_centroids.get(),
      device_centroid_norms.get(),
      clusters,
      dimensions
    );
    cuda_check(cudaGetLastError(), "native CUDA k-means centroid norms");
    assign_kmeans_rows(
      handle.get(),
      device_data,
      device_centroids.get(),
      device_data_norms.get(),
      device_centroid_norms.get(),
      device_scores.get(),
      device_assignments.get(),
      device_changed.get(),
      rows,
      dimensions,
      clusters,
      score_batch_rows
    );

    unsigned int changed = 0;
    device_changed.copy_to_host(&changed, 1);
    if (changed == 0 || iteration + 1 >= max_iterations) break;

    cuda_check(cudaMemset(device_sums.get(), 0, centroid_items * sizeof(float)), "cudaMemset k-means sums");
    cuda_check(cudaMemset(device_counts.get(), 0, static_cast<std::size_t>(clusters) * sizeof(unsigned int)),
               "cudaMemset k-means counts");
    accumulate_centroids_kernel<<<blocks_for(data_items), 256>>>(
      device_data,
      device_assignments.get(),
      device_sums.get(),
      device_counts.get(),
      rows,
      dimensions,
      clusters
    );
    cuda_check(cudaGetLastError(), "native CUDA k-means accumulation");
    finalize_centroids_kernel<<<blocks_for(centroid_items), 256>>>(
      device_data,
      device_sums.get(),
      device_counts.get(),
      device_initial_indices.get(),
      device_centroids.get(),
      rows,
      dimensions,
      clusters,
      iteration
    );
    cuda_check(cudaGetLastError(), "native CUDA k-means centroid update");
  }

  KMeansOutput output;
  output.centroids.resize(centroid_items);
  output.assignments.resize(static_cast<std::size_t>(rows));
  device_centroids.copy_to_host(output.centroids.data(), output.centroids.size());
  device_assignments.copy_to_host(output.assignments.data(), output.assignments.size());
  return output;
}

class FaissCompatibleRandom {
 public:
  explicit FaissCompatibleRandom(std::uint64_t seed)
      : generator_(static_cast<std::uint32_t>(seed)) {}

  int integer(int maximum) {
    return static_cast<int>(generator_() % static_cast<std::uint32_t>(maximum));
  }

  float uniform() {
    return static_cast<float>(generator_()) /
      static_cast<float>(std::mt19937::max());
  }

 private:
  std::mt19937 generator_;
};

std::vector<int> faiss_compatible_permutation(int rows, std::uint64_t seed) {
  std::vector<int> permutation(static_cast<std::size_t>(rows));
  std::iota(permutation.begin(), permutation.end(), 0);
  FaissCompatibleRandom generator(seed);
  for (int row = 0; row + 1 < rows; ++row) {
    const int replacement = row + generator.integer(rows - row);
    std::swap(
      permutation[static_cast<std::size_t>(row)],
      permutation[static_cast<std::size_t>(replacement)]
    );
  }
  return permutation;
}

class FloatFenwickTree {
 public:
  explicit FloatFenwickTree(const std::vector<float>& values)
      : tree_(values.size() + 1, 0.0f), values_(values) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      add(index, values[index]);
    }
  }

  float total() const {
    float sum = 0.0f;
    for (std::size_t index = tree_.size() - 1; index > 0; index -= index & (~index + 1)) {
      sum += tree_[index];
    }
    return sum;
  }

  void set(std::size_t index, float value) {
    const float delta = value - values_[index];
    values_[index] = value;
    add(index, delta);
  }

  int select(float target) const {
    std::size_t index = 0;
    float prefix = 0.0f;
    std::size_t step = 1;
    while ((step << 1) < tree_.size()) step <<= 1;
    for (; step > 0; step >>= 1) {
      const std::size_t next = index + step;
      if (next < tree_.size() && prefix + tree_[next] <= target) {
        index = next;
        prefix += tree_[next];
      }
    }
    return index < values_.size() ? static_cast<int>(index) : -1;
  }

 private:
  void add(std::size_t index, float delta) {
    for (++index; index < tree_.size(); index += index & (~index + 1)) {
      tree_[index] += delta;
    }
  }

  std::vector<float> tree_;
  std::vector<float> values_;
};

void faiss_compatible_split_empty_clusters(
  int rows,
  int dimensions,
  int clusters,
  std::vector<float>& cluster_sizes,
  std::vector<float>& centroids
) {
  if (rows <= clusters) return;
  constexpr float epsilon = 1.0f / 1024.0f;
  FaissCompatibleRandom generator(1234u);
  std::vector<float> donor_weights(static_cast<std::size_t>(clusters), 0.0f);
  for (int cluster = 0; cluster < clusters; ++cluster) {
    donor_weights[static_cast<std::size_t>(cluster)] = std::max(
      0.0f,
      cluster_sizes[static_cast<std::size_t>(cluster)] - 1.0f
    );
  }
  FloatFenwickTree donor_tree(donor_weights);
  for (int empty = 0; empty < clusters; ++empty) {
    if (cluster_sizes[static_cast<std::size_t>(empty)] != 0.0f) continue;

    const float total_weight = donor_tree.total();
    int donor = -1;
    if (total_weight > 0.0f) {
      const float target = generator.uniform() * total_weight;
      donor = donor_tree.select(target);
    }
    if (donor < 0) {
      donor = 0;
      for (int cluster = 1; cluster < clusters; ++cluster) {
        if (cluster_sizes[static_cast<std::size_t>(cluster)] >
            cluster_sizes[static_cast<std::size_t>(donor)]) {
          donor = cluster;
        }
      }
    }

    float* empty_centroid = centroids.data() +
      static_cast<std::size_t>(empty) * dimensions;
    float* donor_centroid = centroids.data() +
      static_cast<std::size_t>(donor) * dimensions;
    std::copy_n(donor_centroid, dimensions, empty_centroid);
    for (int dimension = 0; dimension < dimensions; ++dimension) {
      if ((dimension & 1) == 0) {
        empty_centroid[dimension] *= 1.0f + epsilon;
        donor_centroid[dimension] *= 1.0f - epsilon;
      } else {
        empty_centroid[dimension] *= 1.0f - epsilon;
        donor_centroid[dimension] *= 1.0f + epsilon;
      }
    }
    cluster_sizes[static_cast<std::size_t>(empty)] =
      cluster_sizes[static_cast<std::size_t>(donor)] / 2.0f;
    cluster_sizes[static_cast<std::size_t>(donor)] -=
      cluster_sizes[static_cast<std::size_t>(empty)];
    donor_tree.set(
      static_cast<std::size_t>(empty),
      std::max(0.0f, cluster_sizes[static_cast<std::size_t>(empty)] - 1.0f)
    );
    donor_tree.set(
      static_cast<std::size_t>(donor),
      std::max(0.0f, cluster_sizes[static_cast<std::size_t>(donor)] - 1.0f)
    );
  }
}

KMeansOutput run_faiss_compatible_kmeans_device(
  const std::vector<float>& data,
  int rows,
  int dimensions,
  int clusters,
  int max_iterations,
  std::uint64_t seed
) {
  if (rows < 1 || dimensions < 1 || clusters < 1 || clusters > rows) {
    throw std::invalid_argument("Invalid FAISS-compatible CUDA k-means dimensions.");
  }
  max_iterations = std::max(1, max_iterations);
  const std::size_t data_items = static_cast<std::size_t>(rows) * dimensions;
  const std::size_t centroid_items = static_cast<std::size_t>(clusters) * dimensions;
  const int score_batch_rows = kmeans_score_batch_rows(rows, clusters);
  const std::size_t score_items = static_cast<std::size_t>(score_batch_rows) * clusters;

  const std::vector<int> initial_indices =
    faiss_compatible_permutation(rows, seed + 1u);
  std::vector<float> centroids(centroid_items, 0.0f);
  for (int cluster = 0; cluster < clusters; ++cluster) {
    std::copy_n(
      data.data() + static_cast<std::size_t>(initial_indices[static_cast<std::size_t>(cluster)]) * dimensions,
      dimensions,
      centroids.data() + static_cast<std::size_t>(cluster) * dimensions
    );
  }

  CudaBuffer<float> device_data(data_items);
  CudaBuffer<float> device_centroids(centroid_items);
  CudaBuffer<int> device_assignments(static_cast<std::size_t>(rows));
  CudaBuffer<float> device_data_norms(static_cast<std::size_t>(rows));
  CudaBuffer<float> device_centroid_norms(static_cast<std::size_t>(clusters));
  CudaBuffer<float> device_scores(score_items);
  CudaBuffer<unsigned int> device_changed(1);
  device_data.copy_from_host(data.data(), data.size());
  device_centroids.copy_from_host(centroids.data(), centroids.size());
  cuda_check(
    cudaMemset(device_assignments.get(), 0xff, static_cast<std::size_t>(rows) * sizeof(int)),
    "cudaMemset FAISS-compatible k-means assignments"
  );
  row_norms_kernel<<<blocks_for(static_cast<std::size_t>(rows)), 256>>>(
    device_data.get(),
    device_data_norms.get(),
    rows,
    dimensions
  );
  cuda_check(cudaGetLastError(), "FAISS-compatible CUDA k-means row norms");

  CublasHandle handle;
  std::vector<int> assignments(static_cast<std::size_t>(rows), -1);
  std::vector<float> cluster_sizes(static_cast<std::size_t>(clusters), 0.0f);

  auto assign = [&]() {
    row_norms_kernel<<<blocks_for(static_cast<std::size_t>(clusters)), 256>>>(
      device_centroids.get(),
      device_centroid_norms.get(),
      clusters,
      dimensions
    );
    cuda_check(cudaGetLastError(), "FAISS-compatible CUDA k-means centroid norms");
    assign_kmeans_rows(
      handle.get(),
      device_data.get(),
      device_centroids.get(),
      device_data_norms.get(),
      device_centroid_norms.get(),
      device_scores.get(),
      device_assignments.get(),
      device_changed.get(),
      rows,
      dimensions,
      clusters,
      score_batch_rows
    );
    device_assignments.copy_to_host(assignments.data(), assignments.size());
    unsigned int changed = 0;
    device_changed.copy_to_host(&changed, 1);
    return changed;
  };

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    const unsigned int changed = assign();
    std::fill(centroids.begin(), centroids.end(), 0.0f);
    std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0.0f);
    for (int row = 0; row < rows; ++row) {
      const int cluster = assignments[static_cast<std::size_t>(row)];
      if (cluster < 0 || cluster >= clusters) {
        throw std::runtime_error("FAISS-compatible CUDA k-means returned an invalid assignment.");
      }
      cluster_sizes[static_cast<std::size_t>(cluster)] += 1.0f;
      float* centroid = centroids.data() + static_cast<std::size_t>(cluster) * dimensions;
      const float* point = data.data() + static_cast<std::size_t>(row) * dimensions;
      for (int dimension = 0; dimension < dimensions; ++dimension) {
        centroid[dimension] += point[dimension];
      }
    }
    for (int cluster = 0; cluster < clusters; ++cluster) {
      const float size = cluster_sizes[static_cast<std::size_t>(cluster)];
      if (size == 0.0f) continue;
      const float scale = 1.0f / size;
      float* centroid = centroids.data() + static_cast<std::size_t>(cluster) * dimensions;
      for (int dimension = 0; dimension < dimensions; ++dimension) {
        centroid[dimension] *= scale;
      }
    }
    faiss_compatible_split_empty_clusters(
      rows,
      dimensions,
      clusters,
      cluster_sizes,
      centroids
    );
    device_centroids.copy_from_host(centroids.data(), centroids.size());
    if (changed == 0) break;
  }

  assign();
  return KMeansOutput{std::move(centroids), std::move(assignments)};
}

std::vector<int> normalized_exclusions(const std::vector<int>& input, int query_rows) {
  if (!input.empty() && static_cast<int>(input.size()) != query_rows) {
    throw std::invalid_argument("Native CUDA KNN exclusion vector size mismatch.");
  }
  if (!input.empty()) return input;
  return std::vector<int>(static_cast<std::size_t>(query_rows), -1);
}

void validate_knn_inputs(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  const std::vector<int>& query_train_indices
) {
  if (train_rows < 1 || query_rows < 0 || dimensions < 1 || k < 1) {
    throw std::invalid_argument("Invalid native CUDA KNN dimensions or neighbor count.");
  }
  if (train.size() != static_cast<std::size_t>(train_rows) * dimensions ||
      query.size() != static_cast<std::size_t>(query_rows) * dimensions) {
    throw std::invalid_argument("Native CUDA KNN matrix size mismatch.");
  }
  if (!query_train_indices.empty() && static_cast<int>(query_train_indices.size()) != query_rows) {
    throw std::invalid_argument("Native CUDA KNN exclusion vector size mismatch.");
  }
  if (k > kMaximumCudaK) {
    throw std::invalid_argument("Native CUDA KNN supports k <= 256.");
  }
}

void launch_exact(
  const float* device_train,
  const float* device_query,
  const int* device_exclusions,
  int* device_ids,
  float* device_distances,
  int train_rows,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric
) {
  if (query_rows == 0 || k == 0) return;
  const std::size_t shared_bytes =
    static_cast<std::size_t>(kCudaWarps) * static_cast<std::size_t>(k) * (sizeof(float) + sizeof(int));
  exact_topk_kernel<<<query_rows, kCudaThreads, shared_bytes>>>(
    device_train,
    device_query,
    device_exclusions,
    device_ids,
    device_distances,
    train_rows,
    query_rows,
    dimensions,
    k,
    metric == DistanceMetric::Euclidean ? 0 : 1
  );
  cuda_check(cudaGetLastError(), "native CUDA exact KNN launch");
}

void launch_ivf(
  const float* device_train,
  const float* device_query,
  const float* device_projected_query,
  const float* device_centroids,
  const std::uint32_t* device_list_offsets,
  const int* device_list_ids,
  const int* device_exclusions,
  int* device_ids,
  float* device_distances,
  int train_rows,
  int query_rows,
  int dimensions,
  int projected_dimensions,
  int nlist,
  int nprobe,
  int k,
  DistanceMetric metric
) {
  if (query_rows == 0 || k == 0) return;
  const std::size_t shared_bytes =
    static_cast<std::size_t>(kCudaWarps) * nprobe * (sizeof(float) + sizeof(int)) +
    static_cast<std::size_t>(nprobe) * sizeof(int) +
    static_cast<std::size_t>(kCudaWarps) * k * (sizeof(float) + sizeof(int));
  ivf_topk_kernel<<<query_rows, kCudaThreads, shared_bytes>>>(
    device_train,
    device_query,
    device_projected_query,
    device_centroids,
    device_list_offsets,
    device_list_ids,
    device_exclusions,
    device_ids,
    device_distances,
    train_rows,
    query_rows,
    dimensions,
    projected_dimensions,
    nlist,
    nprobe,
    k,
    metric == DistanceMetric::Euclidean ? 0 : 1
  );
  cuda_check(cudaGetLastError(), "native CUDA IVF-Flat KNN launch");
}

int projection_dimension(int dimensions) {
  int projected = 1;
  const int limit = std::max(1, std::min(dimensions, kCudaProjectionDimension));
  while (projected * 2 <= limit) projected *= 2;
  return projected;
}

void make_projection_map(
  int dimensions,
  int projected_dimensions,
  std::vector<std::uint32_t>& offsets,
  std::vector<std::uint32_t>& ids,
  std::vector<std::int8_t>& signs
) {
  std::vector<std::vector<std::pair<std::uint32_t, std::int8_t>>> buckets(
    static_cast<std::size_t>(projected_dimensions)
  );
  for (int dimension = 0; dimension < dimensions; ++dimension) {
    const std::uint32_t hash = static_cast<std::uint32_t>(dimension + 1) * 2654435761u;
    const int bucket = static_cast<int>(hash & static_cast<std::uint32_t>(projected_dimensions - 1));
    const std::int8_t sign = ((hash >> 17u) & 1u) != 0u ? std::int8_t(1) : std::int8_t(-1);
    buckets[static_cast<std::size_t>(bucket)].push_back({static_cast<std::uint32_t>(dimension), sign});
  }
  offsets.assign(static_cast<std::size_t>(projected_dimensions) + 1, 0);
  ids.clear();
  signs.clear();
  ids.reserve(static_cast<std::size_t>(dimensions));
  signs.reserve(static_cast<std::size_t>(dimensions));
  for (int bucket = 0; bucket < projected_dimensions; ++bucket) {
    offsets[static_cast<std::size_t>(bucket)] = static_cast<std::uint32_t>(ids.size());
    for (const auto& feature : buckets[static_cast<std::size_t>(bucket)]) {
      ids.push_back(feature.first);
      signs.push_back(feature.second);
    }
  }
  offsets.back() = static_cast<std::uint32_t>(ids.size());
}

double recall_at_k(
  const std::vector<int>& exact,
  const std::vector<int>& approximate,
  int rows,
  int k
) {
  if (rows < 1 || k < 1) return 0.0;
  std::size_t hits = 0;
  for (int row = 0; row < rows; ++row) {
    const std::size_t base = static_cast<std::size_t>(row) * k;
    for (int rank = 0; rank < k; ++rank) {
      const int candidate = approximate[base + static_cast<std::size_t>(rank)];
      if (candidate < 0) continue;
      for (int exact_rank = 0; exact_rank < k; ++exact_rank) {
        if (candidate == exact[base + static_cast<std::size_t>(exact_rank)]) {
          ++hits;
          break;
        }
      }
    }
  }
  return static_cast<double>(hits) /
    static_cast<double>(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k));
}

}  // namespace

bool native_cuda_backend_available(int device) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return device >= 0 && device < count;
}

NativeKNNResult native_cuda_exact_knn_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  int device,
  const std::vector<int>& query_train_indices
) {
  validate_knn_inputs(train, train_rows, query, query_rows, dimensions, k, query_train_indices);
  const int available = train_rows - (query_train_indices.empty() ? 0 : 1);
  k = std::min(k, std::max(0, available));
  NativeKNNResult output;
  output.queries = query_rows;
  output.neighbors = k;
  output.indices.assign(static_cast<std::size_t>(query_rows) * k, -1);
  output.distances.assign(
    static_cast<std::size_t>(query_rows) * k,
    std::numeric_limits<float>::infinity()
  );
  if (query_rows == 0 || k == 0) return output;

  CudaDeviceScope device_scope(device);
  const std::vector<int> exclusions = normalized_exclusions(query_train_indices, query_rows);
  CudaBuffer<float> device_train(train.size());
  CudaBuffer<float> device_query(query.size());
  CudaBuffer<int> device_exclusions(exclusions.size());
  CudaBuffer<int> device_ids(output.indices.size());
  CudaBuffer<float> device_distances(output.distances.size());
  device_train.copy_from_host(train.data(), train.size());
  device_query.copy_from_host(query.data(), query.size());
  device_exclusions.copy_from_host(exclusions.data(), exclusions.size());
  launch_exact(
    device_train.get(),
    device_query.get(),
    device_exclusions.get(),
    device_ids.get(),
    device_distances.get(),
    train_rows,
    query_rows,
    dimensions,
    k,
    metric
  );
  device_ids.copy_to_host(output.indices.data(), output.indices.size());
  device_distances.copy_to_host(output.distances.data(), output.distances.size());
  return output;
}

NativeKNNResult native_cuda_ivf_knn_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  int requested_nlist,
  int requested_nprobe,
  double target_recall,
  int device,
  const std::vector<int>& query_train_indices,
  NativeCudaIVFStats* stats
) {
  validate_knn_inputs(train, train_rows, query, query_rows, dimensions, k, query_train_indices);
  if (requested_nlist > kMaximumCudaLists || requested_nprobe > kMaximumCudaProbe) {
    throw std::invalid_argument("Native CUDA IVF supports nlist <= 4096 and nprobe <= 256.");
  }
  const int available = train_rows - (query_train_indices.empty() ? 0 : 1);
  k = std::min(k, std::max(0, available));
  NativeKNNResult output;
  output.queries = query_rows;
  output.neighbors = k;
  output.indices.assign(static_cast<std::size_t>(query_rows) * k, -1);
  output.distances.assign(
    static_cast<std::size_t>(query_rows) * k,
    std::numeric_limits<float>::infinity()
  );
  if (query_rows == 0 || k == 0) return output;

  int nlist = requested_nlist > 0 ? requested_nlist : static_cast<int>(
    std::ceil(std::sqrt(static_cast<double>(train_rows)))
  );
  nlist = std::max(1, std::min({nlist, kMaximumCudaLists, train_rows}));
  if (requested_nlist <= 0) nlist = std::min(nlist, kMaximumCudaProbe);
  const int max_probe = std::min(nlist, kMaximumCudaProbe);
  // A twice-square-root starting budget keeps the automatic path close to exact
  // recall while allowing the pilot search to increase it when needed.
  int nprobe = requested_nprobe > 0 ? requested_nprobe :
    std::max(1, 2 * static_cast<int>(std::ceil(std::sqrt(static_cast<double>(nlist)))));
  nprobe = std::max(1, std::min(nprobe, max_probe));
  target_recall = std::max(0.0, std::min(1.0, target_recall));
  const int projected_dimensions = projection_dimension(dimensions);
  const std::vector<int> exclusions = normalized_exclusions(query_train_indices, query_rows);

  std::vector<std::uint32_t> feature_offsets;
  std::vector<std::uint32_t> feature_ids;
  std::vector<std::int8_t> feature_signs;
  make_projection_map(dimensions, projected_dimensions, feature_offsets, feature_ids, feature_signs);

  CudaDeviceScope device_scope(device);
  CudaBuffer<float> device_train(train.size());
  CudaBuffer<float> device_query(query.size());
  CudaBuffer<int> device_exclusions(exclusions.size());
  CudaBuffer<std::uint32_t> device_feature_offsets(feature_offsets.size());
  CudaBuffer<std::uint32_t> device_feature_ids(feature_ids.size());
  CudaBuffer<std::int8_t> device_feature_signs(feature_signs.size());
  CudaBuffer<float> device_projected_train(static_cast<std::size_t>(train_rows) * projected_dimensions);
  CudaBuffer<float> device_projected_query(static_cast<std::size_t>(query_rows) * projected_dimensions);
  device_train.copy_from_host(train.data(), train.size());
  device_query.copy_from_host(query.data(), query.size());
  device_exclusions.copy_from_host(exclusions.data(), exclusions.size());
  device_feature_offsets.copy_from_host(feature_offsets.data(), feature_offsets.size());
  device_feature_ids.copy_from_host(feature_ids.data(), feature_ids.size());
  device_feature_signs.copy_from_host(feature_signs.data(), feature_signs.size());

  const std::size_t projected_train_items = static_cast<std::size_t>(train_rows) * projected_dimensions;
  const std::size_t projected_query_items = static_cast<std::size_t>(query_rows) * projected_dimensions;
  signed_hash_project_kernel<<<blocks_for(projected_train_items), 256>>>(
    device_train.get(),
    device_feature_offsets.get(),
    device_feature_ids.get(),
    device_feature_signs.get(),
    device_projected_train.get(),
    train_rows,
    dimensions,
    projected_dimensions
  );
  signed_hash_project_kernel<<<blocks_for(projected_query_items), 256>>>(
    device_query.get(),
    device_feature_offsets.get(),
    device_feature_ids.get(),
    device_feature_signs.get(),
    device_projected_query.get(),
    query_rows,
    dimensions,
    projected_dimensions
  );
  cuda_check(cudaGetLastError(), "native CUDA IVF signed-hash projection");

  KMeansOutput clustering = run_kmeans_device(
    device_projected_train.get(),
    train_rows,
    projected_dimensions,
    nlist,
    8,
    4u
  );
  std::vector<std::uint32_t> list_offsets(static_cast<std::size_t>(nlist) + 1, 0);
  for (int assignment : clustering.assignments) {
    if (assignment >= 0 && assignment < nlist) {
      ++list_offsets[static_cast<std::size_t>(assignment) + 1];
    }
  }
  for (int list = 0; list < nlist; ++list) {
    list_offsets[static_cast<std::size_t>(list) + 1] += list_offsets[static_cast<std::size_t>(list)];
  }
  std::vector<std::uint32_t> cursor = list_offsets;
  std::vector<int> list_ids(static_cast<std::size_t>(train_rows), -1);
  for (int row = 0; row < train_rows; ++row) {
    const int assignment = clustering.assignments[static_cast<std::size_t>(row)];
    if (assignment >= 0 && assignment < nlist) {
      list_ids[cursor[static_cast<std::size_t>(assignment)]++] = row;
    }
  }

  CudaBuffer<float> device_centroids(clustering.centroids.size());
  CudaBuffer<std::uint32_t> device_list_offsets(list_offsets.size());
  CudaBuffer<int> device_list_ids(list_ids.size());
  device_centroids.copy_from_host(clustering.centroids.data(), clustering.centroids.size());
  device_list_offsets.copy_from_host(list_offsets.data(), list_offsets.size());
  device_list_ids.copy_from_host(list_ids.data(), list_ids.size());

  const int pilot_rows = std::min(query_rows, 128);
  const std::size_t pilot_items = static_cast<std::size_t>(pilot_rows) * k;
  CudaBuffer<int> device_pilot_exact_ids(pilot_items);
  CudaBuffer<float> device_pilot_exact_distances(pilot_items);
  CudaBuffer<int> device_pilot_ivf_ids(pilot_items);
  CudaBuffer<float> device_pilot_ivf_distances(pilot_items);
  launch_exact(
    device_train.get(),
    device_query.get(),
    device_exclusions.get(),
    device_pilot_exact_ids.get(),
    device_pilot_exact_distances.get(),
    train_rows,
    pilot_rows,
    dimensions,
    k,
    metric
  );
  std::vector<int> pilot_exact(pilot_items, -1);
  std::vector<int> pilot_ivf(pilot_items, -1);
  device_pilot_exact_ids.copy_to_host(pilot_exact.data(), pilot_exact.size());

  auto evaluate_probe = [&](int probes) {
    launch_ivf(
      device_train.get(),
      device_query.get(),
      device_projected_query.get(),
      device_centroids.get(),
      device_list_offsets.get(),
      device_list_ids.get(),
      device_exclusions.get(),
      device_pilot_ivf_ids.get(),
      device_pilot_ivf_distances.get(),
      train_rows,
      pilot_rows,
      dimensions,
      projected_dimensions,
      nlist,
      probes,
      k,
      metric
    );
    device_pilot_ivf_ids.copy_to_host(pilot_ivf.data(), pilot_ivf.size());
    return recall_at_k(pilot_exact, pilot_ivf, pilot_rows, k);
  };

  double pilot_recall = 0.0;
  if (requested_nprobe <= 0) {
    int low_fail = nprobe - 1;
    int high = nprobe;
    while (true) {
      pilot_recall = evaluate_probe(high);
      if (pilot_recall >= target_recall || high >= max_probe) break;
      low_fail = high;
      high = std::min(max_probe, std::max(high + 1, static_cast<int>(std::ceil(high * 1.5))));
    }
    if (pilot_recall >= target_recall) {
      while (high - low_fail > 1) {
        const int middle = low_fail + (high - low_fail) / 2;
        const double middle_recall = evaluate_probe(middle);
        if (middle_recall >= target_recall) {
          high = middle;
          pilot_recall = middle_recall;
        } else {
          low_fail = middle;
        }
      }
    }
    nprobe = high;
  } else {
    pilot_recall = evaluate_probe(nprobe);
  }

  CudaBuffer<int> device_output_ids(output.indices.size());
  CudaBuffer<float> device_output_distances(output.distances.size());
  launch_ivf(
    device_train.get(),
    device_query.get(),
    device_projected_query.get(),
    device_centroids.get(),
    device_list_offsets.get(),
    device_list_ids.get(),
    device_exclusions.get(),
    device_output_ids.get(),
    device_output_distances.get(),
    train_rows,
    query_rows,
    dimensions,
    projected_dimensions,
    nlist,
    nprobe,
    k,
    metric
  );
  device_output_ids.copy_to_host(output.indices.data(), output.indices.size());
  device_output_distances.copy_to_host(output.distances.data(), output.distances.size());
  if (stats != nullptr) {
    stats->nlist = nlist;
    stats->nprobe = nprobe;
    stats->pilot_recall = pilot_recall;
  }
  return output;
}

std::vector<int> native_cuda_kmeans_labels(
  const std::vector<float>& data,
  int rows,
  int dimensions,
  int clusters,
  int max_iterations,
  std::uint64_t seed,
  int device
) {
  if (rows < 1 || dimensions < 1 || clusters < 1 || clusters > rows ||
      data.size() != static_cast<std::size_t>(rows) * dimensions) {
    throw std::invalid_argument("Invalid native CUDA k-means input.");
  }
  CudaDeviceScope device_scope(device);
  KMeansOutput output = run_faiss_compatible_kmeans_device(
    data,
    rows,
    dimensions,
    clusters,
    max_iterations,
    seed
  );
  for (int& label : output.assignments) ++label;
  return output.assignments;
}

}  // namespace kodama::detail
