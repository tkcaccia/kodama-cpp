/*
 * Native float32 Metal primitives adapted from fastEmbedR and fastPLS.
 *
 * The top-k organization is informed by FAISS 1.14.3 (MIT) and the
 * fastEmbedR native Metal backend. MPS matrix multiplication follows the
 * fastPLS Metal backend. Modifications and standalone adaptation are
 * Copyright (c) 2026 Stefano Caccia and distributed under the MIT License.
 */

#include "metal_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kodama::detail {
namespace {

constexpr int kMaximumMetalK = 128;
constexpr int kMaximumMetalLists = 1024;
constexpr int kMaximumMetalProbe = 128;
constexpr int kMetalProjectionDimension = 128;

const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint NSG = 4;
constant uint SIMD_WIDTH = 32;
constant uint MAX_K = 128;
constant uint MAX_PROBE = 128;

struct ExactParams {
  uint n_train;
  uint n_query;
  uint dimensions;
  uint k;
  uint metric;
};

struct ProjectParams {
  uint rows;
  uint dimensions;
  uint projected_dimensions;
};

struct IVFSearchParams {
  uint n_train;
  uint n_query;
  uint dimensions;
  uint projected_dimensions;
  uint nlist;
  uint nprobe;
  uint k;
  uint metric;
};

struct KMeansParams {
  uint rows;
  uint dimensions;
  uint clusters;
};

inline bool better_pair(float da, int ia, float db, int ib) {
  return da < db || (da == db && ia < ib);
}

inline void insert_sorted(
    threadgroup float* values,
    threadgroup int* ids,
    uint base,
    uint count,
    float value,
    int id) {
  if (count == 0 || !better_pair(value, id, values[base + count - 1], ids[base + count - 1])) return;
  uint pos = count - 1;
  while (pos > 0 && better_pair(value, id, values[base + pos - 1], ids[base + pos - 1])) {
    values[base + pos] = values[base + pos - 1];
    ids[base + pos] = ids[base + pos - 1];
    --pos;
  }
  values[base + pos] = value;
  ids[base + pos] = id;
}

kernel void exact_topk_train_query(
    device const float* train [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device const int* excluded_train_id [[buffer(2)]],
    device int* out_ids [[buffer(3)]],
    device float* out_distances [[buffer(4)]],
    constant ExactParams& params [[buffer(5)]],
    uint query_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (query_id >= params.n_query) return;
  threadgroup float local_distance[NSG * MAX_K];
  threadgroup int local_id[NSG * MAX_K];
  for (uint pos = tid; pos < NSG * params.k; pos += NSG * SIMD_WIDTH) {
    local_distance[pos] = INFINITY;
    local_id[pos] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device float* query_row = query + query_id * params.dimensions;
  const int excluded = excluded_train_id[query_id];
  const uint base = simd_group * params.k;
  for (uint candidate = simd_group; candidate < params.n_train; candidate += NSG) {
    if (int(candidate) == excluded) continue;
    const device float* train_row = train + candidate * params.dimensions;
    float partial = 0.0f;
    if (params.metric == 0) {
      for (uint d = lane; d < params.dimensions; d += SIMD_WIDTH) {
        const float delta = query_row[d] - train_row[d];
        partial = fma(delta, delta, partial);
      }
    } else {
      for (uint d = lane; d < params.dimensions; d += SIMD_WIDTH) {
        partial = fma(query_row[d], train_row[d], partial);
      }
      partial = -partial;
    }
    const float distance = simd_sum(partial);
    if (lane == 0) insert_sorted(local_distance, local_id, base, params.k, distance, int(candidate));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG] = {0, 0, 0, 0};
    for (uint rank = 0; rank < params.k; ++rank) {
      float best_distance = INFINITY;
      int best_id = -1;
      uint best_group = 0;
      for (uint group = 0; group < NSG; ++group) {
        const uint head = heads[group];
        if (head >= params.k) continue;
        const uint pos = group * params.k + head;
        if (better_pair(local_distance[pos], local_id[pos], best_distance, best_id)) {
          best_distance = local_distance[pos];
          best_id = local_id[pos];
          best_group = group;
        }
      }
      const uint output = query_id * params.k + rank;
      out_distances[output] = best_distance;
      out_ids[output] = best_id;
      ++heads[best_group];
    }
  }
}

kernel void signed_hash_project(
    device const float* data [[buffer(0)]],
    device const uint* feature_offsets [[buffer(1)]],
    device const uint* feature_ids [[buffer(2)]],
    device const char* feature_signs [[buffer(3)]],
    device float* projected [[buffer(4)]],
    constant ProjectParams& params [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  if (row >= params.rows || tid >= params.projected_dimensions) return;
  float value = 0.0f;
  const uint begin = feature_offsets[tid];
  const uint end = feature_offsets[tid + 1];
  for (uint position = begin; position < end; ++position) {
    value = fma(float(feature_signs[position]), data[row * params.dimensions + feature_ids[position]], value);
  }
  projected[row * params.projected_dimensions + tid] = value;
}

kernel void ivf_topk_train_query(
    device const float* train [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device const float* projected_query [[buffer(2)]],
    device const float* centroids [[buffer(3)]],
    device const uint* list_offsets [[buffer(4)]],
    device const int* list_ids [[buffer(5)]],
    device const int* excluded_train_id [[buffer(6)]],
    device int* out_ids [[buffer(7)]],
    device float* out_distances [[buffer(8)]],
    constant IVFSearchParams& params [[buffer(9)]],
    uint query_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (query_id >= params.n_query) return;
  threadgroup float coarse_distance[NSG * MAX_PROBE];
  threadgroup int coarse_id[NSG * MAX_PROBE];
  threadgroup int probes[MAX_PROBE];
  threadgroup float local_distance[NSG * MAX_K];
  threadgroup int local_id[NSG * MAX_K];

  for (uint position = tid; position < NSG * params.nprobe; position += NSG * SIMD_WIDTH) {
    coarse_distance[position] = INFINITY;
    coarse_id[position] = -1;
  }
  for (uint position = tid; position < NSG * params.k; position += NSG * SIMD_WIDTH) {
    local_distance[position] = INFINITY;
    local_id[position] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device float* query_projected_row =
    projected_query + query_id * params.projected_dimensions;
  const uint coarse_base = simd_group * params.nprobe;
  for (uint centroid_id = simd_group; centroid_id < params.nlist; centroid_id += NSG) {
    const device float* centroid = centroids + centroid_id * params.projected_dimensions;
    float partial = 0.0f;
    for (uint dimension = lane; dimension < params.projected_dimensions; dimension += SIMD_WIDTH) {
      const float delta = query_projected_row[dimension] - centroid[dimension];
      partial = fma(delta, delta, partial);
    }
    const float distance = simd_sum(partial);
    if (lane == 0) {
      insert_sorted(coarse_distance, coarse_id, coarse_base, params.nprobe, distance, int(centroid_id));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG] = {0, 0, 0, 0};
    for (uint rank = 0; rank < params.nprobe; ++rank) {
      float best_distance = INFINITY;
      int best_id = -1;
      uint best_group = 0;
      for (uint group = 0; group < NSG; ++group) {
        const uint head = heads[group];
        if (head >= params.nprobe) continue;
        const uint position = group * params.nprobe + head;
        if (better_pair(coarse_distance[position], coarse_id[position], best_distance, best_id)) {
          best_distance = coarse_distance[position];
          best_id = coarse_id[position];
          best_group = group;
        }
      }
      probes[rank] = best_id;
      ++heads[best_group];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device float* query_row = query + query_id * params.dimensions;
  const int excluded = excluded_train_id[query_id];
  const uint fine_base = simd_group * params.k;
  for (uint probe = 0; probe < params.nprobe; ++probe) {
    const int list = probes[probe];
    if (list < 0 || uint(list) >= params.nlist) continue;
    const uint begin = list_offsets[list];
    const uint end = list_offsets[list + 1];
    for (uint position = begin + simd_group; position < end; position += NSG) {
      const int candidate = list_ids[position];
      if (candidate < 0 || candidate == excluded) continue;
      const device float* train_row = train + uint(candidate) * params.dimensions;
      float partial = 0.0f;
      if (params.metric == 0) {
        for (uint dimension = lane; dimension < params.dimensions; dimension += SIMD_WIDTH) {
          const float delta = query_row[dimension] - train_row[dimension];
          partial = fma(delta, delta, partial);
        }
      } else {
        for (uint dimension = lane; dimension < params.dimensions; dimension += SIMD_WIDTH) {
          partial = fma(query_row[dimension], train_row[dimension], partial);
        }
        partial = -partial;
      }
      const float distance = simd_sum(partial);
      if (lane == 0) insert_sorted(local_distance, local_id, fine_base, params.k, distance, candidate);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG] = {0, 0, 0, 0};
    for (uint rank = 0; rank < params.k; ++rank) {
      float best_distance = INFINITY;
      int best_id = -1;
      uint best_group = 0;
      for (uint group = 0; group < NSG; ++group) {
        const uint head = heads[group];
        if (head >= params.k) continue;
        const uint position = group * params.k + head;
        if (better_pair(local_distance[position], local_id[position], best_distance, best_id)) {
          best_distance = local_distance[position];
          best_id = local_id[position];
          best_group = group;
        }
      }
      const uint output = query_id * params.k + rank;
      out_distances[output] = best_distance;
      out_ids[output] = best_id;
      ++heads[best_group];
    }
  }
}

kernel void clear_kmeans_changed(
    device atomic_uint* changed [[buffer(0)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid == 0) atomic_store_explicit(changed, 0u, memory_order_relaxed);
}

kernel void assign_kmeans_centroid(
    device const float* data [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device int* assignments [[buffer(2)]],
    device atomic_uint* changed [[buffer(3)]],
    constant KMeansParams& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (row >= params.rows) return;
  threadgroup float best_distance[NSG];
  threadgroup int best_cluster[NSG];
  if (tid < NSG) {
    best_distance[tid] = INFINITY;
    best_cluster[tid] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device float* point = data + row * params.dimensions;
  for (uint cluster = simd_group; cluster < params.clusters; cluster += NSG) {
    const device float* centroid = centroids + cluster * params.dimensions;
    float partial = 0.0f;
    for (uint dimension = lane; dimension < params.dimensions; dimension += SIMD_WIDTH) {
      const float delta = point[dimension] - centroid[dimension];
      partial = fma(delta, delta, partial);
    }
    const float distance = simd_sum(partial);
    if (lane == 0 && better_pair(
          distance,
          int(cluster),
          best_distance[simd_group],
          best_cluster[simd_group])) {
      best_distance[simd_group] = distance;
      best_cluster[simd_group] = int(cluster);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float distance = best_distance[0];
    int cluster = best_cluster[0];
    for (uint group = 1; group < NSG; ++group) {
      if (better_pair(best_distance[group], best_cluster[group], distance, cluster)) {
        distance = best_distance[group];
        cluster = best_cluster[group];
      }
    }
    if (assignments[row] != cluster) {
      assignments[row] = cluster;
      atomic_fetch_add_explicit(changed, 1u, memory_order_relaxed);
    }
  }
}

kernel void clear_kmeans_accumulators(
    device atomic_float* sums [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant KMeansParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = params.clusters * params.dimensions;
  if (gid < total) atomic_store_explicit(&sums[gid], 0.0f, memory_order_relaxed);
  if (gid < params.clusters) atomic_store_explicit(&counts[gid], 0u, memory_order_relaxed);
}

kernel void accumulate_kmeans_centroids(
    device const float* data [[buffer(0)]],
    device const int* assignments [[buffer(1)]],
    device atomic_float* sums [[buffer(2)]],
    device atomic_uint* counts [[buffer(3)]],
    constant KMeansParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = params.rows * params.dimensions;
  if (gid >= total) return;
  const uint row = gid / params.dimensions;
  const uint dimension = gid - row * params.dimensions;
  const int cluster = assignments[row];
  if (cluster < 0 || uint(cluster) >= params.clusters) return;
  atomic_fetch_add_explicit(
    &sums[uint(cluster) * params.dimensions + dimension],
    data[gid],
    memory_order_relaxed
  );
  if (dimension == 0) atomic_fetch_add_explicit(&counts[cluster], 1u, memory_order_relaxed);
}

kernel void finalize_kmeans_centroids(
    device const float* data [[buffer(0)]],
    device const atomic_float* sums [[buffer(1)]],
    device const atomic_uint* counts [[buffer(2)]],
    device float* centroids [[buffer(3)]],
    device const int* initial_point_indices [[buffer(4)]],
    constant KMeansParams& params [[buffer(5)]],
    constant uint& iteration [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = params.clusters * params.dimensions;
  if (gid >= total) return;
  const uint cluster = gid / params.dimensions;
  const uint dimension = gid - cluster * params.dimensions;
  const uint count = atomic_load_explicit(&counts[cluster], memory_order_relaxed);
  if (count > 0) {
    centroids[gid] = atomic_load_explicit(&sums[gid], memory_order_relaxed) / float(count);
  } else {
    const uint replacement = uint(initial_point_indices[(cluster + iteration) % params.rows]);
    centroids[gid] = data[replacement * params.dimensions + dimension];
  }
}
)METAL";

struct ExactParamsHost {
  std::uint32_t n_train;
  std::uint32_t n_query;
  std::uint32_t dimensions;
  std::uint32_t k;
  std::uint32_t metric;
};

struct ProjectParamsHost {
  std::uint32_t rows;
  std::uint32_t dimensions;
  std::uint32_t projected_dimensions;
};

struct IVFSearchParamsHost {
  std::uint32_t n_train;
  std::uint32_t n_query;
  std::uint32_t dimensions;
  std::uint32_t projected_dimensions;
  std::uint32_t nlist;
  std::uint32_t nprobe;
  std::uint32_t k;
  std::uint32_t metric;
};

struct KMeansParamsHost {
  std::uint32_t rows;
  std::uint32_t dimensions;
  std::uint32_t clusters;
};

std::runtime_error metal_error(const std::string& context, NSError* error = nil) {
  const char* message = error == nil ? "unknown Metal error" : [[error localizedDescription] UTF8String];
  return std::runtime_error(context + ": " + message);
}

struct MetalState {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> exact_pipeline = nil;
  id<MTLComputePipelineState> project_pipeline = nil;
  id<MTLComputePipelineState> ivf_pipeline = nil;
  id<MTLComputePipelineState> clear_changed_pipeline = nil;
  id<MTLComputePipelineState> assign_kmeans_pipeline = nil;
  id<MTLComputePipelineState> clear_kmeans_pipeline = nil;
  id<MTLComputePipelineState> accumulate_kmeans_pipeline = nil;
  id<MTLComputePipelineState> finalize_kmeans_pipeline = nil;
};

id<MTLDevice> select_metal_device() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device != nil) return device;
  NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
  return devices.count == 0 ? nil : devices.firstObject;
}

MetalState& metal_state() {
  static MetalState state;
  static std::once_flag once;
  static std::exception_ptr initialization_error;
  std::call_once(once, [&]() {
    try {
      state.device = select_metal_device();
      if (state.device == nil) throw std::runtime_error("No Apple Metal device is available.");
      state.queue = [state.device newCommandQueue];
      if (state.queue == nil) throw std::runtime_error("Failed to create the Metal command queue.");
      MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
      options.fastMathEnabled = YES;
      NSError* error = nil;
      state.library = [state.device
        newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
        options:options
        error:&error];
      if (state.library == nil) throw metal_error("Failed to compile KODAMA Metal kernels", error);
      auto make_pipeline = [&](NSString* name) {
        id<MTLFunction> function = [state.library newFunctionWithName:name];
        if (function == nil) {
          throw std::runtime_error(std::string("The Metal function is missing: ") + [name UTF8String]);
        }
        id<MTLComputePipelineState> pipeline = [state.device
          newComputePipelineStateWithFunction:function
          error:&error];
        if (pipeline == nil) {
          throw metal_error(std::string("Failed to create Metal pipeline ") + [name UTF8String], error);
        }
        return pipeline;
      };
      state.exact_pipeline = make_pipeline(@"exact_topk_train_query");
      state.project_pipeline = make_pipeline(@"signed_hash_project");
      state.ivf_pipeline = make_pipeline(@"ivf_topk_train_query");
      state.clear_changed_pipeline = make_pipeline(@"clear_kmeans_changed");
      state.assign_kmeans_pipeline = make_pipeline(@"assign_kmeans_centroid");
      state.clear_kmeans_pipeline = make_pipeline(@"clear_kmeans_accumulators");
      state.accumulate_kmeans_pipeline = make_pipeline(@"accumulate_kmeans_centroids");
      state.finalize_kmeans_pipeline = make_pipeline(@"finalize_kmeans_centroids");
    } catch (...) {
      initialization_error = std::current_exception();
    }
  });
  if (initialization_error) std::rethrow_exception(initialization_error);
  return state;
}

void wait_for_command(id<MTLCommandBuffer> command, const char* context) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) throw metal_error(context, command.error);
}

NSUInteger matrix_row_bytes(int columns) {
  return [MPSMatrixDescriptor rowBytesFromColumns:static_cast<NSUInteger>(columns) dataType:MPSDataTypeFloat32];
}

id<MTLBuffer> make_matrix_buffer(
  id<MTLDevice> device,
  const std::vector<float>& values,
  int rows,
  int columns,
  NSUInteger row_bytes
) {
  id<MTLBuffer> buffer = [device
    newBufferWithLength:row_bytes * static_cast<NSUInteger>(rows)
    options:MTLResourceStorageModeShared];
  if (buffer == nil) throw std::runtime_error("Failed to allocate a Metal matrix buffer.");
  char* base = static_cast<char*>([buffer contents]);
  std::memset(base, 0, row_bytes * static_cast<NSUInteger>(rows));
  for (int row = 0; row < rows; ++row) {
    std::memcpy(
      base + static_cast<NSUInteger>(row) * row_bytes,
      values.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(columns),
      static_cast<std::size_t>(columns) * sizeof(float)
    );
  }
  return buffer;
}

float vector_dot(const std::vector<float>& left, const std::vector<float>& right) {
  float result = 0.0f;
  for (std::size_t i = 0; i < left.size(); ++i) {
    result += left[i] * right[i];
  }
  return result;
}

float vector_norm(const std::vector<float>& values) {
  return std::sqrt(std::max(0.0f, vector_dot(values, values)));
}

void normalize_vector(std::vector<float>& values) {
  const float norm = vector_norm(values);
  if (norm <= 1e-6f || !std::isfinite(norm)) return;
  const float scale = 1.0f / norm;
  for (float& value : values) value *= scale;
}

std::vector<float> cross_product_times(
  const std::vector<float>& cross_product,
  int predictors,
  int responses,
  const std::vector<float>& right
) {
  std::vector<float> result(static_cast<std::size_t>(predictors), 0.0f);
  for (int predictor = 0; predictor < predictors; ++predictor) {
    float value = 0.0f;
    const float* row = cross_product.data() + static_cast<std::size_t>(predictor) * static_cast<std::size_t>(responses);
    for (int response = 0; response < responses; ++response) {
      value += row[response] * right[static_cast<std::size_t>(response)];
    }
    result[static_cast<std::size_t>(predictor)] = value;
  }
  return result;
}

std::vector<float> cross_product_transpose_times(
  const std::vector<float>& cross_product,
  int predictors,
  int responses,
  const std::vector<float>& right
) {
  std::vector<float> result(static_cast<std::size_t>(responses), 0.0f);
  for (int response = 0; response < responses; ++response) {
    float value = 0.0f;
    for (int predictor = 0; predictor < predictors; ++predictor) {
      value += cross_product[
        static_cast<std::size_t>(predictor) * static_cast<std::size_t>(responses) + static_cast<std::size_t>(response)
      ] * right[static_cast<std::size_t>(predictor)];
    }
    result[static_cast<std::size_t>(response)] = value;
  }
  return result;
}

void remove_stored_columns(
  std::vector<float>& vector,
  const std::vector<float>& matrix,
  int rows,
  int matrix_columns,
  int used_columns
) {
  for (int column = 0; column < used_columns; ++column) {
    float projection = 0.0f;
    for (int row = 0; row < rows; ++row) {
      projection += vector[static_cast<std::size_t>(row)] *
        matrix[static_cast<std::size_t>(row) * static_cast<std::size_t>(matrix_columns) + static_cast<std::size_t>(column)];
    }
    for (int row = 0; row < rows; ++row) {
      vector[static_cast<std::size_t>(row)] -= projection *
        matrix[static_cast<std::size_t>(row) * static_cast<std::size_t>(matrix_columns) + static_cast<std::size_t>(column)];
    }
  }
}

int metal_projection_dimension(int dimensions) {
  int projected = 1;
  while (projected * 2 <= dimensions && projected * 2 <= kMetalProjectionDimension) projected *= 2;
  return std::max(1, projected);
}

double recall_at_k(
  const std::vector<int>& truth,
  const int* approximate,
  int rows,
  int k
) {
  if (rows < 1 || k < 1) return 0.0;
  std::size_t hits = 0;
  for (int row = 0; row < rows; ++row) {
    const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(k);
    for (int rank = 0; rank < k; ++rank) {
      const int candidate = approximate[base + static_cast<std::size_t>(rank)];
      for (int exact_rank = 0; exact_rank < k; ++exact_rank) {
        if (candidate == truth[base + static_cast<std::size_t>(exact_rank)]) {
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

bool metal_backend_available() {
  @autoreleasepool {
    id<MTLDevice> device = select_metal_device();
    return device != nil;
  }
}

NativeKNNResult metal_exact_knn_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  const std::vector<int>& query_train_indices
) {
  if (train_rows < 1 || query_rows < 0 || dimensions < 1 || k < 1) {
    throw std::invalid_argument("Invalid Metal KNN dimensions or neighbor count.");
  }
  if (train.size() != static_cast<std::size_t>(train_rows) * static_cast<std::size_t>(dimensions) ||
      query.size() != static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(dimensions)) {
    throw std::invalid_argument("Metal KNN matrix size mismatch.");
  }
  if (!query_train_indices.empty() && static_cast<int>(query_train_indices.size()) != query_rows) {
    throw std::invalid_argument("Metal KNN exclusion vector size mismatch.");
  }
  const int available = train_rows - (query_train_indices.empty() ? 0 : 1);
  k = std::min(k, std::max(0, available));
  if (k > kMaximumMetalK) throw std::invalid_argument("Metal exact KNN supports k <= 128.");

  NativeKNNResult output;
  output.queries = query_rows;
  output.neighbors = k;
  output.indices.assign(static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k), -1);
  output.distances.assign(
    static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k),
    std::numeric_limits<float>::infinity()
  );
  if (query_rows == 0 || k == 0) return output;

  std::vector<int> exclusions = query_train_indices;
  if (exclusions.empty()) exclusions.assign(static_cast<std::size_t>(query_rows), -1);

  @autoreleasepool {
    MetalState& state = metal_state();
    id<MTLBuffer> train_buffer = [state.device
      newBufferWithBytes:train.data()
      length:train.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_buffer = [state.device
      newBufferWithBytes:query.data()
      length:query.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> exclusion_buffer = [state.device
      newBufferWithBytes:exclusions.data()
      length:exclusions.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> index_buffer = [state.device
      newBufferWithLength:output.indices.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> distance_buffer = [state.device
      newBufferWithLength:output.distances.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    if (train_buffer == nil || query_buffer == nil || exclusion_buffer == nil ||
        index_buffer == nil || distance_buffer == nil) {
      throw std::runtime_error("Failed to allocate Metal KNN buffers.");
    }

    const ExactParamsHost parameters{
      static_cast<std::uint32_t>(train_rows),
      static_cast<std::uint32_t>(query_rows),
      static_cast<std::uint32_t>(dimensions),
      static_cast<std::uint32_t>(k),
      metric == DistanceMetric::Euclidean ? 0u : 1u
    };
    id<MTLBuffer> parameter_buffer = [state.device
      newBufferWithBytes:&parameters
      length:sizeof(parameters)
      options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state.exact_pipeline];
    [encoder setBuffer:train_buffer offset:0 atIndex:0];
    [encoder setBuffer:query_buffer offset:0 atIndex:1];
    [encoder setBuffer:exclusion_buffer offset:0 atIndex:2];
    [encoder setBuffer:index_buffer offset:0 atIndex:3];
    [encoder setBuffer:distance_buffer offset:0 atIndex:4];
    [encoder setBuffer:parameter_buffer offset:0 atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(query_rows), 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    wait_for_command(command, "Metal exact KNN command failed");
    std::memcpy(output.indices.data(), [index_buffer contents], output.indices.size() * sizeof(int));
    std::memcpy(output.distances.data(), [distance_buffer contents], output.distances.size() * sizeof(float));
  }
  return output;
}

NativeKNNResult metal_ivf_knn_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  int requested_nlist,
  int requested_nprobe,
  const std::vector<int>& query_train_indices,
  MetalIVFStats* stats
) {
  if (train_rows < 1 || query_rows < 0 || dimensions < 1 || k < 1) {
    throw std::invalid_argument("Invalid Metal IVF dimensions or neighbor count.");
  }
  if (train.size() != static_cast<std::size_t>(train_rows) * static_cast<std::size_t>(dimensions) ||
      query.size() != static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(dimensions)) {
    throw std::invalid_argument("Metal IVF matrix size mismatch.");
  }
  if (!query_train_indices.empty() && static_cast<int>(query_train_indices.size()) != query_rows) {
    throw std::invalid_argument("Metal IVF exclusion vector size mismatch.");
  }
  if (requested_nlist > kMaximumMetalLists || requested_nprobe > kMaximumMetalProbe) {
    throw std::invalid_argument("Metal IVF supports nlist <= 1024 and nprobe <= 128.");
  }

  const int available = train_rows - (query_train_indices.empty() ? 0 : 1);
  k = std::min(k, std::max(0, available));
  if (k > kMaximumMetalK) throw std::invalid_argument("Metal IVF KNN supports k <= 128.");
  NativeKNNResult output;
  output.queries = query_rows;
  output.neighbors = k;
  output.indices.assign(static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k), -1);
  output.distances.assign(
    static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k),
    std::numeric_limits<float>::infinity()
  );
  if (query_rows == 0 || k == 0) return output;

  int nlist = requested_nlist > 0 ? requested_nlist : static_cast<int>(
    std::ceil(4.0 * std::sqrt(static_cast<double>(train_rows)))
  );
  nlist = std::max(1, std::min({nlist, kMaximumMetalLists, train_rows}));
  int nprobe = requested_nprobe > 0 ? requested_nprobe : std::min(8, nlist);
  nprobe = std::max(1, std::min({nprobe, nlist, kMaximumMetalProbe}));
  const int projected_dimensions = metal_projection_dimension(dimensions);

  std::vector<int> exclusions = query_train_indices;
  if (exclusions.empty()) exclusions.assign(static_cast<std::size_t>(query_rows), -1);

  std::vector<std::uint32_t> feature_offsets(static_cast<std::size_t>(projected_dimensions) + 1, 0);
  std::vector<std::vector<std::pair<std::uint32_t, std::int8_t>>> feature_buckets(
    static_cast<std::size_t>(projected_dimensions)
  );
  for (int dimension = 0; dimension < dimensions; ++dimension) {
    const std::uint32_t hash = static_cast<std::uint32_t>(dimension + 1) * 2654435761u;
    const int bucket = static_cast<int>(hash & static_cast<std::uint32_t>(projected_dimensions - 1));
    const std::int8_t sign = ((hash >> 17u) & 1u) != 0u ? std::int8_t(1) : std::int8_t(-1);
    feature_buckets[static_cast<std::size_t>(bucket)].push_back({static_cast<std::uint32_t>(dimension), sign});
  }
  std::vector<std::uint32_t> feature_ids;
  std::vector<std::int8_t> feature_signs;
  feature_ids.reserve(static_cast<std::size_t>(dimensions));
  feature_signs.reserve(static_cast<std::size_t>(dimensions));
  for (int bucket = 0; bucket < projected_dimensions; ++bucket) {
    feature_offsets[static_cast<std::size_t>(bucket)] = static_cast<std::uint32_t>(feature_ids.size());
    for (const auto& feature : feature_buckets[static_cast<std::size_t>(bucket)]) {
      feature_ids.push_back(feature.first);
      feature_signs.push_back(feature.second);
    }
  }
  feature_offsets.back() = static_cast<std::uint32_t>(feature_ids.size());

  const int pilot_rows = std::min(query_rows, 128);
  std::vector<float> pilot_query(
    query.begin(),
    query.begin() + static_cast<std::ptrdiff_t>(pilot_rows) * static_cast<std::ptrdiff_t>(dimensions)
  );
  std::vector<int> pilot_exclusions(exclusions.begin(), exclusions.begin() + pilot_rows);
  const NativeKNNResult pilot_exact = metal_exact_knn_search(
    train,
    train_rows,
    pilot_query,
    pilot_rows,
    dimensions,
    k,
    metric,
    pilot_exclusions
  );
  double pilot_recall = 0.0;

  @autoreleasepool {
    MetalState& state = metal_state();
    auto require_buffer = [](id<MTLBuffer> buffer, const char* name) {
      if (buffer == nil) throw std::runtime_error(std::string("Failed to allocate Metal IVF buffer: ") + name);
    };

    id<MTLBuffer> train_buffer = [state.device
      newBufferWithBytes:train.data()
      length:train.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_buffer = [state.device
      newBufferWithBytes:query.data()
      length:query.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> exclusion_buffer = [state.device
      newBufferWithBytes:exclusions.data()
      length:exclusions.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> feature_offset_buffer = [state.device
      newBufferWithBytes:feature_offsets.data()
      length:feature_offsets.size() * sizeof(std::uint32_t)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> feature_id_buffer = [state.device
      newBufferWithBytes:feature_ids.data()
      length:feature_ids.size() * sizeof(std::uint32_t)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> feature_sign_buffer = [state.device
      newBufferWithBytes:feature_signs.data()
      length:feature_signs.size() * sizeof(std::int8_t)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> projected_train_buffer = [state.device
      newBufferWithLength:static_cast<std::size_t>(train_rows) * static_cast<std::size_t>(projected_dimensions) * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> projected_query_buffer = [state.device
      newBufferWithLength:static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(projected_dimensions) * sizeof(float)
      options:MTLResourceStorageModeShared];
    require_buffer(train_buffer, "train");
    require_buffer(query_buffer, "query");
    require_buffer(exclusion_buffer, "exclusions");
    require_buffer(feature_offset_buffer, "feature offsets");
    require_buffer(feature_id_buffer, "feature ids");
    require_buffer(feature_sign_buffer, "feature signs");
    require_buffer(projected_train_buffer, "projected train");
    require_buffer(projected_query_buffer, "projected query");

    const ProjectParamsHost train_project_params{
      static_cast<std::uint32_t>(train_rows),
      static_cast<std::uint32_t>(dimensions),
      static_cast<std::uint32_t>(projected_dimensions)
    };
    const ProjectParamsHost query_project_params{
      static_cast<std::uint32_t>(query_rows),
      static_cast<std::uint32_t>(dimensions),
      static_cast<std::uint32_t>(projected_dimensions)
    };
    id<MTLBuffer> train_project_params_buffer = [state.device
      newBufferWithBytes:&train_project_params
      length:sizeof(train_project_params)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_project_params_buffer = [state.device
      newBufferWithBytes:&query_project_params
      length:sizeof(query_project_params)
      options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> project_command = [state.queue commandBuffer];
    auto encode_projection = [&](id<MTLBuffer> input, id<MTLBuffer> projected, id<MTLBuffer> parameters, int rows) {
      id<MTLComputeCommandEncoder> encoder = [project_command computeCommandEncoder];
      [encoder setComputePipelineState:state.project_pipeline];
      [encoder setBuffer:input offset:0 atIndex:0];
      [encoder setBuffer:feature_offset_buffer offset:0 atIndex:1];
      [encoder setBuffer:feature_id_buffer offset:0 atIndex:2];
      [encoder setBuffer:feature_sign_buffer offset:0 atIndex:3];
      [encoder setBuffer:projected offset:0 atIndex:4];
      [encoder setBuffer:parameters offset:0 atIndex:5];
      [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(projected_dimensions), 1, 1)];
      [encoder endEncoding];
    };
    encode_projection(train_buffer, projected_train_buffer, train_project_params_buffer, train_rows);
    encode_projection(query_buffer, projected_query_buffer, query_project_params_buffer, query_rows);
    wait_for_command(project_command, "Metal IVF projection failed");

    const float* projected_train = static_cast<const float*>([projected_train_buffer contents]);
    std::vector<int> initial_indices(static_cast<std::size_t>(train_rows));
    std::iota(initial_indices.begin(), initial_indices.end(), 0);
    std::mt19937 generator(4u);
    std::shuffle(initial_indices.begin(), initial_indices.end(), generator);
    std::vector<float> centroids(
      static_cast<std::size_t>(nlist) * static_cast<std::size_t>(projected_dimensions),
      0.0f
    );
    for (int centroid = 0; centroid < nlist; ++centroid) {
      std::copy_n(
        projected_train + static_cast<std::size_t>(initial_indices[static_cast<std::size_t>(centroid)]) *
          static_cast<std::size_t>(projected_dimensions),
        projected_dimensions,
        centroids.data() + static_cast<std::size_t>(centroid) * static_cast<std::size_t>(projected_dimensions)
      );
    }
    std::vector<int> assignments(static_cast<std::size_t>(train_rows), -1);
    const std::size_t centroid_items =
      static_cast<std::size_t>(nlist) * static_cast<std::size_t>(projected_dimensions);
    id<MTLBuffer> centroid_buffer = [state.device
      newBufferWithBytes:centroids.data()
      length:centroids.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> assignment_buffer = [state.device
      newBufferWithBytes:assignments.data()
      length:assignments.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> centroid_sum_buffer = [state.device
      newBufferWithLength:centroid_items * sizeof(float)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> centroid_count_buffer = [state.device
      newBufferWithLength:static_cast<std::size_t>(nlist) * sizeof(std::uint32_t)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> changed_buffer = [state.device
      newBufferWithLength:sizeof(std::uint32_t)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> initial_index_buffer = [state.device
      newBufferWithBytes:initial_indices.data()
      length:initial_indices.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    const KMeansParamsHost kmeans_params{
      static_cast<std::uint32_t>(train_rows),
      static_cast<std::uint32_t>(projected_dimensions),
      static_cast<std::uint32_t>(nlist)
    };
    id<MTLBuffer> kmeans_params_buffer = [state.device
      newBufferWithBytes:&kmeans_params
      length:sizeof(kmeans_params)
      options:MTLResourceStorageModeShared];
    require_buffer(centroid_buffer, "centroids");
    require_buffer(assignment_buffer, "assignments");
    require_buffer(centroid_sum_buffer, "centroid sums");
    require_buffer(centroid_count_buffer, "centroid counts");
    require_buffer(changed_buffer, "changed count");
    require_buffer(initial_index_buffer, "initial indices");
    require_buffer(kmeans_params_buffer, "kmeans parameters");

    for (int iteration = 0; iteration < 4; ++iteration) {
      const std::uint32_t iteration_value = static_cast<std::uint32_t>(iteration);
      id<MTLCommandBuffer> command = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> changed_encoder = [command computeCommandEncoder];
      [changed_encoder setComputePipelineState:state.clear_changed_pipeline];
      [changed_encoder setBuffer:changed_buffer offset:0 atIndex:0];
      [changed_encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [changed_encoder endEncoding];

      id<MTLComputeCommandEncoder> assignment_encoder = [command computeCommandEncoder];
      [assignment_encoder setComputePipelineState:state.assign_kmeans_pipeline];
      [assignment_encoder setBuffer:projected_train_buffer offset:0 atIndex:0];
      [assignment_encoder setBuffer:centroid_buffer offset:0 atIndex:1];
      [assignment_encoder setBuffer:assignment_buffer offset:0 atIndex:2];
      [assignment_encoder setBuffer:changed_buffer offset:0 atIndex:3];
      [assignment_encoder setBuffer:kmeans_params_buffer offset:0 atIndex:4];
      [assignment_encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(train_rows), 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [assignment_encoder endEncoding];

      if (iteration < 3) {
        id<MTLComputeCommandEncoder> clear_encoder = [command computeCommandEncoder];
        [clear_encoder setComputePipelineState:state.clear_kmeans_pipeline];
        [clear_encoder setBuffer:centroid_sum_buffer offset:0 atIndex:0];
        [clear_encoder setBuffer:centroid_count_buffer offset:0 atIndex:1];
        [clear_encoder setBuffer:kmeans_params_buffer offset:0 atIndex:2];
        [clear_encoder dispatchThreads:MTLSizeMake(std::max<std::size_t>(centroid_items, nlist), 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [clear_encoder endEncoding];

        id<MTLComputeCommandEncoder> accumulate_encoder = [command computeCommandEncoder];
        [accumulate_encoder setComputePipelineState:state.accumulate_kmeans_pipeline];
        [accumulate_encoder setBuffer:projected_train_buffer offset:0 atIndex:0];
        [accumulate_encoder setBuffer:assignment_buffer offset:0 atIndex:1];
        [accumulate_encoder setBuffer:centroid_sum_buffer offset:0 atIndex:2];
        [accumulate_encoder setBuffer:centroid_count_buffer offset:0 atIndex:3];
        [accumulate_encoder setBuffer:kmeans_params_buffer offset:0 atIndex:4];
        [accumulate_encoder dispatchThreads:MTLSizeMake(
          static_cast<std::size_t>(train_rows) * static_cast<std::size_t>(projected_dimensions), 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [accumulate_encoder endEncoding];

        id<MTLComputeCommandEncoder> finalize_encoder = [command computeCommandEncoder];
        [finalize_encoder setComputePipelineState:state.finalize_kmeans_pipeline];
        [finalize_encoder setBuffer:projected_train_buffer offset:0 atIndex:0];
        [finalize_encoder setBuffer:centroid_sum_buffer offset:0 atIndex:1];
        [finalize_encoder setBuffer:centroid_count_buffer offset:0 atIndex:2];
        [finalize_encoder setBuffer:centroid_buffer offset:0 atIndex:3];
        [finalize_encoder setBuffer:initial_index_buffer offset:0 atIndex:4];
        [finalize_encoder setBuffer:kmeans_params_buffer offset:0 atIndex:5];
        [finalize_encoder setBytes:&iteration_value length:sizeof(iteration_value) atIndex:6];
        [finalize_encoder dispatchThreads:MTLSizeMake(centroid_items, 1, 1)
                              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [finalize_encoder endEncoding];
      }
      wait_for_command(command, "Metal IVF training failed");
    }

    std::memcpy(assignments.data(), [assignment_buffer contents], assignments.size() * sizeof(int));
    std::vector<std::uint32_t> list_offsets(static_cast<std::size_t>(nlist) + 1, 0);
    for (int assignment : assignments) {
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
      const int assignment = assignments[static_cast<std::size_t>(row)];
      if (assignment >= 0 && assignment < nlist) {
        list_ids[cursor[static_cast<std::size_t>(assignment)]++] = row;
      }
    }
    id<MTLBuffer> list_offset_buffer = [state.device
      newBufferWithBytes:list_offsets.data()
      length:list_offsets.size() * sizeof(std::uint32_t)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> list_id_buffer = [state.device
      newBufferWithBytes:list_ids.data()
      length:list_ids.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    require_buffer(list_offset_buffer, "list offsets");
    require_buffer(list_id_buffer, "list ids");

    auto run_search = [&](int rows, int probes, id<MTLBuffer> ids, id<MTLBuffer> distances) {
      const IVFSearchParamsHost parameters{
        static_cast<std::uint32_t>(train_rows),
        static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(dimensions),
        static_cast<std::uint32_t>(projected_dimensions),
        static_cast<std::uint32_t>(nlist),
        static_cast<std::uint32_t>(probes),
        static_cast<std::uint32_t>(k),
        metric == DistanceMetric::Euclidean ? 0u : 1u
      };
      id<MTLBuffer> parameter_buffer = [state.device
        newBufferWithBytes:&parameters
        length:sizeof(parameters)
        options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> command = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:state.ivf_pipeline];
      [encoder setBuffer:train_buffer offset:0 atIndex:0];
      [encoder setBuffer:query_buffer offset:0 atIndex:1];
      [encoder setBuffer:projected_query_buffer offset:0 atIndex:2];
      [encoder setBuffer:centroid_buffer offset:0 atIndex:3];
      [encoder setBuffer:list_offset_buffer offset:0 atIndex:4];
      [encoder setBuffer:list_id_buffer offset:0 atIndex:5];
      [encoder setBuffer:exclusion_buffer offset:0 atIndex:6];
      [encoder setBuffer:ids offset:0 atIndex:7];
      [encoder setBuffer:distances offset:0 atIndex:8];
      [encoder setBuffer:parameter_buffer offset:0 atIndex:9];
      [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder endEncoding];
      wait_for_command(command, "Metal IVF search failed");
    };

    id<MTLBuffer> pilot_id_buffer = [state.device
      newBufferWithLength:static_cast<std::size_t>(pilot_rows) * static_cast<std::size_t>(k) * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> pilot_distance_buffer = [state.device
      newBufferWithLength:static_cast<std::size_t>(pilot_rows) * static_cast<std::size_t>(k) * sizeof(float)
      options:MTLResourceStorageModeShared];
    require_buffer(pilot_id_buffer, "pilot ids");
    require_buffer(pilot_distance_buffer, "pilot distances");
    auto evaluate_probe = [&](int probes) {
      run_search(pilot_rows, probes, pilot_id_buffer, pilot_distance_buffer);
      return recall_at_k(
        pilot_exact.indices,
        static_cast<const int*>([pilot_id_buffer contents]),
        pilot_rows,
        k
      );
    };

    if (requested_nprobe <= 0) {
      constexpr double target_recall = 0.999;
      int low_fail = 0;
      int high = nprobe;
      while (true) {
        pilot_recall = evaluate_probe(high);
        if (pilot_recall >= target_recall || high >= std::min(nlist, kMaximumMetalProbe)) break;
        low_fail = high;
        high = std::min(
          std::min(nlist, kMaximumMetalProbe),
          std::max(high + 1, static_cast<int>(std::ceil(static_cast<double>(high) * 1.5)))
        );
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

    id<MTLBuffer> output_id_buffer = [state.device
      newBufferWithLength:output.indices.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_distance_buffer = [state.device
      newBufferWithLength:output.distances.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    require_buffer(output_id_buffer, "output ids");
    require_buffer(output_distance_buffer, "output distances");
    run_search(query_rows, nprobe, output_id_buffer, output_distance_buffer);
    std::memcpy(output.indices.data(), [output_id_buffer contents], output.indices.size() * sizeof(int));
    std::memcpy(
      output.distances.data(),
      [output_distance_buffer contents],
      output.distances.size() * sizeof(float)
    );
  }

  if (stats != nullptr) {
    stats->nlist = nlist;
    stats->nprobe = nprobe;
    stats->pilot_recall = pilot_recall;
  }
  return output;
}

std::vector<int> metal_kmeans_labels(
  const std::vector<float>& data,
  int rows,
  int dimensions,
  int clusters,
  const std::vector<int>& initial_point_indices,
  int max_iterations
) {
  if (rows < 1 || dimensions < 1 || clusters < 1 || clusters > rows || max_iterations < 1 ||
      data.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(dimensions) ||
      static_cast<int>(initial_point_indices.size()) < clusters) {
    throw std::invalid_argument("Invalid Metal k-means dimensions or initialization.");
  }
  for (int cluster = 0; cluster < clusters; ++cluster) {
    const int index = initial_point_indices[static_cast<std::size_t>(cluster)];
    if (index < 0 || index >= rows) throw std::invalid_argument("Metal k-means initialization index is out of range.");
  }

  std::vector<float> centroids(
    static_cast<std::size_t>(clusters) * static_cast<std::size_t>(dimensions),
    0.0f
  );
  for (int cluster = 0; cluster < clusters; ++cluster) {
    std::copy_n(
      data.data() + static_cast<std::size_t>(initial_point_indices[static_cast<std::size_t>(cluster)]) *
        static_cast<std::size_t>(dimensions),
      dimensions,
      centroids.data() + static_cast<std::size_t>(cluster) * static_cast<std::size_t>(dimensions)
    );
  }

  std::vector<int> labels(static_cast<std::size_t>(rows), -1);
  @autoreleasepool {
    MetalState& state = metal_state();
    const std::size_t centroid_items = static_cast<std::size_t>(clusters) * static_cast<std::size_t>(dimensions);
    id<MTLBuffer> data_buffer = [state.device
      newBufferWithBytes:data.data()
      length:data.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> centroid_buffer = [state.device
      newBufferWithBytes:centroids.data()
      length:centroids.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> assignment_buffer = [state.device
      newBufferWithBytes:labels.data()
      length:labels.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> sum_buffer = [state.device
      newBufferWithLength:centroid_items * sizeof(float)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> count_buffer = [state.device
      newBufferWithLength:static_cast<std::size_t>(clusters) * sizeof(std::uint32_t)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> changed_buffer = [state.device
      newBufferWithLength:sizeof(std::uint32_t)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> initial_index_buffer = [state.device
      newBufferWithBytes:initial_point_indices.data()
      length:initial_point_indices.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    const KMeansParamsHost parameters{
      static_cast<std::uint32_t>(rows),
      static_cast<std::uint32_t>(dimensions),
      static_cast<std::uint32_t>(clusters)
    };
    id<MTLBuffer> parameter_buffer = [state.device
      newBufferWithBytes:&parameters
      length:sizeof(parameters)
      options:MTLResourceStorageModeShared];
    if (data_buffer == nil || centroid_buffer == nil || assignment_buffer == nil || sum_buffer == nil ||
        count_buffer == nil || changed_buffer == nil || initial_index_buffer == nil || parameter_buffer == nil) {
      throw std::runtime_error("Failed to allocate Metal k-means buffers.");
    }

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
      const std::uint32_t iteration_value = static_cast<std::uint32_t>(iteration);
      id<MTLCommandBuffer> command = [state.queue commandBuffer];

      id<MTLComputeCommandEncoder> changed_encoder = [command computeCommandEncoder];
      [changed_encoder setComputePipelineState:state.clear_changed_pipeline];
      [changed_encoder setBuffer:changed_buffer offset:0 atIndex:0];
      [changed_encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [changed_encoder endEncoding];

      id<MTLComputeCommandEncoder> assignment_encoder = [command computeCommandEncoder];
      [assignment_encoder setComputePipelineState:state.assign_kmeans_pipeline];
      [assignment_encoder setBuffer:data_buffer offset:0 atIndex:0];
      [assignment_encoder setBuffer:centroid_buffer offset:0 atIndex:1];
      [assignment_encoder setBuffer:assignment_buffer offset:0 atIndex:2];
      [assignment_encoder setBuffer:changed_buffer offset:0 atIndex:3];
      [assignment_encoder setBuffer:parameter_buffer offset:0 atIndex:4];
      [assignment_encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [assignment_encoder endEncoding];

      id<MTLComputeCommandEncoder> clear_encoder = [command computeCommandEncoder];
      [clear_encoder setComputePipelineState:state.clear_kmeans_pipeline];
      [clear_encoder setBuffer:sum_buffer offset:0 atIndex:0];
      [clear_encoder setBuffer:count_buffer offset:0 atIndex:1];
      [clear_encoder setBuffer:parameter_buffer offset:0 atIndex:2];
      [clear_encoder dispatchThreads:MTLSizeMake(std::max<std::size_t>(centroid_items, clusters), 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [clear_encoder endEncoding];

      id<MTLComputeCommandEncoder> accumulate_encoder = [command computeCommandEncoder];
      [accumulate_encoder setComputePipelineState:state.accumulate_kmeans_pipeline];
      [accumulate_encoder setBuffer:data_buffer offset:0 atIndex:0];
      [accumulate_encoder setBuffer:assignment_buffer offset:0 atIndex:1];
      [accumulate_encoder setBuffer:sum_buffer offset:0 atIndex:2];
      [accumulate_encoder setBuffer:count_buffer offset:0 atIndex:3];
      [accumulate_encoder setBuffer:parameter_buffer offset:0 atIndex:4];
      [accumulate_encoder dispatchThreads:MTLSizeMake(data.size(), 1, 1)
                              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [accumulate_encoder endEncoding];

      id<MTLComputeCommandEncoder> finalize_encoder = [command computeCommandEncoder];
      [finalize_encoder setComputePipelineState:state.finalize_kmeans_pipeline];
      [finalize_encoder setBuffer:data_buffer offset:0 atIndex:0];
      [finalize_encoder setBuffer:sum_buffer offset:0 atIndex:1];
      [finalize_encoder setBuffer:count_buffer offset:0 atIndex:2];
      [finalize_encoder setBuffer:centroid_buffer offset:0 atIndex:3];
      [finalize_encoder setBuffer:initial_index_buffer offset:0 atIndex:4];
      [finalize_encoder setBuffer:parameter_buffer offset:0 atIndex:5];
      [finalize_encoder setBytes:&iteration_value length:sizeof(iteration_value) atIndex:6];
      [finalize_encoder dispatchThreads:MTLSizeMake(centroid_items, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [finalize_encoder endEncoding];

      wait_for_command(command, "Metal k-means command failed");
      const auto changed = *static_cast<const std::uint32_t*>([changed_buffer contents]);
      if (changed == 0) break;
    }
    std::memcpy(labels.data(), [assignment_buffer contents], labels.size() * sizeof(int));
  }
  for (int& label : labels) label += 1;
  return labels;
}

std::vector<float> metal_matrix_multiply(
  const std::vector<float>& left,
  int left_rows,
  int left_cols,
  const std::vector<float>& right,
  int right_rows,
  int right_cols,
  bool transpose_left,
  bool transpose_right
) {
  if (left_rows < 1 || left_cols < 1 || right_rows < 1 || right_cols < 1 ||
      left.size() != static_cast<std::size_t>(left_rows) * static_cast<std::size_t>(left_cols) ||
      right.size() != static_cast<std::size_t>(right_rows) * static_cast<std::size_t>(right_cols)) {
    throw std::invalid_argument("Invalid Metal matrix dimensions.");
  }
  const int result_rows = transpose_left ? left_cols : left_rows;
  const int inner_left = transpose_left ? left_rows : left_cols;
  const int inner_right = transpose_right ? right_cols : right_rows;
  const int result_cols = transpose_right ? right_rows : right_cols;
  if (inner_left != inner_right) throw std::invalid_argument("Non-conformable Metal matrices.");

  @autoreleasepool {
    MetalState& state = metal_state();
    const NSUInteger left_row_bytes = matrix_row_bytes(left_cols);
    const NSUInteger right_row_bytes = matrix_row_bytes(right_cols);
    const NSUInteger result_row_bytes = matrix_row_bytes(result_cols);
    id<MTLBuffer> left_buffer = make_matrix_buffer(state.device, left, left_rows, left_cols, left_row_bytes);
    id<MTLBuffer> right_buffer = make_matrix_buffer(state.device, right, right_rows, right_cols, right_row_bytes);
    id<MTLBuffer> result_buffer = [state.device
      newBufferWithLength:result_row_bytes * static_cast<NSUInteger>(result_rows)
      options:MTLResourceStorageModeShared];
    if (result_buffer == nil) throw std::runtime_error("Failed to allocate the Metal result matrix.");
    std::memset([result_buffer contents], 0, result_row_bytes * static_cast<NSUInteger>(result_rows));

    MPSMatrixDescriptor* left_descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:static_cast<NSUInteger>(left_rows)
      columns:static_cast<NSUInteger>(left_cols)
      rowBytes:left_row_bytes
      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* right_descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:static_cast<NSUInteger>(right_rows)
      columns:static_cast<NSUInteger>(right_cols)
      rowBytes:right_row_bytes
      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* result_descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:static_cast<NSUInteger>(result_rows)
      columns:static_cast<NSUInteger>(result_cols)
      rowBytes:result_row_bytes
      dataType:MPSDataTypeFloat32];
    MPSMatrix* left_matrix = [[MPSMatrix alloc] initWithBuffer:left_buffer descriptor:left_descriptor];
    MPSMatrix* right_matrix = [[MPSMatrix alloc] initWithBuffer:right_buffer descriptor:right_descriptor];
    MPSMatrix* result_matrix = [[MPSMatrix alloc] initWithBuffer:result_buffer descriptor:result_descriptor];
    MPSMatrixMultiplication* multiplication = [[MPSMatrixMultiplication alloc]
      initWithDevice:state.device
      transposeLeft:transpose_left
      transposeRight:transpose_right
      resultRows:static_cast<NSUInteger>(result_rows)
      resultColumns:static_cast<NSUInteger>(result_cols)
      interiorColumns:static_cast<NSUInteger>(inner_left)
      alpha:1.0
      beta:0.0];
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    [multiplication encodeToCommandBuffer:command
                              leftMatrix:left_matrix
                             rightMatrix:right_matrix
                            resultMatrix:result_matrix];
    wait_for_command(command, "Metal matrix multiplication failed");

    std::vector<float> output(
      static_cast<std::size_t>(result_rows) * static_cast<std::size_t>(result_cols),
      0.0f
    );
    const char* base = static_cast<const char*>([result_buffer contents]);
    for (int row = 0; row < result_rows; ++row) {
      std::memcpy(
        output.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(result_cols),
        base + static_cast<NSUInteger>(row) * result_row_bytes,
        static_cast<std::size_t>(result_cols) * sizeof(float)
      );
    }
    return output;
  }
}

MetalSIMPLSResult metal_simpls_fit(
  const std::vector<float>& x,
  int rows,
  int predictors,
  const std::vector<float>& cross_product,
  int responses,
  int max_components
) {
  if (rows < 2 || predictors < 1 || responses < 1 || max_components < 1 ||
      x.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(predictors) ||
      cross_product.size() != static_cast<std::size_t>(predictors) * static_cast<std::size_t>(responses)) {
    throw std::invalid_argument("Invalid Metal SIMPLS matrix dimensions.");
  }
  const int components = std::min({max_components, predictors, std::max(1, rows - 1)});
  MetalSIMPLSResult result;
  result.predictors = predictors;
  result.responses = responses;
  result.components = components;
  result.weights.assign(static_cast<std::size_t>(predictors) * static_cast<std::size_t>(components), 0.0f);
  result.y_weights.assign(static_cast<std::size_t>(responses) * static_cast<std::size_t>(components), 0.0f);
  std::vector<float> loadings(result.weights.size(), 0.0f);
  std::vector<float> s = cross_product;

  @autoreleasepool {
    MetalState& state = metal_state();
    const NSUInteger x_row_bytes = matrix_row_bytes(predictors);
    const NSUInteger predictor_vector_row_bytes = matrix_row_bytes(1);
    const NSUInteger sample_vector_row_bytes = matrix_row_bytes(1);
    id<MTLBuffer> x_buffer = make_matrix_buffer(state.device, x, rows, predictors, x_row_bytes);
    id<MTLBuffer> weight_buffer = [state.device
      newBufferWithLength:predictor_vector_row_bytes * static_cast<NSUInteger>(predictors)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> score_buffer = [state.device
      newBufferWithLength:sample_vector_row_bytes * static_cast<NSUInteger>(rows)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> loading_buffer = [state.device
      newBufferWithLength:predictor_vector_row_bytes * static_cast<NSUInteger>(predictors)
      options:MTLResourceStorageModeShared];
    if (weight_buffer == nil || score_buffer == nil || loading_buffer == nil) {
      throw std::runtime_error("Failed to allocate resident Metal SIMPLS buffers.");
    }

    MPSMatrixDescriptor* x_descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:static_cast<NSUInteger>(rows)
      columns:static_cast<NSUInteger>(predictors)
      rowBytes:x_row_bytes
      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* weight_descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:static_cast<NSUInteger>(predictors)
      columns:1
      rowBytes:predictor_vector_row_bytes
      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* score_descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:static_cast<NSUInteger>(rows)
      columns:1
      rowBytes:sample_vector_row_bytes
      dataType:MPSDataTypeFloat32];
    MPSMatrix* x_matrix = [[MPSMatrix alloc] initWithBuffer:x_buffer descriptor:x_descriptor];
    MPSMatrix* weight_matrix = [[MPSMatrix alloc] initWithBuffer:weight_buffer descriptor:weight_descriptor];
    MPSMatrix* score_matrix = [[MPSMatrix alloc] initWithBuffer:score_buffer descriptor:score_descriptor];
    MPSMatrix* loading_matrix = [[MPSMatrix alloc] initWithBuffer:loading_buffer descriptor:weight_descriptor];
    MPSMatrixMultiplication* project = [[MPSMatrixMultiplication alloc]
      initWithDevice:state.device
      transposeLeft:false
      transposeRight:false
      resultRows:static_cast<NSUInteger>(rows)
      resultColumns:1
      interiorColumns:static_cast<NSUInteger>(predictors)
      alpha:1.0
      beta:0.0];
    MPSMatrixMultiplication* transpose_project = [[MPSMatrixMultiplication alloc]
      initWithDevice:state.device
      transposeLeft:true
      transposeRight:false
      resultRows:static_cast<NSUInteger>(predictors)
      resultColumns:1
      interiorColumns:static_cast<NSUInteger>(rows)
      alpha:1.0
      beta:0.0];

    auto write_vector = [](id<MTLBuffer> buffer, NSUInteger row_bytes, const std::vector<float>& values) {
      char* base = static_cast<char*>([buffer contents]);
      for (std::size_t i = 0; i < values.size(); ++i) {
        *reinterpret_cast<float*>(base + static_cast<NSUInteger>(i) * row_bytes) = values[i];
      }
    };
    auto read_vector = [](id<MTLBuffer> buffer, NSUInteger row_bytes, int size) {
      std::vector<float> values(static_cast<std::size_t>(size), 0.0f);
      const char* base = static_cast<const char*>([buffer contents]);
      for (int i = 0; i < size; ++i) {
        values[static_cast<std::size_t>(i)] =
          *reinterpret_cast<const float*>(base + static_cast<NSUInteger>(i) * row_bytes);
      }
      return values;
    };
    auto x_products = [&](const std::vector<float>& vector) {
      write_vector(weight_buffer, predictor_vector_row_bytes, vector);
      id<MTLCommandBuffer> command = [state.queue commandBuffer];
      [project encodeToCommandBuffer:command
                          leftMatrix:x_matrix
                         rightMatrix:weight_matrix
                        resultMatrix:score_matrix];
      [transpose_project encodeToCommandBuffer:command
                                    leftMatrix:x_matrix
                                   rightMatrix:score_matrix
                                  resultMatrix:loading_matrix];
      wait_for_command(command, "Metal SIMPLS projection pair failed");
      return std::make_pair(
        read_vector(score_buffer, sample_vector_row_bytes, rows),
        read_vector(loading_buffer, predictor_vector_row_bytes, predictors)
      );
    };

    for (int component = 0; component < components; ++component) {
      std::vector<float> right(static_cast<std::size_t>(responses),
        1.0f / std::sqrt(static_cast<float>(std::max(1, responses))));
      std::vector<float> weight(static_cast<std::size_t>(predictors), 0.0f);
      for (int iteration = 0; iteration < 80; ++iteration) {
        weight = cross_product_times(s, predictors, responses, right);
        if (vector_norm(weight) <= 1e-6f) break;
        normalize_vector(weight);
        right = cross_product_transpose_times(s, predictors, responses, weight);
        if (vector_norm(right) <= 1e-6f) break;
        normalize_vector(right);
      }
      if (vector_norm(weight) <= 1e-6f) {
        std::fill(weight.begin(), weight.end(), 0.0f);
        weight.front() = 1.0f;
      }
      remove_stored_columns(weight, result.weights, predictors, components, component);
      if (vector_norm(weight) <= 1e-6f) {
        std::fill(weight.begin(), weight.end(), 0.0f);
        weight[static_cast<std::size_t>(component % predictors)] = 1.0f;
      } else {
        normalize_vector(weight);
      }

      for (int iteration = 0; iteration < 120; ++iteration) {
        std::vector<float> next = cross_product_times(
          s,
          predictors,
          responses,
          cross_product_transpose_times(s, predictors, responses, weight)
        );
        remove_stored_columns(next, result.weights, predictors, components, component);
        if (vector_norm(next) <= 1e-6f) break;
        normalize_vector(next);
        weight.swap(next);
      }

      right = cross_product_transpose_times(s, predictors, responses, weight);
      const float singular_value = std::max(1e-6f, vector_norm(right));
      for (float& value : right) value /= singular_value;

      auto products = x_products(weight);
      const std::vector<float>& scores = products.first;
      const float score_norm_squared = vector_dot(scores, scores);
      std::vector<float> loading = score_norm_squared > 1e-20 ? std::move(products.second) : weight;
      if (score_norm_squared > 1e-20) {
        const float scale = 1.0f / score_norm_squared;
        for (float& value : loading) value *= scale;
      }
      remove_stored_columns(loading, loadings, predictors, components, component);
      if (vector_norm(loading) <= 1e-6f) {
        loading = weight;
        remove_stored_columns(loading, loadings, predictors, components, component);
      }
      if (vector_norm(loading) <= 1e-6f) {
        std::fill(loading.begin(), loading.end(), 0.0f);
        loading[static_cast<std::size_t>(component % predictors)] = 1.0f;
        remove_stored_columns(loading, loadings, predictors, components, component);
      }
      const float loading_norm = std::max(1e-6f, vector_norm(loading));
      for (float& value : loading) value /= loading_norm;

      for (int predictor = 0; predictor < predictors; ++predictor) {
        const std::size_t position = static_cast<std::size_t>(predictor) * static_cast<std::size_t>(components) +
          static_cast<std::size_t>(component);
        result.weights[position] = weight[static_cast<std::size_t>(predictor)];
        loadings[position] = loading[static_cast<std::size_t>(predictor)];
      }
      for (int response = 0; response < responses; ++response) {
        result.y_weights[static_cast<std::size_t>(response) * static_cast<std::size_t>(components) +
          static_cast<std::size_t>(component)] = right[static_cast<std::size_t>(response)];
      }

      const std::vector<float> loading_cross = cross_product_transpose_times(
        s,
        predictors,
        responses,
        loading
      );
      for (int predictor = 0; predictor < predictors; ++predictor) {
        for (int response = 0; response < responses; ++response) {
          s[static_cast<std::size_t>(predictor) * static_cast<std::size_t>(responses) +
            static_cast<std::size_t>(response)] -= loading[static_cast<std::size_t>(predictor)] *
              loading_cross[static_cast<std::size_t>(response)];
        }
      }
    }
  }
  return result;
}

}  // namespace kodama::detail
