/*
 * SPDX-FileCopyrightText: Meta Platforms, Inc. and affiliates
 * SPDX-FileCopyrightText: 2024 Sydney Bach, The Solace Project
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Native float32 Metal primitives adapted from fastEmbedR and fastPLS.
 *
 * The top-k organization is informed by FAISS 1.14.3 (MIT) and the
 * fastEmbedR native Metal backend. MPS matrix multiplication follows the
 * fastPLS Metal backend. Portions inherited from the Faiss-mlx fused
 * list-scan/top-k organization remain under Apache-2.0; FAISS-derived and
 * kodama-cpp portions remain under MIT. This file is a modified standalone
 * adaptation. See THIRD_PARTY_NOTICES.md and licenses/.
 */

#include "metal_backend.hpp"
#include "spatial_grid_knn.hpp"

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
#include <unordered_map>
#include <vector>

namespace kodama::detail {
namespace {

constexpr int kMaximumMetalK = 128;
constexpr int kMaximumMetalLists = 1024;
constexpr int kMaximumMetalProbe = 128;
constexpr int kMetalProjectionDimension = 128;

const char* kMetalSourcePart1 = R"METAL(
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
  uint filtered;
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
  uint filtered;
};

struct KMeansParams {
  uint rows;
  uint dimensions;
  uint clusters;
};

struct KNNVoteParams {
  uint samples;
  uint neighbors;
  int fallback_label;
};

struct SpatialGridParams {
  uint rows;
  uint dimensions;
  uint k;
  uint nonself_k;
  uint bins;
  uint include_self;
  float min_x;
  float min_y;
  float min_z;
  float cell_x;
  float cell_y;
  float cell_z;
};

int spatial_grid_coord(float value, float minimum, float cell_size, uint bins) {
  return clamp(int((value - minimum) / cell_size), 0, int(bins) - 1);
}

int spatial_grid_cell(int x, int y, int z, constant SpatialGridParams& p) {
  return p.dimensions == 3u ? (z * int(p.bins) + y) * int(p.bins) + x : y * int(p.bins) + x;
}

void spatial_grid_insert(
  thread float* distances,
  thread int* indices,
  uint k,
  float distance,
  int index
) {
  if (index < 0 || (distance > distances[k - 1u]) ||
      (distance == distances[k - 1u] && index >= indices[k - 1u])) return;
  uint position = k - 1u;
  while (position > 0u &&
         (distance < distances[position - 1u] ||
          (distance == distances[position - 1u] && index < indices[position - 1u]))) {
    distances[position] = distances[position - 1u];
    indices[position] = indices[position - 1u];
    --position;
  }
  distances[position] = distance;
  indices[position] = index;
}

void spatial_grid_add_cell(
  device const float* data,
  device const int* offsets,
  device const int* cell_rows,
  constant SpatialGridParams& p,
  uint query,
  int x,
  int y,
  int z,
  thread float* best_distances,
  thread int* best_indices
) {
  if (x < 0 || y < 0 || x >= int(p.bins) || y >= int(p.bins)) return;
  if (p.dimensions == 3u && (z < 0 || z >= int(p.bins))) return;
  const int cell = spatial_grid_cell(x, y, z, p);
  const uint query_base = query * p.dimensions;
  const float qx = data[query_base];
  const float qy = data[query_base + 1u];
  const float qz = p.dimensions == 3u ? data[query_base + 2u] : 0.0f;
  for (int position = offsets[cell]; position < offsets[cell + 1]; ++position) {
    const int candidate = cell_rows[position];
    if (candidate == int(query) || candidate < 0 || candidate >= int(p.rows)) continue;
    const uint candidate_base = uint(candidate) * p.dimensions;
    const float dx = qx - data[candidate_base];
    const float dy = qy - data[candidate_base + 1u];
    float distance = dx * dx + dy * dy;
    if (p.dimensions == 3u) {
      const float dz = qz - data[candidate_base + 2u];
      distance += dz * dz;
    }
    spatial_grid_insert(best_distances, best_indices, p.nonself_k, distance, candidate);
  }
}

float spatial_grid_lower_outside(
  float x,
  float y,
  float z,
  constant SpatialGridParams& p,
  int x0,
  int x1,
  int y0,
  int y1,
  int z0,
  int z1
) {
  float best = INFINITY;
  if (x0 > 0) {
    const float delta = max(0.0f, x - (p.min_x + float(x0) * p.cell_x));
    best = min(best, delta * delta);
  }
  if (x1 + 1 < int(p.bins)) {
    const float delta = max(0.0f, p.min_x + float(x1 + 1) * p.cell_x - x);
    best = min(best, delta * delta);
  }
  if (y0 > 0) {
    const float delta = max(0.0f, y - (p.min_y + float(y0) * p.cell_y));
    best = min(best, delta * delta);
  }
  if (y1 + 1 < int(p.bins)) {
    const float delta = max(0.0f, p.min_y + float(y1 + 1) * p.cell_y - y);
    best = min(best, delta * delta);
  }
  if (p.dimensions == 3u) {
    if (z0 > 0) {
      const float delta = max(0.0f, z - (p.min_z + float(z0) * p.cell_z));
      best = min(best, delta * delta);
    }
    if (z1 + 1 < int(p.bins)) {
      const float delta = max(0.0f, p.min_z + float(z1 + 1) * p.cell_z - z);
      best = min(best, delta * delta);
    }
  }
  return best;
}

kernel void spatial_grid_self_knn(
  device const float* data [[buffer(0)]],
  device const int* offsets [[buffer(1)]],
  device const int* cell_rows [[buffer(2)]],
  device int* output_indices [[buffer(3)]],
  device float* output_distances [[buffer(4)]],
  constant SpatialGridParams& p [[buffer(5)]],
  uint query [[thread_position_in_grid]]
) {
  if (query >= p.rows) return;
  thread float best_distances[MAX_K];
  thread int best_indices[MAX_K];
  for (uint i = 0u; i < p.nonself_k; ++i) {
    best_distances[i] = INFINITY;
    best_indices[i] = INT_MAX;
  }

  const uint base = query * p.dimensions;
  const float qx = data[base];
  const float qy = data[base + 1u];
  const float qz = p.dimensions == 3u ? data[base + 2u] : 0.0f;
  const int cx = spatial_grid_coord(qx, p.min_x, p.cell_x, p.bins);
  const int cy = spatial_grid_coord(qy, p.min_y, p.cell_y, p.bins);
  const int cz = p.dimensions == 3u ? spatial_grid_coord(qz, p.min_z, p.cell_z, p.bins) : 0;

  for (int radius = 0; radius <= int(p.bins) && p.nonself_k > 0u; ++radius) {
    const int raw_x0 = cx - radius;
    const int raw_x1 = cx + radius;
    const int raw_y0 = cy - radius;
    const int raw_y1 = cy + radius;
    const int raw_z0 = cz - radius;
    const int raw_z1 = cz + radius;
    const int x0 = max(0, raw_x0);
    const int x1 = min(int(p.bins) - 1, raw_x1);
    const int y0 = max(0, raw_y0);
    const int y1 = min(int(p.bins) - 1, raw_y1);
    const int z0 = p.dimensions == 3u ? max(0, raw_z0) : 0;
    const int z1 = p.dimensions == 3u ? min(int(p.bins) - 1, raw_z1) : 0;

    if (p.dimensions == 2u) {
      if (radius == 0) {
        spatial_grid_add_cell(data, offsets, cell_rows, p, query, cx, cy, 0, best_distances, best_indices);
      } else {
        for (int ix = raw_x0; ix <= raw_x1; ++ix) {
          spatial_grid_add_cell(data, offsets, cell_rows, p, query, ix, raw_y0, 0, best_distances, best_indices);
          if (raw_y1 != raw_y0) {
            spatial_grid_add_cell(data, offsets, cell_rows, p, query, ix, raw_y1, 0, best_distances, best_indices);
          }
        }
        for (int iy = raw_y0 + 1; iy <= raw_y1 - 1; ++iy) {
          spatial_grid_add_cell(data, offsets, cell_rows, p, query, raw_x0, iy, 0, best_distances, best_indices);
          if (raw_x1 != raw_x0) {
            spatial_grid_add_cell(data, offsets, cell_rows, p, query, raw_x1, iy, 0, best_distances, best_indices);
          }
        }
      }
    } else {
      for (int iz = raw_z0; iz <= raw_z1; ++iz) {
        for (int iy = raw_y0; iy <= raw_y1; ++iy) {
          for (int ix = raw_x0; ix <= raw_x1; ++ix) {
            if (ix != raw_x0 && ix != raw_x1 && iy != raw_y0 && iy != raw_y1 &&
                iz != raw_z0 && iz != raw_z1) continue;
            spatial_grid_add_cell(data, offsets, cell_rows, p, query, ix, iy, iz, best_distances, best_indices);
          }
        }
      }
    }
    if (best_indices[p.nonself_k - 1u] != INT_MAX &&
        spatial_grid_lower_outside(qx, qy, qz, p, x0, x1, y0, y1, z0, z1) >
          best_distances[p.nonself_k - 1u]) break;
  }

  const uint output_base = query * p.k;
  const uint first_nonself = p.include_self != 0u ? 1u : 0u;
  if (p.include_self != 0u) {
    output_indices[output_base] = int(query);
    output_distances[output_base] = 0.0f;
  }
  for (uint i = 0u; i < p.nonself_k; ++i) {
    output_indices[output_base + first_nonself + i] =
      best_indices[i] == INT_MAX ? -1 : best_indices[i];
    output_distances[output_base + first_nonself + i] = best_distances[i];
  }
}

struct LandmarkProjectionParams {
  uint samples;
  uint neighbors;
  uint projection_k;
  int fallback_label;
  uint epoch;
};

struct LandmarkScatterParams {
  uint landmarks;
  uint samples;
  uint epoch;
};

struct KODAMADissimilarityParams {
  uint runs;
  uint samples;
  uint neighbors;
  uint sort_width;
  uint input_one_based;
  uint output_one_based;
};

struct UMAPParams {
  uint samples;
  uint width;
  uint epochs;
  uint negative_sample_rate;
  uint seed;
  float learning_rate;
  float a;
  float b;
  float max_weight;
  float repulsion_strength;
};

uint umap_mix_uint(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

uint umap_negative_vertex(
    uint samples, uint seed, uint epoch, uint head, uint tail,
    uint edge, uint draw) {
  uint x = seed;
  x ^= (epoch + 1u) * 0x9e3779b9u;
  x ^= (head + 1u) * 0x85ebca6bu;
  x ^= (tail + 1u) * 0xc2b2ae35u;
  x ^= (edge + 1u) * 0x27d4eb2du;
  x ^= (draw + 1u) * 0x165667b1u;
  return umap_mix_uint(x) % samples;
}

float umap_positive_pow(float x, float exponent) {
  if (x <= 0.0f) return 0.0f;
  const uint x_bits = as_type<uint>(x);
  constexpr float exponent_bias_word = 1064866805.0f;
  const int whole = int(exponent);
  const float fractional = exponent - float(whole);
  const uint interp_bits = uint(
    fractional * (float(x_bits) - exponent_bias_word) + exponent_bias_word
  );
  float integer_power = 1.0f;
  float base = x;
  int remaining = whole;
  while (remaining > 0) {
    if ((remaining & 1) != 0) integer_power *= base;
    base *= base;
    remaining >>= 1;
  }
  return integer_power * as_type<float>(interp_bits);
}

int umap_samples_at_epoch(float period, uint epoch) {
  if (period <= 0.0f || !isfinite(period)) return 0;
  const int current = int(floor(float(epoch + 1u) / period));
  const int previous = int(floor(float(epoch) / period));
  return max(0, current - previous);
}

int umap_negative_samples_at_epoch(
    float period, uint negative_sample_rate, uint epoch) {
  if (period <= 0.0f || !isfinite(period) || negative_sample_rate == 0u) {
    return 0;
  }
  const int current_positive = int(floor(float(epoch + 1u) / period));
  const int previous_positive = int(floor(float(epoch) / period));
  if (current_positive <= previous_positive) return 0;
  const float negative_period = period / float(negative_sample_rate);
  const int current_total = max(
    0,
    int(floor(((float(epoch + 1u) - negative_period) / negative_period) + 1.0e-6f))
  );
  int previous_total = 0;
  if (previous_positive > 0) {
    const float previous_active = ceil(float(previous_positive) * period);
    previous_total = max(
      0,
      int(floor(((previous_active - negative_period) / negative_period) + 1.0e-6f))
    );
  }
  return max(0, current_total - previous_total);
}

int umap_fixed_delta(float value) {
  return int(clamp(value * 65536.0f, -2140000000.0f, 2140000000.0f));
}

struct PLSLDAParams {
  uint rows;
  uint components;
  uint score_stride;
  uint classes;
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
    device const int* allowed_local_ids [[buffer(5)]],
    constant ExactParams& params [[buffer(6)]],
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
    const int output_id = params.filtered == 0 ? int(candidate) : allowed_local_ids[candidate];
    if (output_id < 0) continue;
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
    if (lane == 0) insert_sorted(local_distance, local_id, base, params.k, distance, output_id);
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

kernel void gather_kmeans_centroids(
    device const float* data [[buffer(0)]],
    device const int* initial_point_indices [[buffer(1)]],
    device float* centroids [[buffer(2)]],
    constant KMeansParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = params.clusters * params.dimensions;
  if (gid >= total) return;
  const uint cluster = gid / params.dimensions;
  const uint dimension = gid - cluster * params.dimensions;
  const uint row = uint(initial_point_indices[cluster]);
  centroids[gid] = data[row * params.dimensions + dimension];
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
    device const int* allowed_local_ids [[buffer(9)]],
    constant IVFSearchParams& params [[buffer(10)]],
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
      const int output_id = params.filtered == 0 ? candidate : allowed_local_ids[candidate];
      if (output_id < 0) continue;
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
      if (lane == 0) insert_sorted(local_distance, local_id, fine_base, params.k, distance, output_id);
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

inline void atomic_add_float_relaxed(device atomic_uint* address, float value) {
  uint expected = atomic_load_explicit(address, memory_order_relaxed);
  while (true) {
    const uint desired = as_type<uint>(as_type<float>(expected) + value);
    if (atomic_compare_exchange_weak_explicit(
          address,
          &expected,
          desired,
          memory_order_relaxed,
          memory_order_relaxed)) {
      return;
    }
  }
}

kernel void clear_kmeans_accumulators(
    device atomic_uint* sums [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant KMeansParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = params.clusters * params.dimensions;
  if (gid < total) atomic_store_explicit(&sums[gid], 0u, memory_order_relaxed);
  if (gid < params.clusters) atomic_store_explicit(&counts[gid], 0u, memory_order_relaxed);
}

kernel void accumulate_kmeans_centroids(
    device const float* data [[buffer(0)]],
    device const int* assignments [[buffer(1)]],
    device atomic_uint* sums [[buffer(2)]],
    device atomic_uint* counts [[buffer(3)]],
    constant KMeansParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = params.rows * params.dimensions;
  if (gid >= total) return;
  const uint row = gid / params.dimensions;
  const uint dimension = gid - row * params.dimensions;
  const int cluster = assignments[row];
  if (cluster < 0 || uint(cluster) >= params.clusters) return;
  atomic_add_float_relaxed(
    &sums[uint(cluster) * params.dimensions + dimension], data[gid]);
  if (dimension == 0) atomic_fetch_add_explicit(&counts[cluster], 1u, memory_order_relaxed);
}

kernel void finalize_kmeans_centroids(
    device const float* data [[buffer(0)]],
    device const atomic_uint* sums [[buffer(1)]],
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
    centroids[gid] = as_type<float>(atomic_load_explicit(&sums[gid], memory_order_relaxed)) /
      float(count);
  } else {
    const uint replacement = uint(initial_point_indices[(cluster + iteration) % params.rows]);
    centroids[gid] = data[replacement * params.dimensions + dimension];
  }
}

kernel void clear_ivf_list_counts(
    device atomic_uint* counts [[buffer(0)]],
    constant KMeansParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < params.clusters) {
    atomic_store_explicit(&counts[gid], 0u, memory_order_relaxed);
  }
}

kernel void count_ivf_assignments(
    device const int* assignments [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant KMeansParams& params [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= params.rows) return;
  const int cluster = assignments[row];
  if (cluster >= 0 && uint(cluster) < params.clusters) {
    atomic_fetch_add_explicit(&counts[cluster], 1u, memory_order_relaxed);
  }
}

kernel void prefix_ivf_list_counts(
    device const atomic_uint* counts [[buffer(0)]],
    device uint* offsets [[buffer(1)]],
    device atomic_uint* cursor [[buffer(2)]],
    constant KMeansParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid != 0) return;
  uint running = 0;
  offsets[0] = 0;
  for (uint cluster = 0; cluster < params.clusters; ++cluster) {
    atomic_store_explicit(&cursor[cluster], running, memory_order_relaxed);
    running += atomic_load_explicit(&counts[cluster], memory_order_relaxed);
    offsets[cluster + 1] = running;
  }
}

kernel void scatter_ivf_list_ids(
    device const int* assignments [[buffer(0)]],
    device atomic_uint* cursor [[buffer(1)]],
    device int* list_ids [[buffer(2)]],
    constant KMeansParams& params [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= params.rows) return;
  const int cluster = assignments[row];
  if (cluster < 0 || uint(cluster) >= params.clusters) return;
  const uint position = atomic_fetch_add_explicit(
    &cursor[cluster],
    1u,
    memory_order_relaxed
  );
  list_ids[position] = int(row);
}

kernel void resident_knn_vote(
    device const int* neighbor_rows [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device const int* labels [[buffer(2)]],
    device int* predictions [[buffer(3)]],
    constant KNNVoteParams& params [[buffer(4)]],
    uint query [[thread_position_in_grid]]) {
  if (query >= params.samples) return;
  const uint base = query * params.neighbors;
  int best_label = params.fallback_label;
  int best_count = -1;
  float best_score = -INFINITY;
  for (uint rank = 0; rank < params.neighbors; ++rank) {
    const int row = neighbor_rows[base + rank];
    if (row < 0 || uint(row) >= params.samples) continue;
    const int candidate = labels[row];
    bool seen = false;
    for (uint prior = 0; prior < rank; ++prior) {
      const int prior_row = neighbor_rows[base + prior];
      if (prior_row >= 0 && uint(prior_row) < params.samples &&
          labels[prior_row] == candidate) {
        seen = true;
        break;
      }
    }
    if (seen) continue;
    int count = 0;
    float score = 0.0f;
    for (uint other = rank; other < params.neighbors; ++other) {
      const int other_row = neighbor_rows[base + other];
      if (other_row < 0 || uint(other_row) >= params.samples ||
          labels[other_row] != candidate) {
        continue;
      }
      ++count;
      score += scores[base + other];
    }
    if (count > best_count ||
        (count == best_count && score > best_score) ||
        (count == best_count && score == best_score &&
         candidate < best_label)) {
      best_label = candidate;
      best_count = count;
      best_score = score;
    }
  }
  predictions[query] = best_count < 0 ? params.fallback_label : best_label;
}

kernel void pls_class_score_sums(
    device const float* scores [[buffer(0)]],
    device const int* labels [[buffer(1)]],
    device float* class_sums [[buffer(2)]],
    constant PLSLDAParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = params.classes * params.components;
  if (gid >= total) return;
  const uint cls = gid / params.components;
  const uint component = gid - cls * params.components;
  float sum = 0.0f;
  for (uint row = 0; row < params.rows; ++row) {
    if (labels[row] == int(cls)) {
      sum += scores[row * params.score_stride + component];
    }
  }
  class_sums[gid] = sum;
}

kernel void pls_lda_score_labels(
    device const float* scores [[buffer(0)]],
    device const float* linear [[buffer(1)]],
    device const float* constants [[buffer(2)]],
    device const int* class_labels [[buffer(3)]],
    device int* predictions [[buffer(4)]],
    constant PLSLDAParams& params [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= params.rows) return;
  const uint score_base = row * params.score_stride;
  int best = 0;
  float best_score = -INFINITY;
  for (uint cls = 0; cls < params.classes; ++cls) {
    float value = constants[cls];
    const uint linear_base = cls * params.components;
    for (uint component = 0; component < params.components; ++component) {
      value += scores[score_base + component] * linear[linear_base + component];
    }
    if (value > best_score) {
      best_score = value;
      best = int(cls);
    }
  }
  predictions[row] = class_labels[best];
}

 kernel void scatter_landmark_labels(
    device const int* landmark_rows [[buffer(0)]],
    device const int* landmark_input_labels [[buffer(1)]],
    device int* landmark_epoch [[buffer(2)]],
    device int* labels [[buffer(3)]],
    constant LandmarkScatterParams& params [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= params.landmarks) return;
  const int row = landmark_rows[i];
  if (row < 0 || uint(row) >= params.samples) return;
  labels[row] = landmark_input_labels[i];
  landmark_epoch[row] = int(params.epoch);
}

kernel void project_landmark_labels(
    device const int* graph_indices [[buffer(0)]],
    device const int* landmark_epoch [[buffer(1)]],
    device const int* labels [[buffer(2)]],
    device int* projected [[buffer(3)]],
    constant LandmarkProjectionParams& params [[buffer(4)]],
    uint query [[thread_position_in_grid]]) {
  if (query >= params.samples) return;
  if (landmark_epoch[query] == int(params.epoch)) {
    projected[query] = labels[query];
    return;
  }
  const uint base = query * params.neighbors;
  uint cutoff = params.neighbors;
  uint accepted = 0;
  for (uint rank = 0; rank < params.neighbors; ++rank) {
    const int neighbor = graph_indices[base + rank];
    if (neighbor < 0 || uint(neighbor) >= params.samples ||
        landmark_epoch[neighbor] != int(params.epoch)) {
      continue;
    }
    ++accepted;
    if (accepted == params.projection_k) {
      cutoff = rank + 1;
      break;
    }
  }
  if (accepted == 0) {
    for (uint first_rank = 0; first_rank < params.neighbors; ++first_rank) {
      const int first = graph_indices[base + first_rank];
      if (first < 0 || uint(first) >= params.samples || uint(first) == query) {
        continue;
      }
      const uint second_base = uint(first) * params.neighbors;
      for (uint second_rank = 0; second_rank < params.neighbors; ++second_rank) {
        const int second = graph_indices[second_base + second_rank];
        if (second >= 0 && uint(second) < params.samples &&
            landmark_epoch[second] == int(params.epoch)) {
          projected[query] = labels[second];
          return;
        }
      }
    }
    projected[query] = params.fallback_label;
    return;
  }

  int best_label = params.fallback_label;
  int best_count = -1;
  for (uint rank = 0; rank < cutoff; ++rank) {
    const int neighbor = graph_indices[base + rank];
    if (neighbor < 0 || uint(neighbor) >= params.samples ||
        landmark_epoch[neighbor] != int(params.epoch)) {
      continue;
    }
    const int candidate = labels[neighbor];
    bool seen = false;
    for (uint prior = 0; prior < rank; ++prior) {
      const int prior_neighbor = graph_indices[base + prior];
      if (prior_neighbor >= 0 && uint(prior_neighbor) < params.samples &&
          landmark_epoch[prior_neighbor] == int(params.epoch) &&
          labels[prior_neighbor] == candidate) {
        seen = true;
        break;
      }
    }
    if (seen) continue;
    int count = 0;
    for (uint other = rank; other < cutoff; ++other) {
      const int other_neighbor = graph_indices[base + other];
      if (other_neighbor >= 0 && uint(other_neighbor) < params.samples &&
          landmark_epoch[other_neighbor] == int(params.epoch) &&
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
  projected[query] = best_count < 0 ? params.fallback_label : best_label;
}

struct ConstrainedMajorityParams {
  uint samples;
  uint groups;
  uint label_width;
};

kernel void count_constrained_labels(
    device const int* labels [[buffer(0)]],
    device const int* groups [[buffer(1)]],
    device atomic_uint* counts [[buffer(2)]],
    constant ConstrainedMajorityParams& params [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= params.samples) return;
  const int label = labels[row];
  if (label >= 0 && uint(label) < params.label_width) {
    atomic_fetch_add_explicit(
      counts + uint(groups[row]) * params.label_width + uint(label),
      1u, memory_order_relaxed);
  }
}

kernel void select_constrained_majority(
    device const uint* counts [[buffer(0)]],
    device int* group_labels [[buffer(1)]],
    constant ConstrainedMajorityParams& params [[buffer(2)]],
    uint group [[thread_position_in_grid]]) {
  if (group >= params.groups) return;
  int best_label = 0;
  uint best_count = 0u;
  const uint base = group * params.label_width;
  for (uint label = 0; label < params.label_width; ++label) {
    const uint count = counts[base + label];
    if (label == 0u || count > best_count) {
      best_count = count;
      best_label = int(label);
    }
  }
  group_labels[group] = best_label;
}

kernel void apply_constrained_majority(
    device int* labels [[buffer(0)]],
    device const int* groups [[buffer(1)]],
    device const int* group_labels [[buffer(2)]],
    constant ConstrainedMajorityParams& params [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
  if (row < params.samples) labels[row] = group_labels[groups[row]];
}

)METAL";

const char* kMetalSourcePart2 = R"METAL(

kernel void kodama_dissimilarity_resident(
    device int* indices [[buffer(0)]],
    device float* distances [[buffer(1)]],
    device const int* result_labels [[buffer(2)]],
    constant KODAMADissimilarityParams& params [[buffer(3)]],
    threadgroup float* row_dist [[threadgroup(0)]],
    threadgroup int* row_idx [[threadgroup(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  if (row >= params.samples) return;
  const uint row_offset = row * params.neighbors;
  if (tid < params.neighbors) {
    const uint offset = row_offset + tid;
    const int stored_neighbor = indices[offset];
    const int neighbor =
      stored_neighbor >= 0 && params.input_one_based != 0 ?
        stored_neighbor - 1 :
        stored_neighbor;
    float distance = distances[offset];
    if (neighbor < 0 || uint(neighbor) >= params.samples ||
        !isfinite(distance)) {
      row_dist[tid] = INFINITY;
      row_idx[tid] = neighbor;
    } else {
      uint same = 0;
      uint valid = 0;
      for (uint run = 0; run < params.runs; ++run) {
        const int lhs = result_labels[run * params.samples + row];
        const int rhs = result_labels[run * params.samples + uint(neighbor)];
        if (lhs == 0 || rhs == 0) continue;
        ++valid;
        if (lhs == rhs) ++same;
      }
      if (same == 0 || valid == 0) {
        distance = INFINITY;
      } else {
        const float agreement = float(same) / float(valid);
        distance = (1.0f + distance) / (agreement * agreement);
      }
      row_dist[tid] = distance;
      row_idx[tid] = neighbor;
    }
  } else if (tid < params.sort_width) {
    row_dist[tid] = INFINITY;
    row_idx[tid] = 0x7fffffff;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint width = 2; width <= params.sort_width; width <<= 1) {
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {
      const uint other = tid ^ stride;
      if (other > tid && other < params.sort_width) {
        const bool ascending = (tid & width) == 0;
        const float self_dist = row_dist[tid];
        const int self_idx = row_idx[tid];
        const float other_dist = row_dist[other];
        const int other_idx = row_idx[other];
        const bool self_greater =
          self_dist > other_dist ||
          (self_dist == other_dist && self_idx > other_idx);
        const bool other_greater =
          other_dist > self_dist ||
          (other_dist == self_dist && other_idx > self_idx);
        const bool swap_pair = ascending ? self_greater : other_greater;
        if (swap_pair) {
          row_dist[tid] = other_dist;
          row_idx[tid] = other_idx;
          row_dist[other] = self_dist;
          row_idx[other] = self_idx;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  if (tid < params.neighbors) {
    const uint offset = row_offset + tid;
    distances[offset] = row_dist[tid];
    indices[offset] =
      row_idx[tid] >= 0 && params.output_one_based != 0 ?
        row_idx[tid] + 1 :
        row_idx[tid];
  }
}

uint umap_clean_hash(
    uint seed, uint epoch, uint head, uint tail, uint edge,
    uint sample, uint stream) {
  uint x = seed;
  x ^= (epoch + 1u) * 0x9e3779b9u;
  x ^= (head + 1u) * 0x85ebca6bu;
  x ^= (tail + 1u) * 0xc2b2ae35u;
  x ^= (edge + 1u) * 0x27d4eb2du;
  x ^= (sample + 1u) * 0x165667b1u;
  x ^= (stream + 1u) * 0xd3a2646cu;
  return umap_mix_uint(x);
}

float umap_clean_uniform01(
    uint seed, uint epoch, uint head, uint tail, uint edge,
    uint sample, uint stream) {
  return float(
    umap_clean_hash(seed, epoch, head, tail, edge, sample, stream) &
    0x00ffffffu
  ) / 16777216.0f;
}

kernel void kodama_umap_clean_epoch(
    device atomic_int* layout [[buffer(0)]],
    device const int* neighbors [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    constant UMAPParams& params [[buffer(3)]],
    constant uint& epoch [[buffer(4)]],
    uint head [[thread_position_in_grid]]) {
  if (head >= params.samples) return;
  constexpr float inverse_scale = 1.0f / 65536.0f;
  const float progress = float(epoch) / max(1.0f, float(params.epochs - 1u));
  const float alpha = params.learning_rate * (1.0f - progress);
  const uint head_base = head * 2u;

  for (uint edge = 0; edge < params.width; ++edge) {
    const uint position = head * params.width + edge;
    const int tail_value = neighbors[position];
    const float weight = weights[position];
    if (tail_value < 0 || uint(tail_value) >= params.samples ||
        uint(tail_value) == head || !isfinite(weight) || weight <= 0.0f) {
      continue;
    }
    const uint tail = uint(tail_value);
    const float edge_probability = clamp(
      weight / max(params.max_weight, 1.0e-6f), 0.0f, 1.0f
    );
    if (edge_probability < 1.0f &&
        umap_clean_uniform01(
          params.seed, epoch, head, tail, edge, 0u, 3u
        ) >= edge_probability) {
      continue;
    }

    const uint tail_base = tail * 2u;
    float head_x = float(atomic_load_explicit(
      &layout[head_base], memory_order_relaxed)) * inverse_scale;
    float head_y = float(atomic_load_explicit(
      &layout[head_base + 1u], memory_order_relaxed)) * inverse_scale;
    const float tail_x = float(atomic_load_explicit(
      &layout[tail_base], memory_order_relaxed)) * inverse_scale;
    const float tail_y = float(atomic_load_explicit(
      &layout[tail_base + 1u], memory_order_relaxed)) * inverse_scale;
    const float2 difference = float2(head_x - tail_x, head_y - tail_y);
    const float distance_squared = max(
      1.1920928955078125e-7f, dot(difference, difference)
    );
    const float distance_power = umap_positive_pow(distance_squared, params.b);
    const float attraction =
      -2.0f * params.a * params.b * (distance_power / distance_squared) /
      (params.a * distance_power + 1.0f);
    const float2 attractive_delta = alpha * float2(
      clamp(attraction * difference.x, -4.0f, 4.0f),
      clamp(attraction * difference.y, -4.0f, 4.0f)
    );
    atomic_fetch_add_explicit(
      &layout[head_base], umap_fixed_delta(attractive_delta.x), memory_order_relaxed);
    atomic_fetch_add_explicit(
      &layout[head_base + 1u], umap_fixed_delta(attractive_delta.y), memory_order_relaxed);
    atomic_fetch_add_explicit(
      &layout[tail_base], umap_fixed_delta(-attractive_delta.x), memory_order_relaxed);
    atomic_fetch_add_explicit(
      &layout[tail_base + 1u], umap_fixed_delta(-attractive_delta.y), memory_order_relaxed);

    for (uint draw = 0; draw < params.negative_sample_rate; ++draw) {
      const uint negative = umap_clean_hash(
        params.seed, epoch, head, tail, edge, draw, 17u
      ) % params.samples;
      if (negative == head || negative == tail) continue;
      const uint negative_base = negative * 2u;
      head_x = float(atomic_load_explicit(
        &layout[head_base], memory_order_relaxed)) * inverse_scale;
      head_y = float(atomic_load_explicit(
        &layout[head_base + 1u], memory_order_relaxed)) * inverse_scale;
      const float negative_x = float(atomic_load_explicit(
        &layout[negative_base], memory_order_relaxed)) * inverse_scale;
      const float negative_y = float(atomic_load_explicit(
        &layout[negative_base + 1u], memory_order_relaxed)) * inverse_scale;
      const float2 negative_difference =
        float2(head_x - negative_x, head_y - negative_y);
      const float negative_distance_squared = max(
        1.1920928955078125e-7f,
        dot(negative_difference, negative_difference)
      );
      const float negative_power = umap_positive_pow(
        negative_distance_squared, params.b);
      const float repulsion =
        params.repulsion_strength * 2.0f * params.b /
        ((0.001f + negative_distance_squared) *
         (params.a * negative_power + 1.0f));
      const float2 repulsive_delta = alpha * float2(
        clamp(repulsion * negative_difference.x, -4.0f, 4.0f),
        clamp(repulsion * negative_difference.y, -4.0f, 4.0f)
      );
      atomic_fetch_add_explicit(
        &layout[head_base], umap_fixed_delta(repulsive_delta.x), memory_order_relaxed);
      atomic_fetch_add_explicit(
        &layout[head_base + 1u], umap_fixed_delta(repulsive_delta.y), memory_order_relaxed);
    }
  }
}

struct OpenTsneMetalParams {
  uint n;
  uint seed;
  float learning_rate;
  float exaggeration;
  float momentum;
  float min_gain;
  float max_step_norm;
  float inv_sum_q;
};

struct OpenTsneFFTGridParams {
  uint n;
  uint grid_size;
  uint fft_size;
  float lower_x;
  float lower_y;
  float inv_spacing;
  float spacing;
  float inv_sum_q;
};

struct Center2 {
  float x;
  float y;
};

struct OpenTsneLayoutStats {
  float min_x;
  float max_x;
  float min_y;
  float max_y;
  float sum_x;
  float sum_y;
};

float sign_component(float x) {
  if (x > 0.0f) return 1.0f;
  if (x < 0.0f) return -1.0f;
  return 0.0f;
}

kernel void opentsne_sum_q_rows(
  device const float2* current [[buffer(0)]],
  device float* row_sums [[buffer(1)]],
  constant OpenTsneMetalParams& p [[buffer(2)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= p.n) return;
  float2 yi = current[row];
  float sum_q = 0.0f;
  for (uint j = 0u; j < p.n; ++j) {
    if (j == row) continue;
    float2 diff = yi - current[j];
    float d2 = dot(diff, diff);
    sum_q += 1.0f / (1.0f + d2);
  }
  row_sums[row] = sum_q;
}

kernel void opentsne_epoch_exact(
  device const int* row_ptr [[buffer(0)]],
  device const int* col_idx [[buffer(1)]],
  device const float* p_val [[buffer(2)]],
  device const float2* current [[buffer(3)]],
  device float2* next_current [[buffer(4)]],
  device float2* gains [[buffer(5)]],
  device float2* updates [[buffer(6)]],
  constant OpenTsneMetalParams& p [[buffer(7)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= p.n) return;
  constexpr float eps = 1.0e-12f;
  float2 yi = current[row];
  float2 grad = float2(0.0f, 0.0f);

  for (uint j = 0u; j < p.n; ++j) {
    if (j == row) continue;
    float2 diff = yi - current[j];
    float d2 = dot(diff, diff);
    float q = 1.0f / (1.0f + d2);
    grad += (-(q * q) * p.inv_sum_q) * diff;
  }

  int begin = row_ptr[row];
  int end = row_ptr[row + 1u];
  for (int pos = begin; pos < end; ++pos) {
    int j = col_idx[pos];
    if (j < 0 || uint(j) >= p.n || uint(j) == row) continue;
    float2 diff = yi - current[uint(j)];
    float d2 = dot(diff, diff);
    float q = 1.0f / (1.0f + d2);
    grad += (p.exaggeration * p_val[pos] * q) * diff;
  }

  float2 gain = gains[row];
  float2 update = updates[row];
  float sx0 = sign_component(update.x);
  float sx1 = sign_component(update.y);
  float sg0 = sign_component(grad.x);
  float sg1 = sign_component(grad.y);
  gain.x = sx0 != sg0 ? gain.x + 0.2f : gain.x * 0.8f + p.min_gain;
  gain.y = sx1 != sg1 ? gain.y + 0.2f : gain.y * 0.8f + p.min_gain;
  gain = max(gain, float2(p.min_gain, p.min_gain));

  update = p.momentum * update - p.learning_rate * gain * grad;
  float step_norm2 = dot(update, update);
  float max_step2 = p.max_step_norm * p.max_step_norm;
  if (isfinite(max_step2) && max_step2 > 0.0f && step_norm2 > max_step2) {
    update *= p.max_step_norm / (sqrt(step_norm2) + eps);
  }

  next_current[row] = yi + update;
  gains[row] = gain;
  updates[row] = update;
}

kernel void opentsne_apply_center(
  device float2* current [[buffer(0)]],
  constant float2& center [[buffer(1)]],
  constant uint& n [[buffer(2)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n) return;
  current[row] -= center;
}

float2 complex_mul(float2 a, float2 b) {
  return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

uint reverse_bits_limited(uint value, uint n_bits) {
  uint out = 0u;
  for (uint bit = 0u; bit < n_bits; ++bit) {
    out = (out << 1u) | (value & 1u);
    value >>= 1u;
  }
  return out;
}

kernel void opentsne_fft_clear_grids(
  device atomic_uint* mass [[buffer(0)]],
  device atomic_uint* mass_x [[buffer(1)]],
  device atomic_uint* mass_y [[buffer(2)]],
  constant OpenTsneFFTGridParams& g [[buffer(3)]],
  uint cell [[thread_position_in_grid]]
) {
  uint total = g.grid_size * g.grid_size;
  if (cell >= total) return;
  atomic_store_explicit(&mass[cell], as_type<uint>(0.0f), memory_order_relaxed);
  atomic_store_explicit(&mass_x[cell], as_type<uint>(0.0f), memory_order_relaxed);
  atomic_store_explicit(&mass_y[cell], as_type<uint>(0.0f), memory_order_relaxed);
}

kernel void opentsne_fft_scatter_bilinear(
  device const float2* current [[buffer(0)]],
  device atomic_uint* mass [[buffer(1)]],
  device atomic_uint* mass_x [[buffer(2)]],
  device atomic_uint* mass_y [[buffer(3)]],
  constant OpenTsneFFTGridParams& g [[buffer(4)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= g.n) return;
  float2 yi = current[row];
  if (!isfinite(yi.x)) yi.x = 0.0f;
  if (!isfinite(yi.y)) yi.y = 0.0f;
  float raw_x = (yi.x - g.lower_x) * g.inv_spacing;
  float raw_y = (yi.y - g.lower_y) * g.inv_spacing;
  float max_cell = float(g.grid_size - 1u);
  raw_x = clamp(raw_x, 0.0f, max_cell);
  raw_y = clamp(raw_y, 0.0f, max_cell);
  uint x0 = min(uint(floor(raw_x)), g.grid_size - 2u);
  uint y0 = min(uint(floor(raw_y)), g.grid_size - 2u);
  uint x1 = x0 + 1u;
  uint y1 = y0 + 1u;
  float tx = raw_x - float(x0);
  float ty = raw_y - float(y0);
  float w00 = (1.0f - tx) * (1.0f - ty);
  float w10 = tx * (1.0f - ty);
  float w01 = (1.0f - tx) * ty;
  float w11 = tx * ty;
  uint p00 = y0 * g.grid_size + x0;
  uint p10 = y0 * g.grid_size + x1;
  uint p01 = y1 * g.grid_size + x0;
  uint p11 = y1 * g.grid_size + x1;
  atomic_add_float_relaxed(&mass[p00], w00);
  atomic_add_float_relaxed(&mass[p10], w10);
  atomic_add_float_relaxed(&mass[p01], w01);
  atomic_add_float_relaxed(&mass[p11], w11);
  atomic_add_float_relaxed(&mass_x[p00], w00 * yi.x);
  atomic_add_float_relaxed(&mass_x[p10], w10 * yi.x);
  atomic_add_float_relaxed(&mass_x[p01], w01 * yi.x);
  atomic_add_float_relaxed(&mass_x[p11], w11 * yi.x);
  atomic_add_float_relaxed(&mass_y[p00], w00 * yi.y);
  atomic_add_float_relaxed(&mass_y[p10], w10 * yi.y);
  atomic_add_float_relaxed(&mass_y[p01], w01 * yi.y);
  atomic_add_float_relaxed(&mass_y[p11], w11 * yi.y);
}

kernel void opentsne_fft_load_inputs(
  device const atomic_uint* mass [[buffer(0)]],
  device const atomic_uint* mass_x [[buffer(1)]],
  device const atomic_uint* mass_y [[buffer(2)]],
  device float2* mass_fft [[buffer(3)]],
  device float2* mass_x_fft [[buffer(4)]],
  device float2* mass_y_fft [[buffer(5)]],
  device float2* kernel_q [[buffer(6)]],
  device float2* kernel_q2 [[buffer(7)]],
  constant OpenTsneFFTGridParams& g [[buffer(8)]],
  uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= g.fft_size || gid.y >= g.fft_size) return;
  uint fft_pos = gid.y * g.fft_size + gid.x;
  float m = 0.0f;
  float mx = 0.0f;
  float my = 0.0f;
  if (gid.x < g.grid_size && gid.y < g.grid_size) {
    uint grid_pos = gid.y * g.grid_size + gid.x;
    m = as_type<float>(atomic_load_explicit(&mass[grid_pos], memory_order_relaxed));
    mx = as_type<float>(atomic_load_explicit(&mass_x[grid_pos], memory_order_relaxed));
    my = as_type<float>(atomic_load_explicit(&mass_y[grid_pos], memory_order_relaxed));
  }
  mass_fft[fft_pos] = float2(m, 0.0f);
  mass_x_fft[fft_pos] = float2(mx, 0.0f);
  mass_y_fft[fft_pos] = float2(my, 0.0f);

  bool x_ok = gid.x < g.grid_size || gid.x > g.grid_size;
  bool y_ok = gid.y < g.grid_size || gid.y > g.grid_size;
  float q = 0.0f;
  float q2 = 0.0f;
  if (x_ok && y_ok) {
    int dx = gid.x < g.grid_size ? int(gid.x) : int(gid.x) - int(g.fft_size);
    int dy = gid.y < g.grid_size ? int(gid.y) : int(gid.y) - int(g.fft_size);
    if (abs(dx) < int(g.grid_size) && abs(dy) < int(g.grid_size)) {
      float x_offset = float(dx) * g.spacing;
      float y_offset = float(dy) * g.spacing;
      float d2 = x_offset * x_offset + y_offset * y_offset;
      q = 1.0f / (1.0f + d2);
      q2 = q * q;
    }
  }
  kernel_q[fft_pos] = float2(q, 0.0f);
  kernel_q2[fft_pos] = float2(q2, 0.0f);
}

kernel void opentsne_mpsgraph_load_real_inputs(
  device const atomic_uint* mass [[buffer(0)]],
  device const atomic_uint* mass_x [[buffer(1)]],
  device const atomic_uint* mass_y [[buffer(2)]],
  device float* mass_real [[buffer(3)]],
  device float* mass_x_real [[buffer(4)]],
  device float* mass_y_real [[buffer(5)]],
  device float* kernel_q_real [[buffer(6)]],
  device float* kernel_q2_real [[buffer(7)]],
  constant OpenTsneFFTGridParams& g [[buffer(8)]],
  uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= g.fft_size || gid.y >= g.fft_size) return;
  uint fft_pos = gid.y * g.fft_size + gid.x;
  float m = 0.0f;
  float mx = 0.0f;
  float my = 0.0f;
  if (gid.x < g.grid_size && gid.y < g.grid_size) {
    uint grid_pos = gid.y * g.grid_size + gid.x;
    m = as_type<float>(atomic_load_explicit(&mass[grid_pos], memory_order_relaxed));
    mx = as_type<float>(atomic_load_explicit(&mass_x[grid_pos], memory_order_relaxed));
    my = as_type<float>(atomic_load_explicit(&mass_y[grid_pos], memory_order_relaxed));
  }
  mass_real[fft_pos] = m;
  mass_x_real[fft_pos] = mx;
  mass_y_real[fft_pos] = my;

  bool x_ok = gid.x < g.grid_size || gid.x > g.grid_size;
  bool y_ok = gid.y < g.grid_size || gid.y > g.grid_size;
  float q = 0.0f;
  float q2 = 0.0f;
  if (x_ok && y_ok) {
    int dx = gid.x < g.grid_size ? int(gid.x) : int(gid.x) - int(g.fft_size);
    int dy = gid.y < g.grid_size ? int(gid.y) : int(gid.y) - int(g.fft_size);
    if (abs(dx) < int(g.grid_size) && abs(dy) < int(g.grid_size)) {
      float x_offset = float(dx) * g.spacing;
      float y_offset = float(dy) * g.spacing;
      float d2 = x_offset * x_offset + y_offset * y_offset;
      q = 1.0f / (1.0f + d2);
      q2 = q * q;
    }
  }
  kernel_q_real[fft_pos] = q;
  kernel_q2_real[fft_pos] = q2;
}

kernel void opentsne_fft_pack_real_to_complex4(
  device const float* q_real [[buffer(0)]],
  device const float* q2_real [[buffer(1)]],
  device const float* xq2_real [[buffer(2)]],
  device const float* yq2_real [[buffer(3)]],
  device float2* q_complex [[buffer(4)]],
  device float2* q2_complex [[buffer(5)]],
  device float2* xq2_complex [[buffer(6)]],
  device float2* yq2_complex [[buffer(7)]],
  constant uint& n_total [[buffer(8)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= n_total) return;
  q_complex[row] = float2(q_real[row], 0.0f);
  q2_complex[row] = float2(q2_real[row], 0.0f);
  xq2_complex[row] = float2(xq2_real[row], 0.0f);
  yq2_complex[row] = float2(yq2_real[row], 0.0f);
}

kernel void opentsne_fft_bit_reverse_rows(
  device const float2* input [[buffer(0)]],
  device float2* output [[buffer(1)]],
  constant uint& n_fft [[buffer(2)]],
  constant uint& log_n [[buffer(3)]],
  uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= n_fft || gid.y >= n_fft) return;
  uint rev = reverse_bits_limited(gid.x, log_n);
  output[gid.y * n_fft + rev] = input[gid.y * n_fft + gid.x];
}

kernel void opentsne_fft_bit_reverse_cols(
  device const float2* input [[buffer(0)]],
  device float2* output [[buffer(1)]],
  constant uint& n_fft [[buffer(2)]],
  constant uint& log_n [[buffer(3)]],
  uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= n_fft || gid.y >= n_fft) return;
  uint rev = reverse_bits_limited(gid.y, log_n);
  output[rev * n_fft + gid.x] = input[gid.y * n_fft + gid.x];
}

kernel void opentsne_fft_butterfly_rows(
  device float2* values [[buffer(0)]],
  constant uint& n_fft [[buffer(1)]],
  constant uint& stage [[buffer(2)]],
  constant uint& inverse [[buffer(3)]],
  device const float2* twiddles [[buffer(4)]],
  uint2 gid [[thread_position_in_grid]]
) {
  uint half_count = n_fft >> 1u;
  if (gid.x >= half_count || gid.y >= n_fft) return;
  uint span_half = 1u << (stage - 1u);
  uint width = span_half << 1u;
  uint group = gid.x / span_half;
  uint j = gid.x - group * span_half;
  uint base = gid.y * n_fft + group * width + j;
  float2 w = twiddles[(stage - 1u) * half_count + j];
  if (inverse != 0u) w.y = -w.y;
  float2 u = values[base];
  float2 v = complex_mul(values[base + span_half], w);
  values[base] = u + v;
  values[base + span_half] = u - v;
}

kernel void opentsne_fft_butterfly_cols(
  device float2* values [[buffer(0)]],
  constant uint& n_fft [[buffer(1)]],
  constant uint& stage [[buffer(2)]],
  constant uint& inverse [[buffer(3)]],
  device const float2* twiddles [[buffer(4)]],
  uint2 gid [[thread_position_in_grid]]
) {
  uint half_count = n_fft >> 1u;
  if (gid.x >= n_fft || gid.y >= half_count) return;
  uint span_half = 1u << (stage - 1u);
  uint width = span_half << 1u;
  uint group = gid.y / span_half;
  uint j = gid.y - group * span_half;
  uint row0 = group * width + j;
  uint idx0 = row0 * n_fft + gid.x;
  uint idx1 = (row0 + span_half) * n_fft + gid.x;
  float2 w = twiddles[(stage - 1u) * half_count + j];
  if (inverse != 0u) w.y = -w.y;
  float2 u = values[idx0];
  float2 v = complex_mul(values[idx1], w);
  values[idx0] = u + v;
  values[idx1] = u - v;
}

kernel void opentsne_fft_multiply(
  device const float2* a [[buffer(0)]],
  device const float2* b [[buffer(1)]],
  device float2* out [[buffer(2)]],
  constant uint& total [[buffer(3)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= total) return;
  out[gid] = complex_mul(a[gid], b[gid]);
}

kernel void opentsne_fft_scale(
  device float2* values [[buffer(0)]],
  constant uint& total [[buffer(1)]],
  constant float& scale [[buffer(2)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= total) return;
  values[gid] *= scale;
}

// Stockham 512 kernels implemented for fastEmbedR using the MIT-licensed
// AppleSiliconFFT radix-4 Stockham design by Mohamed Amine Bergach as a
// reference. Attribution is recorded in inst/NOTICE. They are used only for
// validated 512x512 openTSNE FFT grids; other sizes stay on the generic
// Cooley-Tukey Metal path.
inline void opentsne_fft_radix4(thread float2& x0, thread float2& x1,
                                thread float2& x2, thread float2& x3,
                                bool inverse) {
  float2 t0 = x0 + x2;
  float2 t1 = x1 + x3;
  float2 t2 = x0 - x2;
  float2 t3 = x1 - x3;
  float2 t3r = inverse ? float2(t3.y, -t3.x) : float2(-t3.y, t3.x);
  x0 = t0 + t1;
  x1 = t2 + t3r;
  x2 = t0 - t1;
  x3 = t2 - t3r;
}

inline void opentsne_fft_radix2(thread float2& x0, thread float2& x1) {
  float2 t = x0;
  x0 = t + x1;
  x1 = t - x1;
}

inline void opentsne_fft_apply_twiddle3(thread float2& x1,
                                        thread float2& x2,
                                        thread float2& x3,
                                        float2 w1) {
  float2 w2 = complex_mul(w1, w1);
  float2 w3 = complex_mul(w2, w1);
  x1 = complex_mul(x1, w1);
  x2 = complex_mul(x2, w2);
  x3 = complex_mul(x3, w3);
}

inline void opentsne_fft_stockham512_core(device const float2* input,
                                          device float2* output,
                                          threadgroup float2* buf,
                                          uint tid,
                                          uint lane,
                                          bool column_major,
                                          bool inverse) {
  constexpr uint N = 512u;
  constexpr uint T = 128u;
  float sign = inverse ? -2.0f : 2.0f;
  float two_pi_over_n = sign * M_PI_F / float(N);

  {
    uint off0 = tid;
    uint off1 = tid + T;
    uint off2 = tid + 2u * T;
    uint off3 = tid + 3u * T;
    float2 x0 = column_major ? input[off0 * N + lane] : input[lane * N + off0];
    float2 x1 = column_major ? input[off1 * N + lane] : input[lane * N + off1];
    float2 x2 = column_major ? input[off2 * N + lane] : input[lane * N + off2];
    float2 x3 = column_major ? input[off3 * N + lane] : input[lane * N + off3];
    opentsne_fft_radix4(x0, x1, x2, x3, inverse);
    uint wr = tid << 2u;
    buf[wr] = x0;
    buf[wr + 1u] = x1;
    buf[wr + 2u] = x2;
    buf[wr + 3u] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  {
    uint pos = tid & 3u;
    uint grp = tid >> 2u;
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + T];
    float2 x2 = buf[tid + 2u * T];
    float2 x3 = buf[tid + 3u * T];
    float a1 = two_pi_over_n * float(pos * 32u);
    float c1;
    float s1 = sincos(a1, c1);
    opentsne_fft_apply_twiddle3(x1, x2, x3, float2(c1, s1));
    opentsne_fft_radix4(x0, x1, x2, x3, inverse);
    uint wr = grp * 16u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0;
    buf[wr + 4u] = x1;
    buf[wr + 8u] = x2;
    buf[wr + 12u] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  {
    uint pos = tid & 15u;
    uint grp = tid >> 4u;
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + T];
    float2 x2 = buf[tid + 2u * T];
    float2 x3 = buf[tid + 3u * T];
    float a1 = two_pi_over_n * float(pos * 8u);
    float c1;
    float s1 = sincos(a1, c1);
    opentsne_fft_apply_twiddle3(x1, x2, x3, float2(c1, s1));
    opentsne_fft_radix4(x0, x1, x2, x3, inverse);
    uint wr = grp * 64u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0;
    buf[wr + 16u] = x1;
    buf[wr + 32u] = x2;
    buf[wr + 48u] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  {
    uint pos = tid & 63u;
    uint grp = tid >> 6u;
    float2 x0 = buf[tid];
    float2 x1 = buf[tid + T];
    float2 x2 = buf[tid + 2u * T];
    float2 x3 = buf[tid + 3u * T];
    float a1 = two_pi_over_n * float(pos * 2u);
    float c1;
    float s1 = sincos(a1, c1);
    opentsne_fft_apply_twiddle3(x1, x2, x3, float2(c1, s1));
    opentsne_fft_radix4(x0, x1, x2, x3, inverse);
    uint wr = grp * 256u + pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[wr] = x0;
    buf[wr + 64u] = x1;
    buf[wr + 128u] = x2;
    buf[wr + 192u] = x3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint b = 0u; b < 2u; ++b) {
    uint j = tid + b * T;
    float2 x0 = buf[j];
    float2 x1 = buf[j + 256u];
    float a1 = two_pi_over_n * float(j);
    float c1;
    float s1 = sincos(a1, c1);
    x1 = complex_mul(x1, float2(c1, s1));
    opentsne_fft_radix2(x0, x1);
    uint off0 = j;
    uint off1 = j + 256u;
    if (column_major) {
      output[off0 * N + lane] = x0;
      output[off1 * N + lane] = x1;
    } else {
      output[lane * N + off0] = x0;
      output[lane * N + off1] = x1;
    }
  }
}

kernel void opentsne_fft_512_rows_stockham(
  device const float2* input [[buffer(0)]],
  device float2* output [[buffer(1)]],
  constant uint& inverse_u [[buffer(2)]],
  uint tid [[thread_index_in_threadgroup]],
  uint row [[threadgroup_position_in_grid]]
) {
  threadgroup float2 buf[512];
  opentsne_fft_stockham512_core(input, output, buf, tid, row, false, inverse_u != 0u);
}

kernel void opentsne_fft_512_cols_stockham(
  device const float2* input [[buffer(0)]],
  device float2* output [[buffer(1)]],
  constant uint& inverse_u [[buffer(2)]],
  uint tid [[thread_index_in_threadgroup]],
  uint col [[threadgroup_position_in_grid]]
) {
  threadgroup float2 buf[512];
  opentsne_fft_stockham512_core(input, output, buf, tid, col, true, inverse_u != 0u);
}

float opentsne_sample_grid_value(device const float2* grid,
                                 constant OpenTsneFFTGridParams& g,
                                 float2 yi) {
  float raw_x = (yi.x - g.lower_x) * g.inv_spacing;
  float raw_y = (yi.y - g.lower_y) * g.inv_spacing;
  float max_cell = float(g.grid_size - 1u);
  raw_x = clamp(raw_x, 0.0f, max_cell);
  raw_y = clamp(raw_y, 0.0f, max_cell);
  uint x0 = min(uint(floor(raw_x)), g.grid_size - 2u);
  uint y0 = min(uint(floor(raw_y)), g.grid_size - 2u);
  uint x1 = x0 + 1u;
  uint y1 = y0 + 1u;
  float tx = raw_x - float(x0);
  float ty = raw_y - float(y0);
  float v00 = grid[y0 * g.fft_size + x0].x;
  float v10 = grid[y0 * g.fft_size + x1].x;
  float v01 = grid[y1 * g.fft_size + x0].x;
  float v11 = grid[y1 * g.fft_size + x1].x;
  return (1.0f - tx) * (1.0f - ty) * v00 +
    tx * (1.0f - ty) * v10 +
    (1.0f - tx) * ty * v01 +
    tx * ty * v11;
}

kernel void opentsne_fft_sum_q_rows(
  device const float2* current [[buffer(0)]],
  device const float2* q_grid [[buffer(1)]],
  device float* row_sums [[buffer(2)]],
  constant OpenTsneFFTGridParams& g [[buffer(3)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= g.n) return;
  float2 yi = current[row];
  if (!isfinite(yi.x)) yi.x = 0.0f;
  if (!isfinite(yi.y)) yi.y = 0.0f;
  row_sums[row] = opentsne_sample_grid_value(q_grid, g, yi);
}

kernel void opentsne_fft_sum_q_blocks(
  device const float2* current [[buffer(0)]],
  device const float2* q_grid [[buffer(1)]],
  device float* block_sums [[buffer(2)]],
  constant OpenTsneFFTGridParams& g [[buffer(3)]],
  constant uint& block_size [[buffer(4)]],
  uint block_id [[thread_position_in_grid]]
) {
  uint begin = block_id * block_size;
  if (begin >= g.n) return;
  uint end = min(g.n, begin + block_size);
  float sum = 0.0f;
  for (uint row = begin; row < end; ++row) {
    float2 yi = current[row];
    if (!isfinite(yi.x)) yi.x = 0.0f;
    if (!isfinite(yi.y)) yi.y = 0.0f;
    sum += opentsne_sample_grid_value(q_grid, g, yi);
  }
  block_sums[block_id] = sum;
}

kernel void opentsne_fft_finalize_sum_q(
  device const float* block_sums [[buffer(0)]],
  device float* inv_sum_q [[buffer(1)]],
  constant uint& block_count [[buffer(2)]],
  constant uint& n [[buffer(3)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid > 0u) return;
  float sum_q = -float(n);
  for (uint i = 0; i < block_count; ++i) {
    sum_q += block_sums[i];
  }
  if (!isfinite(sum_q) || sum_q <= 0.0f) {
    sum_q = 1.1754943508222875e-38f;
  }
  inv_sum_q[0] = 1.0f / sum_q;
}

kernel void opentsne_fft_layout_stats_blocks(
  device const float2* current [[buffer(0)]],
  device OpenTsneLayoutStats* block_stats [[buffer(1)]],
  constant uint& n [[buffer(2)]],
  constant uint& block_size [[buffer(3)]],
  uint block_id [[thread_position_in_grid]]
) {
  uint begin = block_id * block_size;
  if (begin >= n) return;
  uint end = min(n, begin + block_size);
  float min_x = INFINITY;
  float max_x = -INFINITY;
  float min_y = INFINITY;
  float max_y = -INFINITY;
  float sum_x = 0.0f;
  float sum_y = 0.0f;
  for (uint row = begin; row < end; ++row) {
    float2 yi = current[row];
    if (!isfinite(yi.x)) yi.x = 0.0f;
    if (!isfinite(yi.y)) yi.y = 0.0f;
    min_x = min(min_x, yi.x);
    max_x = max(max_x, yi.x);
    min_y = min(min_y, yi.y);
    max_y = max(max_y, yi.y);
    sum_x += yi.x;
    sum_y += yi.y;
  }
  block_stats[block_id] = OpenTsneLayoutStats{min_x, max_x, min_y, max_y, sum_x, sum_y};
}

kernel void opentsne_fft_finalize_layout_stats(
  device const OpenTsneLayoutStats* block_stats [[buffer(0)]],
  device OpenTsneFFTGridParams* grid_params_out [[buffer(1)]],
  device float2* center_out [[buffer(2)]],
  constant uint& block_count [[buffer(3)]],
  constant uint& n [[buffer(4)]],
  constant uint& grid_size [[buffer(5)]],
  constant uint& fft_size [[buffer(6)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid > 0u) return;
  float min_x = INFINITY;
  float max_x = -INFINITY;
  float min_y = INFINITY;
  float max_y = -INFINITY;
  float sum_x = 0.0f;
  float sum_y = 0.0f;
  for (uint i = 0u; i < block_count; ++i) {
    OpenTsneLayoutStats s = block_stats[i];
    min_x = min(min_x, s.min_x);
    max_x = max(max_x, s.max_x);
    min_y = min(min_y, s.min_y);
    max_y = max(max_y, s.max_y);
    sum_x += s.sum_x;
    sum_y += s.sum_y;
  }
  float cx = 0.5f * (min_x + max_x);
  float cy = 0.5f * (min_y + max_y);
  float span = max(max_x - min_x, max_y - min_y);
  if (!isfinite(span) || span <= 0.0f) span = 1.0f;
  float half_span = 0.55f * span + 1.0e-3f;
  float spacing = (2.0f * half_span) / float(max(2u, grid_size) - 1u);
  if (!isfinite(spacing) || spacing <= 0.0f) spacing = 1.0f;
  grid_params_out[0] = OpenTsneFFTGridParams{
    n,
    grid_size,
    fft_size,
    cx - half_span,
    cy - half_span,
    1.0f / spacing,
    spacing,
    1.0f
  };
  float inv_n = n > 0u ? 1.0f / float(n) : 0.0f;
  center_out[0] = float2(sum_x * inv_n, sum_y * inv_n);
}

kernel void opentsne_epoch_fft_grid(
  device const int* row_ptr [[buffer(0)]],
  device const int* col_idx [[buffer(1)]],
  device const float* p_val [[buffer(2)]],
  device const float2* current [[buffer(3)]],
  device float2* next_current [[buffer(4)]],
  device float2* gains [[buffer(5)]],
  device float2* updates [[buffer(6)]],
  device const float2* q2_grid [[buffer(7)]],
  device const float2* xq2_grid [[buffer(8)]],
  device const float2* yq2_grid [[buffer(9)]],
  constant OpenTsneMetalParams& p [[buffer(10)]],
  constant OpenTsneFFTGridParams& g [[buffer(11)]],
  device const float* inv_sum_q_device [[buffer(12)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= p.n) return;
  constexpr float eps = 1.0e-12f;
  float2 yi = current[row];
  if (!isfinite(yi.x)) yi.x = 0.0f;
  if (!isfinite(yi.y)) yi.y = 0.0f;
  float q2_value = opentsne_sample_grid_value(q2_grid, g, yi);
  float xq2_value = opentsne_sample_grid_value(xq2_grid, g, yi);
  float yq2_value = opentsne_sample_grid_value(yq2_grid, g, yi);
  float inv_sum_q = inv_sum_q_device[0];
  float2 grad = float2(
    -(yi.x * q2_value - xq2_value) * inv_sum_q,
    -(yi.y * q2_value - yq2_value) * inv_sum_q
  );

  int begin = row_ptr[row];
  int end = row_ptr[row + 1u];
  for (int pos = begin; pos < end; ++pos) {
    int j = col_idx[pos];
    if (j < 0 || uint(j) >= p.n || uint(j) == row) continue;
    float2 diff = yi - current[uint(j)];
    float d2 = dot(diff, diff);
    float q = 1.0f / (1.0f + d2);
    grad += (p.exaggeration * p_val[pos] * q) * diff;
  }

  float2 gain = gains[row];
  float2 update = updates[row];
  float sx0 = sign_component(update.x);
  float sx1 = sign_component(update.y);
  float sg0 = sign_component(grad.x);
  float sg1 = sign_component(grad.y);
  gain.x = sx0 != sg0 ? gain.x + 0.2f : gain.x * 0.8f + p.min_gain;
  gain.y = sx1 != sg1 ? gain.y + 0.2f : gain.y * 0.8f + p.min_gain;
  gain = max(gain, float2(p.min_gain, p.min_gain));

  update = p.momentum * update - p.learning_rate * gain * grad;
  float step_norm2 = dot(update, update);
  float max_step2 = p.max_step_norm * p.max_step_norm;
  if (isfinite(max_step2) && max_step2 > 0.0f && step_norm2 > max_step2) {
    update *= p.max_step_norm / (sqrt(step_norm2) + eps);
  }

  next_current[row] = yi + update;
  gains[row] = gain;
  updates[row] = update;
}

kernel void opentsne_epoch_fft_grid_debug(
  device const int* row_ptr [[buffer(0)]],
  device const int* col_idx [[buffer(1)]],
  device const float* p_val [[buffer(2)]],
  device const float2* current [[buffer(3)]],
  device float2* next_current [[buffer(4)]],
  device float2* gains [[buffer(5)]],
  device float2* updates [[buffer(6)]],
  device const float2* q2_grid [[buffer(7)]],
  device const float2* xq2_grid [[buffer(8)]],
  device const float2* yq2_grid [[buffer(9)]],
  constant OpenTsneMetalParams& p [[buffer(10)]],
  constant OpenTsneFFTGridParams& g [[buffer(11)]],
  device const float* inv_sum_q_device [[buffer(12)]],
  device float* repulsive_norm2 [[buffer(13)]],
  device float* attractive_norm2 [[buffer(14)]],
  device float* gradient_norm2 [[buffer(15)]],
  device float* update_norm2 [[buffer(16)]],
  device float* layout_norm2 [[buffer(17)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= p.n) return;
  constexpr float eps = 1.0e-12f;
  float2 yi = current[row];
  if (!isfinite(yi.x)) yi.x = 0.0f;
  if (!isfinite(yi.y)) yi.y = 0.0f;

  float q2_value = opentsne_sample_grid_value(q2_grid, g, yi);
  float xq2_value = opentsne_sample_grid_value(xq2_grid, g, yi);
  float yq2_value = opentsne_sample_grid_value(yq2_grid, g, yi);
  float inv_sum_q = inv_sum_q_device[0];
  float2 repulsive = float2(
    -(yi.x * q2_value - xq2_value) * inv_sum_q,
    -(yi.y * q2_value - yq2_value) * inv_sum_q
  );
  float2 attractive = float2(0.0f, 0.0f);

  int begin = row_ptr[row];
  int end = row_ptr[row + 1u];
  for (int pos = begin; pos < end; ++pos) {
    int j = col_idx[pos];
    if (j < 0 || uint(j) >= p.n || uint(j) == row) continue;
    float2 diff = yi - current[uint(j)];
    float d2 = dot(diff, diff);
    float q = 1.0f / (1.0f + d2);
    attractive += (p.exaggeration * p_val[pos] * q) * diff;
  }

  float2 grad = repulsive + attractive;
  float2 gain = gains[row];
  float2 update = updates[row];
  float sx0 = sign_component(update.x);
  float sx1 = sign_component(update.y);
  float sg0 = sign_component(grad.x);
  float sg1 = sign_component(grad.y);
  gain.x = sx0 != sg0 ? gain.x + 0.2f : gain.x * 0.8f + p.min_gain;
  gain.y = sx1 != sg1 ? gain.y + 0.2f : gain.y * 0.8f + p.min_gain;
  gain = max(gain, float2(p.min_gain, p.min_gain));

  update = p.momentum * update - p.learning_rate * gain * grad;
  float step_norm2 = dot(update, update);
  float max_step2 = p.max_step_norm * p.max_step_norm;
  if (isfinite(max_step2) && max_step2 > 0.0f && step_norm2 > max_step2) {
    update *= p.max_step_norm / (sqrt(step_norm2) + eps);
    step_norm2 = dot(update, update);
  }

  float2 next = yi + update;
  next_current[row] = next;
  gains[row] = gain;
  updates[row] = update;

  repulsive_norm2[row] = dot(repulsive, repulsive);
  attractive_norm2[row] = dot(attractive, attractive);
  gradient_norm2[row] = dot(grad, grad);
  update_norm2[row] = step_norm2;
  layout_norm2[row] = dot(next, next);
}
)METAL";

struct ExactParamsHost {
  std::uint32_t n_train;
  std::uint32_t n_query;
  std::uint32_t dimensions;
  std::uint32_t k;
  std::uint32_t metric;
  std::uint32_t filtered;
};

struct SpatialGridParamsHost {
  std::uint32_t rows;
  std::uint32_t dimensions;
  std::uint32_t k;
  std::uint32_t nonself_k;
  std::uint32_t bins;
  std::uint32_t include_self;
  float min_x;
  float min_y;
  float min_z;
  float cell_x;
  float cell_y;
  float cell_z;
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
  std::uint32_t filtered;
};

struct KMeansParamsHost {
  std::uint32_t rows;
  std::uint32_t dimensions;
  std::uint32_t clusters;
};

struct KNNVoteParamsHost {
  std::uint32_t samples;
  std::uint32_t neighbors;
  std::int32_t fallback_label;
};

struct LandmarkProjectionParamsHost {
  std::uint32_t samples;
  std::uint32_t neighbors;
  std::uint32_t projection_k;
  std::int32_t fallback_label;
  std::uint32_t epoch;
};

struct LandmarkScatterParamsHost {
  std::uint32_t landmarks;
  std::uint32_t samples;
  std::uint32_t epoch;
};

struct ConstrainedMajorityParamsHost {
  std::uint32_t samples;
  std::uint32_t groups;
  std::uint32_t label_width;
};

struct KODAMADissimilarityParamsHost {
  std::uint32_t runs;
  std::uint32_t samples;
  std::uint32_t neighbors;
  std::uint32_t sort_width;
  std::uint32_t input_one_based;
  std::uint32_t output_one_based;
};

struct UMAPParamsHost {
  std::uint32_t samples;
  std::uint32_t width;
  std::uint32_t epochs;
  std::uint32_t negative_sample_rate;
  std::uint32_t seed;
  float learning_rate;
  float a;
  float b;
  float max_weight;
  float repulsion_strength;
};

struct OpenTsneMetalParamsHost {
  std::uint32_t n;
  std::uint32_t seed;
  float learning_rate;
  float exaggeration;
  float momentum;
  float min_gain;
  float max_step_norm;
  float inv_sum_q;
};

struct OpenTsneFFTGridParamsHost {
  std::uint32_t n;
  std::uint32_t grid_size;
  std::uint32_t fft_size;
  float lower_x;
  float lower_y;
  float inv_spacing;
  float spacing;
  float inv_sum_q;
};

struct OpenTsneCenterHost {
  float x;
  float y;
};

struct OpenTsneLayoutStatsHost {
  float min_x;
  float max_x;
  float min_y;
  float max_y;
  float sum_x;
  float sum_y;
};

struct PLSLDAParamsHost {
  std::uint32_t rows;
  std::uint32_t components;
  std::uint32_t score_stride;
  std::uint32_t classes;
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
  id<MTLComputePipelineState> spatial_grid_pipeline = nil;
  id<MTLComputePipelineState> project_pipeline = nil;
  id<MTLComputePipelineState> gather_centroids_pipeline = nil;
  id<MTLComputePipelineState> ivf_pipeline = nil;
  id<MTLComputePipelineState> clear_changed_pipeline = nil;
  id<MTLComputePipelineState> assign_kmeans_pipeline = nil;
  id<MTLComputePipelineState> clear_kmeans_pipeline = nil;
  id<MTLComputePipelineState> accumulate_kmeans_pipeline = nil;
  id<MTLComputePipelineState> finalize_kmeans_pipeline = nil;
  id<MTLComputePipelineState> clear_ivf_counts_pipeline = nil;
  id<MTLComputePipelineState> count_ivf_pipeline = nil;
  id<MTLComputePipelineState> prefix_ivf_pipeline = nil;
  id<MTLComputePipelineState> scatter_ivf_pipeline = nil;
  id<MTLComputePipelineState> knn_vote_pipeline = nil;
  id<MTLComputePipelineState> landmark_projection_pipeline = nil;
  id<MTLComputePipelineState> landmark_scatter_pipeline = nil;
  id<MTLComputePipelineState> constrained_count_pipeline = nil;
  id<MTLComputePipelineState> constrained_select_pipeline = nil;
  id<MTLComputePipelineState> constrained_apply_pipeline = nil;
  id<MTLComputePipelineState> kodama_dissimilarity_pipeline = nil;
  id<MTLComputePipelineState> umap_pipeline = nil;
  id<MTLComputePipelineState> opentsne_center_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_clear_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_scatter_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_load_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_bit_reverse_rows_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_bit_reverse_cols_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_butterfly_rows_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_butterfly_cols_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_512_rows_stockham_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_512_cols_stockham_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_multiply_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_scale_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_sum_q_blocks_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_finalize_sum_q_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_layout_stats_blocks_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_finalize_layout_stats_pipeline = nil;
  id<MTLComputePipelineState> opentsne_fft_epoch_pipeline = nil;
  id<MTLComputePipelineState> pls_class_sums_pipeline = nil;
  id<MTLComputePipelineState> pls_lda_predict_pipeline = nil;
};

}  // namespace

struct NativeMetalIVFIndex::Impl {
  int rows = 0;
  int dimensions = 0;
  int projected_dimensions = 0;
  int nlist = 0;
  DistanceMetric metric = DistanceMetric::Euclidean;
  id<MTLBuffer> train = nil;
  id<MTLBuffer> projected_train = nil;
  id<MTLBuffer> centroids = nil;
  id<MTLBuffer> list_offsets = nil;
  id<MTLBuffer> list_ids = nil;
  id<MTLBuffer> feature_offsets = nil;
  id<MTLBuffer> feature_ids = nil;
  id<MTLBuffer> feature_signs = nil;
};

struct NativeMetalKNNVoteGraph::Impl {
  int samples = 0;
  int neighbors = 0;
  id<MTLBuffer> neighbor_rows = nil;
  id<MTLBuffer> scores = nil;
  id<MTLBuffer> labels = nil;
  id<MTLBuffer> predictions = nil;
};

struct NativeMetalKODAMAGraph::Impl {
  struct Lane {
    id<MTLBuffer> landmark_epoch = nil;
    id<MTLBuffer> labels = nil;
    id<MTLBuffer> landmark_rows = nil;
    id<MTLBuffer> landmark_input_labels = nil;
    std::size_t landmark_capacity = 0;
    id<MTLBuffer> constrain = nil;
    id<MTLBuffer> constrain_counts = nil;
    id<MTLBuffer> constrain_labels = nil;
    std::size_t constrain_count_capacity = 0;
    std::size_t constrain_group_capacity = 0;
  };

  int samples = 0;
  int neighbors = 0;
  id<MTLBuffer> indices = nil;
  id<MTLBuffer> distances = nil;
  id<MTLBuffer> base_indices = nil;
  id<MTLBuffer> base_distances = nil;
  id<MTLBuffer> result_labels = nil;
  std::size_t result_capacity = 0;
  int result_runs = 0;
  std::unique_ptr<NativeMetalIVFIndex> ivf_index;
  DistanceMetric metric = DistanceMetric::Euclidean;
  int dimensions = 0;
  std::vector<Lane> lane_buffers;
};

struct NativeMetalKMeansContext::Impl {
  struct Lane {
    id<MTLBuffer> centroids = nil;
    id<MTLBuffer> assignments = nil;
    id<MTLBuffer> sums = nil;
    id<MTLBuffer> counts = nil;
    id<MTLBuffer> changed = nil;
    id<MTLBuffer> initial_indices = nil;
    id<MTLBuffer> parameters = nil;
  };

  int rows = 0;
  int dimensions = 0;
  int max_clusters = 0;
  id<MTLBuffer> data = nil;
  std::vector<Lane> lane;
  std::uint64_t input_uploads = 0;
};

namespace {

id<MTLDevice> select_metal_device() {
  static id<MTLDevice> selected = nil;
  static std::once_flag once;
  std::call_once(once, []() {
    selected = MTLCreateSystemDefaultDevice();
    if (selected == nil) {
      NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
      if (devices.count != 0) selected = devices.firstObject;
    }
  });
  return selected;
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
      const std::string metal_source =
        std::string(kMetalSourcePart1) + kMetalSourcePart2;
      state.library = [state.device
        newLibraryWithSource:[NSString stringWithUTF8String:metal_source.c_str()]
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
      state.spatial_grid_pipeline = make_pipeline(@"spatial_grid_self_knn");
      state.project_pipeline = make_pipeline(@"signed_hash_project");
      state.gather_centroids_pipeline = make_pipeline(@"gather_kmeans_centroids");
      state.ivf_pipeline = make_pipeline(@"ivf_topk_train_query");
      state.clear_changed_pipeline = make_pipeline(@"clear_kmeans_changed");
      state.assign_kmeans_pipeline = make_pipeline(@"assign_kmeans_centroid");
      state.clear_kmeans_pipeline = make_pipeline(@"clear_kmeans_accumulators");
      state.accumulate_kmeans_pipeline = make_pipeline(@"accumulate_kmeans_centroids");
      state.finalize_kmeans_pipeline = make_pipeline(@"finalize_kmeans_centroids");
      state.clear_ivf_counts_pipeline = make_pipeline(@"clear_ivf_list_counts");
      state.count_ivf_pipeline = make_pipeline(@"count_ivf_assignments");
      state.prefix_ivf_pipeline = make_pipeline(@"prefix_ivf_list_counts");
      state.scatter_ivf_pipeline = make_pipeline(@"scatter_ivf_list_ids");
      state.knn_vote_pipeline = make_pipeline(@"resident_knn_vote");
      state.landmark_projection_pipeline =
        make_pipeline(@"project_landmark_labels");
      state.landmark_scatter_pipeline =
        make_pipeline(@"scatter_landmark_labels");
      state.constrained_count_pipeline =
        make_pipeline(@"count_constrained_labels");
      state.constrained_select_pipeline =
        make_pipeline(@"select_constrained_majority");
      state.constrained_apply_pipeline =
        make_pipeline(@"apply_constrained_majority");
      state.kodama_dissimilarity_pipeline =
        make_pipeline(@"kodama_dissimilarity_resident");
      state.umap_pipeline = make_pipeline(@"kodama_umap_clean_epoch");
      state.opentsne_center_pipeline = make_pipeline(@"opentsne_apply_center");
      state.opentsne_fft_clear_pipeline = make_pipeline(@"opentsne_fft_clear_grids");
      state.opentsne_fft_scatter_pipeline = make_pipeline(@"opentsne_fft_scatter_bilinear");
      state.opentsne_fft_load_pipeline = make_pipeline(@"opentsne_fft_load_inputs");
      state.opentsne_fft_bit_reverse_rows_pipeline =
        make_pipeline(@"opentsne_fft_bit_reverse_rows");
      state.opentsne_fft_bit_reverse_cols_pipeline =
        make_pipeline(@"opentsne_fft_bit_reverse_cols");
      state.opentsne_fft_butterfly_rows_pipeline =
        make_pipeline(@"opentsne_fft_butterfly_rows");
      state.opentsne_fft_butterfly_cols_pipeline =
        make_pipeline(@"opentsne_fft_butterfly_cols");
      state.opentsne_fft_512_rows_stockham_pipeline =
        make_pipeline(@"opentsne_fft_512_rows_stockham");
      state.opentsne_fft_512_cols_stockham_pipeline =
        make_pipeline(@"opentsne_fft_512_cols_stockham");
      state.opentsne_fft_multiply_pipeline = make_pipeline(@"opentsne_fft_multiply");
      state.opentsne_fft_scale_pipeline = make_pipeline(@"opentsne_fft_scale");
      state.opentsne_fft_sum_q_blocks_pipeline =
        make_pipeline(@"opentsne_fft_sum_q_blocks");
      state.opentsne_fft_finalize_sum_q_pipeline =
        make_pipeline(@"opentsne_fft_finalize_sum_q");
      state.opentsne_fft_layout_stats_blocks_pipeline =
        make_pipeline(@"opentsne_fft_layout_stats_blocks");
      state.opentsne_fft_finalize_layout_stats_pipeline =
        make_pipeline(@"opentsne_fft_finalize_layout_stats");
      state.opentsne_fft_epoch_pipeline = make_pipeline(@"opentsne_epoch_fft_grid");
      state.pls_class_sums_pipeline = make_pipeline(@"pls_class_score_sums");
      state.pls_lda_predict_pipeline = make_pipeline(@"pls_lda_score_labels");
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

struct MetalIVFLists {
  id<MTLBuffer> offsets = nil;
  id<MTLBuffer> ids = nil;
};

MetalIVFLists build_metal_ivf_lists(
  MetalState& state,
  id<MTLBuffer> assignments,
  id<MTLBuffer> parameters,
  int rows,
  int clusters
) {
  id<MTLBuffer> counts = [state.device
    newBufferWithLength:static_cast<std::size_t>(clusters) * sizeof(std::uint32_t)
    options:MTLResourceStorageModePrivate];
  id<MTLBuffer> cursor = [state.device
    newBufferWithLength:static_cast<std::size_t>(clusters) * sizeof(std::uint32_t)
    options:MTLResourceStorageModePrivate];
  id<MTLBuffer> offsets = [state.device
    newBufferWithLength:(static_cast<std::size_t>(clusters) + 1) * sizeof(std::uint32_t)
    options:MTLResourceStorageModePrivate];
  id<MTLBuffer> ids = [state.device
    newBufferWithLength:static_cast<std::size_t>(rows) * sizeof(int)
    options:MTLResourceStorageModePrivate];
  if (counts == nil || cursor == nil || offsets == nil || ids == nil) {
    throw std::runtime_error("Failed to allocate Metal IVF list buffers.");
  }

  id<MTLCommandBuffer> command = [state.queue commandBuffer];
  id<MTLComputeCommandEncoder> clear_encoder = [command computeCommandEncoder];
  [clear_encoder setComputePipelineState:state.clear_ivf_counts_pipeline];
  [clear_encoder setBuffer:counts offset:0 atIndex:0];
  [clear_encoder setBuffer:parameters offset:0 atIndex:1];
  [clear_encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(clusters), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [clear_encoder endEncoding];

  id<MTLComputeCommandEncoder> count_encoder = [command computeCommandEncoder];
  [count_encoder setComputePipelineState:state.count_ivf_pipeline];
  [count_encoder setBuffer:assignments offset:0 atIndex:0];
  [count_encoder setBuffer:counts offset:0 atIndex:1];
  [count_encoder setBuffer:parameters offset:0 atIndex:2];
  [count_encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [count_encoder endEncoding];

  id<MTLComputeCommandEncoder> prefix_encoder = [command computeCommandEncoder];
  [prefix_encoder setComputePipelineState:state.prefix_ivf_pipeline];
  [prefix_encoder setBuffer:counts offset:0 atIndex:0];
  [prefix_encoder setBuffer:offsets offset:0 atIndex:1];
  [prefix_encoder setBuffer:cursor offset:0 atIndex:2];
  [prefix_encoder setBuffer:parameters offset:0 atIndex:3];
  [prefix_encoder dispatchThreads:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  [prefix_encoder endEncoding];

  id<MTLComputeCommandEncoder> scatter_encoder = [command computeCommandEncoder];
  [scatter_encoder setComputePipelineState:state.scatter_ivf_pipeline];
  [scatter_encoder setBuffer:assignments offset:0 atIndex:0];
  [scatter_encoder setBuffer:cursor offset:0 atIndex:1];
  [scatter_encoder setBuffer:ids offset:0 atIndex:2];
  [scatter_encoder setBuffer:parameters offset:0 atIndex:3];
  [scatter_encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [scatter_encoder endEncoding];
  wait_for_command(command, "Metal IVF list construction failed");
  return MetalIVFLists{offsets, ids};
}

NSUInteger matrix_row_bytes(int columns) {
  return [MPSMatrixDescriptor rowBytesFromColumns:static_cast<NSUInteger>(columns) dataType:MPSDataTypeFloat32];
}

struct MetalResidentMatrixSlot {
  std::uint64_t epoch = 0;
  const float* host = nullptr;
  int rows = 0;
  int columns = 0;
  NSUInteger row_bytes = 0;
  id<MTLBuffer> buffer = nil;
  std::uint64_t last_use = 0;
};

struct MetalPLSResidentWorkspace {
  std::uint64_t epoch = 0;
  std::vector<MetalResidentMatrixSlot> matrices;
  id<MTLCommandQueue> queue = nil;
  id<MTLBuffer> transient_left_matrix = nil;
  id<MTLBuffer> right_matrix = nil;
  id<MTLBuffer> result_matrix = nil;
  id<MTLBuffer> simpls_weight = nil;
  id<MTLBuffer> simpls_score = nil;
  id<MTLBuffer> simpls_loading = nil;
  id<MTLBuffer> cross_product = nil;
  id<MTLBuffer> pls_weights = nil;
  id<MTLBuffer> labels = nil;
  id<MTLBuffer> lda_scores = nil;
  id<MTLBuffer> lda_class_sums = nil;
  id<MTLBuffer> lda_score_crossprod = nil;
  id<MTLBuffer> lda_linear = nil;
  id<MTLBuffer> lda_constants = nil;
  id<MTLBuffer> lda_class_labels = nil;
  id<MTLBuffer> lda_predictions = nil;
  std::uint64_t matrix_uploads = 0;
  std::uint64_t matrix_reuses = 0;
  std::uint64_t cache_clock = 0;
};

thread_local MetalPLSResidentWorkspace g_metal_pls_workspace;

id<MTLCommandQueue> metal_pls_worker_queue(id<MTLDevice> device) {
  MetalPLSResidentWorkspace& workspace = g_metal_pls_workspace;
  if (workspace.queue == nil) workspace.queue = [device newCommandQueue];
  if (workspace.queue == nil) {
    throw std::runtime_error("Failed to create the Metal PLS worker command queue.");
  }
  return workspace.queue;
}

void ensure_shared_buffer(
  id<MTLDevice> device,
  __strong id<MTLBuffer>& buffer,
  std::size_t bytes
) {
  if (buffer != nil && [buffer length] >= bytes) return;
  buffer = [device
    newBufferWithLength:bytes
    options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    throw std::runtime_error("Failed to allocate a resident Metal buffer.");
  }
}

void write_matrix_buffer(
  id<MTLBuffer> buffer,
  const std::vector<float>& values,
  int rows,
  int columns,
  NSUInteger row_bytes
) {
  char* base = static_cast<char*>([buffer contents]);
  std::memset(base, 0, row_bytes * static_cast<NSUInteger>(rows));
  for (int row = 0; row < rows; ++row) {
    std::memcpy(
      base + static_cast<NSUInteger>(row) * row_bytes,
      values.data() +
        static_cast<std::size_t>(row) * static_cast<std::size_t>(columns),
      static_cast<std::size_t>(columns) * sizeof(float)
    );
  }
}

id<MTLBuffer> resident_matrix_buffer(
  id<MTLDevice> device,
  const std::vector<float>& values,
  int rows,
  int columns,
  NSUInteger row_bytes
) {
  constexpr std::size_t kMaximumResidentFoldMatrices = 16;
  MetalPLSResidentWorkspace& workspace = g_metal_pls_workspace;
  for (MetalResidentMatrixSlot& slot : workspace.matrices) {
    if (slot.epoch == workspace.epoch &&
        slot.host == values.data() &&
        slot.rows == rows &&
        slot.columns == columns &&
        slot.row_bytes == row_bytes) {
      ++workspace.matrix_reuses;
      slot.last_use = ++workspace.cache_clock;
      return slot.buffer;
    }
  }
  MetalResidentMatrixSlot* target = nullptr;
  if (workspace.matrices.size() < kMaximumResidentFoldMatrices) {
    workspace.matrices.emplace_back();
    target = &workspace.matrices.back();
  } else {
    target = &*std::min_element(
      workspace.matrices.begin(),
      workspace.matrices.end(),
      [](const MetalResidentMatrixSlot& left, const MetalResidentMatrixSlot& right) {
        return left.last_use < right.last_use;
      }
    );
  }
  ensure_shared_buffer(
    device,
    target->buffer,
    row_bytes * static_cast<std::size_t>(rows)
  );
  write_matrix_buffer(target->buffer, values, rows, columns, row_bytes);
  target->epoch = workspace.epoch;
  target->host = values.data();
  target->rows = rows;
  target->columns = columns;
  target->row_bytes = row_bytes;
  target->last_use = ++workspace.cache_clock;
  ++workspace.matrix_uploads;
  return target->buffer;
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

float power_vector_norm(const std::vector<float>& values) {
  float float_sum = 0.0f;
  double double_sum = 0.0;
  for (float value : values) {
    float_sum += value * value;
    double_sum += static_cast<double>(value) * static_cast<double>(value);
  }
  if (std::isfinite(float_sum)) return std::sqrt(std::max(0.0f, float_sum));
  if (!std::isfinite(double_sum) || double_sum < 0.0) {
    return std::numeric_limits<float>::infinity();
  }
  return static_cast<float>(std::sqrt(double_sum));
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

NativeKNNResult search_metal_ivf_buffers(
  id<MTLBuffer> train,
  id<MTLBuffer> query,
  id<MTLBuffer> projected_query,
  id<MTLBuffer> centroids,
  id<MTLBuffer> list_offsets,
  id<MTLBuffer> list_ids,
  int train_rows,
  int query_rows,
  int dimensions,
  int projected_dimensions,
  int nlist,
  int k,
  DistanceMetric metric,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices,
  MetalIVFStats* stats,
  id<MTLBuffer> resident_output_ids = nil,
  id<MTLBuffer> resident_output_distances = nil,
  bool materialize_host = true,
  const std::vector<int>* allowed_local_ids = nullptr
) {
  if (requested_nprobe > kMaximumMetalProbe) {
    throw std::invalid_argument("Metal IVF supports nprobe <= 128.");
  }
  if (allowed_local_ids != nullptr && static_cast<int>(allowed_local_ids->size()) != train_rows) {
    throw std::invalid_argument("Metal IVF landmark mask size mismatch.");
  }
  const int allowed = allowed_local_ids == nullptr ? train_rows :
    static_cast<int>(std::count_if(
      allowed_local_ids->begin(), allowed_local_ids->end(), [](int id) { return id >= 0; }
    ));
  const int available = allowed - (query_train_indices.empty() ? 0 : 1);
  k = std::min(k, std::max(0, available));
  if (k > kMaximumMetalK) throw std::invalid_argument("Metal IVF supports k <= 128.");
  NativeKNNResult output;
  output.queries = query_rows;
  output.neighbors = k;
  const std::size_t output_items =
    static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k);
  if (materialize_host) {
    output.indices.assign(output_items, -1);
    output.distances.assign(output_items, std::numeric_limits<float>::infinity());
  }
  if (query_rows == 0 || k == 0) return output;

  std::vector<int> exclusions = query_train_indices;
  if (exclusions.empty()) exclusions.assign(static_cast<std::size_t>(query_rows), -1);
  target_recall = std::max(0.0, std::min(1.0, target_recall));
  const int max_probe = std::min(nlist, kMaximumMetalProbe);
  int nprobe = requested_nprobe > 0 ? requested_nprobe : std::min(8, max_probe);
  nprobe = std::max(1, std::min(nprobe, max_probe));

  @autoreleasepool {
    MetalState& state = metal_state();
    id<MTLBuffer> exclusion_buffer = [state.device
      newBufferWithBytes:exclusions.data()
      length:exclusions.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    if (exclusion_buffer == nil) {
      throw std::runtime_error("Failed to allocate Metal IVF exclusion buffer.");
    }
    const int dummy_allowed = -1;
    id<MTLBuffer> allowed_buffer = [state.device
      newBufferWithBytes:allowed_local_ids == nullptr ? &dummy_allowed : allowed_local_ids->data()
      length:allowed_local_ids == nullptr ? sizeof(dummy_allowed) :
        allowed_local_ids->size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    if (allowed_buffer == nil) {
      throw std::runtime_error("Failed to allocate Metal IVF landmark mask buffer.");
    }

    auto run_exact = [&](int rows, id<MTLBuffer> ids, id<MTLBuffer> distances) {
      const ExactParamsHost parameters{
        static_cast<std::uint32_t>(train_rows),
        static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(dimensions),
        static_cast<std::uint32_t>(k),
        metric == DistanceMetric::Euclidean ? 0u : 1u,
        allowed_local_ids == nullptr ? 0u : 1u
      };
      id<MTLBuffer> parameter_buffer = [state.device
        newBufferWithBytes:&parameters
        length:sizeof(parameters)
        options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> command = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:state.exact_pipeline];
      [encoder setBuffer:train offset:0 atIndex:0];
      [encoder setBuffer:query offset:0 atIndex:1];
      [encoder setBuffer:exclusion_buffer offset:0 atIndex:2];
      [encoder setBuffer:ids offset:0 atIndex:3];
      [encoder setBuffer:distances offset:0 atIndex:4];
      [encoder setBuffer:allowed_buffer offset:0 atIndex:5];
      [encoder setBuffer:parameter_buffer offset:0 atIndex:6];
      [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder endEncoding];
      wait_for_command(command, "Metal resident IVF exact pilot failed");
    };

    auto run_search = [&](int rows, int probes, id<MTLBuffer> ids, id<MTLBuffer> distances) {
      const IVFSearchParamsHost parameters{
        static_cast<std::uint32_t>(train_rows),
        static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(dimensions),
        static_cast<std::uint32_t>(projected_dimensions),
        static_cast<std::uint32_t>(nlist),
        static_cast<std::uint32_t>(probes),
        static_cast<std::uint32_t>(k),
        metric == DistanceMetric::Euclidean ? 0u : 1u,
        allowed_local_ids == nullptr ? 0u : 1u
      };
      id<MTLBuffer> parameter_buffer = [state.device
        newBufferWithBytes:&parameters
        length:sizeof(parameters)
        options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> command = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:state.ivf_pipeline];
      [encoder setBuffer:train offset:0 atIndex:0];
      [encoder setBuffer:query offset:0 atIndex:1];
      [encoder setBuffer:projected_query offset:0 atIndex:2];
      [encoder setBuffer:centroids offset:0 atIndex:3];
      [encoder setBuffer:list_offsets offset:0 atIndex:4];
      [encoder setBuffer:list_ids offset:0 atIndex:5];
      [encoder setBuffer:exclusion_buffer offset:0 atIndex:6];
      [encoder setBuffer:ids offset:0 atIndex:7];
      [encoder setBuffer:distances offset:0 atIndex:8];
      [encoder setBuffer:allowed_buffer offset:0 atIndex:9];
      [encoder setBuffer:parameter_buffer offset:0 atIndex:10];
      [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder endEncoding];
      wait_for_command(command, "Metal resident IVF search failed");
    };

    const int pilot_rows = std::min(query_rows, 128);
    const std::size_t pilot_items = static_cast<std::size_t>(pilot_rows) * k;
    id<MTLBuffer> exact_ids = [state.device
      newBufferWithLength:pilot_items * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> exact_distances = [state.device
      newBufferWithLength:pilot_items * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> approximate_ids = [state.device
      newBufferWithLength:pilot_items * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> approximate_distances = [state.device
      newBufferWithLength:pilot_items * sizeof(float)
      options:MTLResourceStorageModeShared];
    if (exact_ids == nil || exact_distances == nil ||
        approximate_ids == nil || approximate_distances == nil) {
      throw std::runtime_error("Failed to allocate Metal resident IVF pilot buffers.");
    }
    run_exact(pilot_rows, exact_ids, exact_distances);
    std::vector<int> pilot_exact(pilot_items, -1);
    std::memcpy(
      pilot_exact.data(),
      [exact_ids contents],
      pilot_items * sizeof(int)
    );
    auto evaluate_probe = [&](int probes) {
      run_search(pilot_rows, probes, approximate_ids, approximate_distances);
      return recall_at_k(
        pilot_exact,
        static_cast<const int*>([approximate_ids contents]),
        pilot_rows,
        k
      );
    };
    auto pilot_is_complete = [&]() {
      const int* ids = static_cast<const int*>([approximate_ids contents]);
      return std::none_of(ids, ids + pilot_items, [](int id) { return id < 0; });
    };

    double pilot_recall = 0.0;
    if (requested_nprobe <= 0) {
      int low_fail = nprobe - 1;
      int high = nprobe;
      while (true) {
        pilot_recall = evaluate_probe(high);
        if (pilot_recall >= target_recall || high >= max_probe) break;
        low_fail = high;
        high = std::min(
          max_probe,
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
      while (!pilot_is_complete() && nprobe < max_probe) {
        nprobe = std::min(
          max_probe,
          std::max(nprobe + 1, static_cast<int>(std::ceil(static_cast<double>(nprobe) * 1.5)))
        );
        pilot_recall = evaluate_probe(nprobe);
      }
    }

    id<MTLBuffer> output_ids = resident_output_ids;
    id<MTLBuffer> output_distances = resident_output_distances;
    if (output_ids == nil) {
      output_ids = [state.device
        newBufferWithLength:output_items * sizeof(int)
        options:MTLResourceStorageModeShared];
    }
    if (output_distances == nil) {
      output_distances = [state.device
        newBufferWithLength:output_items * sizeof(float)
        options:MTLResourceStorageModeShared];
    }
    if (output_ids == nil || output_distances == nil) {
      throw std::runtime_error("Failed to allocate Metal resident IVF output buffers.");
    }
    run_search(query_rows, nprobe, output_ids, output_distances);
    if (allowed_local_ids != nullptr) {
      const int* ids = static_cast<const int*>([output_ids contents]);
      if (std::any_of(ids, ids + output_items, [](int id) { return id < 0; })) {
        run_exact(query_rows, output_ids, output_distances);
        pilot_recall = 1.0;
      }
    }
    if (materialize_host) {
      std::memcpy(output.indices.data(), [output_ids contents], output_items * sizeof(int));
      std::memcpy(output.distances.data(), [output_distances contents], output_items * sizeof(float));
    }
    if (stats != nullptr) {
      stats->nlist = nlist;
      stats->nprobe = nprobe;
      stats->pilot_recall = pilot_recall;
    }
  }
  return output;
}

}  // namespace

NativeMetalIVFIndex::NativeMetalIVFIndex() = default;
NativeMetalIVFIndex::~NativeMetalIVFIndex() = default;
NativeMetalIVFIndex::NativeMetalIVFIndex(NativeMetalIVFIndex&&) noexcept = default;
NativeMetalIVFIndex& NativeMetalIVFIndex::operator=(NativeMetalIVFIndex&&) noexcept = default;
NativeMetalIVFIndex::NativeMetalIVFIndex(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool NativeMetalIVFIndex::valid() const noexcept { return impl_ != nullptr; }
int NativeMetalIVFIndex::rows() const noexcept { return impl_ == nullptr ? 0 : impl_->rows; }
int NativeMetalIVFIndex::dimensions() const noexcept { return impl_ == nullptr ? 0 : impl_->dimensions; }
int NativeMetalIVFIndex::nlist() const noexcept { return impl_ == nullptr ? 0 : impl_->nlist; }
DistanceMetric NativeMetalIVFIndex::metric() const noexcept {
  return impl_ == nullptr ? DistanceMetric::Euclidean : impl_->metric;
}

NativeMetalKNNVoteGraph::NativeMetalKNNVoteGraph() = default;
NativeMetalKNNVoteGraph::~NativeMetalKNNVoteGraph() = default;
NativeMetalKNNVoteGraph::NativeMetalKNNVoteGraph(
  NativeMetalKNNVoteGraph&&
) noexcept = default;
NativeMetalKNNVoteGraph& NativeMetalKNNVoteGraph::operator=(
  NativeMetalKNNVoteGraph&&
) noexcept = default;
NativeMetalKNNVoteGraph::NativeMetalKNNVoteGraph(
  std::unique_ptr<Impl> impl
) : impl_(std::move(impl)) {}

bool NativeMetalKNNVoteGraph::valid() const noexcept {
  return impl_ != nullptr;
}
int NativeMetalKNNVoteGraph::samples() const noexcept {
  return impl_ == nullptr ? 0 : impl_->samples;
}
int NativeMetalKNNVoteGraph::neighbors() const noexcept {
  return impl_ == nullptr ? 0 : impl_->neighbors;
}

NativeMetalKODAMAGraph::NativeMetalKODAMAGraph() = default;
NativeMetalKODAMAGraph::~NativeMetalKODAMAGraph() = default;
NativeMetalKODAMAGraph::NativeMetalKODAMAGraph(
  NativeMetalKODAMAGraph&&
) noexcept = default;
NativeMetalKODAMAGraph& NativeMetalKODAMAGraph::operator=(
  NativeMetalKODAMAGraph&&
) noexcept = default;
NativeMetalKODAMAGraph::NativeMetalKODAMAGraph(
  std::unique_ptr<Impl> impl
) : impl_(std::move(impl)) {}

bool NativeMetalKODAMAGraph::valid() const noexcept {
  return impl_ != nullptr;
}
int NativeMetalKODAMAGraph::samples() const noexcept {
  return impl_ == nullptr ? 0 : impl_->samples;
}
int NativeMetalKODAMAGraph::neighbors() const noexcept {
  return impl_ == nullptr ? 0 : impl_->neighbors;
}
int NativeMetalKODAMAGraph::lanes() const noexcept {
  return impl_ == nullptr ? 0 : static_cast<int>(impl_->lane_buffers.size());
}
bool NativeMetalKODAMAGraph::has_landmark_index() const noexcept {
  return impl_ != nullptr && impl_->ivf_index != nullptr;
}

NativeMetalKMeansContext::NativeMetalKMeansContext() = default;
NativeMetalKMeansContext::~NativeMetalKMeansContext() = default;
NativeMetalKMeansContext::NativeMetalKMeansContext(NativeMetalKMeansContext&&) noexcept = default;
NativeMetalKMeansContext& NativeMetalKMeansContext::operator=(NativeMetalKMeansContext&&) noexcept = default;
NativeMetalKMeansContext::NativeMetalKMeansContext(
  std::unique_ptr<Impl> impl
) : impl_(std::move(impl)) {}

bool NativeMetalKMeansContext::valid() const noexcept { return impl_ != nullptr; }
int NativeMetalKMeansContext::rows() const noexcept { return impl_ == nullptr ? 0 : impl_->rows; }
int NativeMetalKMeansContext::dimensions() const noexcept {
  return impl_ == nullptr ? 0 : impl_->dimensions;
}
int NativeMetalKMeansContext::lanes() const noexcept {
  return impl_ == nullptr ? 0 : static_cast<int>(impl_->lane.size());
}
std::uint64_t NativeMetalKMeansContext::input_uploads() const noexcept {
  return impl_ == nullptr ? 0 : impl_->input_uploads;
}

bool metal_backend_available() {
  @autoreleasepool {
    id<MTLDevice> device = select_metal_device();
    return device != nil;
  }
}

int metal_recommended_worker_count(
  const std::size_t estimated_worker_bytes,
  const int max_workers
) {
  @autoreleasepool {
    id<MTLDevice> device = select_metal_device();
    if (device == nil || max_workers <= 1 || estimated_worker_bytes == 0) return 1;
    std::uint64_t budget = 0;
    if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
      budget = device.recommendedMaxWorkingSetSize;
    }
    std::uint64_t allocated = 0;
    if ([device respondsToSelector:@selector(currentAllocatedSize)]) {
      allocated = device.currentAllocatedSize;
    }
    if (budget <= allocated) return 1;
    const double available = 0.70 * static_cast<double>(budget - allocated);
    const int memory_cap = std::max(
      1,
      static_cast<int>(available / static_cast<double>(estimated_worker_bytes))
    );
    // Independent M runs submit small, synchronization-heavy command streams.
    // Four queues underutilize current Apple GPUs; memory remains the primary
    // safety bound. Independent command streams hide the synchronization
    // latency of SIMPLS component updates without changing any run.
    constexpr int kMaximumConcurrentMetalRuns = 32;
    return std::max(1, std::min({max_workers, memory_cap, kMaximumConcurrentMetalRuns}));
  }
}

void metal_set_pls_residency_epoch(std::uint64_t epoch) {
  g_metal_pls_workspace.epoch = epoch;
}

NativeMetalKNNVoteGraph metal_build_knn_vote_graph(
  const std::vector<int>& neighbor_rows,
  const std::vector<float>& scores,
  int samples,
  int neighbors
) {
  const std::size_t expected =
    static_cast<std::size_t>(samples) * static_cast<std::size_t>(neighbors);
  if (samples < 1 || neighbors < 1 ||
      neighbor_rows.size() != expected || scores.size() != expected) {
    throw std::invalid_argument("Invalid resident Metal KNN vote graph.");
  }
  @autoreleasepool {
    MetalState& state = metal_state();
    auto impl = std::make_unique<NativeMetalKNNVoteGraph::Impl>();
    impl->samples = samples;
    impl->neighbors = neighbors;
    impl->neighbor_rows = [state.device
      newBufferWithBytes:neighbor_rows.data()
      length:expected * sizeof(int)
      options:MTLResourceStorageModeShared];
    impl->scores = [state.device
      newBufferWithBytes:scores.data()
      length:expected * sizeof(float)
      options:MTLResourceStorageModeShared];
    impl->labels = [state.device
      newBufferWithLength:static_cast<std::size_t>(samples) * sizeof(int)
      options:MTLResourceStorageModeShared];
    impl->predictions = [state.device
      newBufferWithLength:static_cast<std::size_t>(samples) * sizeof(int)
      options:MTLResourceStorageModeShared];
    if (impl->neighbor_rows == nil || impl->scores == nil ||
        impl->labels == nil || impl->predictions == nil) {
      throw std::runtime_error(
        "Failed to allocate resident Metal KNN vote buffers."
      );
    }
    return NativeMetalKNNVoteGraph(std::move(impl));
  }
}

std::vector<int> metal_knn_vote_predict(
  const NativeMetalKNNVoteGraph& graph,
  const std::vector<int>& labels,
  int fallback_label
) {
  std::vector<int> predictions;
  metal_knn_vote_predict_into(graph, labels, fallback_label, predictions);
  return predictions;
}

void metal_knn_vote_predict_into(
  const NativeMetalKNNVoteGraph& graph,
  const std::vector<int>& labels,
  int fallback_label,
  std::vector<int>& predictions
) {
  if (!graph.valid()) {
    throw std::invalid_argument("Resident Metal KNN vote graph is empty.");
  }
  NativeMetalKNNVoteGraph::Impl& impl = *graph.impl_;
  if (labels.size() != static_cast<std::size_t>(impl.samples)) {
    throw std::invalid_argument("Resident Metal KNN label size mismatch.");
  }
  @autoreleasepool {
    MetalState& state = metal_state();
    std::memcpy(
      [impl.labels contents],
      labels.data(),
      labels.size() * sizeof(int)
    );
    const KNNVoteParamsHost parameters{
      static_cast<std::uint32_t>(impl.samples),
      static_cast<std::uint32_t>(impl.neighbors),
      static_cast<std::int32_t>(fallback_label)
    };
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state.knn_vote_pipeline];
    [encoder setBuffer:impl.neighbor_rows offset:0 atIndex:0];
    [encoder setBuffer:impl.scores offset:0 atIndex:1];
    [encoder setBuffer:impl.labels offset:0 atIndex:2];
    [encoder setBuffer:impl.predictions offset:0 atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
    const NSUInteger threads = std::min<NSUInteger>(
      256,
      state.knn_vote_pipeline.maxTotalThreadsPerThreadgroup
    );
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(impl.samples), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    wait_for_command(command, "Resident Metal KNN vote failed");
    predictions.resize(static_cast<std::size_t>(impl.samples));
    std::memcpy(
      predictions.data(),
      [impl.predictions contents],
      predictions.size() * sizeof(int)
    );
  }
}

NativeMetalKODAMAGraph metal_build_resident_kodama_graph(
  const NeighborGraph& graph,
  int samples,
  int lanes
) {
  const std::size_t expected =
    static_cast<std::size_t>(samples) *
    static_cast<std::size_t>(graph.neighbors);
  if (samples < 1 || graph.neighbors < 1 ||
      graph.indices.size() != expected ||
      graph.distances.size() != expected) {
    throw std::invalid_argument("Invalid resident Metal KODAMA graph.");
  }
  @autoreleasepool {
    MetalState& state = metal_state();
    auto impl = std::make_unique<NativeMetalKODAMAGraph::Impl>();
    impl->samples = samples;
    impl->neighbors = graph.neighbors;
    impl->indices = [state.device
      newBufferWithBytes:graph.indices.data()
      length:expected * sizeof(int)
      options:MTLResourceStorageModeShared];
    impl->distances = [state.device
      newBufferWithBytes:graph.distances.data()
      length:expected * sizeof(float)
      options:MTLResourceStorageModeShared];
    impl->base_indices = [state.device
      newBufferWithBytes:graph.indices.data()
      length:expected * sizeof(int)
      options:MTLResourceStorageModeShared];
    impl->base_distances = [state.device
      newBufferWithBytes:graph.distances.data()
      length:expected * sizeof(float)
      options:MTLResourceStorageModeShared];
    impl->lane_buffers.resize(static_cast<std::size_t>(std::max(1, lanes)));
    for (NativeMetalKODAMAGraph::Impl::Lane& lane : impl->lane_buffers) {
      lane.landmark_epoch = [state.device
        newBufferWithLength:static_cast<std::size_t>(samples) * sizeof(int)
        options:MTLResourceStorageModeShared];
      lane.labels = [state.device
        newBufferWithLength:static_cast<std::size_t>(samples) * sizeof(int)
        options:MTLResourceStorageModeShared];
      lane.constrain = [state.device
        newBufferWithLength:static_cast<std::size_t>(samples) * sizeof(int)
        options:MTLResourceStorageModeShared];
      if (lane.landmark_epoch == nil || lane.labels == nil || lane.constrain == nil) {
        throw std::runtime_error(
          "Failed to allocate resident Metal KODAMA projection buffers."
        );
      }
      std::memset([lane.landmark_epoch contents], 0,
                  static_cast<std::size_t>(samples) * sizeof(int));
    }
    if (impl->indices == nil || impl->distances == nil ||
        impl->base_indices == nil || impl->base_distances == nil) {
      throw std::runtime_error(
        "Failed to allocate resident Metal KODAMA graph buffers."
      );
    }
    return NativeMetalKODAMAGraph(std::move(impl));
  }
}

NativeMetalKODAMAGraph metal_build_resident_kodama_graph_ivf(
  const std::vector<float>& data,
  int samples,
  int dimensions,
  int neighbors,
  DistanceMetric metric,
  int requested_nlist,
  int requested_nprobe,
  int lanes,
  MetalIVFStats* stats
) {
  if (samples < 2 || dimensions < 1 || neighbors < 1 || neighbors >= samples ||
      data.size() != static_cast<std::size_t>(samples) * dimensions) {
    throw std::invalid_argument("Invalid resident Metal IVF KODAMA graph input.");
  }
  NativeMetalIVFIndex index = metal_build_ivf_index(
    data,
    samples,
    dimensions,
    metric,
    requested_nlist
  );
  const NativeMetalIVFIndex::Impl& index_impl = *index.impl_;
  @autoreleasepool {
    MetalState& state = metal_state();
    auto impl = std::make_unique<NativeMetalKODAMAGraph::Impl>();
    impl->samples = samples;
    impl->neighbors = neighbors;
    impl->metric = metric;
    impl->dimensions = dimensions;
    const std::size_t items = static_cast<std::size_t>(samples) * neighbors;
    impl->indices = [state.device
      newBufferWithLength:items * sizeof(int)
      options:MTLResourceStorageModeShared];
    impl->distances = [state.device
      newBufferWithLength:items * sizeof(float)
      options:MTLResourceStorageModeShared];
    impl->base_indices = [state.device
      newBufferWithLength:items * sizeof(int)
      options:MTLResourceStorageModeShared];
    impl->base_distances = [state.device
      newBufferWithLength:items * sizeof(float)
      options:MTLResourceStorageModeShared];
    if (impl->indices == nil || impl->distances == nil ||
        impl->base_indices == nil || impl->base_distances == nil) {
      throw std::runtime_error("Failed to allocate resident Metal IVF graph buffers.");
    }
    std::vector<int> exclusions(static_cast<std::size_t>(samples));
    std::iota(exclusions.begin(), exclusions.end(), 0);
    (void)search_metal_ivf_buffers(
      index_impl.train,
      index_impl.train,
      index_impl.projected_train,
      index_impl.centroids,
      index_impl.list_offsets,
      index_impl.list_ids,
      samples,
      samples,
      dimensions,
      index_impl.projected_dimensions,
      index_impl.nlist,
      neighbors,
      metric,
      requested_nprobe,
      0.99,
      exclusions,
      stats,
      impl->indices,
      impl->distances,
      false
    );
    impl->ivf_index = std::make_unique<NativeMetalIVFIndex>(std::move(index));
    float* distances = static_cast<float*>([impl->distances contents]);
    for (std::size_t item = 0; item < items; ++item) {
      distances[item] = native_knn_output_distance(distances[item], metric);
    }
    std::memcpy([impl->base_indices contents], [impl->indices contents], items * sizeof(int));
    std::memcpy([impl->base_distances contents], [impl->distances contents], items * sizeof(float));
    impl->lane_buffers.resize(static_cast<std::size_t>(std::max(1, lanes)));
    for (NativeMetalKODAMAGraph::Impl::Lane& lane : impl->lane_buffers) {
      lane.landmark_epoch = [state.device
        newBufferWithLength:static_cast<std::size_t>(samples) * sizeof(int)
        options:MTLResourceStorageModeShared];
      lane.labels = [state.device
        newBufferWithLength:static_cast<std::size_t>(samples) * sizeof(int)
        options:MTLResourceStorageModeShared];
      lane.constrain = [state.device
        newBufferWithLength:static_cast<std::size_t>(samples) * sizeof(int)
        options:MTLResourceStorageModeShared];
      if (lane.landmark_epoch == nil || lane.labels == nil || lane.constrain == nil) {
        throw std::runtime_error("Failed to allocate resident Metal IVF projection buffers.");
      }
      std::memset([lane.landmark_epoch contents], 0,
                  static_cast<std::size_t>(samples) * sizeof(int));
    }
    return NativeMetalKODAMAGraph(std::move(impl));
  }
}

NeighborGraph metal_resident_landmark_knn_graph(
  const NativeMetalKODAMAGraph& graph,
  const std::vector<float>& landmark_data,
  const std::vector<int>& landmark_rows,
  int k,
  int requested_nprobe,
  double target_recall
) {
  if (!graph.valid() || graph.impl_->ivf_index == nullptr) {
    throw std::invalid_argument("Metal resident KODAMA graph has no reusable IVF index.");
  }
  const NativeMetalKODAMAGraph::Impl& impl = *graph.impl_;
  if (landmark_rows.size() < 2 ||
      landmark_data.size() != landmark_rows.size() * static_cast<std::size_t>(impl.dimensions)) {
    throw std::invalid_argument("Invalid Metal resident landmark query matrix.");
  }
  std::vector<int> allowed_local_ids(static_cast<std::size_t>(impl.samples), -1);
  for (std::size_t local = 0; local < landmark_rows.size(); ++local) {
    const int global = landmark_rows[local];
    if (global < 0 || global >= impl.samples) {
      throw std::out_of_range("Metal landmark row is outside the resident index.");
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
  MetalIVFStats stats;
  NativeKNNResult result = metal_ivf_index_filtered_search(
    *impl.ivf_index, prepared_landmarks, static_cast<int>(landmark_rows.size()),
    std::min(k, static_cast<int>(landmark_rows.size()) - 1),
    requested_nprobe, target_recall, landmark_rows, allowed_local_ids, &stats
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

void metal_prepare_resident_results(
  NativeMetalKODAMAGraph& graph,
  int runs,
  int lanes
) {
  if (!graph.valid() || runs < 1 || lanes < 1) {
    throw std::invalid_argument("Invalid Metal resident KODAMA result matrix.");
  }
  @autoreleasepool {
    MetalState& state = metal_state();
    NativeMetalKODAMAGraph::Impl& impl = *graph.impl_;
    const std::size_t prior_lanes = impl.lane_buffers.size();
    if (prior_lanes < static_cast<std::size_t>(lanes)) {
      impl.lane_buffers.resize(static_cast<std::size_t>(lanes));
      for (std::size_t lane_id = prior_lanes;
           lane_id < impl.lane_buffers.size();
           ++lane_id) {
        NativeMetalKODAMAGraph::Impl::Lane& lane = impl.lane_buffers[lane_id];
        lane.landmark_epoch = [state.device
          newBufferWithLength:static_cast<std::size_t>(impl.samples) * sizeof(int)
          options:MTLResourceStorageModeShared];
        lane.labels = [state.device
          newBufferWithLength:static_cast<std::size_t>(impl.samples) * sizeof(int)
          options:MTLResourceStorageModeShared];
        lane.constrain = [state.device
          newBufferWithLength:static_cast<std::size_t>(impl.samples) * sizeof(int)
          options:MTLResourceStorageModeShared];
        if (lane.landmark_epoch == nil || lane.labels == nil || lane.constrain == nil) {
          throw std::runtime_error(
            "Failed to expand reusable Metal KODAMA projection workspaces."
          );
        }
      }
    }
    const std::size_t required = static_cast<std::size_t>(runs) * impl.samples;
    if (impl.result_capacity < required) {
      impl.result_labels = [state.device
        newBufferWithLength:required * sizeof(int)
        options:MTLResourceStorageModeShared];
      if (impl.result_labels == nil) {
        throw std::runtime_error("Failed to allocate resident Metal KODAMA results.");
      }
      impl.result_capacity = required;
    }
    for (NativeMetalKODAMAGraph::Impl::Lane& lane : impl.lane_buffers) {
      std::memset(
        [lane.landmark_epoch contents],
        0,
        static_cast<std::size_t>(impl.samples) * sizeof(int)
      );
    }
    impl.result_runs = runs;
  }
}

void metal_project_landmark_labels_to_result(
  NativeMetalKODAMAGraph& graph,
  const std::vector<int>& landmark_rows,
  const std::vector<int>& landmark_labels,
  int projection_k,
  int fallback_label,
  int run,
  int lane
) {
  if (!graph.valid()) {
    throw std::invalid_argument("Resident Metal KODAMA graph is empty.");
  }
  NativeMetalKODAMAGraph::Impl& impl = *graph.impl_;
  if (landmark_rows.empty() || landmark_rows.size() != landmark_labels.size() ||
      run < 0 || run >= impl.result_runs || impl.result_labels == nil) {
    throw std::invalid_argument(
      "Resident Metal KODAMA projection input size mismatch."
    );
  }
  @autoreleasepool {
    MetalState& state = metal_state();
    const int lane_id =
      std::max(0, lane) % static_cast<int>(impl.lane_buffers.size());
    NativeMetalKODAMAGraph::Impl::Lane& workspace =
      impl.lane_buffers[static_cast<std::size_t>(lane_id)];
    if (workspace.landmark_capacity < landmark_rows.size()) {
      workspace.landmark_rows = [state.device
        newBufferWithLength:landmark_rows.size() * sizeof(int)
        options:MTLResourceStorageModeShared];
      workspace.landmark_input_labels = [state.device
        newBufferWithLength:landmark_rows.size() * sizeof(int)
        options:MTLResourceStorageModeShared];
      if (workspace.landmark_rows == nil || workspace.landmark_input_labels == nil) {
        throw std::runtime_error("Failed to allocate sparse Metal landmark buffers.");
      }
      workspace.landmark_capacity = landmark_rows.size();
    }
    std::memcpy([workspace.landmark_rows contents], landmark_rows.data(),
                landmark_rows.size() * sizeof(int));
    std::memcpy([workspace.landmark_input_labels contents], landmark_labels.data(),
                landmark_labels.size() * sizeof(int));
    const std::uint32_t epoch = static_cast<std::uint32_t>(run + 1);
    const LandmarkScatterParamsHost scatter_parameters{
      static_cast<std::uint32_t>(landmark_rows.size()),
      static_cast<std::uint32_t>(impl.samples), epoch
    };
    const LandmarkProjectionParamsHost parameters{
      static_cast<std::uint32_t>(impl.samples),
      static_cast<std::uint32_t>(impl.neighbors),
      static_cast<std::uint32_t>(
        std::max(1, std::min(projection_k, impl.neighbors))
      ),
      static_cast<std::int32_t>(fallback_label),
      epoch
    };
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> scatter = [command computeCommandEncoder];
    [scatter setComputePipelineState:state.landmark_scatter_pipeline];
    [scatter setBuffer:workspace.landmark_rows offset:0 atIndex:0];
    [scatter setBuffer:workspace.landmark_input_labels offset:0 atIndex:1];
    [scatter setBuffer:workspace.landmark_epoch offset:0 atIndex:2];
    [scatter setBuffer:workspace.labels offset:0 atIndex:3];
    [scatter setBytes:&scatter_parameters length:sizeof(scatter_parameters) atIndex:4];
    const NSUInteger scatter_threads = std::min<NSUInteger>(
      256, state.landmark_scatter_pipeline.maxTotalThreadsPerThreadgroup);
    [scatter dispatchThreads:MTLSizeMake(landmark_rows.size(), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(scatter_threads, 1, 1)];
    [scatter endEncoding];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state.landmark_projection_pipeline];
    [encoder setBuffer:impl.indices offset:0 atIndex:0];
    [encoder setBuffer:workspace.landmark_epoch offset:0 atIndex:1];
    [encoder setBuffer:workspace.labels offset:0 atIndex:2];
    [encoder setBuffer:impl.result_labels
                  offset:static_cast<NSUInteger>(run) * impl.samples * sizeof(int)
                 atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
    const NSUInteger threads = std::min<NSUInteger>(
      256,
      state.landmark_projection_pipeline.maxTotalThreadsPerThreadgroup
    );
    [encoder
      dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(impl.samples), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    wait_for_command(command, "Resident Metal KODAMA projection failed");
  }
}

void metal_store_resident_result_row(
  NativeMetalKODAMAGraph& graph,
  const std::vector<int>& labels,
  int run,
  int
) {
  if (!graph.valid() || labels.size() != static_cast<std::size_t>(graph.impl_->samples) ||
      run < 0 || run >= graph.impl_->result_runs) {
    throw std::invalid_argument("Metal resident KODAMA result row size mismatch.");
  }
  std::memcpy(static_cast<char*>([graph.impl_->result_labels contents]) +
                static_cast<std::size_t>(run) * graph.impl_->samples * sizeof(int),
              labels.data(), labels.size() * sizeof(int));
}

void metal_constrain_resident_result_row(
  NativeMetalKODAMAGraph& graph,
  const std::vector<int>& constrain,
  int max_label,
  int run,
  int lane
) {
  if (!graph.valid() || constrain.size() != static_cast<std::size_t>(graph.impl_->samples) ||
      max_label < 0 || run < 0 || run >= graph.impl_->result_runs) {
    throw std::invalid_argument("Metal resident KODAMA constraint input mismatch.");
  }
  @autoreleasepool {
    MetalState& state = metal_state();
    NativeMetalKODAMAGraph::Impl& impl = *graph.impl_;
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
      workspace.constrain_counts = [state.device
        newBufferWithLength:count_items * sizeof(std::uint32_t)
        options:MTLResourceStorageModeShared];
      if (workspace.constrain_counts == nil) {
        throw std::runtime_error("Failed to allocate Metal constraint counts.");
      }
      workspace.constrain_count_capacity = count_items;
    }
    if (workspace.constrain_group_capacity < static_cast<std::size_t>(groups)) {
      workspace.constrain_labels = [state.device
        newBufferWithLength:static_cast<std::size_t>(groups) * sizeof(int)
        options:MTLResourceStorageModeShared];
      if (workspace.constrain_labels == nil) {
        throw std::runtime_error("Failed to allocate Metal constraint labels.");
      }
      workspace.constrain_group_capacity = static_cast<std::size_t>(groups);
    }
    std::memcpy([workspace.constrain contents], compact.data(), compact.size() * sizeof(int));
    std::memset([workspace.constrain_counts contents], 0, count_items * sizeof(std::uint32_t));
    const ConstrainedMajorityParamsHost parameters{
      static_cast<std::uint32_t>(impl.samples),
      static_cast<std::uint32_t>(groups),
      static_cast<std::uint32_t>(label_width)
    };
    const NSUInteger row_offset =
      static_cast<NSUInteger>(run) * impl.samples * sizeof(int);
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    auto encode = [&](id<MTLComputePipelineState> pipeline,
                      NSUInteger items,
                      int phase) {
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      if (phase == 0) {
        [encoder setBuffer:impl.result_labels offset:row_offset atIndex:0];
        [encoder setBuffer:workspace.constrain offset:0 atIndex:1];
        [encoder setBuffer:workspace.constrain_counts offset:0 atIndex:2];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
      } else if (phase == 1) {
        [encoder setBuffer:workspace.constrain_counts offset:0 atIndex:0];
        [encoder setBuffer:workspace.constrain_labels offset:0 atIndex:1];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
      } else {
        [encoder setBuffer:impl.result_labels offset:row_offset atIndex:0];
        [encoder setBuffer:workspace.constrain offset:0 atIndex:1];
        [encoder setBuffer:workspace.constrain_labels offset:0 atIndex:2];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
      }
      const NSUInteger threads = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
      [encoder dispatchThreads:MTLSizeMake(items, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [encoder endEncoding];
    };
    encode(state.constrained_count_pipeline, static_cast<NSUInteger>(impl.samples), 0);
    encode(state.constrained_select_pipeline, static_cast<NSUInteger>(groups), 1);
    encode(state.constrained_apply_pipeline, static_cast<NSUInteger>(impl.samples), 2);
    wait_for_command(command, "Resident Metal constrained majority failed");
  }
}

std::vector<int> metal_download_resident_results(
  const NativeMetalKODAMAGraph& graph,
  int runs
) {
  if (!graph.valid() || runs != graph.impl_->result_runs || graph.impl_->result_labels == nil) {
    throw std::invalid_argument("Metal resident KODAMA results are unavailable.");
  }
  std::vector<int> out(static_cast<std::size_t>(runs) * graph.impl_->samples);
  std::memcpy(out.data(), [graph.impl_->result_labels contents], out.size() * sizeof(int));
  return out;
}

std::vector<int> metal_download_resident_result_row(
  const NativeMetalKODAMAGraph& graph,
  int run,
  int
) {
  if (!graph.valid() || run < 0 || run >= graph.impl_->result_runs) {
    throw std::invalid_argument("Metal resident KODAMA result row is unavailable.");
  }
  std::vector<int> out(static_cast<std::size_t>(graph.impl_->samples));
  std::memcpy(out.data(),
              static_cast<const char*>([graph.impl_->result_labels contents]) +
                static_cast<std::size_t>(run) * graph.impl_->samples * sizeof(int),
              out.size() * sizeof(int));
  return out;
}

void metal_apply_resident_kodama_dissimilarity(
  NativeMetalKODAMAGraph& graph,
  int runs,
  bool input_one_based_indices,
  bool output_one_based_indices
) {
  if (!graph.valid()) {
    throw std::invalid_argument("Resident Metal KODAMA graph is empty.");
  }
  NativeMetalKODAMAGraph::Impl& impl = *graph.impl_;
  if (runs < 1 || runs != impl.result_runs || impl.result_labels == nil) {
    throw std::invalid_argument(
      "Resident Metal KODAMA dissimilarity label size mismatch."
    );
  }
  @autoreleasepool {
    MetalState& state = metal_state();
    std::uint32_t sort_width = 1;
    while (sort_width < static_cast<std::uint32_t>(impl.neighbors)) {
      sort_width <<= 1;
    }
    if (sort_width >
        state.kodama_dissimilarity_pipeline.maxTotalThreadsPerThreadgroup) {
      throw std::invalid_argument(
        "The Metal device cannot sort the requested KODAMA graph width."
      );
    }
    const KODAMADissimilarityParamsHost parameters{
      static_cast<std::uint32_t>(runs),
      static_cast<std::uint32_t>(impl.samples),
      static_cast<std::uint32_t>(impl.neighbors),
      sort_width,
      input_one_based_indices ? 1u : 0u,
      output_one_based_indices ? 1u : 0u
    };
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state.kodama_dissimilarity_pipeline];
    [encoder setBuffer:impl.indices offset:0 atIndex:0];
    [encoder setBuffer:impl.distances offset:0 atIndex:1];
    [encoder setBuffer:impl.result_labels offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder
      setThreadgroupMemoryLength:static_cast<NSUInteger>(sort_width) *
        sizeof(float)
      atIndex:0];
    [encoder
      setThreadgroupMemoryLength:static_cast<NSUInteger>(sort_width) *
        sizeof(int)
      atIndex:1];
    [encoder
      dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(impl.samples), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(sort_width, 1, 1)];
    [encoder endEncoding];
    wait_for_command(
      command,
      "Resident Metal KODAMA dissimilarity failed"
    );
  }
}

NeighborGraph metal_download_resident_kodama_graph(
  const NativeMetalKODAMAGraph& graph
) {
  if (!graph.valid()) {
    throw std::invalid_argument("Resident Metal KODAMA graph is empty.");
  }
  const NativeMetalKODAMAGraph::Impl& impl = *graph.impl_;
  @autoreleasepool {
    NeighborGraph out;
    out.neighbors = impl.neighbors;
    const std::size_t items =
      static_cast<std::size_t>(impl.samples) *
      static_cast<std::size_t>(impl.neighbors);
    out.indices.resize(items);
    out.distances.resize(items);
    std::memcpy(
      out.indices.data(),
      [impl.indices contents],
      items * sizeof(int)
    );
    std::memcpy(
      out.distances.data(),
      [impl.distances contents],
      items * sizeof(float)
    );
    return out;
  }
}

void metal_replace_resident_kodama_graph(
  NativeMetalKODAMAGraph& graph,
  const NeighborGraph& replacement
) {
  if (!graph.valid()) throw std::invalid_argument("Metal resident graph is empty.");
  NativeMetalKODAMAGraph::Impl& impl = *graph.impl_;
  const std::size_t items = static_cast<std::size_t>(impl.samples) * impl.neighbors;
  if (replacement.neighbors != impl.neighbors || replacement.indices.size() != items ||
      replacement.distances.size() != items) {
    throw std::invalid_argument("Metal resident replacement graph dimensions mismatch.");
  }
  std::memcpy([impl.indices contents], replacement.indices.data(), items * sizeof(int));
  std::memcpy([impl.distances contents], replacement.distances.data(), items * sizeof(float));
  std::memcpy([impl.base_indices contents], replacement.indices.data(), items * sizeof(int));
  std::memcpy([impl.base_distances contents], replacement.distances.data(), items * sizeof(float));
}

void metal_reset_resident_kodama_graph(NativeMetalKODAMAGraph& graph) {
  if (!graph.valid()) throw std::invalid_argument("Metal resident graph is empty.");
  NativeMetalKODAMAGraph::Impl& impl = *graph.impl_;
  const std::size_t items = static_cast<std::size_t>(impl.samples) * impl.neighbors;
  std::memcpy([impl.indices contents], [impl.base_indices contents], items * sizeof(int));
  std::memcpy([impl.distances contents], [impl.base_distances contents], items * sizeof(float));
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
      metric == DistanceMetric::Euclidean ? 0u : 1u,
      0u
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
    [encoder setBuffer:exclusion_buffer offset:0 atIndex:5];
    [encoder setBuffer:parameter_buffer offset:0 atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(query_rows), 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    wait_for_command(command, "Metal exact KNN command failed");
    std::memcpy(output.indices.data(), [index_buffer contents], output.indices.size() * sizeof(int));
    std::memcpy(output.distances.data(), [distance_buffer contents], output.distances.size() * sizeof(float));
  }
  return output;
}

NativeKNNResult metal_spatial_grid_self_knn(
  const std::vector<float>& data,
  const int rows,
  const int dimensions,
  int k,
  const bool include_self
) {
  if (rows < 2 || (dimensions != 2 && dimensions != 3) ||
      data.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(dimensions)) {
    throw std::invalid_argument("Metal spatial grid KNN requires a valid 2D/3D matrix.");
  }
  k = include_self ? std::min(k, rows) : std::min(k, rows - 1);
  if (k < 1 || k > kMaximumMetalK) {
    throw std::invalid_argument("Metal spatial grid KNN supports 1 <= k <= 128.");
  }
  const int nonself_k = include_self ? k - 1 : k;
  const int bins = spatial_grid_bins_per_dim(rows, std::max(1, nonself_k), dimensions);
  const SpatialGridIndex grid = build_spatial_grid_index(data.data(), rows, dimensions, bins);

  NativeKNNResult output;
  output.queries = rows;
  output.neighbors = k;
  output.indices.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k), -1);
  output.distances.assign(
    static_cast<std::size_t>(rows) * static_cast<std::size_t>(k),
    std::numeric_limits<float>::infinity()
  );

  @autoreleasepool {
    MetalState& state = metal_state();
    id<MTLBuffer> data_buffer = [state.device
      newBufferWithBytes:data.data()
      length:data.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> offsets_buffer = [state.device
      newBufferWithBytes:grid.offsets.data()
      length:grid.offsets.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> rows_buffer = [state.device
      newBufferWithBytes:grid.rows.data()
      length:grid.rows.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> indices_buffer = [state.device
      newBufferWithLength:output.indices.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> distances_buffer = [state.device
      newBufferWithLength:output.distances.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    if (data_buffer == nil || offsets_buffer == nil || rows_buffer == nil ||
        indices_buffer == nil || distances_buffer == nil) {
      throw std::runtime_error("Failed to allocate Metal spatial grid KNN buffers.");
    }

    const SpatialGridParamsHost parameters{
      static_cast<std::uint32_t>(rows),
      static_cast<std::uint32_t>(dimensions),
      static_cast<std::uint32_t>(k),
      static_cast<std::uint32_t>(nonself_k),
      static_cast<std::uint32_t>(grid.bins),
      include_self ? 1u : 0u,
      grid.min_x,
      grid.min_y,
      grid.min_z,
      grid.cell_x,
      grid.cell_y,
      grid.cell_z
    };
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state.spatial_grid_pipeline];
    [encoder setBuffer:data_buffer offset:0 atIndex:0];
    [encoder setBuffer:offsets_buffer offset:0 atIndex:1];
    [encoder setBuffer:rows_buffer offset:0 atIndex:2];
    [encoder setBuffer:indices_buffer offset:0 atIndex:3];
    [encoder setBuffer:distances_buffer offset:0 atIndex:4];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
    const NSUInteger width = std::min<NSUInteger>(
      256,
      state.spatial_grid_pipeline.maxTotalThreadsPerThreadgroup
    );
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    wait_for_command(command, "Metal spatial grid KNN command failed");
    std::memcpy(output.indices.data(), [indices_buffer contents], output.indices.size() * sizeof(int));
    std::memcpy(output.distances.data(), [distances_buffer contents], output.distances.size() * sizeof(float));
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

    std::vector<int> initial_indices(static_cast<std::size_t>(train_rows));
    std::iota(initial_indices.begin(), initial_indices.end(), 0);
    std::mt19937 generator(4u);
    std::shuffle(initial_indices.begin(), initial_indices.end(), generator);
    std::vector<int> assignments(static_cast<std::size_t>(train_rows), -1);
    const std::size_t centroid_items =
      static_cast<std::size_t>(nlist) * static_cast<std::size_t>(projected_dimensions);
    id<MTLBuffer> centroid_buffer = [state.device
      newBufferWithLength:centroid_items * sizeof(float)
      options:MTLResourceStorageModePrivate];
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

    id<MTLCommandBuffer> gather_command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> gather_encoder = [gather_command computeCommandEncoder];
    [gather_encoder setComputePipelineState:state.gather_centroids_pipeline];
    [gather_encoder setBuffer:projected_train_buffer offset:0 atIndex:0];
    [gather_encoder setBuffer:initial_index_buffer offset:0 atIndex:1];
    [gather_encoder setBuffer:centroid_buffer offset:0 atIndex:2];
    [gather_encoder setBuffer:kmeans_params_buffer offset:0 atIndex:3];
    [gather_encoder dispatchThreads:MTLSizeMake(centroid_items, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [gather_encoder endEncoding];
    wait_for_command(gather_command, "Metal IVF centroid initialization failed");

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

    const MetalIVFLists lists = build_metal_ivf_lists(
      state,
      assignment_buffer,
      kmeans_params_buffer,
      train_rows,
      nlist
    );

    auto run_search = [&](int rows, int probes, id<MTLBuffer> ids, id<MTLBuffer> distances) {
      const IVFSearchParamsHost parameters{
        static_cast<std::uint32_t>(train_rows),
        static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(dimensions),
        static_cast<std::uint32_t>(projected_dimensions),
        static_cast<std::uint32_t>(nlist),
        static_cast<std::uint32_t>(probes),
        static_cast<std::uint32_t>(k),
        metric == DistanceMetric::Euclidean ? 0u : 1u,
        0u
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
      [encoder setBuffer:lists.offsets offset:0 atIndex:4];
      [encoder setBuffer:lists.ids offset:0 atIndex:5];
      [encoder setBuffer:exclusion_buffer offset:0 atIndex:6];
      [encoder setBuffer:ids offset:0 atIndex:7];
      [encoder setBuffer:distances offset:0 atIndex:8];
      [encoder setBuffer:exclusion_buffer offset:0 atIndex:9];
      [encoder setBuffer:parameter_buffer offset:0 atIndex:10];
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

NativeMetalIVFIndex metal_build_ivf_index(
  const std::vector<float>& train,
  int train_rows,
  int dimensions,
  DistanceMetric metric,
  int requested_nlist
) {
  if (train_rows < 1 || dimensions < 1 ||
      train.size() != static_cast<std::size_t>(train_rows) * static_cast<std::size_t>(dimensions)) {
    throw std::invalid_argument("Invalid Metal IVF training matrix.");
  }
  if (requested_nlist > kMaximumMetalLists) {
    throw std::invalid_argument("Metal IVF supports nlist <= 1024.");
  }
  int nlist = requested_nlist > 0 ? requested_nlist : static_cast<int>(
    std::ceil(4.0 * std::sqrt(static_cast<double>(train_rows)))
  );
  nlist = std::max(1, std::min({nlist, kMaximumMetalLists, train_rows}));
  const int projected_dimensions = metal_projection_dimension(dimensions);

  std::vector<std::uint32_t> feature_offsets(
    static_cast<std::size_t>(projected_dimensions) + 1,
    0
  );
  std::vector<std::vector<std::pair<std::uint32_t, std::int8_t>>> feature_buckets(
    static_cast<std::size_t>(projected_dimensions)
  );
  for (int dimension = 0; dimension < dimensions; ++dimension) {
    const std::uint32_t hash = static_cast<std::uint32_t>(dimension + 1) * 2654435761u;
    const int bucket = static_cast<int>(
      hash & static_cast<std::uint32_t>(projected_dimensions - 1)
    );
    const std::int8_t sign =
      ((hash >> 17u) & 1u) != 0u ? std::int8_t(1) : std::int8_t(-1);
    feature_buckets[static_cast<std::size_t>(bucket)].push_back(
      {static_cast<std::uint32_t>(dimension), sign}
    );
  }
  std::vector<std::uint32_t> feature_ids;
  std::vector<std::int8_t> feature_signs;
  feature_ids.reserve(static_cast<std::size_t>(dimensions));
  feature_signs.reserve(static_cast<std::size_t>(dimensions));
  for (int bucket = 0; bucket < projected_dimensions; ++bucket) {
    feature_offsets[static_cast<std::size_t>(bucket)] =
      static_cast<std::uint32_t>(feature_ids.size());
    for (const auto& feature : feature_buckets[static_cast<std::size_t>(bucket)]) {
      feature_ids.push_back(feature.first);
      feature_signs.push_back(feature.second);
    }
  }
  feature_offsets.back() = static_cast<std::uint32_t>(feature_ids.size());

  std::vector<int> initial_indices(static_cast<std::size_t>(train_rows));
  std::iota(initial_indices.begin(), initial_indices.end(), 0);
  std::mt19937 generator(4u);
  std::shuffle(initial_indices.begin(), initial_indices.end(), generator);

  @autoreleasepool {
    MetalState& state = metal_state();
    auto impl = std::make_unique<NativeMetalIVFIndex::Impl>();
    impl->rows = train_rows;
    impl->dimensions = dimensions;
    impl->projected_dimensions = projected_dimensions;
    impl->nlist = nlist;
    impl->metric = metric;
    impl->train = [state.device
      newBufferWithBytes:train.data()
      length:train.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    impl->feature_offsets = [state.device
      newBufferWithBytes:feature_offsets.data()
      length:feature_offsets.size() * sizeof(std::uint32_t)
      options:MTLResourceStorageModeShared];
    impl->feature_ids = [state.device
      newBufferWithBytes:feature_ids.data()
      length:feature_ids.size() * sizeof(std::uint32_t)
      options:MTLResourceStorageModeShared];
    impl->feature_signs = [state.device
      newBufferWithBytes:feature_signs.data()
      length:feature_signs.size() * sizeof(std::int8_t)
      options:MTLResourceStorageModeShared];
    impl->projected_train = [state.device
      newBufferWithLength:
        static_cast<std::size_t>(train_rows) *
        static_cast<std::size_t>(projected_dimensions) *
        sizeof(float)
      options:MTLResourceStorageModePrivate];
    if (impl->train == nil || impl->feature_offsets == nil ||
        impl->feature_ids == nil || impl->feature_signs == nil ||
        impl->projected_train == nil) {
      throw std::runtime_error("Failed to allocate resident Metal IVF input buffers.");
    }

    const ProjectParamsHost project_parameters{
      static_cast<std::uint32_t>(train_rows),
      static_cast<std::uint32_t>(dimensions),
      static_cast<std::uint32_t>(projected_dimensions)
    };
    id<MTLBuffer> project_parameter_buffer = [state.device
      newBufferWithBytes:&project_parameters
      length:sizeof(project_parameters)
      options:MTLResourceStorageModeShared];
    const KMeansParamsHost kmeans_parameters{
      static_cast<std::uint32_t>(train_rows),
      static_cast<std::uint32_t>(projected_dimensions),
      static_cast<std::uint32_t>(nlist)
    };
    id<MTLBuffer> kmeans_parameter_buffer = [state.device
      newBufferWithBytes:&kmeans_parameters
      length:sizeof(kmeans_parameters)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> initial_index_buffer = [state.device
      newBufferWithBytes:initial_indices.data()
      length:initial_indices.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    const std::size_t centroid_items =
      static_cast<std::size_t>(nlist) * static_cast<std::size_t>(projected_dimensions);
    impl->centroids = [state.device
      newBufferWithLength:centroid_items * sizeof(float)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> assignments = [state.device
      newBufferWithLength:static_cast<std::size_t>(train_rows) * sizeof(int)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> sums = [state.device
      newBufferWithLength:centroid_items * sizeof(float)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> counts = [state.device
      newBufferWithLength:static_cast<std::size_t>(nlist) * sizeof(std::uint32_t)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> changed = [state.device
      newBufferWithLength:sizeof(std::uint32_t)
      options:MTLResourceStorageModePrivate];
    if (project_parameter_buffer == nil || kmeans_parameter_buffer == nil ||
        initial_index_buffer == nil || impl->centroids == nil ||
        assignments == nil || sums == nil || counts == nil || changed == nil) {
      throw std::runtime_error("Failed to allocate resident Metal IVF training buffers.");
    }

    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit fillBuffer:assignments
                range:NSMakeRange(0, static_cast<std::size_t>(train_rows) * sizeof(int))
                value:0xff];
    [blit endEncoding];

    id<MTLComputeCommandEncoder> project_encoder = [command computeCommandEncoder];
    [project_encoder setComputePipelineState:state.project_pipeline];
    [project_encoder setBuffer:impl->train offset:0 atIndex:0];
    [project_encoder setBuffer:impl->feature_offsets offset:0 atIndex:1];
    [project_encoder setBuffer:impl->feature_ids offset:0 atIndex:2];
    [project_encoder setBuffer:impl->feature_signs offset:0 atIndex:3];
    [project_encoder setBuffer:impl->projected_train offset:0 atIndex:4];
    [project_encoder setBuffer:project_parameter_buffer offset:0 atIndex:5];
    [project_encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(train_rows), 1, 1)
                         threadsPerThreadgroup:
                           MTLSizeMake(static_cast<NSUInteger>(projected_dimensions), 1, 1)];
    [project_encoder endEncoding];

    id<MTLComputeCommandEncoder> gather_encoder = [command computeCommandEncoder];
    [gather_encoder setComputePipelineState:state.gather_centroids_pipeline];
    [gather_encoder setBuffer:impl->projected_train offset:0 atIndex:0];
    [gather_encoder setBuffer:initial_index_buffer offset:0 atIndex:1];
    [gather_encoder setBuffer:impl->centroids offset:0 atIndex:2];
    [gather_encoder setBuffer:kmeans_parameter_buffer offset:0 atIndex:3];
    [gather_encoder dispatchThreads:MTLSizeMake(centroid_items, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [gather_encoder endEncoding];

    for (int iteration = 0; iteration < 4; ++iteration) {
      const std::uint32_t iteration_value = static_cast<std::uint32_t>(iteration);
      id<MTLComputeCommandEncoder> changed_encoder = [command computeCommandEncoder];
      [changed_encoder setComputePipelineState:state.clear_changed_pipeline];
      [changed_encoder setBuffer:changed offset:0 atIndex:0];
      [changed_encoder dispatchThreads:MTLSizeMake(1, 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [changed_encoder endEncoding];

      id<MTLComputeCommandEncoder> assignment_encoder = [command computeCommandEncoder];
      [assignment_encoder setComputePipelineState:state.assign_kmeans_pipeline];
      [assignment_encoder setBuffer:impl->projected_train offset:0 atIndex:0];
      [assignment_encoder setBuffer:impl->centroids offset:0 atIndex:1];
      [assignment_encoder setBuffer:assignments offset:0 atIndex:2];
      [assignment_encoder setBuffer:changed offset:0 atIndex:3];
      [assignment_encoder setBuffer:kmeans_parameter_buffer offset:0 atIndex:4];
      [assignment_encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(train_rows), 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [assignment_encoder endEncoding];

      if (iteration < 3) {
        id<MTLComputeCommandEncoder> clear_encoder = [command computeCommandEncoder];
        [clear_encoder setComputePipelineState:state.clear_kmeans_pipeline];
        [clear_encoder setBuffer:sums offset:0 atIndex:0];
        [clear_encoder setBuffer:counts offset:0 atIndex:1];
        [clear_encoder setBuffer:kmeans_parameter_buffer offset:0 atIndex:2];
        [clear_encoder dispatchThreads:
          MTLSizeMake(std::max<std::size_t>(centroid_items, nlist), 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [clear_encoder endEncoding];

        id<MTLComputeCommandEncoder> accumulate_encoder = [command computeCommandEncoder];
        [accumulate_encoder setComputePipelineState:state.accumulate_kmeans_pipeline];
        [accumulate_encoder setBuffer:impl->projected_train offset:0 atIndex:0];
        [accumulate_encoder setBuffer:assignments offset:0 atIndex:1];
        [accumulate_encoder setBuffer:sums offset:0 atIndex:2];
        [accumulate_encoder setBuffer:counts offset:0 atIndex:3];
        [accumulate_encoder setBuffer:kmeans_parameter_buffer offset:0 atIndex:4];
        [accumulate_encoder dispatchThreads:
          MTLSizeMake(
            static_cast<std::size_t>(train_rows) *
            static_cast<std::size_t>(projected_dimensions),
            1,
            1
          )
                         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [accumulate_encoder endEncoding];

        id<MTLComputeCommandEncoder> finalize_encoder = [command computeCommandEncoder];
        [finalize_encoder setComputePipelineState:state.finalize_kmeans_pipeline];
        [finalize_encoder setBuffer:impl->projected_train offset:0 atIndex:0];
        [finalize_encoder setBuffer:sums offset:0 atIndex:1];
        [finalize_encoder setBuffer:counts offset:0 atIndex:2];
        [finalize_encoder setBuffer:impl->centroids offset:0 atIndex:3];
        [finalize_encoder setBuffer:initial_index_buffer offset:0 atIndex:4];
        [finalize_encoder setBuffer:kmeans_parameter_buffer offset:0 atIndex:5];
        [finalize_encoder setBytes:&iteration_value length:sizeof(iteration_value) atIndex:6];
        [finalize_encoder dispatchThreads:MTLSizeMake(centroid_items, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [finalize_encoder endEncoding];
      }
    }
    wait_for_command(command, "Resident Metal IVF training failed");

    const MetalIVFLists lists = build_metal_ivf_lists(
      state,
      assignments,
      kmeans_parameter_buffer,
      train_rows,
      nlist
    );
    impl->list_offsets = lists.offsets;
    impl->list_ids = lists.ids;
    return NativeMetalIVFIndex(std::move(impl));
  }
}

NativeKNNResult metal_ivf_index_search(
  const NativeMetalIVFIndex& index,
  const std::vector<float>& query,
  int query_rows,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices,
  MetalIVFStats* stats
) {
  if (!index.valid()) throw std::invalid_argument("Metal IVF index is empty.");
  if (query_rows < 1) {
    throw std::invalid_argument("Metal IVF search requires at least one query.");
  }
  const NativeMetalIVFIndex::Impl& impl = *index.impl_;
  if (query.size() !=
        static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(impl.dimensions)) {
    throw std::invalid_argument("Metal IVF query matrix size mismatch.");
  }
  if (!query_train_indices.empty() &&
      static_cast<int>(query_train_indices.size()) != query_rows) {
    throw std::invalid_argument("Metal IVF exclusion vector size mismatch.");
  }

  @autoreleasepool {
    MetalState& state = metal_state();
    id<MTLBuffer> query_buffer = [state.device
      newBufferWithBytes:query.data()
      length:query.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> projected_query = [state.device
      newBufferWithLength:
        static_cast<std::size_t>(query_rows) *
        static_cast<std::size_t>(impl.projected_dimensions) *
        sizeof(float)
      options:MTLResourceStorageModePrivate];
    const ProjectParamsHost parameters{
      static_cast<std::uint32_t>(query_rows),
      static_cast<std::uint32_t>(impl.dimensions),
      static_cast<std::uint32_t>(impl.projected_dimensions)
    };
    id<MTLBuffer> parameter_buffer = [state.device
      newBufferWithBytes:&parameters
      length:sizeof(parameters)
      options:MTLResourceStorageModeShared];
    if (query_buffer == nil || projected_query == nil || parameter_buffer == nil) {
      throw std::runtime_error("Failed to allocate resident Metal IVF query buffers.");
    }
    if (query_rows > 0) {
      id<MTLCommandBuffer> command = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:state.project_pipeline];
      [encoder setBuffer:query_buffer offset:0 atIndex:0];
      [encoder setBuffer:impl.feature_offsets offset:0 atIndex:1];
      [encoder setBuffer:impl.feature_ids offset:0 atIndex:2];
      [encoder setBuffer:impl.feature_signs offset:0 atIndex:3];
      [encoder setBuffer:projected_query offset:0 atIndex:4];
      [encoder setBuffer:parameter_buffer offset:0 atIndex:5];
      [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(query_rows), 1, 1)
                   threadsPerThreadgroup:
                     MTLSizeMake(static_cast<NSUInteger>(impl.projected_dimensions), 1, 1)];
      [encoder endEncoding];
      wait_for_command(command, "Resident Metal IVF query projection failed");
    }
    return search_metal_ivf_buffers(
      impl.train,
      query_buffer,
      projected_query,
      impl.centroids,
      impl.list_offsets,
      impl.list_ids,
      impl.rows,
      query_rows,
      impl.dimensions,
      impl.projected_dimensions,
      impl.nlist,
      k,
      impl.metric,
      requested_nprobe,
      target_recall,
      query_train_indices,
      stats
    );
  }
}

NativeKNNResult metal_ivf_index_filtered_search(
  const NativeMetalIVFIndex& index,
  const std::vector<float>& query,
  int query_rows,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices,
  const std::vector<int>& allowed_local_ids,
  MetalIVFStats* stats
) {
  if (!index.valid()) throw std::invalid_argument("Metal IVF index is empty.");
  const NativeMetalIVFIndex::Impl& impl = *index.impl_;
  if (query_rows < 1 ||
      query.size() != static_cast<std::size_t>(query_rows) * impl.dimensions) {
    throw std::invalid_argument("Metal filtered IVF query matrix size mismatch.");
  }
  if (static_cast<int>(query_train_indices.size()) != query_rows ||
      static_cast<int>(allowed_local_ids.size()) != impl.rows) {
    throw std::invalid_argument("Metal filtered IVF landmark mapping size mismatch.");
  }
  @autoreleasepool {
    MetalState& state = metal_state();
    id<MTLBuffer> query_buffer = [state.device
      newBufferWithBytes:query.data()
      length:query.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> projected_query = [state.device
      newBufferWithLength:static_cast<std::size_t>(query_rows) *
        static_cast<std::size_t>(impl.projected_dimensions) * sizeof(float)
      options:MTLResourceStorageModePrivate];
    const ProjectParamsHost parameters{
      static_cast<std::uint32_t>(query_rows),
      static_cast<std::uint32_t>(impl.dimensions),
      static_cast<std::uint32_t>(impl.projected_dimensions)
    };
    id<MTLBuffer> parameter_buffer = [state.device
      newBufferWithBytes:&parameters
      length:sizeof(parameters)
      options:MTLResourceStorageModeShared];
    if (query_buffer == nil || projected_query == nil || parameter_buffer == nil) {
      throw std::runtime_error("Failed to allocate filtered Metal IVF query buffers.");
    }
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state.project_pipeline];
    [encoder setBuffer:query_buffer offset:0 atIndex:0];
    [encoder setBuffer:impl.feature_offsets offset:0 atIndex:1];
    [encoder setBuffer:impl.feature_ids offset:0 atIndex:2];
    [encoder setBuffer:impl.feature_signs offset:0 atIndex:3];
    [encoder setBuffer:projected_query offset:0 atIndex:4];
    [encoder setBuffer:parameter_buffer offset:0 atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(query_rows), 1, 1)
                 threadsPerThreadgroup:
                   MTLSizeMake(static_cast<NSUInteger>(impl.projected_dimensions), 1, 1)];
    [encoder endEncoding];
    wait_for_command(command, "Filtered Metal IVF query projection failed");
    return search_metal_ivf_buffers(
      impl.train, query_buffer, projected_query, impl.centroids,
      impl.list_offsets, impl.list_ids, impl.rows, query_rows,
      impl.dimensions, impl.projected_dimensions, impl.nlist, k,
      impl.metric, requested_nprobe, target_recall, query_train_indices,
      stats, nil, nil, true, &allowed_local_ids
    );
  }
}

NativeKNNResult metal_ivf_index_self_search(
  const NativeMetalIVFIndex& index,
  int k,
  int requested_nprobe,
  double target_recall,
  const std::vector<int>& query_train_indices,
  MetalIVFStats* stats
) {
  if (!index.valid()) throw std::invalid_argument("Metal IVF index is empty.");
  const NativeMetalIVFIndex::Impl& impl = *index.impl_;
  if (!query_train_indices.empty() &&
      static_cast<int>(query_train_indices.size()) != impl.rows) {
    throw std::invalid_argument("Metal IVF exclusion vector size mismatch.");
  }
  return search_metal_ivf_buffers(
    impl.train,
    impl.train,
    impl.projected_train,
    impl.centroids,
    impl.list_offsets,
    impl.list_ids,
    impl.rows,
    impl.rows,
    impl.dimensions,
    impl.projected_dimensions,
    impl.nlist,
    k,
    impl.metric,
    requested_nprobe,
    target_recall,
    query_train_indices,
    stats
  );
}

NativeMetalKMeansContext metal_build_kmeans_context(
  const std::vector<float>& data,
  int rows,
  int dimensions,
  int lanes,
  int max_clusters
) {
  if (rows < 1 || dimensions < 1 || lanes < 1 || max_clusters < 1 ||
      max_clusters > rows ||
      data.size() != static_cast<std::size_t>(rows) * dimensions) {
    throw std::invalid_argument("Invalid Metal k-means context dimensions.");
  }
  @autoreleasepool {
    MetalState& state = metal_state();
    auto impl = std::make_unique<NativeMetalKMeansContext::Impl>();
    impl->rows = rows;
    impl->dimensions = dimensions;
    impl->max_clusters = max_clusters;
    impl->data = [state.device
      newBufferWithBytes:data.data()
      length:data.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    if (impl->data == nil) throw std::runtime_error("Failed to allocate resident Metal k-means input.");
    impl->input_uploads = 1;

    const std::size_t centroid_items =
      static_cast<std::size_t>(max_clusters) * dimensions;
    impl->lane.resize(static_cast<std::size_t>(lanes));
    for (auto& workspace : impl->lane) {
      workspace.centroids = [state.device
        newBufferWithLength:centroid_items * sizeof(float)
        options:MTLResourceStorageModeShared];
      workspace.assignments = [state.device
        newBufferWithLength:static_cast<std::size_t>(rows) * sizeof(int)
        options:MTLResourceStorageModeShared];
      workspace.sums = [state.device
        newBufferWithLength:centroid_items * sizeof(float)
        options:MTLResourceStorageModePrivate];
      workspace.counts = [state.device
        newBufferWithLength:static_cast<std::size_t>(max_clusters) * sizeof(std::uint32_t)
        options:MTLResourceStorageModePrivate];
      workspace.changed = [state.device
        newBufferWithLength:sizeof(std::uint32_t)
        options:MTLResourceStorageModeShared];
      workspace.initial_indices = [state.device
        newBufferWithLength:static_cast<std::size_t>(rows) * sizeof(int)
        options:MTLResourceStorageModeShared];
      workspace.parameters = [state.device
        newBufferWithLength:sizeof(KMeansParamsHost)
        options:MTLResourceStorageModeShared];
      if (workspace.centroids == nil || workspace.assignments == nil ||
          workspace.sums == nil || workspace.counts == nil || workspace.changed == nil ||
          workspace.initial_indices == nil || workspace.parameters == nil) {
        throw std::runtime_error("Failed to allocate resident Metal k-means workspace.");
      }
    }
    return NativeMetalKMeansContext(std::move(impl));
  }
}

std::vector<int> metal_kmeans_context_labels(
  NativeMetalKMeansContext& context,
  int lane,
  int clusters,
  const std::vector<int>& initial_point_indices,
  int max_iterations
) {
  if (!context.valid()) throw std::invalid_argument("Metal k-means context is empty.");
  NativeMetalKMeansContext::Impl& impl = *context.impl_;
  if (lane < 0 || lane >= static_cast<int>(impl.lane.size()) ||
      clusters < 1 || clusters > impl.max_clusters || max_iterations < 1 ||
      static_cast<int>(initial_point_indices.size()) < clusters) {
    throw std::invalid_argument("Invalid Metal k-means context request.");
  }
  for (int cluster = 0; cluster < clusters; ++cluster) {
    const int row = initial_point_indices[static_cast<std::size_t>(cluster)];
    if (row < 0 || row >= impl.rows) {
      throw std::invalid_argument("Metal k-means initialization index is out of range.");
    }
  }

  @autoreleasepool {
    MetalState& state = metal_state();
    auto& workspace = impl.lane[static_cast<std::size_t>(lane)];
    const std::size_t centroid_items =
      static_cast<std::size_t>(clusters) * impl.dimensions;
    std::memcpy(
      [workspace.initial_indices contents], initial_point_indices.data(),
      initial_point_indices.size() * sizeof(int)
    );
    std::memset(
      [workspace.assignments contents], 0xff,
      static_cast<std::size_t>(impl.rows) * sizeof(int)
    );
    const KMeansParamsHost parameters{
      static_cast<std::uint32_t>(impl.rows),
      static_cast<std::uint32_t>(impl.dimensions),
      static_cast<std::uint32_t>(clusters)
    };
    std::memcpy([workspace.parameters contents], &parameters, sizeof(parameters));

    id<MTLCommandBuffer> gather_command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> gather_encoder = [gather_command computeCommandEncoder];
    [gather_encoder setComputePipelineState:state.gather_centroids_pipeline];
    [gather_encoder setBuffer:impl.data offset:0 atIndex:0];
    [gather_encoder setBuffer:workspace.initial_indices offset:0 atIndex:1];
    [gather_encoder setBuffer:workspace.centroids offset:0 atIndex:2];
    [gather_encoder setBuffer:workspace.parameters offset:0 atIndex:3];
    [gather_encoder dispatchThreads:MTLSizeMake(centroid_items, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [gather_encoder endEncoding];
    wait_for_command(gather_command, "Resident Metal k-means initialization failed");

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
      const std::uint32_t iteration_value = static_cast<std::uint32_t>(iteration);
      id<MTLCommandBuffer> command = [state.queue commandBuffer];

      id<MTLComputeCommandEncoder> changed_encoder = [command computeCommandEncoder];
      [changed_encoder setComputePipelineState:state.clear_changed_pipeline];
      [changed_encoder setBuffer:workspace.changed offset:0 atIndex:0];
      [changed_encoder dispatchThreads:MTLSizeMake(1, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [changed_encoder endEncoding];

      id<MTLComputeCommandEncoder> assignment_encoder = [command computeCommandEncoder];
      [assignment_encoder setComputePipelineState:state.assign_kmeans_pipeline];
      [assignment_encoder setBuffer:impl.data offset:0 atIndex:0];
      [assignment_encoder setBuffer:workspace.centroids offset:0 atIndex:1];
      [assignment_encoder setBuffer:workspace.assignments offset:0 atIndex:2];
      [assignment_encoder setBuffer:workspace.changed offset:0 atIndex:3];
      [assignment_encoder setBuffer:workspace.parameters offset:0 atIndex:4];
      [assignment_encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(impl.rows), 1, 1)
                                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [assignment_encoder endEncoding];

      id<MTLComputeCommandEncoder> clear_encoder = [command computeCommandEncoder];
      [clear_encoder setComputePipelineState:state.clear_kmeans_pipeline];
      [clear_encoder setBuffer:workspace.sums offset:0 atIndex:0];
      [clear_encoder setBuffer:workspace.counts offset:0 atIndex:1];
      [clear_encoder setBuffer:workspace.parameters offset:0 atIndex:2];
      [clear_encoder dispatchThreads:MTLSizeMake(std::max<std::size_t>(centroid_items, clusters), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [clear_encoder endEncoding];

      id<MTLComputeCommandEncoder> accumulate_encoder = [command computeCommandEncoder];
      [accumulate_encoder setComputePipelineState:state.accumulate_kmeans_pipeline];
      [accumulate_encoder setBuffer:impl.data offset:0 atIndex:0];
      [accumulate_encoder setBuffer:workspace.assignments offset:0 atIndex:1];
      [accumulate_encoder setBuffer:workspace.sums offset:0 atIndex:2];
      [accumulate_encoder setBuffer:workspace.counts offset:0 atIndex:3];
      [accumulate_encoder setBuffer:workspace.parameters offset:0 atIndex:4];
      [accumulate_encoder dispatchThreads:MTLSizeMake(
          static_cast<std::size_t>(impl.rows) * impl.dimensions, 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [accumulate_encoder endEncoding];

      id<MTLComputeCommandEncoder> finalize_encoder = [command computeCommandEncoder];
      [finalize_encoder setComputePipelineState:state.finalize_kmeans_pipeline];
      [finalize_encoder setBuffer:impl.data offset:0 atIndex:0];
      [finalize_encoder setBuffer:workspace.sums offset:0 atIndex:1];
      [finalize_encoder setBuffer:workspace.counts offset:0 atIndex:2];
      [finalize_encoder setBuffer:workspace.centroids offset:0 atIndex:3];
      [finalize_encoder setBuffer:workspace.initial_indices offset:0 atIndex:4];
      [finalize_encoder setBuffer:workspace.parameters offset:0 atIndex:5];
      [finalize_encoder setBytes:&iteration_value length:sizeof(iteration_value) atIndex:6];
      [finalize_encoder dispatchThreads:MTLSizeMake(centroid_items, 1, 1)
                              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [finalize_encoder endEncoding];

      wait_for_command(command, "Resident Metal k-means command failed");
      const auto changed = *static_cast<const std::uint32_t*>([workspace.changed contents]);
      if (changed == 0) break;
    }
    std::vector<int> labels(static_cast<std::size_t>(impl.rows), 0);
    std::memcpy(
      labels.data(), [workspace.assignments contents], labels.size() * sizeof(int)
    );
    for (int& label : labels) ++label;
    return labels;
  }
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
  bool transpose_right,
  bool reuse_left
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
    id<MTLCommandQueue> queue = metal_pls_worker_queue(state.device);
    const NSUInteger left_row_bytes = matrix_row_bytes(left_cols);
    const NSUInteger right_row_bytes = matrix_row_bytes(right_cols);
    const NSUInteger result_row_bytes = matrix_row_bytes(result_cols);
    MetalPLSResidentWorkspace& workspace = g_metal_pls_workspace;
    id<MTLBuffer> left_buffer = nil;
    if (reuse_left) {
      left_buffer = resident_matrix_buffer(
        state.device,
        left,
        left_rows,
        left_cols,
        left_row_bytes
      );
    } else {
      ensure_shared_buffer(
        state.device,
        workspace.transient_left_matrix,
        left_row_bytes * static_cast<std::size_t>(left_rows)
      );
      write_matrix_buffer(
        workspace.transient_left_matrix,
        left,
        left_rows,
        left_cols,
        left_row_bytes
      );
      left_buffer = workspace.transient_left_matrix;
    }
    ensure_shared_buffer(
      state.device,
      workspace.right_matrix,
      right_row_bytes * static_cast<std::size_t>(right_rows)
    );
    write_matrix_buffer(
      workspace.right_matrix,
      right,
      right_rows,
      right_cols,
      right_row_bytes
    );
    ensure_shared_buffer(
      state.device,
      workspace.result_matrix,
      result_row_bytes * static_cast<std::size_t>(result_rows)
    );
    id<MTLBuffer> right_buffer = workspace.right_matrix;
    id<MTLBuffer> result_buffer = workspace.result_matrix;
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
    id<MTLCommandBuffer> command = [queue commandBuffer];
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

namespace {

NSUInteger encode_metal_pls_projection(
  MetalState& state,
  MetalPLSResidentWorkspace& workspace,
  id<MTLCommandBuffer> command,
  const std::vector<float>& x,
  int rows,
  int predictors,
  const std::vector<float>& weights,
  int components
) {
  const NSUInteger x_row_bytes = matrix_row_bytes(predictors);
  const NSUInteger weight_row_bytes = matrix_row_bytes(components);
  const NSUInteger score_row_bytes = matrix_row_bytes(components);
  id<MTLBuffer> x_buffer = resident_matrix_buffer(
    state.device,
    x,
    rows,
    predictors,
    x_row_bytes
  );
  ensure_shared_buffer(
    state.device,
    workspace.right_matrix,
    weight_row_bytes * static_cast<std::size_t>(predictors)
  );
  write_matrix_buffer(
    workspace.right_matrix,
    weights,
    predictors,
    components,
    weight_row_bytes
  );
  ensure_shared_buffer(
    state.device,
    workspace.lda_scores,
    score_row_bytes * static_cast<std::size_t>(rows)
  );

  MPSMatrixDescriptor* x_descriptor = [MPSMatrixDescriptor
    matrixDescriptorWithRows:static_cast<NSUInteger>(rows)
    columns:static_cast<NSUInteger>(predictors)
    rowBytes:x_row_bytes
    dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor* weight_descriptor = [MPSMatrixDescriptor
    matrixDescriptorWithRows:static_cast<NSUInteger>(predictors)
    columns:static_cast<NSUInteger>(components)
    rowBytes:weight_row_bytes
    dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor* score_descriptor = [MPSMatrixDescriptor
    matrixDescriptorWithRows:static_cast<NSUInteger>(rows)
    columns:static_cast<NSUInteger>(components)
    rowBytes:score_row_bytes
    dataType:MPSDataTypeFloat32];
  MPSMatrix* x_matrix = [[MPSMatrix alloc] initWithBuffer:x_buffer descriptor:x_descriptor];
  MPSMatrix* weight_matrix = [[MPSMatrix alloc]
    initWithBuffer:workspace.right_matrix
    descriptor:weight_descriptor];
  MPSMatrix* score_matrix = [[MPSMatrix alloc]
    initWithBuffer:workspace.lda_scores
    descriptor:score_descriptor];
  MPSMatrixMultiplication* project = [[MPSMatrixMultiplication alloc]
    initWithDevice:state.device
    transposeLeft:false
    transposeRight:false
    resultRows:static_cast<NSUInteger>(rows)
    resultColumns:static_cast<NSUInteger>(components)
    interiorColumns:static_cast<NSUInteger>(predictors)
    alpha:1.0
    beta:0.0];
  [project encodeToCommandBuffer:command
                      leftMatrix:x_matrix
                     rightMatrix:weight_matrix
                    resultMatrix:score_matrix];
  return score_row_bytes;
}

}  // namespace

MetalPLSScoreStatistics metal_pls_score_statistics(
  const std::vector<float>& x,
  int rows,
  int predictors,
  const std::vector<float>& weights,
  int components,
  const std::vector<int>& encoded_labels,
  int classes
) {
  if (rows < 1 || predictors < 1 || components < 1 || classes < 1 ||
      x.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(predictors) ||
      weights.size() != static_cast<std::size_t>(predictors) * static_cast<std::size_t>(components) ||
      encoded_labels.size() != static_cast<std::size_t>(rows)) {
    throw std::invalid_argument("Invalid Metal PLS score-statistics dimensions.");
  }
  for (int label : encoded_labels) {
    if (label < 0 || label >= classes) {
      throw std::invalid_argument("Metal PLS score-statistics label is out of range.");
    }
  }

  @autoreleasepool {
    MetalState& state = metal_state();
    MetalPLSResidentWorkspace& workspace = g_metal_pls_workspace;
    id<MTLCommandQueue> queue = metal_pls_worker_queue(state.device);
    ensure_shared_buffer(state.device, workspace.labels, encoded_labels.size() * sizeof(int));
    std::memcpy([workspace.labels contents], encoded_labels.data(), encoded_labels.size() * sizeof(int));
    const std::size_t class_items = static_cast<std::size_t>(classes) * static_cast<std::size_t>(components);
    const std::size_t cross_items = static_cast<std::size_t>(components) * static_cast<std::size_t>(components);
    const NSUInteger cross_row_bytes = matrix_row_bytes(components);
    ensure_shared_buffer(state.device, workspace.lda_class_sums, class_items * sizeof(float));
    ensure_shared_buffer(
      state.device,
      workspace.lda_score_crossprod,
      cross_row_bytes * static_cast<std::size_t>(components)
    );

    id<MTLCommandBuffer> command = [queue commandBuffer];
    const NSUInteger score_row_bytes = encode_metal_pls_projection(
      state, workspace, command, x, rows, predictors, weights, components
    );
    const PLSLDAParamsHost parameters{
      static_cast<std::uint32_t>(rows),
      static_cast<std::uint32_t>(components),
      static_cast<std::uint32_t>(score_row_bytes / sizeof(float)),
      static_cast<std::uint32_t>(classes)
    };

    id<MTLComputeCommandEncoder> sums_encoder = [command computeCommandEncoder];
    [sums_encoder setComputePipelineState:state.pls_class_sums_pipeline];
    [sums_encoder setBuffer:workspace.lda_scores offset:0 atIndex:0];
    [sums_encoder setBuffer:workspace.labels offset:0 atIndex:1];
    [sums_encoder setBuffer:workspace.lda_class_sums offset:0 atIndex:2];
    [sums_encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [sums_encoder dispatchThreads:MTLSizeMake(class_items, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<std::size_t>(256, class_items), 1, 1)];
    [sums_encoder endEncoding];

    MPSMatrixDescriptor* score_descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:static_cast<NSUInteger>(rows)
      columns:static_cast<NSUInteger>(components)
      rowBytes:score_row_bytes
      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* cross_descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:static_cast<NSUInteger>(components)
      columns:static_cast<NSUInteger>(components)
      rowBytes:cross_row_bytes
      dataType:MPSDataTypeFloat32];
    MPSMatrix* score_matrix = [[MPSMatrix alloc]
      initWithBuffer:workspace.lda_scores descriptor:score_descriptor];
    MPSMatrix* cross_matrix = [[MPSMatrix alloc]
      initWithBuffer:workspace.lda_score_crossprod descriptor:cross_descriptor];
    MPSMatrixMultiplication* score_gram = [[MPSMatrixMultiplication alloc]
      initWithDevice:state.device
      transposeLeft:true
      transposeRight:false
      resultRows:static_cast<NSUInteger>(components)
      resultColumns:static_cast<NSUInteger>(components)
      interiorColumns:static_cast<NSUInteger>(rows)
      alpha:1.0
      beta:0.0];
    [score_gram encodeToCommandBuffer:command
                           leftMatrix:score_matrix
                          rightMatrix:score_matrix
                         resultMatrix:cross_matrix];
    wait_for_command(command, "Metal PLS-LDA sufficient-statistics reduction failed");

    MetalPLSScoreStatistics output;
    output.classes = classes;
    output.components = components;
    output.class_sums.resize(class_items);
    output.score_crossprod.resize(cross_items);
    std::memcpy(output.class_sums.data(), [workspace.lda_class_sums contents], class_items * sizeof(float));
    const char* cross_base = static_cast<const char*>([workspace.lda_score_crossprod contents]);
    for (int row = 0; row < components; ++row) {
      std::memcpy(
        output.score_crossprod.data() +
          static_cast<std::size_t>(row) * static_cast<std::size_t>(components),
        cross_base + static_cast<NSUInteger>(row) * cross_row_bytes,
        static_cast<std::size_t>(components) * sizeof(float)
      );
    }
    return output;
  }
}

std::vector<int> metal_pls_lda_predict(
  const std::vector<float>& x,
  int rows,
  int predictors,
  const std::vector<float>& weights,
  int components,
  const std::vector<float>& linear,
  const std::vector<float>& constants,
  const std::vector<int>& class_labels
) {
  const int classes = static_cast<int>(class_labels.size());
  if (rows < 1 || predictors < 1 || components < 1 || classes < 1 ||
      x.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(predictors) ||
      weights.size() != static_cast<std::size_t>(predictors) * static_cast<std::size_t>(components) ||
      linear.size() != static_cast<std::size_t>(classes) * static_cast<std::size_t>(components) ||
      constants.size() != static_cast<std::size_t>(classes)) {
    throw std::invalid_argument("Invalid Metal PLS-LDA prediction dimensions.");
  }

  @autoreleasepool {
    MetalState& state = metal_state();
    MetalPLSResidentWorkspace& workspace = g_metal_pls_workspace;
    id<MTLCommandQueue> queue = metal_pls_worker_queue(state.device);
    ensure_shared_buffer(state.device, workspace.lda_linear, linear.size() * sizeof(float));
    ensure_shared_buffer(state.device, workspace.lda_constants, constants.size() * sizeof(float));
    ensure_shared_buffer(state.device, workspace.lda_class_labels, class_labels.size() * sizeof(int));
    ensure_shared_buffer(state.device, workspace.lda_predictions, static_cast<std::size_t>(rows) * sizeof(int));
    std::memcpy([workspace.lda_linear contents], linear.data(), linear.size() * sizeof(float));
    std::memcpy([workspace.lda_constants contents], constants.data(), constants.size() * sizeof(float));
    std::memcpy([workspace.lda_class_labels contents], class_labels.data(), class_labels.size() * sizeof(int));

    id<MTLCommandBuffer> command = [queue commandBuffer];
    const NSUInteger score_row_bytes = encode_metal_pls_projection(
      state, workspace, command, x, rows, predictors, weights, components
    );
    const PLSLDAParamsHost parameters{
      static_cast<std::uint32_t>(rows),
      static_cast<std::uint32_t>(components),
      static_cast<std::uint32_t>(score_row_bytes / sizeof(float)),
      static_cast<std::uint32_t>(classes)
    };
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state.pls_lda_predict_pipeline];
    [encoder setBuffer:workspace.lda_scores offset:0 atIndex:0];
    [encoder setBuffer:workspace.lda_linear offset:0 atIndex:1];
    [encoder setBuffer:workspace.lda_constants offset:0 atIndex:2];
    [encoder setBuffer:workspace.lda_class_labels offset:0 atIndex:3];
    [encoder setBuffer:workspace.lda_predictions offset:0 atIndex:4];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min(256, rows), 1, 1)];
    [encoder endEncoding];
    wait_for_command(command, "Metal PLS-LDA prediction failed");
    std::vector<int> predictions(static_cast<std::size_t>(rows));
    std::memcpy(predictions.data(), [workspace.lda_predictions contents], predictions.size() * sizeof(int));
    return predictions;
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
  const std::vector<float> s0 = cross_product;

  @autoreleasepool {
    MetalState& state = metal_state();
    id<MTLCommandQueue> queue = metal_pls_worker_queue(state.device);
    MetalPLSResidentWorkspace& workspace = g_metal_pls_workspace;
    const NSUInteger x_row_bytes = matrix_row_bytes(predictors);
    const NSUInteger predictor_vector_row_bytes = matrix_row_bytes(1);
    const NSUInteger sample_vector_row_bytes = matrix_row_bytes(1);
    id<MTLBuffer> x_buffer = resident_matrix_buffer(
      state.device,
      x,
      rows,
      predictors,
      x_row_bytes
    );
    ensure_shared_buffer(
      state.device,
      workspace.simpls_weight,
      predictor_vector_row_bytes * static_cast<std::size_t>(predictors)
    );
    ensure_shared_buffer(
      state.device,
      workspace.simpls_score,
      sample_vector_row_bytes * static_cast<std::size_t>(rows)
    );
    ensure_shared_buffer(
      state.device,
      workspace.simpls_loading,
      predictor_vector_row_bytes * static_cast<std::size_t>(predictors)
    );
    ensure_shared_buffer(
      state.device,
      workspace.cross_product,
      cross_product.size() * sizeof(float)
    );
    std::memcpy(
      [workspace.cross_product contents],
      cross_product.data(),
      cross_product.size() * sizeof(float)
    );
    id<MTLBuffer> weight_buffer = workspace.simpls_weight;
    id<MTLBuffer> score_buffer = workspace.simpls_score;
    id<MTLBuffer> loading_buffer = workspace.simpls_loading;

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
      id<MTLCommandBuffer> command = [queue commandBuffer];
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

    std::mt19937 generator(1u);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> previous_weight;
    int fitted_components = 0;
    for (int component = 0; component < components; ++component) {
      std::vector<float> weight(static_cast<std::size_t>(predictors), 0.0f);
      if (previous_weight.empty()) {
        for (float& value : weight) value = normal(generator);
      } else {
        weight = previous_weight;
      }

      for (int iteration = 0; iteration < 2; ++iteration) {
        const std::vector<float> response_projection =
          cross_product_transpose_times(s, predictors, responses, weight);
        weight = cross_product_times(s, predictors, responses, response_projection);
      }
      const float refresh_norm = power_vector_norm(weight);
      if (!std::isfinite(refresh_norm) || refresh_norm <= 1.0e-10f) break;
      for (float& value : weight) value /= refresh_norm;

      auto products = x_products(weight);
      const float score_norm = vector_norm(products.first);
      if (!std::isfinite(score_norm) || score_norm <= 1.0e-10f) break;
      const float inverse_score_norm = 1.0f / score_norm;
      for (float& value : weight) value *= inverse_score_norm;
      std::vector<float> loading = std::move(products.second);
      for (float& value : loading) value *= inverse_score_norm;

      const std::vector<float> response_weight =
        cross_product_transpose_times(s0, predictors, responses, weight);
      previous_weight = weight;
      remove_stored_columns(loading, loadings, predictors, components, component);
      const float loading_norm = vector_norm(loading);
      if (!std::isfinite(loading_norm) || loading_norm <= 1.0e-10f) break;
      for (float& value : loading) value /= loading_norm;

      for (int predictor = 0; predictor < predictors; ++predictor) {
        const std::size_t position = static_cast<std::size_t>(predictor) * static_cast<std::size_t>(components) +
          static_cast<std::size_t>(component);
        result.weights[position] = weight[static_cast<std::size_t>(predictor)];
        loadings[position] = loading[static_cast<std::size_t>(predictor)];
      }
      for (int response = 0; response < responses; ++response) {
        result.y_weights[static_cast<std::size_t>(response) * static_cast<std::size_t>(components) +
          static_cast<std::size_t>(component)] = response_weight[static_cast<std::size_t>(response)];
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
      fitted_components = component + 1;
    }

    if (fitted_components < 1) {
      throw std::runtime_error("fastPLS-compatible Metal float32 SIMPLS fit failed.");
    }
    if (fitted_components < components) {
      std::vector<float> fitted_weights(
        static_cast<std::size_t>(predictors) * static_cast<std::size_t>(fitted_components),
        0.0f
      );
      std::vector<float> fitted_y_weights(
        static_cast<std::size_t>(responses) * static_cast<std::size_t>(fitted_components),
        0.0f
      );
      for (int predictor = 0; predictor < predictors; ++predictor) {
        for (int component = 0; component < fitted_components; ++component) {
          fitted_weights[static_cast<std::size_t>(predictor) * static_cast<std::size_t>(fitted_components) +
            static_cast<std::size_t>(component)] =
              result.weights[static_cast<std::size_t>(predictor) * static_cast<std::size_t>(components) +
                static_cast<std::size_t>(component)];
        }
      }
      for (int response = 0; response < responses; ++response) {
        for (int component = 0; component < fitted_components; ++component) {
          fitted_y_weights[static_cast<std::size_t>(response) * static_cast<std::size_t>(fitted_components) +
            static_cast<std::size_t>(component)] =
              result.y_weights[static_cast<std::size_t>(response) * static_cast<std::size_t>(components) +
                static_cast<std::size_t>(component)];
        }
      }
      result.components = fitted_components;
      result.weights = std::move(fitted_weights);
      result.y_weights = std::move(fitted_y_weights);
    }
    ensure_shared_buffer(
      state.device,
      workspace.pls_weights,
      result.weights.size() * sizeof(float)
    );
    std::memcpy(
      [workspace.pls_weights contents],
      result.weights.data(),
      result.weights.size() * sizeof(float)
    );
  }
  return result;
}

namespace {

NSUInteger bounded_metal_threads(id<MTLComputePipelineState> pipeline) {
  return std::max<NSUInteger>(
    1,
    std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup)
  );
}

MTLSize metal_threadgroup_2d(id<MTLComputePipelineState> pipeline) {
  const NSUInteger maximum = pipeline.maxTotalThreadsPerThreadgroup;
  if (maximum >= 256) return MTLSizeMake(16, 16, 1);
  if (maximum >= 64) return MTLSizeMake(8, 8, 1);
  return MTLSizeMake(std::max<NSUInteger>(1, maximum), 1, 1);
}

int metal_tsne_fft_grid_size(int samples) {
  const int requested = samples >= 10000 ? 256 : 64;
  int grid = 32;
  while (grid < requested && grid < 512) grid <<= 1;
  return std::max(32, std::min(512, grid));
}

std::uint32_t log2_power_of_two(std::uint32_t value) {
  std::uint32_t result = 0;
  while (value > 1u) {
    value >>= 1u;
    ++result;
  }
  return result;
}

std::vector<float> make_fft_twiddles(
  std::uint32_t fft_size,
  std::uint32_t log_fft
) {
  const std::uint32_t half_count = fft_size >> 1u;
  std::vector<float> twiddles(
    static_cast<std::size_t>(log_fft) * half_count * 2u,
    0.0f
  );
  constexpr double two_pi = 6.283185307179586476925286766559;
  for (std::uint32_t stage = 1; stage <= log_fft; ++stage) {
    const std::uint32_t span_half = 1u << (stage - 1u);
    const std::uint32_t width = span_half << 1u;
    const std::size_t base =
      static_cast<std::size_t>(stage - 1u) * half_count * 2u;
    for (std::uint32_t j = 0; j < span_half; ++j) {
      const double angle = two_pi * static_cast<double>(j) /
        static_cast<double>(width);
      const std::size_t position = base + static_cast<std::size_t>(j) * 2u;
      twiddles[position] = static_cast<float>(std::cos(angle));
      twiddles[position + 1u] = static_cast<float>(std::sin(angle));
    }
  }
  return twiddles;
}

void encode_fft_512_stockham(
  MetalState& state,
  id<MTLCommandBuffer> command,
  id<MTLBuffer> values,
  id<MTLBuffer> scratch,
  bool inverse
) {
  const std::uint32_t inverse_value = inverse ? 1u : 0u;
  const MTLSize groups = MTLSizeMake(512, 1, 1);
  const MTLSize threads = MTLSizeMake(128, 1, 1);
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:state.opentsne_fft_512_rows_stockham_pipeline];
  [encoder setBuffer:values offset:0 atIndex:0];
  [encoder setBuffer:scratch offset:0 atIndex:1];
  [encoder setBytes:&inverse_value length:sizeof(inverse_value) atIndex:2];
  [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
  [encoder endEncoding];

  encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:state.opentsne_fft_512_cols_stockham_pipeline];
  [encoder setBuffer:scratch offset:0 atIndex:0];
  [encoder setBuffer:values offset:0 atIndex:1];
  [encoder setBytes:&inverse_value length:sizeof(inverse_value) atIndex:2];
  [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
  [encoder endEncoding];

  if (inverse) {
    const std::uint32_t total = 512u * 512u;
    const float scale = 1.0f / static_cast<float>(total);
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_scale_pipeline];
    [encoder setBuffer:values offset:0 atIndex:0];
    [encoder setBytes:&total length:sizeof(total) atIndex:1];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(
        bounded_metal_threads(state.opentsne_fft_scale_pipeline), 1, 1)];
    [encoder endEncoding];
  }
}

void encode_fft_2d(
  MetalState& state,
  id<MTLCommandBuffer> command,
  id<MTLBuffer> values,
  id<MTLBuffer> scratch,
  id<MTLBuffer> twiddles,
  std::uint32_t fft_size,
  std::uint32_t log_fft,
  bool inverse
) {
  if (fft_size == 512u) {
    encode_fft_512_stockham(state, command, values, scratch, inverse);
    return;
  }
  const std::uint32_t inverse_value = inverse ? 1u : 0u;
  const MTLSize full_grid = MTLSizeMake(fft_size, fft_size, 1);
  const MTLSize half_rows = MTLSizeMake(fft_size / 2u, fft_size, 1);
  const MTLSize half_columns = MTLSizeMake(fft_size, fft_size / 2u, 1);

  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:state.opentsne_fft_bit_reverse_rows_pipeline];
  [encoder setBuffer:values offset:0 atIndex:0];
  [encoder setBuffer:scratch offset:0 atIndex:1];
  [encoder setBytes:&fft_size length:sizeof(fft_size) atIndex:2];
  [encoder setBytes:&log_fft length:sizeof(log_fft) atIndex:3];
  [encoder dispatchThreads:full_grid
    threadsPerThreadgroup:metal_threadgroup_2d(
      state.opentsne_fft_bit_reverse_rows_pipeline)];
  [encoder endEncoding];

  for (std::uint32_t stage = 1; stage <= log_fft; ++stage) {
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_butterfly_rows_pipeline];
    [encoder setBuffer:scratch offset:0 atIndex:0];
    [encoder setBytes:&fft_size length:sizeof(fft_size) atIndex:1];
    [encoder setBytes:&stage length:sizeof(stage) atIndex:2];
    [encoder setBytes:&inverse_value length:sizeof(inverse_value) atIndex:3];
    [encoder setBuffer:twiddles offset:0 atIndex:4];
    [encoder dispatchThreads:half_rows
      threadsPerThreadgroup:metal_threadgroup_2d(
        state.opentsne_fft_butterfly_rows_pipeline)];
    [encoder endEncoding];
  }

  encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:state.opentsne_fft_bit_reverse_cols_pipeline];
  [encoder setBuffer:scratch offset:0 atIndex:0];
  [encoder setBuffer:values offset:0 atIndex:1];
  [encoder setBytes:&fft_size length:sizeof(fft_size) atIndex:2];
  [encoder setBytes:&log_fft length:sizeof(log_fft) atIndex:3];
  [encoder dispatchThreads:full_grid
    threadsPerThreadgroup:metal_threadgroup_2d(
      state.opentsne_fft_bit_reverse_cols_pipeline)];
  [encoder endEncoding];

  for (std::uint32_t stage = 1; stage <= log_fft; ++stage) {
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_butterfly_cols_pipeline];
    [encoder setBuffer:values offset:0 atIndex:0];
    [encoder setBytes:&fft_size length:sizeof(fft_size) atIndex:1];
    [encoder setBytes:&stage length:sizeof(stage) atIndex:2];
    [encoder setBytes:&inverse_value length:sizeof(inverse_value) atIndex:3];
    [encoder setBuffer:twiddles offset:0 atIndex:4];
    [encoder dispatchThreads:half_columns
      threadsPerThreadgroup:metal_threadgroup_2d(
        state.opentsne_fft_butterfly_cols_pipeline)];
    [encoder endEncoding];
  }

  if (inverse) {
    const std::uint32_t total = fft_size * fft_size;
    const float scale = 1.0f / static_cast<float>(total);
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state.opentsne_fft_scale_pipeline];
    [encoder setBuffer:values offset:0 atIndex:0];
    [encoder setBytes:&total length:sizeof(total) atIndex:1];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(
        bounded_metal_threads(state.opentsne_fft_scale_pipeline), 1, 1)];
    [encoder endEncoding];
  }
}

void encode_fft_convolution(
  MetalState& state,
  id<MTLCommandBuffer> command,
  id<MTLBuffer> transformed_mass,
  id<MTLBuffer> transformed_kernel,
  id<MTLBuffer> output,
  id<MTLBuffer> scratch,
  id<MTLBuffer> twiddles,
  std::uint32_t fft_size,
  std::uint32_t log_fft,
  std::uint32_t total
) {
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:state.opentsne_fft_multiply_pipeline];
  [encoder setBuffer:transformed_mass offset:0 atIndex:0];
  [encoder setBuffer:transformed_kernel offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:&total length:sizeof(total) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(
      bounded_metal_threads(state.opentsne_fft_multiply_pipeline), 1, 1)];
  [encoder endEncoding];
  encode_fft_2d(
    state, command, output, scratch, twiddles, fft_size, log_fft, true);
}

}  // namespace

std::vector<float> metal_umap_optimize(
  const std::vector<int>& neighbors,
  const std::vector<float>& weights,
  const std::vector<float>& initialization,
  int samples,
  int width,
  int epochs,
  int negative_sample_rate,
  float learning_rate,
  float a,
  float b,
  float max_weight,
  float repulsion_strength,
  std::uint32_t seed
) {
  if (samples < 2 || width < 1 || epochs < 0 || negative_sample_rate < 0 ||
      !std::isfinite(learning_rate) || learning_rate <= 0.0f ||
      !std::isfinite(a) || a <= 0.0f || !std::isfinite(b) || b <= 0.0f ||
      !std::isfinite(max_weight) || max_weight <= 0.0f ||
      !std::isfinite(repulsion_strength) || repulsion_strength <= 0.0f) {
    throw std::invalid_argument("Invalid Metal UMAP parameters.");
  }
  const std::size_t expected_graph_size =
    static_cast<std::size_t>(samples) * static_cast<std::size_t>(width);
  if (neighbors.size() != expected_graph_size ||
      weights.size() != expected_graph_size) {
    throw std::invalid_argument(
      "Metal UMAP rectangular graph arrays have inconsistent dimensions."
    );
  }
  if (initialization.size() != static_cast<std::size_t>(samples) * 2u) {
    throw std::invalid_argument("Metal UMAP requires a two-column initialization.");
  }
  for (std::size_t position = 0; position < neighbors.size(); ++position) {
    const int neighbor = neighbors[position];
    const float weight = weights[position];
    if (neighbor == -1 && weight == 0.0f) continue;
    if (neighbor < 0 || neighbor >= samples || !std::isfinite(weight) ||
        weight <= 0.0f) {
      throw std::invalid_argument("Metal UMAP received an invalid graph entry.");
    }
  }
  if (epochs == 0) return initialization;

  @autoreleasepool {
    MetalState& state = metal_state();
    std::vector<std::int32_t> fixed(initialization.size());
    for (std::size_t i = 0; i < initialization.size(); ++i) {
      if (!std::isfinite(initialization[i])) {
        throw std::invalid_argument("Metal UMAP initialization must be finite.");
      }
      const float scaled = std::max(
        -2140000000.0f,
        std::min(2140000000.0f, initialization[i] * 65536.0f)
      );
      fixed[i] = static_cast<std::int32_t>(scaled);
    }

    id<MTLBuffer> layout_buffer = [state.device
      newBufferWithBytes:fixed.data()
      length:fixed.size() * sizeof(std::int32_t)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> neighbor_buffer = [state.device
      newBufferWithBytes:neighbors.data()
      length:neighbors.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight_buffer = [state.device
      newBufferWithBytes:weights.data()
      length:weights.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    if (layout_buffer == nil || neighbor_buffer == nil || weight_buffer == nil) {
      throw std::runtime_error("Failed to allocate Metal UMAP buffers.");
    }

    const UMAPParamsHost parameters{
      static_cast<std::uint32_t>(samples),
      static_cast<std::uint32_t>(width),
      static_cast<std::uint32_t>(epochs),
      static_cast<std::uint32_t>(negative_sample_rate),
      seed,
      learning_rate,
      a,
      b,
      max_weight,
      repulsion_strength
    };
    const NSUInteger threads = std::min<NSUInteger>(
      256,
      state.umap_pipeline.maxTotalThreadsPerThreadgroup
    );
    constexpr int epochs_per_command = 64;
    for (int epoch_begin = 0; epoch_begin < epochs;
         epoch_begin += epochs_per_command) {
      id<MTLCommandBuffer> command = [state.queue commandBuffer];
      const int epoch_end = std::min(epochs, epoch_begin + epochs_per_command);
      for (int epoch = epoch_begin; epoch < epoch_end; ++epoch) {
        const std::uint32_t epoch_value = static_cast<std::uint32_t>(epoch);
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:state.umap_pipeline];
        [encoder setBuffer:layout_buffer offset:0 atIndex:0];
        [encoder setBuffer:neighbor_buffer offset:0 atIndex:1];
        [encoder setBuffer:weight_buffer offset:0 atIndex:2];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
        [encoder setBytes:&epoch_value length:sizeof(epoch_value) atIndex:4];
        [encoder
          dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(samples), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
      }
      wait_for_command(command, "Metal UMAP optimization failed");
    }

    std::memcpy(
      fixed.data(),
      [layout_buffer contents],
      fixed.size() * sizeof(std::int32_t)
    );
    std::vector<float> result(fixed.size());
    constexpr float inverse_scale = 1.0f / 65536.0f;
    for (std::size_t i = 0; i < fixed.size(); ++i) {
      result[i] = static_cast<float>(fixed[i]) * inverse_scale;
    }
    return result;
  }
}

std::vector<float> metal_opentsne_optimize(
  const std::vector<int>& row_ptr,
  const std::vector<int>& columns,
  const std::vector<float>& probabilities,
  const std::vector<float>& initialization,
  int samples,
  int early_exaggeration_iter,
  int normal_iter,
  float early_exaggeration,
  float exaggeration,
  float learning_rate,
  bool learning_rate_auto,
  float initial_momentum,
  float final_momentum,
  float min_gain,
  float max_step_norm,
  std::uint32_t seed
) {
  if (samples < 2 || early_exaggeration_iter < 0 || normal_iter < 0 ||
      early_exaggeration_iter + normal_iter < 1 ||
      !std::isfinite(early_exaggeration) || early_exaggeration <= 0.0f ||
      !std::isfinite(exaggeration) || exaggeration <= 0.0f ||
      (!learning_rate_auto &&
       (!std::isfinite(learning_rate) || learning_rate <= 0.0f)) ||
      !std::isfinite(initial_momentum) || initial_momentum < 0.0f ||
      !std::isfinite(final_momentum) || final_momentum < 0.0f ||
      !std::isfinite(min_gain) || min_gain <= 0.0f ||
      row_ptr.size() != static_cast<std::size_t>(samples) + 1u ||
      columns.empty() || columns.size() != probabilities.size() ||
      initialization.size() != static_cast<std::size_t>(samples) * 2u) {
    throw std::invalid_argument("Invalid Metal openTSNE inputs or parameters.");
  }
  if (row_ptr.front() != 0 || row_ptr.back() != static_cast<int>(columns.size())) {
    throw std::invalid_argument("Metal openTSNE CSR row pointers are inconsistent.");
  }
  for (int row = 0; row < samples; ++row) {
    if (row_ptr[static_cast<std::size_t>(row)] >
        row_ptr[static_cast<std::size_t>(row + 1)]) {
      throw std::invalid_argument("Metal openTSNE CSR row pointers are not monotone.");
    }
  }
  for (std::size_t edge = 0; edge < columns.size(); ++edge) {
    if (columns[edge] < 0 || columns[edge] >= samples ||
        !std::isfinite(probabilities[edge]) || probabilities[edge] < 0.0f) {
      throw std::invalid_argument("Metal openTSNE received an invalid sparse edge.");
    }
  }

  std::vector<float> centered = initialization;
  double mean_x = 0.0;
  double mean_y = 0.0;
  for (int row = 0; row < samples; ++row) {
    const float x = centered[static_cast<std::size_t>(row) * 2u];
    const float y = centered[static_cast<std::size_t>(row) * 2u + 1u];
    if (!std::isfinite(x) || !std::isfinite(y)) {
      throw std::invalid_argument("Metal openTSNE initialization must be finite.");
    }
    mean_x += x;
    mean_y += y;
  }
  mean_x /= static_cast<double>(samples);
  mean_y /= static_cast<double>(samples);
  for (int row = 0; row < samples; ++row) {
    centered[static_cast<std::size_t>(row) * 2u] -= static_cast<float>(mean_x);
    centered[static_cast<std::size_t>(row) * 2u + 1u] -= static_cast<float>(mean_y);
  }

  @autoreleasepool {
    MetalState& state = metal_state();
    const std::size_t embedding_bytes = centered.size() * sizeof(float);
    const std::uint32_t n = static_cast<std::uint32_t>(samples);
    const std::uint32_t grid_size =
      static_cast<std::uint32_t>(metal_tsne_fft_grid_size(samples));
    const std::uint32_t fft_size = grid_size << 1u;
    const std::uint32_t log_fft = log2_power_of_two(fft_size);
    const std::uint32_t grid_total = grid_size * grid_size;
    const std::uint32_t fft_total = fft_size * fft_size;
    const std::size_t grid_bytes =
      static_cast<std::size_t>(grid_total) * sizeof(float);
    const std::size_t complex_bytes =
      static_cast<std::size_t>(fft_total) * sizeof(float) * 2u;
    const std::vector<float> twiddles = make_fft_twiddles(fft_size, log_fft);

    std::vector<float> gains(centered.size(), 1.0f);
    std::vector<float> updates(centered.size(), 0.0f);
    id<MTLBuffer> row_ptr_buffer = [state.device
      newBufferWithBytes:row_ptr.data()
      length:row_ptr.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> column_buffer = [state.device
      newBufferWithBytes:columns.data()
      length:columns.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> probability_buffer = [state.device
      newBufferWithBytes:probabilities.data()
      length:probabilities.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> current_buffer = [state.device
      newBufferWithBytes:centered.data()
      length:embedding_bytes
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> next_buffer = [state.device
      newBufferWithLength:embedding_bytes
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> gains_buffer = [state.device
      newBufferWithBytes:gains.data()
      length:embedding_bytes
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> updates_buffer = [state.device
      newBufferWithBytes:updates.data()
      length:embedding_bytes
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> partial_sum_buffer = [state.device
      newBufferWithLength:((n + 255u) / 256u) * sizeof(float)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> inverse_sum_buffer = [state.device
      newBufferWithLength:sizeof(float)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> mass_buffer = [state.device
      newBufferWithLength:grid_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> mass_x_buffer = [state.device
      newBufferWithLength:grid_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> mass_y_buffer = [state.device
      newBufferWithLength:grid_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> mass_fft_buffer = [state.device
      newBufferWithLength:complex_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> mass_x_fft_buffer = [state.device
      newBufferWithLength:complex_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> mass_y_fft_buffer = [state.device
      newBufferWithLength:complex_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> kernel_q_buffer = [state.device
      newBufferWithLength:complex_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> kernel_q2_buffer = [state.device
      newBufferWithLength:complex_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> q_grid_buffer = [state.device
      newBufferWithLength:complex_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> q2_grid_buffer = [state.device
      newBufferWithLength:complex_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> xq2_grid_buffer = [state.device
      newBufferWithLength:complex_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> yq2_grid_buffer = [state.device
      newBufferWithLength:complex_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> fft_scratch_buffer = [state.device
      newBufferWithLength:complex_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> twiddle_buffer = [state.device
      newBufferWithBytes:twiddles.data()
      length:twiddles.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    constexpr std::uint32_t stats_block_size = 256u;
    const std::uint32_t stats_blocks = (n + stats_block_size - 1u) /
      stats_block_size;
    id<MTLBuffer> layout_stats_buffer = [state.device
      newBufferWithLength:static_cast<std::size_t>(stats_blocks) *
        sizeof(OpenTsneLayoutStatsHost)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> grid_parameters_buffer = [state.device
      newBufferWithLength:sizeof(OpenTsneFFTGridParamsHost)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> center_buffer = [state.device
      newBufferWithLength:sizeof(OpenTsneCenterHost)
      options:MTLResourceStorageModePrivate];
    if (row_ptr_buffer == nil || column_buffer == nil ||
        probability_buffer == nil || current_buffer == nil ||
        next_buffer == nil || gains_buffer == nil || updates_buffer == nil ||
        partial_sum_buffer == nil || inverse_sum_buffer == nil ||
        mass_buffer == nil || mass_x_buffer == nil || mass_y_buffer == nil ||
        mass_fft_buffer == nil || mass_x_fft_buffer == nil ||
        mass_y_fft_buffer == nil || kernel_q_buffer == nil ||
        kernel_q2_buffer == nil || q_grid_buffer == nil ||
        q2_grid_buffer == nil || xq2_grid_buffer == nil ||
        yq2_grid_buffer == nil || fft_scratch_buffer == nil ||
        twiddle_buffer == nil || layout_stats_buffer == nil ||
        grid_parameters_buffer == nil || center_buffer == nil) {
      throw std::runtime_error("Failed to allocate Metal openTSNE buffers.");
    }

    const MTLSize point_grid = MTLSizeMake(n, 1, 1);
    const MTLSize grid_cells = MTLSizeMake(grid_total, 1, 1);
    const MTLSize fft_grid = MTLSizeMake(fft_size, fft_size, 1);
    const MTLSize stats_grid = MTLSizeMake(stats_blocks, 1, 1);
    constexpr std::uint32_t sum_block_size = 256u;
    const std::uint32_t sum_blocks = (n + sum_block_size - 1u) /
      sum_block_size;
    const MTLSize sum_grid = MTLSizeMake(sum_blocks, 1, 1);
    const NSUInteger clear_threads =
      bounded_metal_threads(state.opentsne_fft_clear_pipeline);
    const NSUInteger scatter_threads =
      bounded_metal_threads(state.opentsne_fft_scatter_pipeline);
    const NSUInteger epoch_threads =
      bounded_metal_threads(state.opentsne_fft_epoch_pipeline);
    const NSUInteger center_threads =
      bounded_metal_threads(state.opentsne_center_pipeline);
    constexpr int iterations_per_command = 16;
    const int total_iterations = early_exaggeration_iter + normal_iter;

    for (int batch_start = 0; batch_start < total_iterations;
         batch_start += iterations_per_command) {
      id<MTLCommandBuffer> command = [state.queue commandBuffer];
      const int batch_end = std::min(
        total_iterations, batch_start + iterations_per_command);
      for (int iteration = batch_start; iteration < batch_end; ++iteration) {
        const bool early = iteration < early_exaggeration_iter;
        const float phase_exaggeration = early ? early_exaggeration : exaggeration;
        const float phase_learning_rate = learning_rate_auto ?
          static_cast<float>(samples) / phase_exaggeration : learning_rate;
        const OpenTsneMetalParamsHost parameters{
          n,
          seed,
          phase_learning_rate,
          phase_exaggeration,
          early ? initial_momentum : final_momentum,
          min_gain,
          std::isfinite(max_step_norm) ?
            max_step_norm : std::numeric_limits<float>::max(),
          1.0f
        };

        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:
          state.opentsne_fft_layout_stats_blocks_pipeline];
        [encoder setBuffer:current_buffer offset:0 atIndex:0];
        [encoder setBuffer:layout_stats_buffer offset:0 atIndex:1];
        [encoder setBytes:&n length:sizeof(n) atIndex:2];
        [encoder setBytes:&stats_block_size length:sizeof(stats_block_size) atIndex:3];
        [encoder dispatchThreads:stats_grid threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:
          state.opentsne_fft_finalize_layout_stats_pipeline];
        [encoder setBuffer:layout_stats_buffer offset:0 atIndex:0];
        [encoder setBuffer:grid_parameters_buffer offset:0 atIndex:1];
        [encoder setBuffer:center_buffer offset:0 atIndex:2];
        [encoder setBytes:&stats_blocks length:sizeof(stats_blocks) atIndex:3];
        [encoder setBytes:&n length:sizeof(n) atIndex:4];
        [encoder setBytes:&grid_size length:sizeof(grid_size) atIndex:5];
        [encoder setBytes:&fft_size length:sizeof(fft_size) atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:state.opentsne_fft_clear_pipeline];
        [encoder setBuffer:mass_buffer offset:0 atIndex:0];
        [encoder setBuffer:mass_x_buffer offset:0 atIndex:1];
        [encoder setBuffer:mass_y_buffer offset:0 atIndex:2];
        [encoder setBuffer:grid_parameters_buffer offset:0 atIndex:3];
        [encoder dispatchThreads:grid_cells
          threadsPerThreadgroup:MTLSizeMake(clear_threads, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:state.opentsne_fft_scatter_pipeline];
        [encoder setBuffer:current_buffer offset:0 atIndex:0];
        [encoder setBuffer:mass_buffer offset:0 atIndex:1];
        [encoder setBuffer:mass_x_buffer offset:0 atIndex:2];
        [encoder setBuffer:mass_y_buffer offset:0 atIndex:3];
        [encoder setBuffer:grid_parameters_buffer offset:0 atIndex:4];
        [encoder dispatchThreads:point_grid
          threadsPerThreadgroup:MTLSizeMake(scatter_threads, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:state.opentsne_fft_load_pipeline];
        [encoder setBuffer:mass_buffer offset:0 atIndex:0];
        [encoder setBuffer:mass_x_buffer offset:0 atIndex:1];
        [encoder setBuffer:mass_y_buffer offset:0 atIndex:2];
        [encoder setBuffer:mass_fft_buffer offset:0 atIndex:3];
        [encoder setBuffer:mass_x_fft_buffer offset:0 atIndex:4];
        [encoder setBuffer:mass_y_fft_buffer offset:0 atIndex:5];
        [encoder setBuffer:kernel_q_buffer offset:0 atIndex:6];
        [encoder setBuffer:kernel_q2_buffer offset:0 atIndex:7];
        [encoder setBuffer:grid_parameters_buffer offset:0 atIndex:8];
        [encoder dispatchThreads:fft_grid
          threadsPerThreadgroup:metal_threadgroup_2d(
            state.opentsne_fft_load_pipeline)];
        [encoder endEncoding];

        encode_fft_2d(state, command, mass_fft_buffer, fft_scratch_buffer,
          twiddle_buffer, fft_size, log_fft, false);
        encode_fft_2d(state, command, mass_x_fft_buffer, fft_scratch_buffer,
          twiddle_buffer, fft_size, log_fft, false);
        encode_fft_2d(state, command, mass_y_fft_buffer, fft_scratch_buffer,
          twiddle_buffer, fft_size, log_fft, false);
        encode_fft_2d(state, command, kernel_q_buffer, fft_scratch_buffer,
          twiddle_buffer, fft_size, log_fft, false);
        encode_fft_2d(state, command, kernel_q2_buffer, fft_scratch_buffer,
          twiddle_buffer, fft_size, log_fft, false);
        encode_fft_convolution(state, command, mass_fft_buffer, kernel_q_buffer,
          q_grid_buffer, fft_scratch_buffer, twiddle_buffer, fft_size,
          log_fft, fft_total);
        encode_fft_convolution(state, command, mass_fft_buffer, kernel_q2_buffer,
          q2_grid_buffer, fft_scratch_buffer, twiddle_buffer, fft_size,
          log_fft, fft_total);
        encode_fft_convolution(state, command, mass_x_fft_buffer, kernel_q2_buffer,
          xq2_grid_buffer, fft_scratch_buffer, twiddle_buffer, fft_size,
          log_fft, fft_total);
        encode_fft_convolution(state, command, mass_y_fft_buffer, kernel_q2_buffer,
          yq2_grid_buffer, fft_scratch_buffer, twiddle_buffer, fft_size,
          log_fft, fft_total);

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:state.opentsne_fft_sum_q_blocks_pipeline];
        [encoder setBuffer:current_buffer offset:0 atIndex:0];
        [encoder setBuffer:q_grid_buffer offset:0 atIndex:1];
        [encoder setBuffer:partial_sum_buffer offset:0 atIndex:2];
        [encoder setBuffer:grid_parameters_buffer offset:0 atIndex:3];
        [encoder setBytes:&sum_block_size length:sizeof(sum_block_size) atIndex:4];
        [encoder dispatchThreads:sum_grid threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:
          state.opentsne_fft_finalize_sum_q_pipeline];
        [encoder setBuffer:partial_sum_buffer offset:0 atIndex:0];
        [encoder setBuffer:inverse_sum_buffer offset:0 atIndex:1];
        [encoder setBytes:&sum_blocks length:sizeof(sum_blocks) atIndex:2];
        [encoder setBytes:&n length:sizeof(n) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:state.opentsne_fft_epoch_pipeline];
        [encoder setBuffer:row_ptr_buffer offset:0 atIndex:0];
        [encoder setBuffer:column_buffer offset:0 atIndex:1];
        [encoder setBuffer:probability_buffer offset:0 atIndex:2];
        [encoder setBuffer:current_buffer offset:0 atIndex:3];
        [encoder setBuffer:next_buffer offset:0 atIndex:4];
        [encoder setBuffer:gains_buffer offset:0 atIndex:5];
        [encoder setBuffer:updates_buffer offset:0 atIndex:6];
        [encoder setBuffer:q2_grid_buffer offset:0 atIndex:7];
        [encoder setBuffer:xq2_grid_buffer offset:0 atIndex:8];
        [encoder setBuffer:yq2_grid_buffer offset:0 atIndex:9];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:10];
        [encoder setBuffer:grid_parameters_buffer offset:0 atIndex:11];
        [encoder setBuffer:inverse_sum_buffer offset:0 atIndex:12];
        [encoder dispatchThreads:point_grid
          threadsPerThreadgroup:MTLSizeMake(epoch_threads, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:
          state.opentsne_fft_layout_stats_blocks_pipeline];
        [encoder setBuffer:next_buffer offset:0 atIndex:0];
        [encoder setBuffer:layout_stats_buffer offset:0 atIndex:1];
        [encoder setBytes:&n length:sizeof(n) atIndex:2];
        [encoder setBytes:&stats_block_size length:sizeof(stats_block_size) atIndex:3];
        [encoder dispatchThreads:stats_grid threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:
          state.opentsne_fft_finalize_layout_stats_pipeline];
        [encoder setBuffer:layout_stats_buffer offset:0 atIndex:0];
        [encoder setBuffer:grid_parameters_buffer offset:0 atIndex:1];
        [encoder setBuffer:center_buffer offset:0 atIndex:2];
        [encoder setBytes:&stats_blocks length:sizeof(stats_blocks) atIndex:3];
        [encoder setBytes:&n length:sizeof(n) atIndex:4];
        [encoder setBytes:&grid_size length:sizeof(grid_size) atIndex:5];
        [encoder setBytes:&fft_size length:sizeof(fft_size) atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:state.opentsne_center_pipeline];
        [encoder setBuffer:next_buffer offset:0 atIndex:0];
        [encoder setBuffer:center_buffer offset:0 atIndex:1];
        [encoder setBytes:&n length:sizeof(n) atIndex:2];
        [encoder dispatchThreads:point_grid
          threadsPerThreadgroup:MTLSizeMake(center_threads, 1, 1)];
        [encoder endEncoding];
        std::swap(current_buffer, next_buffer);
      }
      wait_for_command(command, "Metal openTSNE FFT-grid optimization failed");
    }

    std::vector<float> result(centered.size());
    std::memcpy(result.data(), [current_buffer contents], embedding_bytes);
    for (float value : result) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("Metal openTSNE produced a non-finite layout.");
      }
    }
    return result;
  }
}

}  // namespace kodama::detail
