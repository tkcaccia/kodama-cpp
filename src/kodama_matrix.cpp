#include "common.hpp"
#include "metal_backend.hpp"
#include "native_cuda_backend.hpp"
#include "native_knn.hpp"
#include "spatial_grid_knn.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef KODAMA_ENABLE_CUDA
#include "kodama_matrix_cuda.hpp"

#include <cuda_runtime.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kodama {
namespace {

class OmpThreadScope {
 public:
  explicit OmpThreadScope(int n_threads) {
#ifdef _OPENMP
    previous_ = omp_get_max_threads();
    if (n_threads > 0) omp_set_num_threads(std::max(1, n_threads));
#else
    (void)n_threads;
#endif
  }

  ~OmpThreadScope() {
#ifdef _OPENMP
    if (previous_ > 0) omp_set_num_threads(previous_);
#endif
  }

 private:
#ifdef _OPENMP
  int previous_ = 0;
#endif
};

std::vector<float> copy_float32(MatrixView x, const std::vector<int>& rows = std::vector<int>()) {
  const std::size_t n = rows.empty() ? x.rows : rows.size();
  std::vector<float> out(n * x.cols, 0.0f);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t src = rows.empty() ? i : static_cast<std::size_t>(rows[i]);
    for (std::size_t j = 0; j < x.cols; ++j) {
      out[i * x.cols + j] = x.value_float(src, j);
    }
  }
  return out;
}

std::vector<float> copy_float32_rows(const std::vector<float>& x, std::size_t cols, const std::vector<int>& rows) {
  std::vector<float> out(rows.size() * cols, 0.0f);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const std::size_t src = static_cast<std::size_t>(rows[i]) * cols;
    std::copy_n(x.data() + src, cols, out.data() + i * cols);
  }
  return out;
}

void copy_float32_rows_into(
  const std::vector<float>& x,
  std::size_t cols,
  const std::vector<int>& rows,
  std::vector<float>& out
) {
  out.resize(rows.size() * cols);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const std::size_t src = static_cast<std::size_t>(rows[i]) * cols;
    std::copy_n(x.data() + src, cols, out.data() + i * cols);
  }
}

void normalize_rows_for_cosine(std::vector<float>& x, std::size_t rows, std::size_t cols) {
  for (std::size_t i = 0; i < rows; ++i) {
    long double ss = 0.0;
    for (std::size_t j = 0; j < cols; ++j) {
      const float v = x[i * cols + j];
      ss += static_cast<long double>(v) * static_cast<long double>(v);
    }
    const double norm = std::sqrt(static_cast<double>(ss));
    if (norm <= 0.0 || !std::isfinite(norm)) continue;
    const float scale = static_cast<float>(1.0 / norm);
    for (std::size_t j = 0; j < cols; ++j) x[i * cols + j] *= scale;
  }
}

int majority_label(const std::vector<int>& values) {
  std::map<int, int> counts;
  for (int v : values) counts[v] += 1;
  int best_label = counts.begin()->first;
  int best_count = counts.begin()->second;
  for (const auto& kv : counts) {
    if (kv.second > best_count || (kv.second == best_count && kv.first < best_label)) {
      best_label = kv.first;
      best_count = kv.second;
    }
  }
  return best_label;
}

std::vector<int> normalize_constrain(const std::vector<int>& constrain, std::size_t n) {
  if (constrain.empty()) {
    std::vector<int> out(n);
    std::iota(out.begin(), out.end(), 1);
    return out;
  }
  if (constrain.size() != n) throw std::invalid_argument("constrain size must be zero or match number of rows.");
  std::map<int, int> ids;
  std::vector<int> out(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    auto it = ids.find(constrain[i]);
    if (it == ids.end()) {
      const int id = static_cast<int>(ids.size()) + 1;
      it = ids.emplace(constrain[i], id).first;
    }
    out[i] = it->second;
  }
  return out;
}

std::vector<int> normalize_fixed(const std::vector<int>& fixed, std::size_t n) {
  if (fixed.empty()) return std::vector<int>(n, 0);
  if (fixed.size() != n) throw std::invalid_argument("fixed size must be zero or match number of rows.");
  std::vector<int> out(n, 0);
  for (std::size_t i = 0; i < n; ++i) out[i] = fixed[i] != 0 ? 1 : 0;
  return out;
}

bool is_identity_constrain(const std::vector<int>& constrain) {
  for (std::size_t i = 0; i < constrain.size(); ++i) {
    if (constrain[i] != static_cast<int>(i + 1)) return false;
  }
  return true;
}

NeighborGraph hnsw_graph(
  const std::vector<float>& base,
  const std::vector<float>& query,
  int n_base,
  int n_query,
  int dim,
  int neighbors,
  DistanceMetric metric,
  int n_threads,
  bool self_search = false,
  bool include_self = true
) {
  if (n_base < 1 || n_query < 1 || dim < 1) throw std::invalid_argument("HNSW graph input is empty.");
  neighbors = std::max(1, std::min(neighbors, n_base));
  std::vector<float> xb = base;
  std::vector<float> xq = query;
  if (metric == DistanceMetric::Cosine) {
    normalize_rows_for_cosine(xb, static_cast<std::size_t>(n_base), static_cast<std::size_t>(dim));
    normalize_rows_for_cosine(xq, static_cast<std::size_t>(n_query), static_cast<std::size_t>(dim));
  }

  std::vector<int> query_train_indices;
  if (self_search && !include_self) {
    query_train_indices.resize(static_cast<std::size_t>(n_query));
    std::iota(query_train_indices.begin(), query_train_indices.end(), 0);
  }
  const int m = std::min(32, std::max(2, n_base > 1 ? n_base - 1 : 2));
  const detail::NativeKNNResult search = detail::native_hnsw_search(
    xb,
    n_base,
    xq,
    n_query,
    dim,
    neighbors,
    metric,
    detail::NativeHNSWParameters{m, std::max(200, m), std::max(150, neighbors)},
    n_threads,
    query_train_indices
  );

  NeighborGraph graph;
  graph.neighbors = search.neighbors;
  graph.indices = search.indices;
  graph.distances.resize(search.distances.size());
  for (std::size_t i = 0; i < search.distances.size(); ++i) {
    graph.distances[i] = detail::native_knn_output_distance(search.distances[i], metric);
  }
  return graph;
}

#ifdef KODAMA_ENABLE_CUDA
NeighborGraph native_cuda_flat_graph(
  const std::vector<float>& base,
  const std::vector<float>& query,
  int n_base,
  int n_query,
  int dim,
  int neighbors,
  DistanceMetric metric,
  int gpu_device,
  bool exclude_self
) {
  if (n_base < 1 || n_query < 1 || dim < 1) throw std::invalid_argument("CUDA graph input is empty.");
  neighbors = std::max(1, std::min(neighbors, n_base));
  std::vector<float> xb = base;
  std::vector<float> xq = query;
  if (metric == DistanceMetric::Cosine) {
    normalize_rows_for_cosine(xb, static_cast<std::size_t>(n_base), static_cast<std::size_t>(dim));
    normalize_rows_for_cosine(xq, static_cast<std::size_t>(n_query), static_cast<std::size_t>(dim));
  }

  std::vector<int> exclusions;
  if (exclude_self) {
    exclusions.resize(static_cast<std::size_t>(n_query));
    std::iota(exclusions.begin(), exclusions.end(), 0);
  }
  const detail::NativeKNNResult search = detail::native_cuda_exact_knn_search(
    xb,
    n_base,
    xq,
    n_query,
    dim,
    neighbors,
    metric,
    gpu_device,
    exclusions
  );
  NeighborGraph graph;
  graph.neighbors = search.neighbors;
  graph.indices = search.indices;
  graph.distances.resize(search.distances.size(), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < search.distances.size(); ++i) {
    graph.distances[i] = detail::native_knn_output_distance(search.distances[i], metric);
  }
  return graph;
}
#endif

NeighborGraph self_knn_graph(
  const std::vector<float>& data,
  int n,
  int dim,
  int neighbors,
  DistanceMetric metric,
  int n_threads,
  Backend backend,
  int gpu_device,
  bool include_self,
  KNNIndexType metal_index_type,
  int metal_ivf_nlist,
  int metal_ivf_nprobe
) {
  if (backend != Backend::Metal && detail::should_use_spatial_grid_knn(n, dim, metric)) {
#if defined(KODAMA_ENABLE_CUDA)
    if (backend == Backend::CUDA && neighbors <= 256) {
      return detail::spatial_grid_self_knn_cuda(data, n, dim, neighbors, gpu_device, false, include_self);
    }
#else
    (void)gpu_device;
#endif
    return detail::spatial_grid_self_knn(data.data(), n, dim, neighbors, n_threads, false, include_self);
  }
#if defined(KODAMA_ENABLE_CUDA)
  if (backend == Backend::CUDA) {
    return native_cuda_flat_graph(data, data, n, n, dim, neighbors, metric, gpu_device, !include_self);
  }
#else
  (void)gpu_device;
#endif
  if (backend == Backend::Metal) {
#if defined(KODAMA_ENABLE_METAL)
    std::vector<float> prepared = data;
    if (metric == DistanceMetric::Cosine) normalize_rows_for_cosine(prepared, n, dim);
    std::vector<int> exclusions;
    if (!include_self) {
      exclusions.resize(static_cast<std::size_t>(n));
      std::iota(exclusions.begin(), exclusions.end(), 0);
    }
    const detail::NativeKNNResult search = metal_index_type == KNNIndexType::MetalIVFFlat ?
      detail::metal_ivf_knn_search(
        prepared,
        n,
        prepared,
        n,
        dim,
        neighbors,
        metric,
        metal_ivf_nlist,
        metal_ivf_nprobe,
        exclusions
      ) :
      detail::metal_exact_knn_search(
        prepared,
        n,
        prepared,
        n,
        dim,
        neighbors,
        metric,
        exclusions
      );
    NeighborGraph graph;
    graph.neighbors = search.neighbors;
    graph.indices = search.indices;
    graph.distances.resize(search.distances.size(), std::numeric_limits<float>::infinity());
    for (std::size_t i = 0; i < search.distances.size(); ++i) {
      graph.distances[i] = detail::native_knn_output_distance(search.distances[i], metric);
    }
    return graph;
#else
    throw std::runtime_error("KODAMA Metal graph construction requires an Apple Metal build.");
#endif
  }
  return hnsw_graph(data, data, n, n, dim, neighbors, metric, n_threads, true, include_self);
}

std::vector<int> kmeans_labels(
  const std::vector<float>& x,
  int n,
  int p,
  int k,
  std::mt19937_64& rng,
  int max_iter = 10,
  int n_threads = 1,
  Backend backend = Backend::CPU,
  int gpu_device = 0
) {
  k = std::max(1, std::min(k, n));

#ifdef KODAMA_ENABLE_CUDA
  if (backend == Backend::CUDA) {
    return detail::native_cuda_kmeans_labels(
      x,
      n,
      p,
      k,
      std::max(1, max_iter),
      rng(),
      gpu_device
    );
  }
#else
  (void)gpu_device;
#endif

  std::vector<int> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng);
  if (backend == Backend::Metal) {
#if defined(KODAMA_ENABLE_METAL)
    return detail::metal_kmeans_labels(x, n, p, k, order, std::max(1, max_iter));
#else
    throw std::runtime_error("KODAMA Metal k-means requires an Apple Metal build.");
#endif
  }
  std::vector<float> centroids(static_cast<std::size_t>(k) * static_cast<std::size_t>(p), 0.0f);
  for (int cluster = 0; cluster < k; ++cluster) {
    std::copy_n(
      x.data() + static_cast<std::size_t>(order[static_cast<std::size_t>(cluster)]) * static_cast<std::size_t>(p),
      p,
      centroids.data() + static_cast<std::size_t>(cluster) * static_cast<std::size_t>(p)
    );
  }

  std::vector<int> labels(static_cast<std::size_t>(n), 1);
  std::vector<int> previous(static_cast<std::size_t>(n), -1);
  std::vector<double> sums(static_cast<std::size_t>(k) * static_cast<std::size_t>(p), 0.0);
  std::vector<int> counts(static_cast<std::size_t>(k), 0);
  OmpThreadScope threads(n_threads);
  for (int iteration = 0; iteration < std::max(1, max_iter); ++iteration) {
    std::atomic<int> changed{0};
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row = 0; row < n; ++row) {
      int best_cluster = 0;
      double best_distance = std::numeric_limits<double>::infinity();
      const float* point = x.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(p);
      for (int cluster = 0; cluster < k; ++cluster) {
        const float* centroid = centroids.data() + static_cast<std::size_t>(cluster) * static_cast<std::size_t>(p);
        double distance = 0.0;
        for (int feature = 0; feature < p; ++feature) {
          const double delta = static_cast<double>(point[feature]) - static_cast<double>(centroid[feature]);
          distance += delta * delta;
        }
        if (distance < best_distance || (distance == best_distance && cluster < best_cluster)) {
          best_distance = distance;
          best_cluster = cluster;
        }
      }
      labels[static_cast<std::size_t>(row)] = best_cluster + 1;
      if (previous[static_cast<std::size_t>(row)] != best_cluster) changed.fetch_add(1, std::memory_order_relaxed);
    }

    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);
    for (int row = 0; row < n; ++row) {
      const int cluster = labels[static_cast<std::size_t>(row)] - 1;
      ++counts[static_cast<std::size_t>(cluster)];
      const float* point = x.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(p);
      double* sum = sums.data() + static_cast<std::size_t>(cluster) * static_cast<std::size_t>(p);
      for (int feature = 0; feature < p; ++feature) sum[feature] += point[feature];
      previous[static_cast<std::size_t>(row)] = cluster;
    }
    for (int cluster = 0; cluster < k; ++cluster) {
      float* centroid = centroids.data() + static_cast<std::size_t>(cluster) * static_cast<std::size_t>(p);
      if (counts[static_cast<std::size_t>(cluster)] == 0) {
        const int replacement = order[static_cast<std::size_t>((cluster + iteration) % n)];
        std::copy_n(x.data() + static_cast<std::size_t>(replacement) * static_cast<std::size_t>(p), p, centroid);
        continue;
      }
      const double scale = 1.0 / static_cast<double>(counts[static_cast<std::size_t>(cluster)]);
      const double* sum = sums.data() + static_cast<std::size_t>(cluster) * static_cast<std::size_t>(p);
      for (int feature = 0; feature < p; ++feature) centroid[feature] = static_cast<float>(sum[feature] * scale);
    }
    if (changed.load(std::memory_order_relaxed) == 0) break;
  }
  return labels;
}

std::vector<int> factor_subset(const std::vector<int>& values, const std::vector<int>& rows) {
  std::map<int, int> ids;
  std::vector<int> out(rows.size(), 0);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const int value = values[static_cast<std::size_t>(rows[i])];
    auto it = ids.find(value);
    if (it == ids.end()) {
      const int id = static_cast<int>(ids.size()) + 1;
      it = ids.emplace(value, id).first;
    }
    out[i] = it->second;
  }
  return out;
}

std::vector<int> constrained_majority(const std::vector<int>& labels, const std::vector<int>& constrain) {
  std::map<int, std::vector<int>> by_group;
  for (std::size_t i = 0; i < labels.size(); ++i) by_group[constrain[i]].push_back(labels[i]);
  std::map<int, int> group_label;
  for (const auto& kv : by_group) group_label[kv.first] = majority_label(kv.second);
  std::vector<int> out(labels.size(), 0);
  for (std::size_t i = 0; i < labels.size(); ++i) out[i] = group_label[constrain[i]];
  return out;
}

std::vector<int> majority_by_constrain(const std::vector<int>& values, const std::vector<int>& constrain) {
  if (values.size() != constrain.size()) throw std::invalid_argument("majority_by_constrain size mismatch.");
  std::map<int, std::map<int, int>> counts;
  for (std::size_t i = 0; i < values.size(); ++i) {
    counts[constrain[i]][values[i]] += 1;
  }
  std::map<int, int> group_value;
  for (const auto& group : counts) {
    int best_value = group.second.begin()->first;
    int best_count = group.second.begin()->second;
    for (const auto& kv : group.second) {
      if (kv.second > best_count || (kv.second == best_count && kv.first < best_value)) {
        best_value = kv.first;
        best_count = kv.second;
      }
    }
    group_value[group.first] = best_value;
  }
  std::vector<int> out(values.size(), 0);
  for (std::size_t i = 0; i < values.size(); ++i) out[i] = group_value[constrain[i]];
  return out;
}

struct DisjointSet {
  std::vector<int> parent;
  std::vector<int> size;

  explicit DisjointSet(int n) : parent(static_cast<std::size_t>(n)), size(static_cast<std::size_t>(n), 1) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    while (parent[static_cast<std::size_t>(x)] != x) {
      parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
      x = parent[static_cast<std::size_t>(x)];
    }
    return x;
  }

  bool unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return false;
    if (size[static_cast<std::size_t>(a)] < size[static_cast<std::size_t>(b)]) std::swap(a, b);
    parent[static_cast<std::size_t>(b)] = a;
    size[static_cast<std::size_t>(a)] += size[static_cast<std::size_t>(b)];
    return true;
  }
};

std::vector<int> spatial_graph_components(
  const std::vector<float>& spatial,
  int n,
  int dims,
  int target_components,
  int n_threads,
  Backend backend,
  int gpu_device
) {
  target_components = std::max(1, std::min(target_components, n));
  if (target_components >= n) {
    std::vector<int> out(static_cast<std::size_t>(n));
    std::iota(out.begin(), out.end(), 1);
    return out;
  }
  const int k = std::max(2, std::min(n, 32));
  const NeighborGraph graph = self_knn_graph(
    spatial,
    n,
    dims,
    k,
    DistanceMetric::Euclidean,
    n_threads,
    backend,
    gpu_device,
    true,
    KNNIndexType::MetalExact,
    0,
    0
  );
  struct Edge {
    float distance;
    int a;
    int b;
  };
  std::vector<Edge> edges;
  edges.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < graph.neighbors; ++j) {
      const std::size_t offset = static_cast<std::size_t>(i) * graph.neighbors + static_cast<std::size_t>(j);
      const int b = graph.indices[offset];
      if (b < 0 || b == i || b < i) continue;
      edges.push_back({graph.distances[offset], i, b});
    }
  }
  std::sort(edges.begin(), edges.end(), [](const Edge& lhs, const Edge& rhs) {
    if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
    if (lhs.a != rhs.a) return lhs.a < rhs.a;
    return lhs.b < rhs.b;
  });
  DisjointSet dsu(n);
  int components = n;
  for (const Edge& edge : edges) {
    if (components <= target_components) break;
    if (dsu.unite(edge.a, edge.b)) --components;
  }

  std::map<int, int> ids;
  std::vector<int> out(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    const int root = dsu.find(i);
    auto it = ids.find(root);
    if (it == ids.end()) {
      const int id = static_cast<int>(ids.size()) + 1;
      it = ids.emplace(root, id).first;
    }
    out[static_cast<std::size_t>(i)] = it->second;
  }
  return out;
}

std::vector<float> spatial_jitter_from_graph(
  const std::vector<float>& spatial,
  int n,
  int dims,
  int neighbors,
  int n_threads,
  Backend backend,
  int gpu_device
) {
  const int k = std::max(1, std::min(neighbors, n));
  NeighborGraph graph = self_knn_graph(
    spatial,
    n,
    dims,
    k,
    DistanceMetric::Euclidean,
    n_threads,
    backend,
    gpu_device,
    true,
    KNNIndexType::MetalExact,
    0,
    0
  );
  const int far_col = std::min(19, graph.neighbors - 1);
  std::vector<double> sums(static_cast<std::size_t>(dims), 0.0);
  for (int i = 0; i < n; ++i) {
    const int near_row = graph.indices[static_cast<std::size_t>(i) * graph.neighbors];
    const int far_row = graph.indices[static_cast<std::size_t>(i) * graph.neighbors + static_cast<std::size_t>(far_col)];
    if (near_row < 0 || far_row < 0) continue;
    for (int d = 0; d < dims; ++d) {
      sums[static_cast<std::size_t>(d)] += std::abs(
        spatial[static_cast<std::size_t>(near_row) * dims + static_cast<std::size_t>(d)] -
        spatial[static_cast<std::size_t>(far_row) * dims + static_cast<std::size_t>(d)]
      );
    }
  }
  std::vector<float> jitter(static_cast<std::size_t>(dims), 0.0f);
  for (int d = 0; d < dims; ++d) jitter[static_cast<std::size_t>(d)] = static_cast<float>(3.0 * sums[static_cast<std::size_t>(d)] / std::max(1, n));
  return jitter;
}

void repair_singleton_spatial_clusters(
  std::vector<int>& clusters,
  const std::vector<float>& spatial,
  int n,
  int dims,
  int n_threads
) {
  std::map<int, int> counts;
  for (int c : clusters) counts[c] += 1;
  std::vector<char> keep(static_cast<std::size_t>(n), 0);
  int n_keep = 0;
  for (int i = 0; i < n; ++i) {
    if (counts[clusters[static_cast<std::size_t>(i)]] > 1) {
      keep[static_cast<std::size_t>(i)] = 1;
      ++n_keep;
    }
  }
  if (n_keep == n || n_keep == 0) return;

  std::vector<float> base(static_cast<std::size_t>(n_keep) * dims, 0.0f);
  std::vector<float> query(static_cast<std::size_t>(n - n_keep) * dims, 0.0f);
  std::vector<int> base_rows;
  std::vector<int> query_rows;
  base_rows.reserve(static_cast<std::size_t>(n_keep));
  query_rows.reserve(static_cast<std::size_t>(n - n_keep));
  for (int i = 0; i < n; ++i) {
    if (keep[static_cast<std::size_t>(i)]) base_rows.push_back(i);
    else query_rows.push_back(i);
  }
  for (std::size_t i = 0; i < base_rows.size(); ++i) {
    std::copy_n(spatial.data() + static_cast<std::size_t>(base_rows[i]) * dims, dims, base.data() + i * dims);
  }
  for (std::size_t i = 0; i < query_rows.size(); ++i) {
    std::copy_n(spatial.data() + static_cast<std::size_t>(query_rows[i]) * dims, dims, query.data() + i * dims);
  }
  NeighborGraph nearest = hnsw_graph(base, query, n_keep, static_cast<int>(query_rows.size()), dims, 1, DistanceMetric::Euclidean, n_threads);
  for (std::size_t i = 0; i < query_rows.size(); ++i) {
    const int local = nearest.indices[i];
    if (local >= 0) clusters[static_cast<std::size_t>(query_rows[i])] = clusters[static_cast<std::size_t>(base_rows[static_cast<std::size_t>(local)])];
  }
}

NeighborGraph trim_self_neighbors(const NeighborGraph& graph, int samples, int neighbors) {
  NeighborGraph trimmed;
  trimmed.neighbors = neighbors;
  trimmed.indices.assign(static_cast<std::size_t>(samples) * neighbors, -1);
  trimmed.distances.assign(static_cast<std::size_t>(samples) * neighbors, std::numeric_limits<float>::infinity());
  for (int i = 0; i < samples; ++i) {
    int out_col = 0;
    for (int j = 0; j < graph.neighbors && out_col < neighbors; ++j) {
      const std::size_t in_offset = static_cast<std::size_t>(i) * graph.neighbors + static_cast<std::size_t>(j);
      const int id = graph.indices[in_offset];
      if (id == i) continue;
      const std::size_t out_offset = static_cast<std::size_t>(i) * neighbors + static_cast<std::size_t>(out_col);
      trimmed.indices[out_offset] = id;
      trimmed.distances[out_offset] = graph.distances[in_offset];
      ++out_col;
    }
  }
  return trimmed;
}

NeighborGraph normalize_external_graph(const NeighborGraph& graph, int samples, int max_neighbors) {
  if (samples < 2) throw std::invalid_argument("NeighborGraph samples must be at least 2.");
  if (graph.neighbors <= 0) throw std::invalid_argument("NeighborGraph.neighbors must be positive.");
  const std::size_t expected = static_cast<std::size_t>(samples) * static_cast<std::size_t>(graph.neighbors);
  if (graph.indices.size() != expected || graph.distances.size() != expected) {
    throw std::invalid_argument("NeighborGraph indices/distances size must equal samples * neighbors.");
  }
  int min_index = std::numeric_limits<int>::max();
  int max_index = std::numeric_limits<int>::min();
  for (int id : graph.indices) {
    if (id < 0) continue;
    min_index = std::min(min_index, id);
    max_index = std::max(max_index, id);
  }
  const bool one_based = min_index >= 1 && max_index <= samples;
  const int neighbors = std::max(1, std::min(max_neighbors > 0 ? max_neighbors : graph.neighbors, graph.neighbors));
  NeighborGraph out;
  out.neighbors = neighbors;
  out.indices.assign(static_cast<std::size_t>(samples) * static_cast<std::size_t>(neighbors), -1);
  out.distances.assign(static_cast<std::size_t>(samples) * static_cast<std::size_t>(neighbors), std::numeric_limits<float>::infinity());
  for (int i = 0; i < samples; ++i) {
    int out_col = 0;
    for (int j = 0; j < graph.neighbors && out_col < neighbors; ++j) {
      const std::size_t src = static_cast<std::size_t>(i) * static_cast<std::size_t>(graph.neighbors) + static_cast<std::size_t>(j);
      int id = graph.indices[src];
      if (one_based && id > 0) --id;
      if (id < 0 || id >= samples || id == i) continue;
      const float d = graph.distances[src];
      if (!std::isfinite(d)) continue;
      const std::size_t dst = static_cast<std::size_t>(i) * static_cast<std::size_t>(neighbors) + static_cast<std::size_t>(out_col);
      out.indices[dst] = id;
      out.distances[dst] = std::max(0.0f, d);
      ++out_col;
    }
  }
  return out;
}

NeighborGraph subset_graph_to_rows(const NeighborGraph& graph, const std::vector<int>& rows, int samples) {
  std::vector<int> global_to_local(static_cast<std::size_t>(samples), -1);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    global_to_local[static_cast<std::size_t>(rows[i])] = static_cast<int>(i);
  }
  NeighborGraph out;
  out.neighbors = graph.neighbors;
  out.indices.assign(rows.size() * static_cast<std::size_t>(graph.neighbors), -1);
  out.distances.assign(rows.size() * static_cast<std::size_t>(graph.neighbors), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const int row = rows[i];
    int out_col = 0;
    for (int j = 0; j < graph.neighbors && out_col < graph.neighbors; ++j) {
      const std::size_t src = static_cast<std::size_t>(row) * static_cast<std::size_t>(graph.neighbors) + static_cast<std::size_t>(j);
      const int global_nb = graph.indices[src];
      if (global_nb < 0 || global_nb >= samples) continue;
      const int local_nb = global_to_local[static_cast<std::size_t>(global_nb)];
      if (local_nb < 0 || local_nb == static_cast<int>(i)) continue;
      const std::size_t dst = i * static_cast<std::size_t>(graph.neighbors) + static_cast<std::size_t>(out_col);
      // CoreKNNGraph_CPU accepts public-style graphs and normalizes 1-based
      // indices at its boundary. Emit the local subset graph in that form so
      // index-base auto detection cannot misclassify a zero-based subset when
      // local index 0 is absent from the neighbor list.
      out.indices[dst] = local_nb + 1;
      out.distances[dst] = graph.distances[src];
      ++out_col;
    }
  }
  return out;
}

struct SparseGraphOperator {
  int samples = 0;
  std::vector<int> indptr;
  std::vector<int> indices;
  std::vector<float> weights;
};

std::vector<float> graph_local_scales(const NeighborGraph& graph, int samples) {
  std::vector<float> scales(static_cast<std::size_t>(samples), 1.0f);
  std::vector<float> row_dist;
  row_dist.reserve(static_cast<std::size_t>(graph.neighbors));
  for (int i = 0; i < samples; ++i) {
    row_dist.clear();
    const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(graph.neighbors);
    for (int j = 0; j < graph.neighbors; ++j) {
      const int id = graph.indices[base + static_cast<std::size_t>(j)];
      const float d = graph.distances[base + static_cast<std::size_t>(j)];
      if (id < 0 || !std::isfinite(d)) continue;
      row_dist.push_back(std::max(0.0f, d));
    }
    if (row_dist.empty()) continue;
    const std::size_t mid = row_dist.size() / 2;
    std::nth_element(row_dist.begin(), row_dist.begin() + static_cast<std::ptrdiff_t>(mid), row_dist.end());
    float scale = row_dist[mid];
    if (scale <= 1.0e-6f) {
      double mean = 0.0;
      for (float d : row_dist) mean += static_cast<double>(d);
      scale = static_cast<float>(mean / static_cast<double>(row_dist.size()));
    }
    scales[static_cast<std::size_t>(i)] = std::max(scale, 1.0e-6f);
  }
  return scales;
}

SparseGraphOperator make_sparse_graph_operator(
  const NeighborGraph& graph,
  int samples,
  bool symmetrize,
  bool self_tuning,
  bool symmetric_normalize
) {
  NeighborGraph g = normalize_external_graph(graph, samples, graph.neighbors);
  const std::vector<float> scales = self_tuning ? graph_local_scales(g, samples) : std::vector<float>();
  std::vector<std::unordered_map<int, float>> rows(static_cast<std::size_t>(samples));
  for (auto& row : rows) row.reserve(static_cast<std::size_t>(std::max(1, g.neighbors)));

  auto compute_weight = [&](int i, int j, float d) {
    d = std::max(0.0f, d);
    if (self_tuning) {
      const double denom =
        std::max(1.0e-12, static_cast<double>(scales[static_cast<std::size_t>(i)]) *
                          static_cast<double>(scales[static_cast<std::size_t>(j)]));
      return static_cast<float>(std::exp(-(static_cast<double>(d) * static_cast<double>(d)) / denom));
    }
    return 1.0f / (1.0f + d);
  };
  auto add_edge = [&](int from, int to, float w) {
    if (from < 0 || from >= samples || to < 0 || to >= samples || from == to || !std::isfinite(w) || w <= 0.0f) return;
    auto& row = rows[static_cast<std::size_t>(from)];
    auto it = row.find(to);
    if (it == row.end()) {
      row.emplace(to, w);
    } else {
      it->second = std::max(it->second, w);
    }
  };

  for (int i = 0; i < samples; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(g.neighbors);
    for (int j = 0; j < g.neighbors; ++j) {
      const std::size_t offset = base + static_cast<std::size_t>(j);
      const int nb = g.indices[offset];
      if (nb < 0 || nb >= samples) continue;
      const float d = g.distances[offset];
      if (!std::isfinite(d)) continue;
      const float w = compute_weight(i, nb, d);
      add_edge(i, nb, w);
      if (symmetrize) add_edge(nb, i, w);
    }
  }

  std::vector<double> degree(static_cast<std::size_t>(samples), 0.0);
  for (int i = 0; i < samples; ++i) {
    for (const auto& kv : rows[static_cast<std::size_t>(i)]) degree[static_cast<std::size_t>(i)] += static_cast<double>(kv.second);
  }

  SparseGraphOperator op;
  op.samples = samples;
  op.indptr.assign(static_cast<std::size_t>(samples) + 1, 0);
  std::size_t nnz = 0;
  for (int i = 0; i < samples; ++i) {
    op.indptr[static_cast<std::size_t>(i)] = static_cast<int>(nnz);
    nnz += rows[static_cast<std::size_t>(i)].size();
  }
  op.indptr[static_cast<std::size_t>(samples)] = static_cast<int>(nnz);
  op.indices.reserve(nnz);
  op.weights.reserve(nnz);
  for (int i = 0; i < samples; ++i) {
    std::vector<std::pair<int, float>> ordered(rows[static_cast<std::size_t>(i)].begin(), rows[static_cast<std::size_t>(i)].end());
    std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    const double row_degree = std::max(degree[static_cast<std::size_t>(i)], 1.0e-12);
    for (const auto& kv : ordered) {
      const int nb = kv.first;
      float w = kv.second;
      if (symmetric_normalize) {
        const double denom = std::sqrt(row_degree * std::max(degree[static_cast<std::size_t>(nb)], 1.0e-12));
        w = static_cast<float>(static_cast<double>(w) / denom);
      } else {
        w = static_cast<float>(static_cast<double>(w) / row_degree);
      }
      op.indices.push_back(nb);
      op.weights.push_back(w);
    }
  }
  return op;
}

void standardize_feature_columns(std::vector<float>& x, int n, int p) {
  for (int c = 0; c < p; ++c) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]);
    mean /= static_cast<double>(std::max(1, n));
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
      float& v = x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)];
      v = static_cast<float>(static_cast<double>(v) - mean);
      ss += static_cast<double>(v) * static_cast<double>(v);
    }
    const double scale = ss > 0.0 ? std::sqrt(ss / static_cast<double>(std::max(1, n - 1))) : 1.0;
    const float inv = static_cast<float>(1.0 / std::max(scale, 1.0e-6));
    for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)] *= inv;
  }
}

void center_feature_columns(std::vector<float>& x, int n, int p) {
  for (int c = 0; c < p; ++c) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]);
    mean /= static_cast<double>(std::max(1, n));
    for (int i = 0; i < n; ++i) {
      x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)] =
        static_cast<float>(static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]) - mean);
    }
  }
}

void orthonormalize_feature_columns(std::vector<float>& x, int n, int p) {
  center_feature_columns(x, n, p);
  for (int c = 0; c < p; ++c) {
    for (int prev = 0; prev < c; ++prev) {
      double dot = 0.0;
      for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]) *
               static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(prev)]);
      }
      for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)] =
          static_cast<float>(
            static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]) -
            dot * static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(prev)])
          );
      }
    }
    double norm2 = 0.0;
    for (int i = 0; i < n; ++i) {
      const double v = static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]);
      norm2 += v * v;
    }
    if (norm2 <= 1.0e-20) {
      for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)] = 0.0f;
      if (n > 0) x[static_cast<std::size_t>(c % n) * p + static_cast<std::size_t>(c)] = 1.0f;
      center_feature_columns(x, n, p);
      norm2 = 0.0;
      for (int i = 0; i < n; ++i) {
        const double v = static_cast<double>(x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)]);
        norm2 += v * v;
      }
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(std::max(norm2, 1.0e-20)));
    for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i) * p + static_cast<std::size_t>(c)] *= inv;
  }
}

void apply_sparse_graph_operator(
  const SparseGraphOperator& op,
  const std::vector<float>& current,
  std::vector<float>& next,
  int components,
  int n_threads
) {
  std::fill(next.begin(), next.end(), 0.0f);
  OmpThreadScope threads(n_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < op.samples; ++i) {
    const std::size_t row_base = static_cast<std::size_t>(i) * static_cast<std::size_t>(components);
    for (int ptr = op.indptr[static_cast<std::size_t>(i)]; ptr < op.indptr[static_cast<std::size_t>(i + 1)]; ++ptr) {
      const int nb = op.indices[static_cast<std::size_t>(ptr)];
      const float w = op.weights[static_cast<std::size_t>(ptr)];
      const std::size_t nb_base = static_cast<std::size_t>(nb) * static_cast<std::size_t>(components);
      for (int c = 0; c < components; ++c) {
        next[row_base + static_cast<std::size_t>(c)] += w * current[nb_base + static_cast<std::size_t>(c)];
      }
    }
  }
}

std::vector<float> graph_laplacian_operator_features(
  const NeighborGraph& graph,
  int samples,
  int components,
  int iterations,
  std::uint64_t seed,
  int n_threads
) {
  const SparseGraphOperator op = make_sparse_graph_operator(
    graph,
    samples,
    true,
    true,
    true
  );
  components = std::max(1, std::min(components, samples));
  iterations = std::max(8, iterations);
  std::vector<float> current(static_cast<std::size_t>(samples) * components, 0.0f);
  std::vector<float> next(current.size(), 0.0f);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (float& v : current) v = normal(rng);
  orthonormalize_feature_columns(current, samples, components);
  for (int iter = 0; iter < iterations; ++iter) {
    apply_sparse_graph_operator(op, current, next, components, n_threads);
    orthonormalize_feature_columns(next, samples, components);
    current.swap(next);
  }
  standardize_feature_columns(current, samples, components);
  return current;
}

NeighborGraph merge_feature_spatial_graphs(
  const NeighborGraph& feature,
  const NeighborGraph& spatial,
  int samples,
  int neighbors
) {
  NeighborGraph out;
  out.neighbors = neighbors;
  out.indices.assign(static_cast<std::size_t>(samples) * neighbors, -1);
  out.distances.assign(static_cast<std::size_t>(samples) * neighbors, std::numeric_limits<float>::infinity());
  std::vector<char> seen(static_cast<std::size_t>(samples), 0);
  std::vector<int> touched;
  touched.reserve(static_cast<std::size_t>(neighbors) * 2);
  auto try_add = [&](int row, int id, int rank, int& out_col) {
    if (out_col >= neighbors || id < 0 || id >= samples || id == row || seen[static_cast<std::size_t>(id)]) return;
    seen[static_cast<std::size_t>(id)] = 1;
    touched.push_back(id);
    const std::size_t out_offset = static_cast<std::size_t>(row) * neighbors + static_cast<std::size_t>(out_col);
    out.indices[out_offset] = id;
    out.distances[out_offset] = static_cast<float>(rank + 1) / static_cast<float>(std::max(1, neighbors));
    ++out_col;
  };
  for (int i = 0; i < samples; ++i) {
    touched.clear();
    int out_col = 0;
    for (int rank = 0; rank < neighbors && out_col < neighbors; ++rank) {
      const std::size_t offset = static_cast<std::size_t>(i) * neighbors + static_cast<std::size_t>(rank);
      try_add(i, feature.indices[offset], rank, out_col);
      try_add(i, spatial.indices[offset], rank, out_col);
    }
    for (int rank = 0; rank < neighbors && out_col < neighbors; ++rank) {
      const std::size_t offset = static_cast<std::size_t>(i) * neighbors + static_cast<std::size_t>(rank);
      try_add(i, feature.indices[offset], rank, out_col);
      try_add(i, spatial.indices[offset], rank, out_col);
    }
    for (int id : touched) seen[static_cast<std::size_t>(id)] = 0;
  }
  return out;
}

struct IterationResult {
  std::vector<int> res;
  std::vector<int> constrain;
  std::vector<double> acc;
  double accbest = std::numeric_limits<double>::quiet_NaN();
  double runtime = 0.0;
  double memory = 0.0;
};

struct IterationScratch {
  std::vector<int> cluster_counts;
  std::vector<int> selected_landpoints;
  std::vector<int> landpoints;
  std::vector<char> is_landmark;
  std::vector<int> tpoints;
  std::vector<float> x_land;
  std::vector<float> x_test;
  std::vector<int> x_constrain;
  std::vector<int> x_fixed;
  std::vector<int> tmp_labels;
  std::vector<int> xw;
  std::vector<int> init;
  std::vector<int> global_to_local;
  std::vector<int> projection_votes;
  std::vector<float> spatial_jittered;
  std::vector<int> spatial_clusters;
  std::vector<int> run_constrain;
};

struct GpuWorkerPlan {
  int workers = 1;
  bool automatic = false;
  int sm_count = 0;
  double free_memory_mb = 0.0;
  double total_memory_mb = 0.0;
  double worker_memory_estimate_mb = 0.0;
};

#ifdef KODAMA_ENABLE_CUDA
GpuWorkerPlan resolve_gpu_worker_plan(
  const KODAMAMatrixOptions& options,
  int samples,
  int features,
  int landmarks,
  int graph_neighbors
) {
  GpuWorkerPlan plan;
  plan.workers = std::max(1, std::min(options.n_threads, options.runs));
  if (options.backend != Backend::CUDA || options.n_threads > 0) return plan;

  plan.automatic = true;
  plan.workers = 1;
  cudaSetDevice(options.knn.gpu_device);
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, options.knn.gpu_device) == cudaSuccess) {
    plan.sm_count = prop.multiProcessorCount;
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
    plan.free_memory_mb = static_cast<double>(free_bytes) / (1024.0 * 1024.0);
    plan.total_memory_mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  }

  const double x_bytes = static_cast<double>(samples) * static_cast<double>(features) * sizeof(float);
  const double landmark_bytes = static_cast<double>(landmarks) * static_cast<double>(features) * sizeof(float);
  const double graph_bytes = static_cast<double>(samples) * static_cast<double>(graph_neighbors) *
    static_cast<double>(sizeof(int) + sizeof(float));
  const double component_bytes = static_cast<double>(landmarks) *
    static_cast<double>(std::max(1, options.components)) * sizeof(float);
  double worker_bytes = 512.0 * 1024.0 * 1024.0;
  worker_bytes += 1.5 * x_bytes;
  worker_bytes += 6.0 * landmark_bytes;
  worker_bytes += 0.25 * graph_bytes;
  if (options.classifier != CoreClassifier::KNN) {
    const double feature_component_ratio =
      static_cast<double>(features) / static_cast<double>(std::max(1, options.components));
    worker_bytes += 4.0 * feature_component_ratio * landmark_bytes;
    worker_bytes += 10.0 * component_bytes;
  } else {
    worker_bytes += static_cast<double>(landmarks) * static_cast<double>(std::max(1, options.knn.k)) *
      static_cast<double>(sizeof(int) + sizeof(float));
  }
  plan.worker_memory_estimate_mb = worker_bytes / (1024.0 * 1024.0);

  int sm_cap = 2;
  if (plan.sm_count >= 132) sm_cap = 6;
  else if (plan.sm_count >= 96) sm_cap = 5;
  else if (plan.sm_count >= 72) sm_cap = 4;
  else if (plan.sm_count >= 32) sm_cap = 4;

  int memory_cap = 1;
  if (free_bytes > 0 && worker_bytes > 0.0) {
    memory_cap = static_cast<int>(std::floor((0.70 * static_cast<double>(free_bytes)) / worker_bytes));
    memory_cap = std::max(1, memory_cap);
  }

  plan.workers = std::max(1, std::min({options.runs, sm_cap, memory_cap}));
  return plan;
}
#else
GpuWorkerPlan resolve_gpu_worker_plan(
  const KODAMAMatrixOptions& options,
  int,
  int,
  int,
  int
) {
  GpuWorkerPlan plan;
  plan.workers = std::max(1, std::min(options.n_threads, options.runs));
  return plan;
}
#endif

#ifdef KODAMA_ENABLE_CUDA
class CudaMScheduler {
 public:
  CudaMScheduler(int lanes, int device) :
    lanes_(std::max(1, lanes)),
    device_(device) {}

  int lanes() const { return lanes_; }

  template <typename Runner>
  void run(int runs, Runner&& runner) const {
    std::atomic<int> next_run{1};
    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<std::size_t>(lanes_));
    for (int lane = 0; lane < lanes_; ++lane) {
      futures.emplace_back(std::async(std::launch::async, [&, lane]() {
        cudaSetDevice(device_);
        cudaFree(nullptr);
        IterationScratch scratch;
        while (true) {
          const int run_id = next_run.fetch_add(1);
          if (run_id > runs) break;
          runner(run_id, lane, scratch);
        }
      }));
    }
    for (auto& future : futures) future.get();
  }

 private:
  int lanes_ = 1;
  int device_ = 0;
};
#endif

void apply_kodama_dissimilarity(NeighborGraph& graph, const std::vector<int>& res, int runs, int samples, int n_threads) {
  if (runs <= 0 || samples <= 0 || graph.neighbors <= 0) return;
  OmpThreadScope threads(n_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < samples; ++i) {
    std::vector<std::pair<float, int>> row(static_cast<std::size_t>(graph.neighbors));
    const std::size_t row_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(graph.neighbors);
    for (int j = 0; j < graph.neighbors; ++j) {
      const std::size_t offset = row_offset + static_cast<std::size_t>(j);
      const int neighbor = graph.indices[offset];
      float distance = graph.distances[offset];
      if (neighbor < 0 || neighbor >= samples || !std::isfinite(distance)) {
        row[static_cast<std::size_t>(j)] = {std::numeric_limits<float>::infinity(), neighbor};
        continue;
      }
      int same = 0;
      int valid = 0;
      for (int run = 0; run < runs; ++run) {
        const std::size_t base = static_cast<std::size_t>(run) * static_cast<std::size_t>(samples);
        const int lhs = res[base + static_cast<std::size_t>(i)];
        const int rhs = res[base + static_cast<std::size_t>(neighbor)];
        if (lhs == 0 || rhs == 0) continue;
        ++valid;
        if (lhs == rhs) ++same;
      }
      if (same == 0 || valid == 0) {
        distance = std::numeric_limits<float>::infinity();
      } else {
        const double agreement = static_cast<double>(same) / static_cast<double>(valid);
        distance = static_cast<float>((1.0 + static_cast<double>(distance)) / (agreement * agreement));
      }
      row[static_cast<std::size_t>(j)] = {distance, neighbor};
    }
    std::stable_sort(row.begin(), row.end(), [](const auto& a, const auto& b) {
      if (a.first != b.first) return a.first < b.first;
      return a.second < b.second;
    });
    for (int j = 0; j < graph.neighbors; ++j) {
      const std::size_t offset = row_offset + static_cast<std::size_t>(j);
      graph.distances[offset] = row[static_cast<std::size_t>(j)].first;
      graph.indices[offset] = row[static_cast<std::size_t>(j)].second;
    }
  }
}

IterationResult run_iteration(
  MatrixView full,
  const std::vector<float>& full_float,
  const std::vector<float>& spatial,
  const std::vector<float>& spatial_jitter,
  const NeighborGraph& global_graph,
  bool global_graph_is_input,
  const std::vector<int>& constrain,
  bool constrain_is_identity,
  const std::vector<int>& fixed,
  const std::vector<int>& starting_labels,
  const KODAMAMatrixOptions& options,
  int run_id,
  IterationScratch& scratch
) {
  const int n = static_cast<int>(full.rows);
  const int p = static_cast<int>(full.cols);
  int landmarks = options.landmarks;
  if (n <= landmarks) {
    landmarks = static_cast<int>(std::ceil(static_cast<double>(n) * 0.75));
  }
  landmarks = std::max(2, std::min(landmarks, n - 1));
  const int splitting = options.splitting > 0 ? options.splitting : (n < 40000 ? 100 : 300);
  const bool spatial_flag = !spatial.empty() && options.spatial_cols > 0;
  std::mt19937_64 rng(options.seed + static_cast<std::uint64_t>(run_id));

  detail::Timer iter_timer;
  if (options.progress) {
    std::cerr << "[kodama] M " << run_id << "/" << options.runs
              << " landmark k-means with " << landmarks << " centers" << std::endl;
  }
  const int kmeans_gpu_device = options.knn.gpu_device;
  const std::vector<int> landmark_clusters = kmeans_labels(
    full_float,
    n,
    p,
    landmarks,
    rng,
    10,
    options.n_threads,
    options.backend,
    kmeans_gpu_device
  );
  if (options.progress) {
    std::cerr << "[kodama] M " << run_id << "/" << options.runs
              << " landmark k-means done in " << iter_timer.seconds() << "s" << std::endl;
  }
  const std::vector<int>* run_constrain_ptr = &constrain;
  bool run_constrain_is_identity = constrain_is_identity;
  if (spatial_flag) {
    const int spatial_dims = options.spatial_cols;
    const int nspatialclusters = std::max(1, static_cast<int>(std::llround(static_cast<double>(landmarks) * options.spatial_resolution)));
    scratch.spatial_jittered.resize(spatial.size());
    for (int i = 0; i < n; ++i) {
      for (int d = 0; d < spatial_dims; ++d) {
        const float width = spatial_jitter[static_cast<std::size_t>(d)];
        std::uniform_real_distribution<float> jitter(-width, width);
        scratch.spatial_jittered[static_cast<std::size_t>(i) * spatial_dims + static_cast<std::size_t>(d)] =
          spatial[static_cast<std::size_t>(i) * spatial_dims + static_cast<std::size_t>(d)] + jitter(rng);
      }
    }
    const bool use_graph_spatial_constraints = options.spatial_constraint_mode == 1;
    if (use_graph_spatial_constraints) {
      scratch.spatial_clusters = spatial_graph_components(
        scratch.spatial_jittered,
        n,
        spatial_dims,
        nspatialclusters,
        options.n_threads,
        options.backend,
        kmeans_gpu_device
      );
    } else {
      scratch.spatial_clusters = kmeans_labels(
        scratch.spatial_jittered,
        n,
        spatial_dims,
        nspatialclusters,
        rng,
        10,
        options.n_threads,
        options.backend,
        kmeans_gpu_device
      );
    }
    repair_singleton_spatial_clusters(scratch.spatial_clusters, spatial, n, spatial_dims, options.n_threads);
    scratch.run_constrain = constrain_is_identity ?
      scratch.spatial_clusters :
      majority_by_constrain(scratch.spatial_clusters, constrain);
    run_constrain_ptr = &scratch.run_constrain;
    run_constrain_is_identity = is_identity_constrain(*run_constrain_ptr);
  }
  const std::vector<int>& run_constrain = *run_constrain_ptr;
  std::vector<std::vector<int>> cluster_rows(static_cast<std::size_t>(landmarks) + 1);
  for (int i = 0; i < n; ++i) cluster_rows[static_cast<std::size_t>(landmark_clusters[static_cast<std::size_t>(i)])].push_back(i);
  scratch.landpoints.clear();
  scratch.landpoints.reserve(static_cast<std::size_t>(landmarks));
  for (int c = 1; c <= landmarks; ++c) {
    const std::vector<int>& rows = cluster_rows[static_cast<std::size_t>(c)];
    if (rows.empty()) continue;
    std::uniform_int_distribution<int> pick(0, static_cast<int>(rows.size()) - 1);
    scratch.landpoints.push_back(rows[static_cast<std::size_t>(pick(rng))]);
  }
  std::sort(scratch.landpoints.begin(), scratch.landpoints.end());
  const std::vector<int>& landpoints = scratch.landpoints;

  scratch.is_landmark.assign(static_cast<std::size_t>(n), 0);
  for (int row : landpoints) scratch.is_landmark[static_cast<std::size_t>(row)] = 1;
  scratch.tpoints.clear();
  scratch.tpoints.reserve(static_cast<std::size_t>(n) - landpoints.size());
  for (int i = 0; i < n; ++i) {
    if (!scratch.is_landmark[static_cast<std::size_t>(i)]) scratch.tpoints.push_back(i);
  }
  const std::vector<int>& tpoints = scratch.tpoints;

  copy_float32_rows_into(full_float, full.cols, landpoints, scratch.x_land);
  if (run_constrain_is_identity) {
    scratch.x_constrain.resize(landpoints.size());
    std::iota(scratch.x_constrain.begin(), scratch.x_constrain.end(), 1);
  } else {
    scratch.x_constrain = factor_subset(run_constrain, landpoints);
  }
  scratch.x_fixed.resize(landpoints.size());
  for (std::size_t i = 0; i < landpoints.size(); ++i) scratch.x_fixed[i] = fixed[static_cast<std::size_t>(landpoints[i])];

  scratch.xw.assign(landpoints.size(), 0);
  if (!starting_labels.empty()) {
    scratch.tmp_labels.resize(landpoints.size());
    for (std::size_t i = 0; i < landpoints.size(); ++i) {
      scratch.tmp_labels[i] = starting_labels[static_cast<std::size_t>(landpoints[i])];
    }
    scratch.xw = run_constrain_is_identity ? scratch.tmp_labels : constrained_majority(scratch.tmp_labels, scratch.x_constrain);
  } else {
    const int init_k = std::max(2, std::min(splitting, static_cast<int>(landpoints.size())));
    scratch.init = kmeans_labels(
      scratch.x_land,
      static_cast<int>(landpoints.size()),
      p,
      init_k,
      rng,
      10,
      options.n_threads,
      options.backend,
      kmeans_gpu_device
    );
    scratch.xw = run_constrain_is_identity ? scratch.init : constrained_majority(scratch.init, scratch.x_constrain);
  }

  MatrixView x_view{scratch.x_land.data(), landpoints.size(), full.cols};
  CoreOptions core;
  core.cycles = options.cycles;
  core.seed = options.seed + static_cast<std::uint64_t>(run_id);
  core.classifier = options.classifier;
  core.evolutionary_search = starting_labels.empty();
  core.guarded_diversity = true;
  core.auto_class_coarsening = options.classifier == CoreClassifier::PLS_LDA;
  core.many_to_one_absorption = true;
  core.knn = options.knn;
  core.knn.backend = options.backend;
  core.knn.metric = options.metric;
  core.knn.cv.stratified = false;
  core.knn.cv.seed = options.seed + static_cast<std::uint64_t>(run_id);
  core.knn.k = std::max(core.knn.k, 1);
  core.knn.hnsw_tune_k = 50;
  core.knn.hnsw_target_recall = 0.99;
  core.knn.n_threads = options.backend == Backend::CUDA ?
    options.n_threads : std::max(1, options.knn.n_threads);
  core.pls = options.pls;
  core.pls.backend = options.backend;
  core.pls.cv.stratified = false;
  core.pls.cv.seed = options.seed + static_cast<std::uint64_t>(run_id);
  core.pls.max_components = options.components;
  core.pls.fixed_components = options.components;
  core.pls.n_threads = options.backend == Backend::CUDA ?
    options.n_threads : std::max(1, options.pls.n_threads);

  auto run_knn_core = [&](const std::vector<int>& labels, const CoreOptions& phase) {
    if (options.backend == Backend::CUDA) {
      return CoreKNN_CUDA(x_view, labels, scratch.x_constrain, scratch.x_fixed, phase);
    }
    if (options.backend == Backend::Metal) {
      return CoreKNN_METAL(x_view, labels, scratch.x_constrain, scratch.x_fixed, phase);
    }
    return CoreKNN_CPU(x_view, labels, scratch.x_constrain, scratch.x_fixed, phase);
  };
  auto run_pls_core = [&](const std::vector<int>& labels, const CoreOptions& phase) {
    if (options.backend == Backend::CUDA) {
      return CorePLSLDA_CUDA(x_view, labels, scratch.x_constrain, scratch.x_fixed, phase);
    }
    if (options.backend == Backend::Metal) {
      return CorePLSLDA_METAL(x_view, labels, scratch.x_constrain, scratch.x_fixed, phase);
    }
    return CorePLSLDA_CPU(x_view, labels, scratch.x_constrain, scratch.x_fixed, phase);
  };

  CoreResult core_result;
  if (options.classifier == CoreClassifier::KNN) {
    if (global_graph_is_input) {
      const NeighborGraph local_graph = subset_graph_to_rows(global_graph, landpoints, n);
      core_result = CoreKNNGraph_CPU(local_graph, static_cast<int>(landpoints.size()), scratch.xw, scratch.x_constrain, scratch.x_fixed, core);
    } else {
      core_result = run_knn_core(scratch.xw, core);
    }
  } else if (options.classifier == CoreClassifier::PLS_LDA) {
    core_result = run_pls_core(scratch.xw, core);
  } else {
    throw std::invalid_argument("Unsupported KODAMA.matrix classifier.");
  }

  IterationResult out;
  out.res.assign(static_cast<std::size_t>(n), 0);
  out.constrain = run_constrain;
  for (std::size_t i = 0; i < landpoints.size(); ++i) out.res[static_cast<std::size_t>(landpoints[i])] = core_result.clbest[i];
  if (!tpoints.empty()) {
    if (options.classifier == CoreClassifier::KNN) {
      scratch.global_to_local.assign(static_cast<std::size_t>(n), -1);
      for (std::size_t i = 0; i < landpoints.size(); ++i) {
        scratch.global_to_local[static_cast<std::size_t>(landpoints[i])] = static_cast<int>(i);
      }
      const int projection_k = std::max(1, options.knn.k);
      scratch.projection_votes.clear();
      scratch.projection_votes.reserve(static_cast<std::size_t>(projection_k));
      for (std::size_t i = 0; i < tpoints.size(); ++i) {
        const std::size_t row_offset = static_cast<std::size_t>(tpoints[i]) * static_cast<std::size_t>(global_graph.neighbors);
        scratch.projection_votes.clear();
        for (int j = 0; j < global_graph.neighbors && static_cast<int>(scratch.projection_votes.size()) < projection_k; ++j) {
          const int global_neighbor = global_graph.indices[row_offset + static_cast<std::size_t>(j)];
          if (global_neighbor < 0 || global_neighbor >= n) continue;
          const int local = scratch.global_to_local[static_cast<std::size_t>(global_neighbor)];
          if (local >= 0) scratch.projection_votes.push_back(core_result.clbest[static_cast<std::size_t>(local)]);
        }
        out.res[static_cast<std::size_t>(tpoints[i])] =
          scratch.projection_votes.empty() ? core_result.clbest.front() : majority_label(scratch.projection_votes);
      }
    } else {
      copy_float32_rows_into(full_float, full.cols, tpoints, scratch.x_test);
      MatrixView test_view{scratch.x_test.data(), tpoints.size(), full.cols};
      std::vector<int> projected;
      if (options.backend == Backend::CUDA) {
        projected = PLSLDAPredict_CUDA(x_view, core_result.clbest, test_view, core.pls);
      } else if (options.backend == Backend::Metal) {
        projected = PLSLDAPredict_METAL(x_view, core_result.clbest, test_view, core.pls);
      } else {
        projected = PLSLDAPredict_CPU(x_view, core_result.clbest, test_view, core.pls);
      }
      for (std::size_t i = 0; i < tpoints.size(); ++i) {
        out.res[static_cast<std::size_t>(tpoints[i])] = projected[i];
      }
    }
  }
  if (!run_constrain_is_identity) out.res = constrained_majority(out.res, run_constrain);
  out.acc = core_result.vect_acc;
  out.accbest = core_result.accbest;
  out.runtime = core_result.runtime_seconds;
  out.memory = core_result.peak_memory_mb;
  return out;
}

KODAMAMatrixResult run_kodama_matrix(
  MatrixView x,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain_in,
  const std::vector<int>& fixed_in,
  KODAMAMatrixOptions options,
  const NeighborGraph* input_graph = nullptr
) {
  detail::validate_inputs(x, std::vector<int>(x.rows, 1), std::vector<int>());
  if (!starting_labels.empty() && starting_labels.size() != x.rows) throw std::invalid_argument("starting_labels size must match number of rows.");
  if (x.rows < 3) throw std::invalid_argument("KODAMAMatrix requires at least 3 rows.");
  if (options.runs < 1) throw std::invalid_argument("KODAMAMatrixOptions::runs must be positive.");
  if (options.cycles < 0) throw std::invalid_argument("KODAMAMatrixOptions::cycles must be non-negative.");

  detail::Timer timer;
  if (options.landmarks <= 0) options.landmarks = 10000;
  if (static_cast<std::size_t>(options.landmarks) >= x.rows) {
    options.landmarks = static_cast<int>(std::ceil(static_cast<double>(x.rows) * 0.75));
  }
  options.landmarks = std::max(2, std::min(options.landmarks, static_cast<int>(x.rows) - 1));
  options.components = std::max(1, std::min(options.components, static_cast<int>(std::min(x.rows, x.cols))));
  const int requested_neighbors = options.graph_neighbors > 0 ? options.graph_neighbors : 100;
  GpuWorkerPlan worker_plan;
  if (options.backend == Backend::CUDA && options.n_threads <= 0) {
    worker_plan = resolve_gpu_worker_plan(
      options,
      static_cast<int>(x.rows),
      static_cast<int>(x.cols),
      options.landmarks,
      requested_neighbors
    );
    options.n_threads = worker_plan.workers;
    if (options.progress) {
      std::cerr << "[kodama] CUDA auto workers selected " << worker_plan.workers
                << " (SM=" << worker_plan.sm_count
                << ", free=" << worker_plan.free_memory_mb << " MiB"
                << ", per_worker_est=" << worker_plan.worker_memory_estimate_mb << " MiB)"
                << std::endl;
    }
  } else {
    options.n_threads = std::max(1, options.n_threads);
    worker_plan.workers = std::min(options.n_threads, options.runs);
  }
  if (!options.spatial.empty()) {
    if (options.spatial_cols <= 0) throw std::invalid_argument("KODAMAMatrixOptions::spatial_cols must be positive when spatial is provided.");
    if (options.spatial.size() != x.rows * static_cast<std::size_t>(options.spatial_cols)) {
      throw std::invalid_argument("KODAMAMatrixOptions::spatial size must be rows * spatial_cols.");
    }
    if (options.spatial_resolution <= 0.0 || !std::isfinite(options.spatial_resolution)) {
      throw std::invalid_argument("KODAMAMatrixOptions::spatial_resolution must be positive.");
    }
  }

  detail::Timer input_copy_timer;
  const std::vector<float> full_float = copy_float32(x);
  const std::vector<int> constrain = normalize_constrain(constrain_in, x.rows);
  const bool constrain_is_identity = is_identity_constrain(constrain);
  const std::vector<int> fixed = normalize_fixed(fixed_in, x.rows);
  const double input_copy_seconds = input_copy_timer.seconds();

  detail::Timer spatial_precompute_timer;
  const std::vector<float> spatial_jitter = options.spatial.empty() ?
    std::vector<float>() :
    spatial_jitter_from_graph(
      options.spatial,
      static_cast<int>(x.rows),
      options.spatial_cols,
      std::max(20, options.graph_neighbors > 0 ? options.graph_neighbors : 100),
      options.n_threads,
      options.backend,
      options.knn.gpu_device
    );
  const double spatial_precompute_seconds = spatial_precompute_timer.seconds();
  const int neighbors = std::max(1, static_cast<int>(std::floor(std::min({
    static_cast<double>(options.landmarks),
    static_cast<double>(x.rows) * 0.75 - 1.0,
    static_cast<double>(requested_neighbors)
  }))));

  KODAMAMatrixResult result;
  result.runs = options.runs;
  result.samples = static_cast<int>(x.rows);
  result.cycles = options.cycles;
  result.n_threads = options.n_threads;
  result.backend = options.backend;
  result.gpu_auto_workers = worker_plan.automatic;
  result.gpu_scheduler_enabled = options.backend == Backend::CUDA;
  result.gpu_scheduler_lanes = options.backend == Backend::CUDA ? worker_plan.workers : 0;
  result.gpu_sm_count = worker_plan.sm_count;
  result.gpu_free_memory_mb = worker_plan.free_memory_mb;
  result.gpu_total_memory_mb = worker_plan.total_memory_mb;
  result.gpu_worker_memory_estimate_mb = worker_plan.worker_memory_estimate_mb;
  result.input_copy_seconds = input_copy_seconds;
  result.spatial_precompute_seconds = spatial_precompute_seconds;
  result.acc.assign(static_cast<std::size_t>(options.runs), std::numeric_limits<double>::quiet_NaN());
  result.v.assign(static_cast<std::size_t>(options.runs) * options.cycles, std::numeric_limits<double>::quiet_NaN());
  result.res.assign(static_cast<std::size_t>(options.runs) * x.rows, 0);
  result.res_constrain.assign(static_cast<std::size_t>(options.runs) * x.rows, 0);
  detail::Timer graph_timer;
  NeighborGraph global_graph;
  const bool graph_is_input = input_graph != nullptr;
  if (graph_is_input) {
    if (options.progress) {
      std::cerr << "[kodama] using caller-supplied KNN graph with "
                << input_graph->neighbors << " neighbors" << std::endl;
    }
    const int retained = std::max(
      input_graph->neighbors,
      std::max(neighbors, std::max(1, options.knn.k))
    );
    global_graph = normalize_external_graph(*input_graph, static_cast<int>(x.rows), retained);
  } else {
    if (options.progress) {
      std::cerr << "[kodama] building global KNN graph for " << x.rows
                << " samples and " << neighbors << " neighbors" << std::endl;
    }
    global_graph = self_knn_graph(
      full_float,
      static_cast<int>(x.rows),
      static_cast<int>(x.cols),
      neighbors + 1,
      options.metric,
      options.n_threads,
      options.backend,
      options.knn.gpu_device,
      true,
      options.knn.index_type,
      options.knn.ivf_nlist,
      options.knn.ivf_nprobe
    );
  }
  result.graph_seconds = graph_timer.seconds();

  result.knn = trim_self_neighbors(global_graph, static_cast<int>(x.rows), std::min(neighbors, global_graph.neighbors));
  if (!options.spatial.empty() && options.spatial_graph_mix) {
    if (options.progress) {
      std::cerr << "[kodama] building spatial KNN graph for final KODAMA graph" << std::endl;
    }
    detail::Timer spatial_graph_timer;
    const NeighborGraph spatial_global_graph = self_knn_graph(
      options.spatial,
      static_cast<int>(x.rows),
      options.spatial_cols,
      neighbors + 1,
      DistanceMetric::Euclidean,
      options.n_threads,
      options.backend,
      options.knn.gpu_device,
      true,
      options.knn.index_type,
      options.knn.ivf_nlist,
      options.knn.ivf_nprobe
    );
    NeighborGraph spatial_trimmed = trim_self_neighbors(spatial_global_graph, static_cast<int>(x.rows), neighbors);
    result.knn = merge_feature_spatial_graphs(result.knn, spatial_trimmed, static_cast<int>(x.rows), neighbors);
    result.spatial_graph_seconds = spatial_graph_timer.seconds();
  }
  result.base_knn = result.knn;

  detail::Timer optimization_timer;
  const int workers = std::max(1, std::min(options.n_threads, options.runs));
  std::vector<IterationResult> iterations(static_cast<std::size_t>(options.runs));

  auto execute_run = [&](int run_id, IterationScratch& scratch) {
    if (options.progress) {
      std::cerr << "[kodama] launch M " << run_id << "/" << options.runs << std::endl;
    }
    IterationResult iter = run_iteration(
      x,
      full_float,
      options.spatial,
      spatial_jitter,
      global_graph,
      graph_is_input,
      constrain,
      constrain_is_identity,
      fixed,
      starting_labels,
      options,
      run_id,
      scratch
    );
    if (options.progress) {
      std::cerr << "[kodama] complete M " << run_id << "/" << options.runs
                << " acc=" << iter.accbest
                << " elapsed=" << timer.seconds() << "s" << std::endl;
    }
    iterations[static_cast<std::size_t>(run_id - 1)] = std::move(iter);
  };

#ifdef KODAMA_ENABLE_CUDA
  if (options.backend == Backend::CUDA) {
    CudaMScheduler scheduler(workers, options.knn.gpu_device);
    if (options.progress) {
      std::cerr << "[kodama] CUDA M scheduler using " << scheduler.lanes()
                << " independent lanes" << std::endl;
    }
    scheduler.run(options.runs, [&](int run_id, int, IterationScratch& scratch) {
      execute_run(run_id, scratch);
    });
  } else
#endif
  {
    std::atomic<int> next_run{1};
    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<std::size_t>(workers));
    for (int worker = 0; worker < workers; ++worker) {
      futures.emplace_back(std::async(std::launch::async, [&]() {
        IterationScratch scratch;
        while (true) {
          const int run_id = next_run.fetch_add(1);
          if (run_id > options.runs) break;
          execute_run(run_id, scratch);
        }
      }));
    }
    for (auto& future : futures) future.get();
  }
  result.optimization_wall_seconds = optimization_timer.seconds();

  for (int run_id = 1; run_id <= options.runs; ++run_id) {
    const IterationResult& iter = iterations[static_cast<std::size_t>(run_id - 1)];
    const std::size_t row = static_cast<std::size_t>(run_id - 1);
    result.acc[row] = iter.accbest;
    std::copy(iter.res.begin(), iter.res.end(), result.res.begin() + row * x.rows);
    std::copy(iter.constrain.begin(), iter.constrain.end(), result.res_constrain.begin() + row * x.rows);
    for (int c = 0; c < options.cycles && c < static_cast<int>(iter.acc.size()); ++c) {
      result.v[row * static_cast<std::size_t>(options.cycles) + static_cast<std::size_t>(c)] = iter.acc[static_cast<std::size_t>(c)];
    }
    result.optimization_sum_seconds += iter.runtime;
    result.peak_memory_mb = std::max(result.peak_memory_mb, iter.memory);
  }

  if (options.apply_kodama_dissimilarity) {
    if (options.progress) {
      std::cerr << "[kodama] applying KODAMA dissimilarity to KNN graph" << std::endl;
    }
    detail::Timer dissimilarity_timer;
#if defined(KODAMA_ENABLE_CUDA)
    if (options.backend == Backend::CUDA) {
      detail::apply_kodama_dissimilarity_cuda(result.knn, result.res, result.runs, result.samples, options.knn.gpu_device, true);
    } else
#endif
    {
      apply_kodama_dissimilarity(result.knn, result.res, result.runs, result.samples, options.n_threads);
      for (std::size_t i = 0; i < result.knn.indices.size(); ++i) {
        if (result.knn.indices[i] >= 0) result.knn.indices[i] += 1;
      }
    }
    result.dissimilarity_seconds = dissimilarity_timer.seconds();
  } else if (options.progress) {
    std::cerr << "[kodama] skipping final KODAMA dissimilarity for graph+labels output" << std::endl;
  }
  result.runtime_seconds = timer.seconds();
  result.peak_memory_mb = std::max(result.peak_memory_mb, detail::peak_memory_mb());
  if (options.progress) {
    std::cerr << "[kodama] finished KODAMA.matrix in " << result.runtime_seconds << "s" << std::endl;
  }
  return result;
}

}  // namespace

KODAMAMatrixResult KODAMAMatrix_CPU(
  MatrixView x,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  KODAMAMatrixOptions cpu_options = options;
  cpu_options.backend = Backend::CPU;
  return run_kodama_matrix(x, starting_labels, constrain, fixed, cpu_options);
}

KODAMAMatrixResult KODAMAMatrix_CUDA(
  MatrixView x,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  KODAMAMatrixOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  return run_kodama_matrix(x, starting_labels, constrain, fixed, cuda_options);
#else
  (void)x;
  (void)starting_labels;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("KODAMAMatrix_CUDA requires a CUDA build.");
#endif
}

KODAMAMatrixResult KODAMAMatrix_METAL(
  MatrixView x,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  KODAMAMatrixOptions metal_options = options;
  metal_options.backend = Backend::Metal;
  metal_options.knn.backend = Backend::Metal;
  if (metal_options.knn.index_type != KNNIndexType::MetalIVFFlat) {
    metal_options.knn.index_type = KNNIndexType::MetalExact;
  }
  metal_options.pls.backend = Backend::Metal;
  return run_kodama_matrix(x, starting_labels, constrain, fixed, metal_options);
#else
  (void)x;
  (void)starting_labels;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("KODAMAMatrix_METAL requires an Apple Metal build.");
#endif
}

KODAMAMatrixResult KODAMAMatrix(
  MatrixView x,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  if (options.backend == Backend::CUDA) return KODAMAMatrix_CUDA(x, starting_labels, constrain, fixed, options);
  if (options.backend == Backend::Metal) return KODAMAMatrix_METAL(x, starting_labels, constrain, fixed, options);
  return KODAMAMatrix_CPU(x, starting_labels, constrain, fixed, options);
}

std::vector<float> KODAMAGraphFeatures_CPU(
  const NeighborGraph& graph,
  int samples,
  const KODAMAMatrixOptions& options
) {
  const int components = std::max(
    1,
    options.graph_feature_components > 0 ? options.graph_feature_components : std::max(1, options.components)
  );
  return graph_laplacian_operator_features(
    graph,
    samples,
    components,
    std::max(8, options.graph_feature_steps),
    options.seed,
    options.n_threads
  );
}

KODAMAMatrixResult KODAMAMatrixFromGraph_CPU(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  KODAMAMatrixOptions cpu_options = options;
  cpu_options.backend = Backend::CPU;
  detail::Timer feature_timer;
  std::vector<float> features = KODAMAGraphFeatures_CPU(graph, samples, cpu_options);
  const double graph_feature_seconds = feature_timer.seconds();
  const int components = static_cast<int>(features.size() / static_cast<std::size_t>(samples));
  MatrixView view{features.data(), static_cast<std::size_t>(samples), static_cast<std::size_t>(components)};
  KODAMAMatrixResult result = run_kodama_matrix(view, starting_labels, constrain, fixed, cpu_options, &graph);
  result.graph_feature_seconds = graph_feature_seconds;
  return result;
}

KODAMAMatrixResult KODAMAMatrixFromGraphData_CPU(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  if (x.rows < 2) throw std::invalid_argument("KODAMAMatrixFromGraphData requires at least two rows.");
  KODAMAMatrixOptions cpu_options = options;
  cpu_options.backend = Backend::CPU;
  KODAMAMatrixResult result = run_kodama_matrix(x, starting_labels, constrain, fixed, cpu_options, &graph);
  result.graph_feature_seconds = 0.0;
  return result;
}

KODAMAMatrixResult KODAMAMatrixFromGraphData_CUDA(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  if (x.rows < 2) throw std::invalid_argument("KODAMAMatrixFromGraphData requires at least two rows.");
  KODAMAMatrixOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  KODAMAMatrixResult result = run_kodama_matrix(x, starting_labels, constrain, fixed, cuda_options, &graph);
  result.graph_feature_seconds = 0.0;
  return result;
#else
  (void)x;
  (void)graph;
  (void)starting_labels;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("KODAMAMatrixFromGraphData_CUDA requires a CUDA build.");
#endif
}

KODAMAMatrixResult KODAMAMatrixFromGraphData_METAL(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  if (x.rows < 2) throw std::invalid_argument("KODAMAMatrixFromGraphData requires at least two rows.");
  KODAMAMatrixOptions metal_options = options;
  metal_options.backend = Backend::Metal;
  metal_options.knn.backend = Backend::Metal;
  if (metal_options.knn.index_type != KNNIndexType::MetalIVFFlat) {
    metal_options.knn.index_type = KNNIndexType::MetalExact;
  }
  metal_options.pls.backend = Backend::Metal;
  KODAMAMatrixResult result = run_kodama_matrix(x, starting_labels, constrain, fixed, metal_options, &graph);
  result.graph_feature_seconds = 0.0;
  return result;
#else
  (void)x;
  (void)graph;
  (void)starting_labels;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("KODAMAMatrixFromGraphData_METAL requires an Apple Metal build.");
#endif
}

KODAMAMatrixResult KODAMAMatrixFromGraphData(
  MatrixView x,
  const NeighborGraph& graph,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  if (options.backend == Backend::CUDA) return KODAMAMatrixFromGraphData_CUDA(x, graph, starting_labels, constrain, fixed, options);
  if (options.backend == Backend::Metal) return KODAMAMatrixFromGraphData_METAL(x, graph, starting_labels, constrain, fixed, options);
  return KODAMAMatrixFromGraphData_CPU(x, graph, starting_labels, constrain, fixed, options);
}

KODAMAMatrixResult KODAMAMatrixFromGraph_CUDA(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  KODAMAMatrixOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  detail::Timer feature_timer;
  std::vector<float> features = KODAMAGraphFeatures_CPU(graph, samples, cuda_options);
  const double graph_feature_seconds = feature_timer.seconds();
  const int components = static_cast<int>(features.size() / static_cast<std::size_t>(samples));
  MatrixView view{features.data(), static_cast<std::size_t>(samples), static_cast<std::size_t>(components)};
  KODAMAMatrixResult result = run_kodama_matrix(view, starting_labels, constrain, fixed, cuda_options, &graph);
  result.graph_feature_seconds = graph_feature_seconds;
  return result;
#else
  (void)graph;
  (void)samples;
  (void)starting_labels;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("KODAMAMatrixFromGraph_CUDA requires a CUDA build.");
#endif
}

KODAMAMatrixResult KODAMAMatrixFromGraph_METAL(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
  KODAMAMatrixOptions metal_options = options;
  metal_options.backend = Backend::Metal;
  metal_options.knn.backend = Backend::Metal;
  if (metal_options.knn.index_type != KNNIndexType::MetalIVFFlat) {
    metal_options.knn.index_type = KNNIndexType::MetalExact;
  }
  metal_options.pls.backend = Backend::Metal;
  detail::Timer feature_timer;
  std::vector<float> features = KODAMAGraphFeatures_CPU(graph, samples, metal_options);
  const double graph_feature_seconds = feature_timer.seconds();
  const int components = static_cast<int>(features.size() / static_cast<std::size_t>(samples));
  MatrixView view{features.data(), static_cast<std::size_t>(samples), static_cast<std::size_t>(components)};
  KODAMAMatrixResult result = run_kodama_matrix(view, starting_labels, constrain, fixed, metal_options, &graph);
  result.graph_feature_seconds = graph_feature_seconds;
  return result;
#else
  (void)graph;
  (void)samples;
  (void)starting_labels;
  (void)constrain;
  (void)fixed;
  (void)options;
  throw std::runtime_error("KODAMAMatrixFromGraph_METAL requires an Apple Metal build.");
#endif
}

KODAMAMatrixResult KODAMAMatrixFromGraph(
  const NeighborGraph& graph,
  int samples,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  if (options.backend == Backend::CUDA) return KODAMAMatrixFromGraph_CUDA(graph, samples, starting_labels, constrain, fixed, options);
  if (options.backend == Backend::Metal) return KODAMAMatrixFromGraph_METAL(graph, samples, starting_labels, constrain, fixed, options);
  return KODAMAMatrixFromGraph_CPU(graph, samples, starting_labels, constrain, fixed, options);
}

}  // namespace kodama
