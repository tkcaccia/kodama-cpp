#include "common.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>

#ifdef KODAMA_ENABLE_CUDA
#include "kodama_matrix_cuda.hpp"

#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
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
      out[i * x.cols + j] = static_cast<float>(x(src, j));
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
  int n_threads
) {
  if (n_base < 1 || n_query < 1 || dim < 1) throw std::invalid_argument("HNSW graph input is empty.");
  neighbors = std::max(1, std::min(neighbors, n_base));
  std::vector<float> xb = base;
  std::vector<float> xq = query;
  const faiss::MetricType faiss_metric = metric == DistanceMetric::Euclidean ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
  if (metric == DistanceMetric::Cosine) {
    normalize_rows_for_cosine(xb, static_cast<std::size_t>(n_base), static_cast<std::size_t>(dim));
    normalize_rows_for_cosine(xq, static_cast<std::size_t>(n_query), static_cast<std::size_t>(dim));
  }

  const int m = std::min(32, std::max(1, n_base > 1 ? n_base - 1 : 1));
  faiss::IndexHNSWFlat index(dim, m, faiss_metric);
  index.hnsw.efConstruction = std::max(200, m);
  index.hnsw.efSearch = std::max(150, neighbors);
  index.add(n_base, xb.data());

  NeighborGraph graph;
  graph.neighbors = neighbors;
  graph.indices.assign(static_cast<std::size_t>(n_query) * neighbors, -1);
  graph.distances.assign(static_cast<std::size_t>(n_query) * neighbors, std::numeric_limits<float>::infinity());
  std::vector<faiss::idx_t> idx(static_cast<std::size_t>(n_query) * neighbors, -1);
  OmpThreadScope threads(n_threads);
  index.search(n_query, xq.data(), neighbors, graph.distances.data(), idx.data());
  for (std::size_t i = 0; i < idx.size(); ++i) graph.indices[i] = static_cast<int>(idx[i]);
  if (metric == DistanceMetric::Cosine || metric == DistanceMetric::InnerProduct) {
    for (float& d : graph.distances) d = 1.0f - d;
  } else {
    for (float& d : graph.distances) d = d > 0.0f ? std::sqrt(d) : 0.0f;
  }
  return graph;
}

#ifdef KODAMA_ENABLE_CUDA
NeighborGraph faiss_gpu_flat_graph(
  const std::vector<float>& base,
  const std::vector<float>& query,
  int n_base,
  int n_query,
  int dim,
  int neighbors,
  DistanceMetric metric,
  int gpu_device
) {
  if (n_base < 1 || n_query < 1 || dim < 1) throw std::invalid_argument("CUDA graph input is empty.");
  neighbors = std::max(1, std::min(neighbors, n_base));
  std::vector<float> xb = base;
  std::vector<float> xq = query;
  const faiss::MetricType faiss_metric = metric == DistanceMetric::Euclidean ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
  if (metric == DistanceMetric::Cosine) {
    normalize_rows_for_cosine(xb, static_cast<std::size_t>(n_base), static_cast<std::size_t>(dim));
    normalize_rows_for_cosine(xq, static_cast<std::size_t>(n_query), static_cast<std::size_t>(dim));
  }

  faiss::gpu::StandardGpuResources resources;
  faiss::gpu::GpuIndexFlatConfig config;
  config.device = gpu_device;
  faiss::gpu::GpuIndexFlat index(&resources, dim, faiss_metric, config);
  index.add(n_base, xb.data());

  NeighborGraph graph;
  graph.neighbors = neighbors;
  graph.indices.assign(static_cast<std::size_t>(n_query) * neighbors, -1);
  graph.distances.assign(static_cast<std::size_t>(n_query) * neighbors, std::numeric_limits<float>::infinity());
  std::vector<faiss::idx_t> idx(static_cast<std::size_t>(n_query) * neighbors, -1);
  index.search(n_query, xq.data(), neighbors, graph.distances.data(), idx.data());
  for (std::size_t i = 0; i < idx.size(); ++i) graph.indices[i] = static_cast<int>(idx[i]);
  if (metric == DistanceMetric::Cosine || metric == DistanceMetric::InnerProduct) {
    for (float& d : graph.distances) d = 1.0f - d;
  } else {
    for (float& d : graph.distances) d = d > 0.0f ? std::sqrt(d) : 0.0f;
  }
  return graph;
}
#endif

std::vector<int> kmeans_labels(
  const std::vector<float>& x,
  int n,
  int p,
  int k,
  std::mt19937_64& rng,
  int max_iter = 10,
  int n_threads = 1,
  bool use_gpu = false,
  int gpu_device = 0
) {
  k = std::max(1, std::min(k, n));

  faiss::ClusteringParameters cp;
  cp.niter = std::max(1, max_iter);
  cp.nredo = 1;
  cp.verbose = false;
  cp.spherical = false;
  cp.seed = static_cast<int>(rng() & 0x7fffffffULL);
  cp.min_points_per_centroid = 1;
  cp.max_points_per_centroid = std::max(
    256,
    static_cast<int>((static_cast<long long>(n) + k - 1) / k)
  );
  std::vector<float> distances(static_cast<std::size_t>(n), 0.0f);
  std::vector<faiss::idx_t> idx(static_cast<std::size_t>(n), -1);
  auto train_and_assign = [&](faiss::Index& train_index, faiss::Index& assign_index) {
    faiss::Clustering clustering(p, k, cp);
    clustering.train(n, x.data(), train_index);
    assign_index.add(k, clustering.centroids.data());
    assign_index.search(n, x.data(), 1, distances.data(), idx.data());
  };

#ifdef KODAMA_ENABLE_CUDA
  if (use_gpu) {
    thread_local std::unique_ptr<faiss::gpu::StandardGpuResources> resources;
    thread_local int resources_device = -1;
    if (!resources || resources_device != gpu_device) {
      resources = std::make_unique<faiss::gpu::StandardGpuResources>();
      resources_device = gpu_device;
    }
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = gpu_device;
    faiss::gpu::GpuIndexFlatL2 train_index(resources.get(), p, config);
    faiss::gpu::GpuIndexFlatL2 assign_index(resources.get(), p, config);
    train_and_assign(train_index, assign_index);
  } else
#else
  (void)use_gpu;
  (void)gpu_device;
#endif
  {
    OmpThreadScope threads(n_threads);
    faiss::IndexFlatL2 train_index(p);
    faiss::IndexFlatL2 assign_index(p);
    train_and_assign(train_index, assign_index);
  }

  std::vector<int> labels(static_cast<std::size_t>(n), 1);
  for (int i = 0; i < n; ++i) {
    const faiss::idx_t label = idx[static_cast<std::size_t>(i)];
    if (label < 0 || label >= k) {
      throw std::runtime_error("FAISS k-means returned an invalid cluster label.");
    }
    labels[static_cast<std::size_t>(i)] = static_cast<int>(label) + 1;
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
  bool use_gpu,
  int gpu_device
) {
  target_components = std::max(1, std::min(target_components, n));
  if (target_components >= n) {
    std::vector<int> out(static_cast<std::size_t>(n));
    std::iota(out.begin(), out.end(), 1);
    return out;
  }
  const int k = std::max(2, std::min(n, 32));
  const NeighborGraph graph =
#if defined(KODAMA_ENABLE_CUDA)
    use_gpu ?
    faiss_gpu_flat_graph(spatial, spatial, n, n, dims, k, DistanceMetric::Euclidean, gpu_device) :
#endif
    hnsw_graph(spatial, spatial, n, n, dims, k, DistanceMetric::Euclidean, n_threads);
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
  int n_threads
) {
  const int k = std::max(1, std::min(neighbors, n));
  NeighborGraph graph = hnsw_graph(spatial, spatial, n, n, dims, k, DistanceMetric::Euclidean, n_threads);
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
  const int landmarks = std::max(2, std::min(options.landmarks, n - 1));
  const int splitting = options.splitting > 0 ? options.splitting : (n < 40000 ? 100 : 300);
  const bool spatial_flag = !spatial.empty() && options.spatial_cols > 0;
  std::mt19937_64 rng(options.seed + static_cast<std::uint64_t>(run_id));

  detail::Timer iter_timer;
  if (options.progress) {
    std::cerr << "[kodama] M " << run_id << "/" << options.runs
              << " landmark k-means with " << landmarks << " centers" << std::endl;
  }
  const bool kmeans_on_gpu = options.backend == Backend::CUDA;
  const int kmeans_gpu_device = options.knn.gpu_device;
  const std::vector<int> landmark_clusters = kmeans_labels(
    full_float,
    n,
    p,
    landmarks,
    rng,
    10,
    options.n_threads,
    kmeans_on_gpu,
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
        kmeans_on_gpu,
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
        kmeans_on_gpu,
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
  } else if (landmarks < 200) {
    scratch.xw = scratch.x_constrain;
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
      kmeans_on_gpu,
      kmeans_gpu_device
    );
    scratch.xw = run_constrain_is_identity ? scratch.init : constrained_majority(scratch.init, scratch.x_constrain);
  }

  MatrixView x_view{scratch.x_land.data(), landpoints.size(), full.cols};
  CoreOptions core;
  core.cycles = options.cycles;
  core.seed = options.seed + static_cast<std::uint64_t>(run_id);
  core.classifier = options.classifier;
  core.knn = options.knn;
  core.knn.backend = options.backend == Backend::CUDA ? Backend::CUDA : Backend::CPU;
  core.knn.metric = options.metric;
  core.knn.cv.stratified = false;
  core.knn.cv.seed = options.seed + static_cast<std::uint64_t>(run_id);
  core.knn.k = std::max(core.knn.k, 1);
  core.knn.hnsw_tune_k = 50;
  core.knn.hnsw_target_recall = 0.99;
  core.knn.n_threads = options.backend == Backend::CUDA ?
    options.n_threads : std::max(1, options.knn.n_threads);
  core.pls = options.pls;
  core.pls.backend = options.backend == Backend::CUDA ? Backend::CUDA : Backend::CPU;
  core.pls.cv.stratified = false;
  core.pls.cv.seed = options.seed + static_cast<std::uint64_t>(run_id);
  core.pls.max_components = options.components;
  core.pls.fixed_components = options.components;
  core.pls.n_threads = options.backend == Backend::CUDA ?
    options.n_threads : std::max(1, options.pls.n_threads);

  CoreResult core_result;
  if (options.classifier == CoreClassifier::KNN) {
    core_result = options.backend == Backend::CUDA ?
      CoreKNN_CUDA(x_view, scratch.xw, scratch.x_constrain, scratch.x_fixed, core) :
      CoreKNN_CPU(x_view, scratch.xw, scratch.x_constrain, scratch.x_fixed, core);
  } else {
    core_result = options.backend == Backend::CUDA ?
      CorePLSLDA_CUDA(x_view, scratch.xw, scratch.x_constrain, scratch.x_fixed, core) :
      CorePLSLDA_CPU(x_view, scratch.xw, scratch.x_constrain, scratch.x_fixed, core);
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
      std::vector<int> projected = options.backend == Backend::CUDA ?
        PLSLDAPredict_CUDA(x_view, core_result.clbest, test_view, core.pls) :
        PLSLDAPredict_CPU(x_view, core_result.clbest, test_view, core.pls);
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
  KODAMAMatrixOptions options
) {
  detail::validate_inputs(x, std::vector<int>(x.rows, 1), std::vector<int>());
  if (!starting_labels.empty() && starting_labels.size() != x.rows) throw std::invalid_argument("starting_labels size must match number of rows.");
  if (x.rows < 3) throw std::invalid_argument("KODAMAMatrix requires at least 3 rows.");
  if (options.runs < 1) throw std::invalid_argument("KODAMAMatrixOptions::runs must be positive.");
  if (options.cycles < 0) throw std::invalid_argument("KODAMAMatrixOptions::cycles must be non-negative.");

  detail::Timer timer;
  options.n_threads = std::max(1, options.n_threads);
  if (options.landmarks <= 0) options.landmarks = 10000;
  if (static_cast<std::size_t>(options.landmarks) >= x.rows) {
    options.landmarks = static_cast<int>(std::ceil(static_cast<double>(x.rows) * 0.75));
  }
  options.landmarks = std::max(2, std::min(options.landmarks, static_cast<int>(x.rows) - 1));
  options.components = std::max(1, std::min(options.components, static_cast<int>(std::min(x.rows, x.cols))));
  if (!options.spatial.empty()) {
    if (options.spatial_cols <= 0) throw std::invalid_argument("KODAMAMatrixOptions::spatial_cols must be positive when spatial is provided.");
    if (options.spatial.size() != x.rows * static_cast<std::size_t>(options.spatial_cols)) {
      throw std::invalid_argument("KODAMAMatrixOptions::spatial size must be rows * spatial_cols.");
    }
    if (options.spatial_resolution <= 0.0 || !std::isfinite(options.spatial_resolution)) {
      throw std::invalid_argument("KODAMAMatrixOptions::spatial_resolution must be positive.");
    }
  }

  const std::vector<float> full_float = copy_float32(x);
  const std::vector<int> constrain = normalize_constrain(constrain_in, x.rows);
  const bool constrain_is_identity = is_identity_constrain(constrain);
  const std::vector<int> fixed = normalize_fixed(fixed_in, x.rows);
  const std::vector<float> spatial_jitter = options.spatial.empty() ?
    std::vector<float>() :
    spatial_jitter_from_graph(
      options.spatial,
      static_cast<int>(x.rows),
      options.spatial_cols,
      std::max(20, options.graph_neighbors > 0 ? options.graph_neighbors : 100),
      options.n_threads
    );
  const int requested_neighbors = options.graph_neighbors > 0 ? options.graph_neighbors : 100;
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
  result.acc.assign(static_cast<std::size_t>(options.runs), std::numeric_limits<double>::quiet_NaN());
  result.v.assign(static_cast<std::size_t>(options.runs) * options.cycles, std::numeric_limits<double>::quiet_NaN());
  result.res.assign(static_cast<std::size_t>(options.runs) * x.rows, 0);
  result.res_constrain.assign(static_cast<std::size_t>(options.runs) * x.rows, 0);
  if (options.progress) {
    std::cerr << "[kodama] building global KNN graph for " << x.rows
              << " samples and " << neighbors << " neighbors" << std::endl;
  }
  const NeighborGraph global_graph =
#if defined(KODAMA_ENABLE_CUDA)
    options.backend == Backend::CUDA ?
    faiss_gpu_flat_graph(
      full_float,
      full_float,
      static_cast<int>(x.rows),
      static_cast<int>(x.rows),
      static_cast<int>(x.cols),
      neighbors + 1,
      options.metric,
      options.knn.gpu_device
    ) :
#endif
    hnsw_graph(
      full_float,
      full_float,
      static_cast<int>(x.rows),
      static_cast<int>(x.rows),
      static_cast<int>(x.cols),
      neighbors + 1,
      options.metric,
      options.n_threads
    );

  result.knn = trim_self_neighbors(global_graph, static_cast<int>(x.rows), neighbors);
  if (!options.spatial.empty() && options.spatial_graph_mix) {
    if (options.progress) {
      std::cerr << "[kodama] building spatial KNN graph for final KODAMA graph" << std::endl;
    }
    const NeighborGraph spatial_global_graph =
#if defined(KODAMA_ENABLE_CUDA)
      options.backend == Backend::CUDA ?
      faiss_gpu_flat_graph(
        options.spatial,
        options.spatial,
        static_cast<int>(x.rows),
        static_cast<int>(x.rows),
        options.spatial_cols,
        neighbors + 1,
        DistanceMetric::Euclidean,
        options.knn.gpu_device
      ) :
#endif
      hnsw_graph(
        options.spatial,
        options.spatial,
        static_cast<int>(x.rows),
        static_cast<int>(x.rows),
        options.spatial_cols,
        neighbors + 1,
        DistanceMetric::Euclidean,
        options.n_threads
      );
    NeighborGraph spatial_trimmed = trim_self_neighbors(spatial_global_graph, static_cast<int>(x.rows), neighbors);
    result.knn = merge_feature_spatial_graphs(result.knn, spatial_trimmed, static_cast<int>(x.rows), neighbors);
  }
  result.base_knn = result.knn;

  const int workers = std::max(1, std::min(options.n_threads, options.runs));
  std::atomic<int> next_run{1};
  std::vector<IterationResult> iterations(static_cast<std::size_t>(options.runs));
  std::vector<std::future<void>> futures;
  futures.reserve(static_cast<std::size_t>(workers));
  for (int worker = 0; worker < workers; ++worker) {
    futures.emplace_back(std::async(std::launch::async, [&]() {
      IterationScratch scratch;
      while (true) {
        const int run_id = next_run.fetch_add(1);
        if (run_id > options.runs) break;
        if (options.progress) {
          std::cerr << "[kodama] launch M " << run_id << "/" << options.runs << std::endl;
        }
        IterationResult iter = run_iteration(
          x,
          full_float,
          options.spatial,
          spatial_jitter,
          global_graph,
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
      }
    }));
  }
  for (auto& future : futures) future.get();

  for (int run_id = 1; run_id <= options.runs; ++run_id) {
    const IterationResult& iter = iterations[static_cast<std::size_t>(run_id - 1)];
    const std::size_t row = static_cast<std::size_t>(run_id - 1);
    result.acc[row] = iter.accbest;
    std::copy(iter.res.begin(), iter.res.end(), result.res.begin() + row * x.rows);
    std::copy(iter.constrain.begin(), iter.constrain.end(), result.res_constrain.begin() + row * x.rows);
    for (int c = 0; c < options.cycles && c < static_cast<int>(iter.acc.size()); ++c) {
      result.v[row * static_cast<std::size_t>(options.cycles) + static_cast<std::size_t>(c)] = iter.acc[static_cast<std::size_t>(c)];
    }
    result.runtime_seconds += iter.runtime;
    result.peak_memory_mb = std::max(result.peak_memory_mb, iter.memory);
  }

  if (options.apply_kodama_dissimilarity) {
    if (options.progress) {
      std::cerr << "[kodama] applying KODAMA dissimilarity to KNN graph" << std::endl;
    }
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

KODAMAMatrixResult KODAMAMatrix(
  MatrixView x,
  const std::vector<int>& starting_labels,
  const std::vector<int>& constrain,
  const std::vector<int>& fixed,
  const KODAMAMatrixOptions& options
) {
  if (options.backend == Backend::CUDA) return KODAMAMatrix_CUDA(x, starting_labels, constrain, fixed, options);
  return KODAMAMatrix_CPU(x, starting_labels, constrain, fixed, options);
}

}  // namespace kodama
