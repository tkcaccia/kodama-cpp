// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"
#include "metal_backend.hpp"
#include "native_cuda_backend.hpp"
#include "native_knn.hpp"
#include "spatial_grid_knn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef KODAMA_ENABLE_CUDA
#include "kodama_matrix_cuda.hpp"
#endif

namespace kodama {
namespace {

struct Timer {
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }
};

std::vector<float> copy_float32(MatrixView x) {
  std::vector<float> out(x.rows * x.cols);
  for (std::size_t i = 0; i < x.rows; ++i) {
    for (std::size_t j = 0; j < x.cols; ++j) out[i * x.cols + j] = x.value_float(i, j);
  }
  return out;
}

#if defined(KODAMA_ENABLE_CUDA)
void normalize_rows(std::vector<float>& x, std::size_t rows, std::size_t cols) {
  for (std::size_t i = 0; i < rows; ++i) {
    double norm2 = 0.0;
    for (std::size_t j = 0; j < cols; ++j) {
      const double v = x[i * cols + j];
      norm2 += v * v;
    }
    const double norm = std::sqrt(norm2);
    if (norm <= 0.0) continue;
    for (std::size_t j = 0; j < cols; ++j) x[i * cols + j] = static_cast<float>(x[i * cols + j] / norm);
  }
}
#endif

NeighborGraph build_hnsw_graph(MatrixView x, const GraphClusterOptions& options) {
  if (x.rows < 2 || x.cols < 1) throw std::invalid_argument("KODAMAKNNGraph requires at least two rows.");
  const int n = static_cast<int>(x.rows);
  const int d = static_cast<int>(x.cols);
  const int k = std::max(1, std::min(options.k, n - 1));
  const std::vector<float> data = detail::prepare_native_matrix(x, options.metric);
  std::vector<int> self_indices(static_cast<std::size_t>(n));
  std::iota(self_indices.begin(), self_indices.end(), 0);
  const detail::NativeKNNResult search = detail::native_hnsw_search(
    data,
    n,
    data,
    n,
    d,
    k,
    options.metric,
    detail::NativeHNSWParameters{
      std::min(32, std::max(2, n - 1)),
      200,
      std::max(150, k + 1)
    },
    options.n_threads,
    self_indices
  );

  NeighborGraph out;
  out.neighbors = k;
  out.index_base = GraphIndexBase::One;
  out.indices.assign(static_cast<std::size_t>(n) * k, -1);
  out.distances.assign(static_cast<std::size_t>(n) * k, std::numeric_limits<float>::infinity());
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < search.neighbors; ++j) {
      const std::size_t pos = static_cast<std::size_t>(i) * static_cast<std::size_t>(search.neighbors) + static_cast<std::size_t>(j);
      const int nb = search.indices[pos];
      if (nb < 0 || nb == i) continue;
      const std::size_t off = static_cast<std::size_t>(i) * static_cast<std::size_t>(k) + static_cast<std::size_t>(j);
      out.indices[off] = nb + 1;
      out.distances[off] = detail::native_knn_output_distance(search.distances[pos], options.metric);
    }
  }
  return out;
}

#if defined(KODAMA_ENABLE_METAL)
NeighborGraph build_metal_graph(MatrixView x, const GraphClusterOptions& options) {
  if (x.rows < 2 || x.cols < 1) throw std::invalid_argument("KODAMAKNNGraph requires at least two rows.");
  const int n = static_cast<int>(x.rows);
  const int d = static_cast<int>(x.cols);
  const int k = std::max(1, std::min(options.k, n - 1));
  const std::vector<float> data = detail::prepare_native_matrix(x, options.metric);
  std::vector<int> self_indices(static_cast<std::size_t>(n));
  std::iota(self_indices.begin(), self_indices.end(), 0);
  const detail::NativeKNNResult search = detail::metal_exact_knn_search(
    data,
    n,
    data,
    n,
    d,
    k,
    options.metric,
    self_indices
  );

  NeighborGraph out;
  out.neighbors = search.neighbors;
  out.index_base = GraphIndexBase::One;
  out.indices.resize(search.indices.size(), -1);
  out.distances.resize(search.distances.size(), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < search.indices.size(); ++i) {
    if (search.indices[i] >= 0) out.indices[i] = search.indices[i] + 1;
    out.distances[i] = detail::native_knn_output_distance(search.distances[i], options.metric);
  }
  return out;
}
#endif

std::uint64_t edge_key(int a, int b) {
  const std::uint32_t u = static_cast<std::uint32_t>(std::min(a, b));
  const std::uint32_t v = static_cast<std::uint32_t>(std::max(a, b));
  return (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint64_t>(v);
}

struct Edge {
  std::uint64_t key = 0;
  double weight = 0.0;
};

int edge_from(std::uint64_t key) { return static_cast<int>(static_cast<std::uint32_t>(key >> 32)); }
int edge_to(std::uint64_t key) { return static_cast<int>(static_cast<std::uint32_t>(key & 0xffffffffULL)); }

int normalize_index(int idx, int n, bool one_based) {
  if (idx < 0) return -1;
  const int z = one_based ? idx - 1 : idx;
  return z >= 0 && z < n ? z : -1;
}

bool graph_is_one_based(const NeighborGraph& graph, int n) {
  if (graph.index_base == GraphIndexBase::One) return true;
  if (graph.index_base == GraphIndexBase::Zero) return false;
  for (int idx : graph.indices) {
    if (idx == 0) return false;
    if (idx == n) return true;
  }
  return true;
}

bool contains_neighbor(const NeighborGraph& graph, int row, int target0, int n, bool one_based) {
  for (int col = 0; col < graph.neighbors; ++col) {
    const int nb = normalize_index(graph.indices[static_cast<std::size_t>(row) * graph.neighbors + col], n, one_based);
    if (nb == target0) return true;
  }
  return false;
}

void push_edge(std::vector<Edge>& edges, int u0, int v0, double weight, double prune) {
  if (u0 == v0 || !std::isfinite(weight) || weight <= prune) return;
  edges.push_back({edge_key(u0, v0), weight});
}

std::vector<double> local_sigmas(const NeighborGraph& graph, int n) {
  std::vector<double> sigma(static_cast<std::size_t>(n), 1.0);
  for (int i = 0; i < n; ++i) {
    double last = 0.0;
    double sum = 0.0;
    int count = 0;
    for (int j = 0; j < graph.neighbors; ++j) {
      const float d = graph.distances[static_cast<std::size_t>(i) * graph.neighbors + j];
      if (std::isfinite(d) && d > 0.0f) {
        last = d;
        sum += d;
        ++count;
      }
    }
    if (last > 0.0) sigma[static_cast<std::size_t>(i)] = last;
    else if (count > 0 && sum > 0.0) sigma[static_cast<std::size_t>(i)] = sum / count;
  }
  return sigma;
}

struct EdgeList {
  int n = 0;
  std::vector<int> from;
  std::vector<int> to;
  std::vector<double> weight;
};

struct DisjointSet {
  std::vector<int> parent;
  std::vector<int> rank;

  explicit DisjointSet(int n) : parent(static_cast<std::size_t>(n)), rank(static_cast<std::size_t>(n), 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    int root = x;
    while (parent[static_cast<std::size_t>(root)] != root) root = parent[static_cast<std::size_t>(root)];
    while (parent[static_cast<std::size_t>(x)] != x) {
      const int next = parent[static_cast<std::size_t>(x)];
      parent[static_cast<std::size_t>(x)] = root;
      x = next;
    }
    return root;
  }

  bool unite(int a, int b) {
    int ra = find(a);
    int rb = find(b);
    if (ra == rb) return false;
    if (rank[static_cast<std::size_t>(ra)] < rank[static_cast<std::size_t>(rb)]) std::swap(ra, rb);
    parent[static_cast<std::size_t>(rb)] = ra;
    if (rank[static_cast<std::size_t>(ra)] == rank[static_cast<std::size_t>(rb)]) ++rank[static_cast<std::size_t>(ra)];
    return true;
  }
};

EdgeList edge_list_from_graph(const NeighborGraph& graph, int n, const GraphClusterOptions& options) {
  if (n < 2 || graph.neighbors < 1) throw std::invalid_argument("graph clustering requires a non-empty graph.");
  if (graph.indices.size() != static_cast<std::size_t>(n) * graph.neighbors ||
      graph.distances.size() != static_cast<std::size_t>(n) * graph.neighbors) {
    throw std::invalid_argument("NeighborGraph size does not match sample count.");
  }
  const bool one_based = graph_is_one_based(graph, n);
  const double prune = std::max(0.0, options.prune);
  std::vector<Edge> edges;
  edges.reserve(static_cast<std::size_t>(n) * graph.neighbors);

  if (options.weight_type == GraphWeightType::SNN) {
    std::vector<int> valid(static_cast<std::size_t>(n), 0);
    std::vector<int> reverse_count(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < graph.neighbors; ++j) {
        const int nb = normalize_index(graph.indices[static_cast<std::size_t>(i) * graph.neighbors + j], n, one_based);
        if (nb >= 0 && nb != i) {
          ++valid[static_cast<std::size_t>(i)];
          ++reverse_count[static_cast<std::size_t>(nb)];
        }
      }
    }
    std::vector<int> ptr(static_cast<std::size_t>(n) + 1, 0);
    for (int i = 0; i < n; ++i) ptr[static_cast<std::size_t>(i + 1)] = ptr[static_cast<std::size_t>(i)] + reverse_count[static_cast<std::size_t>(i)];
    std::vector<int> rows(static_cast<std::size_t>(ptr.back()), 0);
    std::fill(reverse_count.begin(), reverse_count.end(), 0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < graph.neighbors; ++j) {
        const int nb = normalize_index(graph.indices[static_cast<std::size_t>(i) * graph.neighbors + j], n, one_based);
        if (nb >= 0 && nb != i) rows[static_cast<std::size_t>(ptr[static_cast<std::size_t>(nb)] + reverse_count[static_cast<std::size_t>(nb)]++)] = i;
      }
    }
    std::vector<int> shared(static_cast<std::size_t>(n), 0);
    std::vector<int> touched;
    touched.reserve(static_cast<std::size_t>(graph.neighbors) * graph.neighbors);
    for (int i = 0; i < n; ++i) {
      touched.clear();
      for (int j = 0; j < graph.neighbors; ++j) {
        const int nb = normalize_index(graph.indices[static_cast<std::size_t>(i) * graph.neighbors + j], n, one_based);
        if (nb < 0 || nb == i) continue;
        if (options.mutual && !contains_neighbor(graph, nb, i, n, one_based)) continue;
        for (int pos = ptr[static_cast<std::size_t>(nb)]; pos < ptr[static_cast<std::size_t>(nb + 1)]; ++pos) {
          const int other = rows[static_cast<std::size_t>(pos)];
          if (other <= i) continue;
          int& s = shared[static_cast<std::size_t>(other)];
          if (s == 0) touched.push_back(other);
          ++s;
        }
      }
      for (int other : touched) {
        const int s = shared[static_cast<std::size_t>(other)];
        shared[static_cast<std::size_t>(other)] = 0;
        const int denom = valid[static_cast<std::size_t>(i)] + valid[static_cast<std::size_t>(other)] - s;
        if (s > 0 && denom > 0) push_edge(edges, i, other, static_cast<double>(s) / denom, prune);
      }
    }
  } else {
    const std::vector<double> sigma = options.weight_type == GraphWeightType::Adaptive ? local_sigmas(graph, n) : std::vector<double>();
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < graph.neighbors; ++j) {
        const std::size_t off = static_cast<std::size_t>(i) * graph.neighbors + j;
        const int nb = normalize_index(graph.indices[off], n, one_based);
        if (nb < 0 || nb == i) continue;
        if (options.mutual && !contains_neighbor(graph, nb, i, n, one_based)) continue;
        double w = 1.0;
        if (options.weight_type == GraphWeightType::Distance) {
          const float d = graph.distances[off];
          if (!std::isfinite(d) || d < 0.0f) continue;
          w = 1.0 / (1.0 + static_cast<double>(d));
        } else if (options.weight_type == GraphWeightType::Adaptive) {
          const float d = graph.distances[off];
          if (!std::isfinite(d) || d < 0.0f) continue;
          const double scale = std::max(1e-12, sigma[static_cast<std::size_t>(i)] * sigma[static_cast<std::size_t>(nb)]);
          w = std::exp(-(static_cast<double>(d) * d) / scale);
        }
        push_edge(edges, i, nb, w, prune);
      }
    }
  }

  std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) { return a.key < b.key; });
  std::size_t unique_count = 0;
  for (std::size_t p = 0; p < edges.size();) {
    const std::uint64_t key = edges[p].key;
    double w = edges[p].weight;
    ++p;
    while (p < edges.size() && edges[p].key == key) {
      w = std::max(w, edges[p].weight);
      ++p;
    }
    edges[unique_count++] = {key, w};
  }
  EdgeList out;
  out.n = n;
  out.from.resize(unique_count);
  out.to.resize(unique_count);
  out.weight.resize(unique_count);
  for (std::size_t i = 0; i < unique_count; ++i) {
    out.from[i] = edge_from(edges[i].key);
    out.to[i] = edge_to(edges[i].key);
    out.weight[i] = edges[i].weight;
  }
  return out;
}

std::vector<int> components_from_edges(const EdgeList& edges) {
  DisjointSet dsu(edges.n);
  for (std::size_t e = 0; e < edges.from.size(); ++e) {
    const int u = edges.from[e];
    const int v = edges.to[e];
    const double w = edges.weight[e];
    if (u >= 0 && u < edges.n && v >= 0 && v < edges.n && u != v && std::isfinite(w) && w > 0.0) dsu.unite(u, v);
  }
  std::unordered_map<int, int> remap;
  remap.reserve(static_cast<std::size_t>(edges.n));
  std::vector<int> comp(static_cast<std::size_t>(edges.n), 0);
  int next = 0;
  for (int i = 0; i < edges.n; ++i) {
    const int root = dsu.find(i);
    auto it = remap.find(root);
    if (it == remap.end()) it = remap.emplace(root, next++).first;
    comp[static_cast<std::size_t>(i)] = it->second;
  }
  return comp;
}

int component_count(const std::vector<int>& comp) {
  int count = 0;
  for (int c : comp) count = std::max(count, c + 1);
  return count;
}

double robust_edge_weight(const EdgeList& edges) {
  std::vector<double> weights;
  weights.reserve(edges.weight.size());
  for (double w : edges.weight) {
    if (std::isfinite(w) && w > 0.0) weights.push_back(w);
  }
  if (weights.empty()) return 1.0;
  const std::size_t mid = weights.size() / 2;
  std::nth_element(weights.begin(), weights.begin() + static_cast<std::ptrdiff_t>(mid), weights.end());
  return std::max(1e-12, weights[mid]);
}

double squared_distance(MatrixView x, int a, int b) {
  double out = 0.0;
  for (std::size_t j = 0; j < x.cols; ++j) {
    const double d = static_cast<double>(x.value_float(static_cast<std::size_t>(a), j)) -
                     static_cast<double>(x.value_float(static_cast<std::size_t>(b), j));
    out += d * d;
  }
  return out;
}

double squared_centroid_distance(const std::vector<double>& centroids, int dims, int a, int b) {
  double out = 0.0;
  const std::size_t ao = static_cast<std::size_t>(a) * dims;
  const std::size_t bo = static_cast<std::size_t>(b) * dims;
  for (int j = 0; j < dims; ++j) {
    const double d = centroids[ao + static_cast<std::size_t>(j)] - centroids[bo + static_cast<std::size_t>(j)];
    out += d * d;
  }
  return out;
}

void bridge_embedding_components(EdgeList& edges, MatrixView embedding) {
  if (edges.n < 2) return;
  if (embedding.rows != static_cast<std::size_t>(edges.n)) throw std::invalid_argument("Embedding row count does not match graph vertices.");
  if (embedding.cols < 1) throw std::invalid_argument("Embedding clustering requires at least one column.");

  const std::vector<int> comp = components_from_edges(edges);
  const int n_comp = component_count(comp);
  if (n_comp <= 1) return;

  const int dims = static_cast<int>(embedding.cols);
  std::vector<int> counts(static_cast<std::size_t>(n_comp), 0);
  std::vector<double> centroids(static_cast<std::size_t>(n_comp) * dims, 0.0);
  for (int i = 0; i < edges.n; ++i) {
    const int c = comp[static_cast<std::size_t>(i)];
    ++counts[static_cast<std::size_t>(c)];
    const std::size_t off = static_cast<std::size_t>(c) * dims;
    for (int j = 0; j < dims; ++j) centroids[off + static_cast<std::size_t>(j)] += embedding(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
  }
  for (int c = 0; c < n_comp; ++c) {
    const double denom = static_cast<double>(std::max(1, counts[static_cast<std::size_t>(c)]));
    const std::size_t off = static_cast<std::size_t>(c) * dims;
    for (int j = 0; j < dims; ++j) centroids[off + static_cast<std::size_t>(j)] /= denom;
  }

  std::vector<int> representative(static_cast<std::size_t>(n_comp), -1);
  std::vector<double> rep_dist(static_cast<std::size_t>(n_comp), std::numeric_limits<double>::infinity());
  for (int i = 0; i < edges.n; ++i) {
    const int c = comp[static_cast<std::size_t>(i)];
    double dist = 0.0;
    const std::size_t off = static_cast<std::size_t>(c) * dims;
    for (int j = 0; j < dims; ++j) {
      const double d = embedding(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) - centroids[off + static_cast<std::size_t>(j)];
      dist += d * d;
    }
    if (dist < rep_dist[static_cast<std::size_t>(c)] || (dist == rep_dist[static_cast<std::size_t>(c)] && i < representative[static_cast<std::size_t>(c)])) {
      rep_dist[static_cast<std::size_t>(c)] = dist;
      representative[static_cast<std::size_t>(c)] = i;
    }
  }

  const double weight = robust_edge_weight(edges);
  std::vector<double> best_dist(static_cast<std::size_t>(n_comp), std::numeric_limits<double>::infinity());
  std::vector<int> parent(static_cast<std::size_t>(n_comp), -1);
  std::vector<char> in_tree(static_cast<std::size_t>(n_comp), 0);
  best_dist[0] = 0.0;
  for (int iter = 0; iter < n_comp; ++iter) {
    int u = -1;
    double best = std::numeric_limits<double>::infinity();
    for (int c = 0; c < n_comp; ++c) {
      if (!in_tree[static_cast<std::size_t>(c)] && best_dist[static_cast<std::size_t>(c)] < best) {
        best = best_dist[static_cast<std::size_t>(c)];
        u = c;
      }
    }
    if (u < 0) break;
    in_tree[static_cast<std::size_t>(u)] = 1;
    if (parent[static_cast<std::size_t>(u)] >= 0) {
      const int a = representative[static_cast<std::size_t>(u)];
      const int b = representative[static_cast<std::size_t>(parent[static_cast<std::size_t>(u)])];
      if (a >= 0 && b >= 0 && a != b) {
        edges.from.push_back(std::min(a, b));
        edges.to.push_back(std::max(a, b));
        edges.weight.push_back(weight);
      }
    }
    for (int v = 0; v < n_comp; ++v) {
      if (in_tree[static_cast<std::size_t>(v)] || v == u) continue;
      double dist = squared_centroid_distance(centroids, dims, u, v);
      const int ru = representative[static_cast<std::size_t>(u)];
      const int rv = representative[static_cast<std::size_t>(v)];
      if (ru >= 0 && rv >= 0) dist = std::min(dist, squared_distance(embedding, ru, rv));
      if (dist < best_dist[static_cast<std::size_t>(v)] ||
          (dist == best_dist[static_cast<std::size_t>(v)] && u < parent[static_cast<std::size_t>(v)])) {
        best_dist[static_cast<std::size_t>(v)] = dist;
        parent[static_cast<std::size_t>(v)] = u;
      }
    }
  }
}

struct CsrGraph {
  int n = 0;
  std::vector<int> ptr;
  std::vector<int> to;
  std::vector<double> weight;
  std::vector<double> degree;
  double total_weight = 0.0;
};

CsrGraph csr_from_edges(const EdgeList& edges) {
  CsrGraph g;
  g.n = edges.n;
  g.ptr.assign(static_cast<std::size_t>(g.n) + 1, 0);
  g.degree.assign(static_cast<std::size_t>(g.n), 0.0);
  for (std::size_t e = 0; e < edges.from.size(); ++e) {
    const int u = edges.from[e];
    const int v = edges.to[e];
    const double w = edges.weight[e];
    if (u < 0 || u >= g.n || v < 0 || v >= g.n || u == v || !std::isfinite(w) || w <= 0.0) continue;
    ++g.ptr[static_cast<std::size_t>(u + 1)];
    ++g.ptr[static_cast<std::size_t>(v + 1)];
    g.degree[static_cast<std::size_t>(u)] += w;
    g.degree[static_cast<std::size_t>(v)] += w;
    g.total_weight += w;
  }
  for (int i = 1; i <= g.n; ++i) g.ptr[static_cast<std::size_t>(i)] += g.ptr[static_cast<std::size_t>(i - 1)];
  g.to.assign(static_cast<std::size_t>(g.ptr.back()), 0);
  g.weight.assign(g.to.size(), 0.0);
  std::vector<int> fill = g.ptr;
  for (std::size_t e = 0; e < edges.from.size(); ++e) {
    const int u = edges.from[e];
    const int v = edges.to[e];
    const double w = edges.weight[e];
    if (u < 0 || u >= g.n || v < 0 || v >= g.n || u == v || !std::isfinite(w) || w <= 0.0) continue;
    int pos = fill[static_cast<std::size_t>(u)]++;
    g.to[static_cast<std::size_t>(pos)] = v;
    g.weight[static_cast<std::size_t>(pos)] = w;
    pos = fill[static_cast<std::size_t>(v)]++;
    g.to[static_cast<std::size_t>(pos)] = u;
    g.weight[static_cast<std::size_t>(pos)] = w;
  }
  return g;
}

std::vector<int> compact(std::vector<int> x) {
  std::unordered_map<int, int> map;
  map.reserve(x.size());
  int next = 0;
  for (int& v : x) {
    auto it = map.find(v);
    if (it == map.end()) it = map.emplace(v, next++).first;
    v = it->second;
  }
  return x;
}

std::vector<double> community_degree(const CsrGraph& g, const std::vector<int>& membership) {
  int max_comm = 0;
  for (int c : membership) max_comm = std::max(max_comm, c);
  std::vector<double> deg(static_cast<std::size_t>(max_comm) + 1, 0.0);
  for (int i = 0; i < g.n; ++i) deg[static_cast<std::size_t>(membership[static_cast<std::size_t>(i)])] += g.degree[static_cast<std::size_t>(i)];
  return deg;
}

double modularity(const CsrGraph& g, const std::vector<int>& membership) {
  if (g.n == 0 || g.total_weight <= 0.0) return 0.0;
  const double two_m = 2.0 * g.total_weight;
  double internal = 0.0;
  for (int u = 0; u < g.n; ++u) {
    const int cu = membership[static_cast<std::size_t>(u)];
    for (int p = g.ptr[static_cast<std::size_t>(u)]; p < g.ptr[static_cast<std::size_t>(u + 1)]; ++p) {
      const int v = g.to[static_cast<std::size_t>(p)];
      if (cu == membership[static_cast<std::size_t>(v)]) internal += g.weight[static_cast<std::size_t>(p)];
    }
  }
  const std::vector<double> cdeg = community_degree(g, membership);
  double expected = 0.0;
  for (double d : cdeg) expected += d * d;
  return internal / two_m - expected / (two_m * two_m);
}

using SparseWalkDistribution = std::vector<std::pair<int, float>>;

struct WalkNeighbor {
  double delta = 0.0;
  double edge_weight = 0.0;
};

struct WalkCommunity {
  bool active = false;
  int size = 0;
  double degree = 0.0;
  SparseWalkDistribution probability;
  std::unordered_map<int, WalkNeighbor> neighbors;
};

struct WalkCandidate {
  double delta = 0.0;
  int left = -1;
  int right = -1;
};

struct WalkCandidateGreater {
  bool operator()(const WalkCandidate& left, const WalkCandidate& right) const {
    if (left.delta != right.delta) return left.delta > right.delta;
    if (left.left != right.left) return left.left > right.left;
    return left.right > right.right;
  }
};

double walk_distance_squared(
  const SparseWalkDistribution& left,
  const SparseWalkDistribution& right,
  const std::vector<double>& degree
) {
  std::size_t i = 0;
  std::size_t j = 0;
  double distance = 0.0;
  while (i < left.size() || j < right.size()) {
    int vertex = -1;
    double difference = 0.0;
    if (j == right.size() ||
        (i < left.size() && left[i].first < right[j].first)) {
      vertex = left[i].first;
      difference = left[i].second;
      ++i;
    } else if (i == left.size() || right[j].first < left[i].first) {
      vertex = right[j].first;
      difference = -right[j].second;
      ++j;
    } else {
      vertex = left[i].first;
      difference = static_cast<double>(left[i].second) - right[j].second;
      ++i;
      ++j;
    }
    distance += difference * difference /
      std::max(degree[static_cast<std::size_t>(vertex)], 1e-30);
  }
  return distance;
}

SparseWalkDistribution merge_walk_distributions(
  const SparseWalkDistribution& left,
  int left_size,
  const SparseWalkDistribution& right,
  int right_size
) {
  SparseWalkDistribution merged;
  merged.reserve(left.size() + right.size());
  const double denominator = static_cast<double>(left_size + right_size);
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < left.size() || j < right.size()) {
    int vertex = -1;
    double value = 0.0;
    if (j == right.size() ||
        (i < left.size() && left[i].first < right[j].first)) {
      vertex = left[i].first;
      value = static_cast<double>(left_size) * left[i].second;
      ++i;
    } else if (i == left.size() || right[j].first < left[i].first) {
      vertex = right[j].first;
      value = static_cast<double>(right_size) * right[j].second;
      ++j;
    } else {
      vertex = left[i].first;
      value = static_cast<double>(left_size) * left[i].second +
        static_cast<double>(right_size) * right[j].second;
      ++i;
      ++j;
    }
    value /= denominator;
    if (value > 0.0) merged.emplace_back(vertex, static_cast<float>(value));
  }
  return merged;
}

std::vector<SparseWalkDistribution> walk_probabilities(
  const CsrGraph& g,
  const std::vector<double>& self_weight,
  const std::vector<double>& walk_degree,
  int steps,
  int n_threads
) {
  (void)n_threads;
  std::vector<SparseWalkDistribution> probability(static_cast<std::size_t>(g.n));
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
  {
    std::vector<double> current_value(static_cast<std::size_t>(g.n), 0.0);
    std::vector<double> next_value(static_cast<std::size_t>(g.n), 0.0);
    std::vector<int> current;
    std::vector<int> next;
    current.reserve(static_cast<std::size_t>(g.n));
    next.reserve(static_cast<std::size_t>(g.n));
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
    for (int source = 0; source < g.n; ++source) {
      current.clear();
      current.push_back(source);
      current_value[static_cast<std::size_t>(source)] = 1.0;
      for (int step = 0; step < steps; ++step) {
        next.clear();
        for (int state : current) {
          const double normalized = current_value[static_cast<std::size_t>(state)] /
            walk_degree[static_cast<std::size_t>(state)];
          auto add_mass = [&](int vertex, double mass) {
            double& slot = next_value[static_cast<std::size_t>(vertex)];
            if (slot == 0.0) next.push_back(vertex);
            slot += mass;
          };
          add_mass(state, normalized * self_weight[static_cast<std::size_t>(state)]);
          for (int p = g.ptr[static_cast<std::size_t>(state)];
               p < g.ptr[static_cast<std::size_t>(state + 1)]; ++p) {
            add_mass(
              g.to[static_cast<std::size_t>(p)],
              normalized * g.weight[static_cast<std::size_t>(p)]
            );
          }
        }
        for (int state : current) current_value[static_cast<std::size_t>(state)] = 0.0;
        current.swap(next);
        current_value.swap(next_value);
      }
      std::sort(current.begin(), current.end());
      SparseWalkDistribution& row = probability[static_cast<std::size_t>(source)];
      row.reserve(current.size());
      for (int state : current) {
        const double value = current_value[static_cast<std::size_t>(state)];
        if (value > 0.0) row.emplace_back(state, static_cast<float>(value));
        current_value[static_cast<std::size_t>(state)] = 0.0;
      }
    }
  }
  return probability;
}

std::vector<int> replay_walk_partition(
  int n,
  const std::vector<std::pair<int, int>>& merges,
  int merge_count
) {
  DisjointSet sets(n);
  std::vector<int> representative(static_cast<std::size_t>(2 * n - 1), -1);
  std::iota(representative.begin(), representative.begin() + n, 0);
  merge_count = std::min(merge_count, static_cast<int>(merges.size()));
  for (int step = 0; step < merge_count; ++step) {
    const int left = merges[static_cast<std::size_t>(step)].first;
    const int right = merges[static_cast<std::size_t>(step)].second;
    const int left_root = representative[static_cast<std::size_t>(left)];
    const int right_root = representative[static_cast<std::size_t>(right)];
    sets.unite(left_root, right_root);
    representative[static_cast<std::size_t>(n + step)] = sets.find(left_root);
  }
  std::vector<int> membership(static_cast<std::size_t>(n), 0);
  std::unordered_map<int, int> labels;
  labels.reserve(static_cast<std::size_t>(n));
  int next = 0;
  for (int vertex = 0; vertex < n; ++vertex) {
    const int root = sets.find(vertex);
    auto inserted = labels.emplace(root, next);
    if (inserted.second) ++next;
    membership[static_cast<std::size_t>(vertex)] = inserted.first->second;
  }
  return membership;
}

// Independent sparse implementation of the adjacent-community random-walk
// agglomeration of Pons and Latapy. A requested count is a direct hierarchy
// cut; without one, the maximum-modularity cut is returned.
std::vector<int> random_walk_cluster(
  const CsrGraph& g,
  int steps,
  int max_iter,
  int n_threads,
  int target_clusters
) {
  (void)max_iter;
  std::vector<int> singleton(static_cast<std::size_t>(g.n));
  std::iota(singleton.begin(), singleton.end(), 0);
  if (g.total_weight <= 0.0 || g.n < 2) return singleton;
  steps = std::max(1, steps);
  n_threads = std::max(1, n_threads);

  std::vector<double> self_weight(static_cast<std::size_t>(g.n), 1.0);
  std::vector<double> walk_degree(static_cast<std::size_t>(g.n), 1.0);
  for (int vertex = 0; vertex < g.n; ++vertex) {
    const int edge_count = g.ptr[static_cast<std::size_t>(vertex + 1)] -
      g.ptr[static_cast<std::size_t>(vertex)];
    if (edge_count > 0) {
      self_weight[static_cast<std::size_t>(vertex)] =
        g.degree[static_cast<std::size_t>(vertex)] / edge_count;
    }
    walk_degree[static_cast<std::size_t>(vertex)] =
      g.degree[static_cast<std::size_t>(vertex)] +
      self_weight[static_cast<std::size_t>(vertex)];
  }

  std::vector<SparseWalkDistribution> probability = walk_probabilities(
    g, self_weight, walk_degree, steps, n_threads
  );
  const int maximum_communities = 2 * g.n - 1;
  std::vector<WalkCommunity> communities(static_cast<std::size_t>(maximum_communities));
  for (int vertex = 0; vertex < g.n; ++vertex) {
    WalkCommunity& community = communities[static_cast<std::size_t>(vertex)];
    community.active = true;
    community.size = 1;
    community.degree = g.degree[static_cast<std::size_t>(vertex)];
    community.probability = std::move(probability[static_cast<std::size_t>(vertex)]);
    const int edge_count = g.ptr[static_cast<std::size_t>(vertex + 1)] -
      g.ptr[static_cast<std::size_t>(vertex)];
    community.neighbors.reserve(static_cast<std::size_t>(edge_count) * 2U + 1U);
  }

  std::vector<int> edge_left;
  std::vector<int> edge_right;
  std::vector<double> edge_weight;
  for (int left = 0; left < g.n; ++left) {
    for (int p = g.ptr[static_cast<std::size_t>(left)];
         p < g.ptr[static_cast<std::size_t>(left + 1)]; ++p) {
      const int right = g.to[static_cast<std::size_t>(p)];
      if (left < right) {
        edge_left.push_back(left);
        edge_right.push_back(right);
        edge_weight.push_back(g.weight[static_cast<std::size_t>(p)]);
      }
    }
  }
  std::vector<double> edge_delta(edge_left.size(), 0.0);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 64)
#endif
  for (std::ptrdiff_t edge = 0;
       edge < static_cast<std::ptrdiff_t>(edge_left.size()); ++edge) {
    const int left = edge_left[static_cast<std::size_t>(edge)];
    const int right = edge_right[static_cast<std::size_t>(edge)];
    edge_delta[static_cast<std::size_t>(edge)] = walk_distance_squared(
      communities[static_cast<std::size_t>(left)].probability,
      communities[static_cast<std::size_t>(right)].probability,
      walk_degree
    ) / (2.0 * g.n);
  }

  std::priority_queue<
    WalkCandidate, std::vector<WalkCandidate>, WalkCandidateGreater
  > heap;
  for (std::size_t edge = 0; edge < edge_left.size(); ++edge) {
    const int left = edge_left[edge];
    const int right = edge_right[edge];
    const WalkNeighbor neighbor{edge_delta[edge], edge_weight[edge]};
    communities[static_cast<std::size_t>(left)].neighbors.emplace(right, neighbor);
    communities[static_cast<std::size_t>(right)].neighbors.emplace(left, neighbor);
    heap.push(WalkCandidate{edge_delta[edge], left, right});
  }

  const double two_m = 2.0 * g.total_weight;
  double modularity_value = 0.0;
  for (double degree : g.degree) {
    modularity_value -= degree * degree / (two_m * two_m);
  }
  double best_modularity = modularity_value;
  int best_merge_count = 0;
  std::vector<std::pair<int, int>> merges;
  merges.reserve(static_cast<std::size_t>(g.n - 1));
  int next_id = g.n;

  while (!heap.empty()) {
    const WalkCandidate candidate = heap.top();
    heap.pop();
    const int left = candidate.left;
    const int right = candidate.right;
    if (left < 0 || right < 0 || left >= next_id || right >= next_id) continue;
    WalkCommunity& left_community = communities[static_cast<std::size_t>(left)];
    WalkCommunity& right_community = communities[static_cast<std::size_t>(right)];
    if (!left_community.active || !right_community.active) continue;
    const auto current = left_community.neighbors.find(right);
    if (current == left_community.neighbors.end()) continue;
    const double tolerance = 1e-12 * std::max(1.0, std::abs(current->second.delta));
    if (std::abs(current->second.delta - candidate.delta) > tolerance) continue;

    const WalkNeighbor between = current->second;
    const int merged = next_id++;
    WalkCommunity& merged_community = communities[static_cast<std::size_t>(merged)];
    merged_community.active = true;
    merged_community.size = left_community.size + right_community.size;
    merged_community.degree = left_community.degree + right_community.degree;
    merged_community.probability = merge_walk_distributions(
      left_community.probability, left_community.size,
      right_community.probability, right_community.size
    );

    std::vector<int> adjacent;
    adjacent.reserve(left_community.neighbors.size() + right_community.neighbors.size());
    for (const auto& item : left_community.neighbors) {
      if (item.first != right) adjacent.push_back(item.first);
    }
    for (const auto& item : right_community.neighbors) {
      if (item.first != left) adjacent.push_back(item.first);
    }
    std::sort(adjacent.begin(), adjacent.end());
    adjacent.erase(std::unique(adjacent.begin(), adjacent.end()), adjacent.end());
    merged_community.neighbors.reserve(adjacent.size() * 2U + 1U);

    for (int other : adjacent) {
      WalkCommunity& other_community = communities[static_cast<std::size_t>(other)];
      if (!other_community.active) continue;
      const auto left_it = left_community.neighbors.find(other);
      const auto right_it = right_community.neighbors.find(other);
      const bool has_left = left_it != left_community.neighbors.end();
      const bool has_right = right_it != right_community.neighbors.end();
      double delta = 0.0;
      if (has_left && has_right) {
        const double other_size = static_cast<double>(other_community.size);
        delta =
          ((left_community.size + other_size) * left_it->second.delta +
           (right_community.size + other_size) * right_it->second.delta -
           other_size * between.delta) /
          (left_community.size + right_community.size + other_size);
      } else {
        const double distance = walk_distance_squared(
          merged_community.probability,
          other_community.probability,
          walk_degree
        );
        delta = (1.0 / g.n) *
          (static_cast<double>(merged_community.size) * other_community.size /
           (merged_community.size + other_community.size)) * distance;
      }
      if (delta < 0.0 && delta > -1e-14) delta = 0.0;
      const double joined_weight =
        (has_left ? left_it->second.edge_weight : 0.0) +
        (has_right ? right_it->second.edge_weight : 0.0);
      const WalkNeighbor neighbor{delta, joined_weight};
      merged_community.neighbors.emplace(other, neighbor);
      other_community.neighbors.erase(left);
      other_community.neighbors.erase(right);
      other_community.neighbors.emplace(merged, neighbor);
      heap.push(WalkCandidate{
        delta, std::min(merged, other), std::max(merged, other)
      });
    }

    modularity_value += between.edge_weight / g.total_weight -
      left_community.degree * right_community.degree * 2.0 /
      (two_m * two_m);
    merges.emplace_back(left, right);
    if (modularity_value > best_modularity + 1e-12) {
      best_modularity = modularity_value;
      best_merge_count = static_cast<int>(merges.size());
    }

    left_community.active = false;
    right_community.active = false;
    left_community.neighbors.clear();
    right_community.neighbors.clear();
    left_community.probability.clear();
    left_community.probability.shrink_to_fit();
    right_community.probability.clear();
    right_community.probability.shrink_to_fit();
  }

  const int merge_count = target_clusters > 0 ?
    g.n - std::min(target_clusters, g.n) : best_merge_count;
  return replay_walk_partition(g.n, merges, merge_count);
}

std::vector<int> merge_communities_to_target(
  const CsrGraph& g,
  std::vector<int> membership,
  int target
) {
  membership = compact(std::move(membership));
  int communities = 0;
  for (int label : membership) communities = std::max(communities, label + 1);
  if (communities == target) return membership;

  if (communities < target) {
    membership.resize(static_cast<std::size_t>(g.n));
    std::iota(membership.begin(), membership.end(), 0);
    communities = g.n;
  }

  while (communities > target) {
    std::vector<double> volume(static_cast<std::size_t>(communities), 0.0);
    std::unordered_map<std::uint64_t, double> between;
    between.reserve(g.weight.size() / 2 + 1);
    for (int u = 0; u < g.n; ++u) {
      const int cu = membership[static_cast<std::size_t>(u)];
      volume[static_cast<std::size_t>(cu)] += g.degree[static_cast<std::size_t>(u)];
      for (int p = g.ptr[static_cast<std::size_t>(u)];
           p < g.ptr[static_cast<std::size_t>(u + 1)]; ++p) {
        const int cv = membership[static_cast<std::size_t>(
          g.to[static_cast<std::size_t>(p)]
        )];
        if (cu == cv) continue;
        const std::uint32_t lo = static_cast<std::uint32_t>(std::min(cu, cv));
        const std::uint32_t hi = static_cast<std::uint32_t>(std::max(cu, cv));
        const std::uint64_t key = (static_cast<std::uint64_t>(lo) << 32) | hi;
        between[key] += g.weight[static_cast<std::size_t>(p)];
      }
    }

    int merge_a = -1;
    int merge_b = -1;
    double best = -1.0;
    for (const auto& item : between) {
      const int a = static_cast<int>(item.first >> 32);
      const int b = static_cast<int>(item.first & 0xffffffffULL);
      const double denom = std::sqrt(std::max(
        volume[static_cast<std::size_t>(a)] *
        volume[static_cast<std::size_t>(b)],
        1e-30
      ));
      const double score = item.second / denom;
      if (score > best + 1e-15 ||
          (std::abs(score - best) <= 1e-15 &&
           (a < merge_a || (a == merge_a && b < merge_b)))) {
        best = score;
        merge_a = a;
        merge_b = b;
      }
    }

    if (merge_a < 0) {
      throw std::runtime_error(
        "Exact-K clustering cannot merge disconnected communities."
      );
    }

    for (int& label : membership) {
      if (label == merge_b) label = merge_a;
    }
    membership = compact(std::move(membership));
    --communities;
  }
  return membership;
}

GraphClusterResult make_result(const CsrGraph& g, const std::vector<int>& membership0, const GraphClusterOptions& options, int n_edges, Backend backend, double elapsed) {
  GraphClusterResult out;
  out.membership.resize(static_cast<std::size_t>(g.n));
  int max_comm = 0;
  for (int i = 0; i < g.n; ++i) {
    out.membership[static_cast<std::size_t>(i)] = membership0[static_cast<std::size_t>(i)] + 1;
    max_comm = std::max(max_comm, membership0[static_cast<std::size_t>(i)] + 1);
  }
  out.modularity = modularity(g, membership0);
  out.n_communities = max_comm;
  out.n_vertices = g.n;
  out.n_edges = n_edges;
  out.target_clusters = options.target_clusters;
  out.target_gap = options.target_clusters > 0 ? std::abs(out.n_communities - options.target_clusters) : 0;
  out.target_exact = options.target_clusters == 0 || out.target_gap == 0;
  out.backend = backend;
  out.runtime_seconds = elapsed;
  return out;
}

int connected_component_count(const CsrGraph& graph) {
  std::vector<unsigned char> visited(static_cast<std::size_t>(graph.n), 0);
  std::vector<int> stack;
  int components = 0;
  for (int root = 0; root < graph.n; ++root) {
    if (visited[static_cast<std::size_t>(root)]) continue;
    ++components;
    stack.clear();
    stack.push_back(root);
    visited[static_cast<std::size_t>(root)] = 1;
    while (!stack.empty()) {
      const int vertex = stack.back();
      stack.pop_back();
      for (int edge = graph.ptr[static_cast<std::size_t>(vertex)];
           edge < graph.ptr[static_cast<std::size_t>(vertex + 1)]; ++edge) {
        const int neighbor = graph.to[static_cast<std::size_t>(edge)];
        if (visited[static_cast<std::size_t>(neighbor)]) continue;
        visited[static_cast<std::size_t>(neighbor)] = 1;
        stack.push_back(neighbor);
      }
    }
  }
  return components;
}

GraphClusterResult run_cpu_cluster(const EdgeList& edge_list, const GraphClusterOptions& options, double elapsed_start = 0.0) {
  Timer timer;
  CsrGraph g = csr_from_edges(edge_list);
  if (g.n == 0) throw std::invalid_argument("empty graph.");
  if (options.target_clusters > g.n) {
    throw std::invalid_argument("target_clusters cannot exceed number of graph vertices.");
  }
  const int graph_components = connected_component_count(g);
  if (options.target_clusters > 0 && options.target_clusters < graph_components) {
    throw std::invalid_argument(
      "target_clusters cannot be smaller than the number of disconnected graph components."
    );
  }

  GraphClusterOptions opts = options;
  opts.n_iterations = std::max(1, opts.n_iterations);
  opts.random_walk_steps = std::max(1, opts.random_walk_steps);
  opts.n_threads = std::max(1, opts.n_threads);
  std::vector<int> membership = random_walk_cluster(
    g,
    opts.random_walk_steps,
    opts.n_iterations,
    opts.n_threads,
    opts.target_clusters
  );
  if (opts.target_clusters > 0) {
    membership = merge_communities_to_target(
      g, std::move(membership), opts.target_clusters
    );
  }
  GraphClusterResult out = make_result(
    g,
    membership,
    opts,
    static_cast<int>(edge_list.from.size()),
    Backend::CPU,
    elapsed_start + timer.seconds()
  );
  return out;
}

}  // namespace

NeighborGraph KODAMAKNNGraph_CPU(MatrixView x, const GraphClusterOptions& options) {
  GraphClusterOptions opts = options;
  opts.backend = Backend::CPU;
  if (detail::should_use_spatial_grid_knn(static_cast<int>(x.rows), static_cast<int>(x.cols), opts.metric)) {
    const std::vector<float> data = copy_float32(x);
    return detail::spatial_grid_self_knn(
      data.data(),
      static_cast<int>(x.rows),
      static_cast<int>(x.cols),
      opts.k,
      opts.n_threads,
      true
    );
  }
  return build_hnsw_graph(x, opts);
}

NeighborGraph KODAMAKNNGraph_CUDA(MatrixView x, const GraphClusterOptions& options) {
#ifdef KODAMA_ENABLE_CUDA
  if (x.rows < 2 || x.cols < 1) throw std::invalid_argument("KODAMAKNNGraph requires at least two rows.");
  const int n = static_cast<int>(x.rows);
  const int d = static_cast<int>(x.cols);
  const int k = std::max(1, std::min(options.k, n - 1));
  std::vector<float> data = copy_float32(x);
  if (detail::should_use_spatial_grid_knn(n, d, options.metric)) {
    if (k <= 256) {
      return detail::spatial_grid_self_knn_cuda(data, n, d, k, options.gpu_device, true);
    }
    return detail::spatial_grid_self_knn(data.data(), n, d, k, options.n_threads, true);
  }
  if (options.metric == DistanceMetric::Cosine) normalize_rows(data, x.rows, x.cols);
  std::vector<int> exclusions(static_cast<std::size_t>(n));
  std::iota(exclusions.begin(), exclusions.end(), 0);
  const detail::NativeKNNResult search = detail::native_cuda_exact_knn_search(
    data,
    n,
    data,
    n,
    d,
    k,
    options.metric,
    options.gpu_device,
    exclusions
  );
  NeighborGraph out;
  out.neighbors = search.neighbors;
  out.index_base = GraphIndexBase::One;
  out.indices = search.indices;
  out.distances.resize(search.distances.size(), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < out.indices.size(); ++i) {
    if (out.indices[i] >= 0) ++out.indices[i];
    out.distances[i] = detail::native_knn_output_distance(search.distances[i], options.metric);
  }
  return out;
#else
  (void)x;
  (void)options;
  throw std::runtime_error("KODAMAKNNGraph_CUDA requires a CUDA build.");
#endif
}

NeighborGraph KODAMAKNNGraph_METAL(MatrixView x, const GraphClusterOptions& options) {
#ifdef KODAMA_ENABLE_METAL
  GraphClusterOptions opts = options;
  opts.backend = Backend::Metal;
  return build_metal_graph(x, opts);
#else
  (void)x;
  (void)options;
  throw std::runtime_error("KODAMAKNNGraph_METAL requires an Apple Metal build.");
#endif
}

NeighborGraph KODAMAKNNGraph(MatrixView x, const GraphClusterOptions& options) {
  if (options.backend == Backend::CUDA) return KODAMAKNNGraph_CUDA(x, options);
  if (options.backend == Backend::Metal) return KODAMAKNNGraph_METAL(x, options);
  return KODAMAKNNGraph_CPU(x, options);
}

GraphClusterResult KODAMAGraphCluster_CPU(const NeighborGraph& graph, int samples, const GraphClusterOptions& options) {
  Timer timer;
  GraphClusterOptions opts = options;
  opts.backend = Backend::CPU;
  EdgeList edges = edge_list_from_graph(graph, samples, opts);
  return run_cpu_cluster(edges, opts, timer.seconds());
}

GraphClusterResult KODAMAGraphCluster(const NeighborGraph& graph, int samples, const GraphClusterOptions& options) {
  if (options.backend != Backend::CPU) {
    throw std::runtime_error(
      "Random-walk clustering is CPU-only. Construct the graph with the requested "
      "accelerator, then call KODAMAGraphCluster_CPU explicitly."
    );
  }
  return KODAMAGraphCluster_CPU(graph, samples, options);
}

GraphClusterResult KODAMAEmbeddingGraphCluster(MatrixView embedding, const NeighborGraph& graph, const GraphClusterOptions& options) {
  Timer timer;
  if (embedding.rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("Embedding has too many rows for graph clustering.");
  }
  const int samples = static_cast<int>(embedding.rows);
  if (samples < 2) throw std::invalid_argument("KODAMAEmbeddingCluster requires at least two rows.");
  GraphClusterOptions opts = options;
  if (opts.backend != Backend::CPU) {
    throw std::runtime_error(
      "Random-walk clustering is CPU-only. Construct the graph with the requested "
      "accelerator, then call KODAMAEmbeddingGraphCluster with Backend::CPU."
    );
  }
  EdgeList edges = edge_list_from_graph(graph, samples, opts);
  if (opts.target_clusters > 0) bridge_embedding_components(edges, embedding);
  GraphClusterResult out = run_cpu_cluster(edges, opts, timer.seconds());
  out.runtime_seconds = timer.seconds();
  return out;
}

GraphClusterResult KODAMAEmbeddingCluster(MatrixView embedding, const GraphClusterOptions& options) {
  Timer timer;
  if (options.backend != Backend::CPU) {
    throw std::runtime_error(
      "KODAMAEmbeddingCluster provides CPU random-walk clustering only. Use "
      "KODAMAKNNGraph_CUDA or KODAMAKNNGraph_METAL separately when accelerated "
      "graph construction is required."
    );
  }
  GraphClusterOptions graph_options = options;
  NeighborGraph graph = KODAMAKNNGraph(embedding, graph_options);
  GraphClusterResult out = KODAMAEmbeddingGraphCluster(embedding, graph, options);
  out.runtime_seconds = timer.seconds();
  return out;
}

}  // namespace kodama
