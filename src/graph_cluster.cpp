#include "kodama/kodama.hpp"
#include "metal_backend.hpp"
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

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef KODAMA_ENABLE_CUDA
#include "kodama_matrix_cuda.hpp"

#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#endif

#ifdef KODAMA_ENABLE_CUGRAPH
#define FALSE CUGRAPH_FALSE
#define TRUE CUGRAPH_TRUE
#include <cugraph_c/array.h>
#include <cugraph_c/community_algorithms.h>
#include <cugraph_c/error.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/random.h>
#include <cugraph_c/resource_handle.h>
#undef FALSE
#undef TRUE
#endif

namespace kodama {
namespace {

struct Timer {
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }
};

struct OmpScope {
  int previous = 0;
  explicit OmpScope(int n_threads) {
#ifdef _OPENMP
    previous = omp_get_max_threads();
    if (n_threads > 0) omp_set_num_threads(n_threads);
#else
    (void)n_threads;
#endif
  }
  ~OmpScope() {
#ifdef _OPENMP
    if (previous > 0) omp_set_num_threads(previous);
#endif
  }
};

std::vector<float> copy_float32(MatrixView x) {
  std::vector<float> out(x.rows * x.cols);
  for (std::size_t i = 0; i < x.rows; ++i) {
    for (std::size_t j = 0; j < x.cols; ++j) out[i * x.cols + j] = x.value_float(i, j);
  }
  return out;
}

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
  out.indices.resize(search.indices.size(), -1);
  out.distances.resize(search.distances.size(), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < search.indices.size(); ++i) {
    if (search.indices[i] >= 0) out.indices[i] = search.indices[i] + 1;
    out.distances[i] = detail::native_knn_output_distance(search.distances[i], options.metric);
  }
  return out;
}

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

int dense_membership_count(const std::vector<int>& membership) {
  int count = 0;
  for (int c : membership) count = std::max(count, c + 1);
  return count;
}

std::vector<double> community_degree(const CsrGraph& g, const std::vector<int>& membership) {
  int max_comm = 0;
  for (int c : membership) max_comm = std::max(max_comm, c);
  std::vector<double> deg(static_cast<std::size_t>(max_comm) + 1, 0.0);
  for (int i = 0; i < g.n; ++i) deg[static_cast<std::size_t>(membership[static_cast<std::size_t>(i)])] += g.degree[static_cast<std::size_t>(i)];
  return deg;
}

double modularity(const CsrGraph& g, const std::vector<int>& membership, double resolution) {
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
  return internal / two_m - resolution * expected / (two_m * two_m);
}

std::vector<int> louvain_local_moving(const CsrGraph& g, int max_iter, double resolution, int n_threads, int seed) {
  std::vector<int> membership(static_cast<std::size_t>(g.n));
  std::iota(membership.begin(), membership.end(), 0);
  if (g.total_weight <= 0.0) return membership;
  const double two_m = 2.0 * g.total_weight;
  max_iter = std::max(1, max_iter);
  (void)n_threads;
  (void)seed;
  std::vector<double> cdeg = community_degree(g, membership);
  for (int iter = 0; iter < max_iter; ++iter) {
    int changed = 0;
    for (int u = 0; u < g.n; ++u) {
      std::unordered_map<int, double> neigh;
      neigh.reserve(static_cast<std::size_t>(g.ptr[static_cast<std::size_t>(u + 1)] - g.ptr[static_cast<std::size_t>(u)] + 1));
      const int old = membership[static_cast<std::size_t>(u)];
      for (int p = g.ptr[static_cast<std::size_t>(u)]; p < g.ptr[static_cast<std::size_t>(u + 1)]; ++p) {
        neigh[membership[static_cast<std::size_t>(g.to[static_cast<std::size_t>(p)])]] += g.weight[static_cast<std::size_t>(p)];
      }
      const double k_i = g.degree[static_cast<std::size_t>(u)];
      cdeg[static_cast<std::size_t>(old)] -= k_i;
      int best = old;
      double best_gain = 0.0;
      for (const auto& kv : neigh) {
        if (kv.first < 0 || kv.first >= static_cast<int>(cdeg.size())) continue;
        const double gain = kv.second - resolution * k_i * cdeg[static_cast<std::size_t>(kv.first)] / two_m;
        if (gain > best_gain + 1e-12 || (std::abs(gain - best_gain) <= 1e-12 && kv.first < best && gain > 0.0)) {
          best_gain = gain;
          best = kv.first;
        }
      }
      cdeg[static_cast<std::size_t>(best)] += k_i;
      if (best != old) {
        membership[static_cast<std::size_t>(u)] = best;
        ++changed;
      }
    }
    membership = compact(std::move(membership));
    cdeg = community_degree(g, membership);
    if (changed == 0) break;
  }
  return compact(std::move(membership));
}

CsrGraph coarsen_graph(const CsrGraph& g, const std::vector<int>& membership, int n_communities) {
  std::unordered_map<std::uint64_t, double> weights;
  weights.reserve(static_cast<std::size_t>(g.ptr.back() / 2 + 1));
  for (int u = 0; u < g.n; ++u) {
    const int cu = membership[static_cast<std::size_t>(u)];
    for (int p = g.ptr[static_cast<std::size_t>(u)]; p < g.ptr[static_cast<std::size_t>(u + 1)]; ++p) {
      const int v = g.to[static_cast<std::size_t>(p)];
      if (v <= u) continue;
      const int cv = membership[static_cast<std::size_t>(v)];
      if (cu == cv) continue;
      weights[edge_key(cu, cv)] += g.weight[static_cast<std::size_t>(p)];
    }
  }

  EdgeList edges;
  edges.n = n_communities;
  edges.from.reserve(weights.size());
  edges.to.reserve(weights.size());
  edges.weight.reserve(weights.size());
  for (const auto& kv : weights) {
    if (kv.second <= 0.0 || !std::isfinite(kv.second)) continue;
    edges.from.push_back(edge_from(kv.first));
    edges.to.push_back(edge_to(kv.first));
    edges.weight.push_back(kv.second);
  }
  return csr_from_edges(edges);
}

std::vector<int> louvain_multilevel(const CsrGraph& g, int max_iter, double resolution, int n_threads, int seed) {
  std::vector<int> original_to_current(static_cast<std::size_t>(g.n));
  std::iota(original_to_current.begin(), original_to_current.end(), 0);
  if (g.total_weight <= 0.0) return original_to_current;

  CsrGraph current = g;
  const int max_levels = std::max(1, std::min(50, max_iter));
  for (int level = 0; level < max_levels; ++level) {
    std::vector<int> level_membership =
      louvain_local_moving(current, max_iter, resolution, n_threads, seed + level * 104729);
    level_membership = compact(std::move(level_membership));
    const int n_communities = dense_membership_count(level_membership);
    if (n_communities >= current.n) break;

    for (int& c : original_to_current) c = level_membership[static_cast<std::size_t>(c)];
    original_to_current = compact(std::move(original_to_current));
    if (n_communities <= 1) break;

    CsrGraph next = coarsen_graph(current, level_membership, n_communities);
    if (next.n >= current.n || next.total_weight <= 0.0) break;
    current = std::move(next);
  }
  return compact(std::move(original_to_current));
}

std::vector<int> split_disconnected(const CsrGraph& g, const std::vector<int>& membership) {
  std::vector<int> refined(static_cast<std::size_t>(g.n), -1);
  std::vector<char> seen(static_cast<std::size_t>(g.n), 0);
  std::queue<int> q;
  int next = 0;
  for (int start = 0; start < g.n; ++start) {
    if (seen[static_cast<std::size_t>(start)]) continue;
    const int comm = membership[static_cast<std::size_t>(start)];
    seen[static_cast<std::size_t>(start)] = 1;
    refined[static_cast<std::size_t>(start)] = next;
    q.push(start);
    while (!q.empty()) {
      const int u = q.front();
      q.pop();
      for (int p = g.ptr[static_cast<std::size_t>(u)]; p < g.ptr[static_cast<std::size_t>(u + 1)]; ++p) {
        const int v = g.to[static_cast<std::size_t>(p)];
        if (!seen[static_cast<std::size_t>(v)] && membership[static_cast<std::size_t>(v)] == comm) {
          seen[static_cast<std::size_t>(v)] = 1;
          refined[static_cast<std::size_t>(v)] = next;
          q.push(v);
        }
      }
    }
    ++next;
  }
  return compact(std::move(refined));
}

std::vector<int> random_walk_cluster(const CsrGraph& g, int steps, int max_iter, int n_threads) {
  std::vector<int> labels(static_cast<std::size_t>(g.n));
  std::iota(labels.begin(), labels.end(), 0);
  if (g.total_weight <= 0.0) return labels;
  steps = std::max(1, steps);
  max_iter = std::max(1, max_iter);
  n_threads = std::max(1, n_threads);
  for (int iter = 0; iter < max_iter; ++iter) {
    std::vector<int> proposed = labels;
    int changed = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 256) reduction(+:changed)
#endif
    for (int u = 0; u < g.n; ++u) {
      std::unordered_map<int, double> score;
      std::unordered_map<int, double> frontier;
      frontier[u] = 1.0;
      for (int step = 0; step < steps; ++step) {
        std::unordered_map<int, double> next;
        next.reserve(frontier.size() * 2 + 1);
        for (const auto& item : frontier) {
          const int x = item.first;
          const double mass = item.second;
          const double deg = g.degree[static_cast<std::size_t>(x)];
          if (deg <= 0.0) continue;
          for (int p = g.ptr[static_cast<std::size_t>(x)]; p < g.ptr[static_cast<std::size_t>(x + 1)]; ++p) {
            next[g.to[static_cast<std::size_t>(p)]] += mass * g.weight[static_cast<std::size_t>(p)] / deg;
          }
        }
        frontier.swap(next);
        for (const auto& item : frontier) score[labels[static_cast<std::size_t>(item.first)]] += item.second;
      }
      int best = labels[static_cast<std::size_t>(u)];
      double best_score = -1.0;
      for (const auto& kv : score) {
        if (kv.second > best_score + 1e-15 || (std::abs(kv.second - best_score) <= 1e-15 && kv.first < best)) {
          best_score = kv.second;
          best = kv.first;
        }
      }
      proposed[static_cast<std::size_t>(u)] = best;
      if (best != labels[static_cast<std::size_t>(u)]) ++changed;
    }
    labels = compact(std::move(proposed));
    if (changed == 0) break;
  }
  return split_disconnected(g, labels);
}

GraphClusterResult make_result(const CsrGraph& g, const std::vector<int>& membership0, const GraphClusterOptions& options, int n_edges, Backend backend, double elapsed) {
  GraphClusterResult out;
  out.membership.resize(static_cast<std::size_t>(g.n));
  int max_comm = 0;
  for (int i = 0; i < g.n; ++i) {
    out.membership[static_cast<std::size_t>(i)] = membership0[static_cast<std::size_t>(i)] + 1;
    max_comm = std::max(max_comm, membership0[static_cast<std::size_t>(i)] + 1);
  }
  out.modularity = modularity(g, membership0, options.resolution);
  out.n_communities = max_comm;
  out.n_vertices = g.n;
  out.n_edges = n_edges;
  out.target_clusters = options.target_clusters;
  out.target_gap = options.target_clusters > 0 ? std::abs(out.n_communities - options.target_clusters) : 0;
  out.target_exact = options.target_clusters > 0 && out.target_gap == 0;
  out.selected_resolution = options.resolution;
  out.backend = backend;
  out.method = options.method;
  out.runtime_seconds = elapsed;
  return out;
}

GraphClusterResult run_cpu_cluster_once(const CsrGraph& g, const EdgeList& edge_list, const GraphClusterOptions& options, double elapsed) {
  std::vector<int> best;
  double best_mod = -std::numeric_limits<double>::infinity();
  GraphClusterOptions opts = options;
  opts.n_runs = std::max(1, opts.n_runs);
  opts.n_iterations = std::max(1, opts.n_iterations);
  opts.random_walk_steps = std::max(1, opts.random_walk_steps);
  opts.n_threads = std::max(1, opts.n_threads);
  if (!std::isfinite(opts.resolution) || opts.resolution <= 0.0) opts.resolution = 1.0;
  std::vector<double> all(static_cast<std::size_t>(opts.n_runs), 0.0);
  int selected = 1;
  for (int run = 0; run < opts.n_runs; ++run) {
    const int run_seed = static_cast<int>(opts.seed + static_cast<std::uint64_t>(run) * 104729ULL);
    std::vector<int> membership;
    if (opts.method == GraphClusterMethod::Louvain) {
      membership = louvain_multilevel(g, opts.n_iterations, opts.resolution, opts.n_threads, run_seed);
    } else if (opts.method == GraphClusterMethod::Leiden) {
      membership = split_disconnected(g, louvain_multilevel(g, std::max(2, opts.n_iterations), opts.resolution, opts.n_threads, run_seed));
    } else {
      membership = random_walk_cluster(g, opts.random_walk_steps, opts.n_iterations, opts.n_threads);
    }
    const double mod = modularity(g, membership, opts.resolution);
    all[static_cast<std::size_t>(run)] = mod;
    if (best.empty() || mod > best_mod) {
      best = std::move(membership);
      best_mod = mod;
      selected = run + 1;
    }
  }
  GraphClusterResult out = make_result(g, best, opts, static_cast<int>(edge_list.from.size()), Backend::CPU, elapsed);
  out.selected_run = selected;
  out.all_modularity = std::move(all);
  return out;
}

template <typename RunAtResolution>
GraphClusterResult search_target_clusters(GraphClusterOptions options, RunAtResolution run_at_resolution, double elapsed_start) {
  if (options.target_clusters < 1) {
    return run_at_resolution(options.resolution, elapsed_start);
  }
  if (options.method == GraphClusterMethod::RandomWalking) {
    GraphClusterResult out = run_at_resolution(options.resolution, elapsed_start);
    if (out.n_communities != options.target_clusters) {
      std::ostringstream oss;
      oss << "random_walking produced " << out.n_communities
          << " clusters, but target_clusters=" << options.target_clusters
          << ". The current random-walking implementation has no resolution parameter to tune exactly.";
      throw std::runtime_error(oss.str());
    }
    return out;
  }

  double delta = std::isfinite(options.target_delta) && options.target_delta > 0.0 ? options.target_delta : 0.2;
  const int max_steps = std::max(1, options.target_max_steps);
  GraphClusterResult best;
  bool have_best = false;
  int steps = 0;

  auto effective_gap = [&](const GraphClusterResult& candidate) {
    if (candidate.membership.empty() || options.target_clusters <= 0) return std::numeric_limits<double>::infinity();
    std::unordered_map<int, int> counts;
    counts.reserve(candidate.membership.size());
    for (int c : candidate.membership) ++counts[c];
    const double n = static_cast<double>(candidate.membership.size());
    double entropy = 0.0;
    for (const auto& kv : counts) {
      const double p = static_cast<double>(kv.second) / n;
      if (p > 0.0) entropy -= p * std::log(p);
    }
    return std::abs(std::exp(entropy) - static_cast<double>(options.target_clusters));
  };

  auto keep_best = [&](const GraphClusterResult& candidate) {
    const double candidate_eff_gap = effective_gap(candidate);
    const double best_eff_gap = have_best ? effective_gap(best) : std::numeric_limits<double>::infinity();
    if (!have_best ||
        candidate.target_gap < best.target_gap ||
        (candidate.target_gap == best.target_gap && candidate_eff_gap < best_eff_gap - 1e-9) ||
        (candidate.target_gap == best.target_gap && std::abs(candidate_eff_gap - best_eff_gap) <= 1e-9 &&
         candidate.modularity > best.modularity)) {
      best = candidate;
      have_best = true;
    }
  };

  auto run_candidate = [&](double resolution) {
    const double safe_res = std::max(resolution, 1e-12);
    GraphClusterResult out = run_at_resolution(safe_res, elapsed_start);
    out.target_clusters = options.target_clusters;
    out.target_gap = std::abs(out.n_communities - options.target_clusters);
    out.target_exact = out.target_gap == 0;
    out.selected_resolution = safe_res;
    keep_best(out);
    ++steps;
    return out;
  };

  double scan_res = options.target_resolution_init - delta;
  double low_res = std::numeric_limits<double>::quiet_NaN();
  double high_res = std::numeric_limits<double>::quiet_NaN();
  bool have_low = false;
  bool have_high = false;

  while (steps < max_steps) {
    scan_res += delta;
    GraphClusterResult out = run_candidate(scan_res);
    if (out.n_communities <= options.target_clusters) {
      low_res = out.selected_resolution;
      have_low = true;
    }
    if (out.n_communities > options.target_clusters) {
      high_res = out.selected_resolution;
      have_high = true;
      break;
    }
    if (out.n_communities == options.target_clusters) {
      if (steps >= max_steps) break;
      scan_res += delta;
      GraphClusterResult probe = run_candidate(scan_res);
      if (probe.n_communities <= options.target_clusters) {
        low_res = probe.selected_resolution;
        have_low = true;
      } else {
        high_res = probe.selected_resolution;
        have_high = true;
        break;
      }
    }
  }

  if (!have_low) {
    low_res = 1e-12;
    have_low = true;
  }
  if (!have_high) {
    high_res = std::max(low_res + delta, delta);
  }

  while (steps < max_steps && high_res > low_res) {
    const double mid = 0.5 * (low_res + high_res);
    if (mid <= low_res || mid >= high_res) break;
    GraphClusterResult out = run_candidate(mid);
    if (out.n_communities <= options.target_clusters) {
      low_res = out.selected_resolution;
    } else {
      high_res = out.selected_resolution;
    }
  }

  if (have_best && best.target_exact) return best;
  std::ostringstream oss;
  oss << "Could not obtain exactly " << options.target_clusters
      << " clusters after " << steps << " resolution-search steps";
  if (have_best) {
    oss << "; closest result had " << best.n_communities
        << " clusters at resolution " << best.selected_resolution << ".";
  } else {
    oss << ".";
  }
  throw std::runtime_error(oss.str());
}

GraphClusterResult run_cpu_cluster(const EdgeList& edge_list, const GraphClusterOptions& options, double elapsed_start = 0.0) {
  Timer timer;
  CsrGraph g = csr_from_edges(edge_list);
  if (g.n == 0) throw std::invalid_argument("empty graph.");
  if (options.target_clusters > g.n) throw std::invalid_argument("target_clusters cannot exceed number of graph vertices.");
  auto run_at = [&](double resolution, double start) {
    GraphClusterOptions opts = options;
    opts.resolution = resolution;
    return run_cpu_cluster_once(g, edge_list, opts, start + timer.seconds());
  };
  return search_target_clusters(options, run_at, elapsed_start);
}

#ifdef KODAMA_ENABLE_CUGRAPH
std::string cugraph_error_string(cugraph_error_t* error) {
  std::string msg = error ? cugraph_error_message(error) : "unknown cuGraph error";
  if (error) cugraph_error_free(error);
  return msg;
}

void check_cugraph(cugraph_error_code_t code, cugraph_error_t* error, const char* what) {
  if (code != CUGRAPH_SUCCESS) throw std::runtime_error(std::string(what) + ": " + cugraph_error_string(error));
  if (error) cugraph_error_free(error);
}

template <typename T>
struct DeviceArray {
  cugraph_type_erased_device_array_t* array = nullptr;
  cugraph_type_erased_device_array_view_t* view = nullptr;
  DeviceArray(const cugraph_resource_handle_t* handle, const std::vector<T>& host, cugraph_data_type_id_t dtype) {
    cugraph_error_t* error = nullptr;
    check_cugraph(cugraph_type_erased_device_array_create(handle, host.size(), dtype, &array, &error), error, "cuGraph device array create");
    view = cugraph_type_erased_device_array_view(array);
    check_cugraph(cugraph_type_erased_device_array_view_copy_from_host(handle, view, reinterpret_cast<const byte_t*>(host.data()), &error), error, "cuGraph host-to-device copy");
  }
  ~DeviceArray() {
    if (view) cugraph_type_erased_device_array_view_free(view);
    if (array) cugraph_type_erased_device_array_free(array);
  }
};

std::vector<int32_t> copy_i32_result(const cugraph_resource_handle_t* handle, cugraph_type_erased_device_array_view_t* view) {
  std::vector<int32_t> out(cugraph_type_erased_device_array_view_size(view));
  cugraph_error_t* error = nullptr;
  check_cugraph(cugraph_type_erased_device_array_view_copy_to_host(handle, reinterpret_cast<byte_t*>(out.data()), view, &error), error, "cuGraph device-to-host copy");
  return out;
}

GraphClusterResult run_cugraph_cluster(const EdgeList& edge_list, const GraphClusterOptions& options, double elapsed_start) {
  if (options.method == GraphClusterMethod::RandomWalking) throw std::runtime_error("CUDA random_walking clustering is not implemented; use CPU.");
  if (edge_list.from.empty()) throw std::invalid_argument("cuGraph clustering requires at least one edge.");
  Timer timer;
  const int m = static_cast<int>(edge_list.from.size());
  std::vector<int32_t> src(static_cast<std::size_t>(m));
  std::vector<int32_t> dst(static_cast<std::size_t>(m));
  std::vector<float> w(static_cast<std::size_t>(m));
  for (int i = 0; i < m; ++i) {
    src[static_cast<std::size_t>(i)] = edge_list.from[static_cast<std::size_t>(i)];
    dst[static_cast<std::size_t>(i)] = edge_list.to[static_cast<std::size_t>(i)];
    w[static_cast<std::size_t>(i)] = static_cast<float>(edge_list.weight[static_cast<std::size_t>(i)]);
  }
  std::vector<int32_t> vertices(static_cast<std::size_t>(edge_list.n));
  for (int i = 0; i < edge_list.n; ++i) vertices[static_cast<std::size_t>(i)] = i;
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(nullptr);
  if (!handle) throw std::runtime_error("cuGraph resource handle creation failed.");
  try {
    DeviceArray<int32_t> d_vertices(handle, vertices, INT32);
    DeviceArray<int32_t> d_src(handle, src, INT32);
    DeviceArray<int32_t> d_dst(handle, dst, INT32);
    DeviceArray<float> d_weight(handle, w, FLOAT32);
    cugraph_graph_properties_t props;
    props.is_symmetric = static_cast<bool_t>(1);
    props.is_multigraph = static_cast<bool_t>(0);
    cugraph_graph_t* graph = nullptr;
    cugraph_error_t* error = nullptr;
    check_cugraph(cugraph_graph_create_sg(handle, &props, d_vertices.view, d_src.view, d_dst.view, d_weight.view,
                                          nullptr, nullptr, static_cast<bool_t>(0), static_cast<bool_t>(0),
                                          static_cast<bool_t>(1), static_cast<bool_t>(1), static_cast<bool_t>(0),
                                          static_cast<bool_t>(0), &graph, &error), error, "cuGraph graph create");
    std::vector<int32_t> best_membership;
    double best_mod = -std::numeric_limits<double>::infinity();
    const int runs = std::max(1, options.n_runs);
    std::vector<double> all(static_cast<std::size_t>(runs), 0.0);
    int selected = 1;
    for (int run = 0; run < runs; ++run) {
      cugraph_hierarchical_clustering_result_t* result = nullptr;
      if (options.method == GraphClusterMethod::Louvain) {
        check_cugraph(cugraph_louvain(handle, graph, static_cast<size_t>(std::max(1, options.n_iterations)), 1e-7,
                                      options.resolution, static_cast<bool_t>(0), &result, &error), error, "cuGraph Louvain");
      } else {
        cugraph_rng_state_t* rng = nullptr;
        check_cugraph(cugraph_rng_state_create(handle, options.seed + static_cast<std::uint64_t>(run) * 104729ULL, &rng, &error), error, "cuGraph RNG create");
        check_cugraph(cugraph_leiden(handle, rng, graph, static_cast<size_t>(std::max(1, options.n_iterations)),
                                     options.resolution, 0.01, static_cast<bool_t>(0), &result, &error), error, "cuGraph Leiden");
        cugraph_rng_state_free(rng);
      }
      const double mod = cugraph_hierarchical_clustering_result_get_modularity(result);
      all[static_cast<std::size_t>(run)] = mod;
      if (best_membership.empty() || mod > best_mod) {
        std::vector<int32_t> rv = copy_i32_result(handle, cugraph_hierarchical_clustering_result_get_vertices(result));
        std::vector<int32_t> rc = copy_i32_result(handle, cugraph_hierarchical_clustering_result_get_clusters(result));
        best_membership.assign(static_cast<std::size_t>(edge_list.n), 0);
        for (std::size_t i = 0; i < rv.size(); ++i) {
          const int v = static_cast<int>(rv[i]);
          if (v >= 0 && v < edge_list.n) best_membership[static_cast<std::size_t>(v)] = rc[i];
        }
        best_mod = mod;
        selected = run + 1;
      }
      cugraph_hierarchical_clustering_result_free(result);
    }
    cugraph_graph_free(graph);
    cugraph_free_resource_handle(handle);
    std::vector<int> membership(best_membership.begin(), best_membership.end());
    membership = compact(std::move(membership));
    CsrGraph cpu_graph = csr_from_edges(edge_list);
    GraphClusterResult out = make_result(cpu_graph, membership, options, m, Backend::CUDA, elapsed_start + timer.seconds());
    out.modularity = best_mod;
    out.selected_run = selected;
    out.all_modularity = std::move(all);
    return out;
  } catch (...) {
    cugraph_free_resource_handle(handle);
    throw;
  }
}
#endif

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
  const faiss::MetricType faiss_metric = options.metric == DistanceMetric::Euclidean ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
  if (options.metric == DistanceMetric::Cosine) normalize_rows(data, x.rows, x.cols);
  faiss::gpu::StandardGpuResources resources;
  faiss::gpu::GpuIndexFlatConfig config;
  config.device = options.gpu_device;
  faiss::gpu::GpuIndexFlat index(&resources, d, faiss_metric, config);
  index.add(n, data.data());
  const int search_k = std::min(n, k + 1);
  std::vector<float> distances(static_cast<std::size_t>(n) * search_k);
  std::vector<faiss::idx_t> idx(static_cast<std::size_t>(n) * search_k);
  index.search(n, data.data(), search_k, distances.data(), idx.data());
  NeighborGraph out;
  out.neighbors = k;
  out.indices.assign(static_cast<std::size_t>(n) * k, -1);
  out.distances.assign(static_cast<std::size_t>(n) * k, std::numeric_limits<float>::infinity());
  for (int i = 0; i < n; ++i) {
    int kept = 0;
    for (int j = 0; j < search_k && kept < k; ++j) {
      const int nb = static_cast<int>(idx[static_cast<std::size_t>(i) * search_k + j]);
      if (nb < 0 || nb == i) continue;
      float dist = distances[static_cast<std::size_t>(i) * search_k + j];
      dist = options.metric == DistanceMetric::Euclidean ? (dist > 0.0f ? std::sqrt(dist) : 0.0f) : 1.0f - dist;
      const std::size_t off = static_cast<std::size_t>(i) * k + kept;
      out.indices[off] = nb + 1;
      out.distances[off] = dist;
      ++kept;
    }
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

GraphClusterResult KODAMAGraphCluster_CUDA(const NeighborGraph& graph, int samples, const GraphClusterOptions& options) {
  Timer timer;
  GraphClusterOptions opts = options;
  opts.backend = Backend::CUDA;
  EdgeList edges = edge_list_from_graph(graph, samples, opts);
  if (opts.target_clusters > samples) throw std::invalid_argument("target_clusters cannot exceed number of graph vertices.");
#ifdef KODAMA_ENABLE_CUGRAPH
  auto run_at = [&](double resolution, double start) {
    GraphClusterOptions run_opts = opts;
    run_opts.resolution = resolution;
    return run_cugraph_cluster(edges, run_opts, start + timer.seconds());
  };
  return search_target_clusters(opts, run_at, timer.seconds());
#else
  (void)edges;
  throw std::runtime_error("KODAMAGraphCluster_CUDA requires a CUDA build with RAPIDS libcugraph.");
#endif
}

GraphClusterResult KODAMAGraphCluster(const NeighborGraph& graph, int samples, const GraphClusterOptions& options) {
  if (options.backend == Backend::CUDA) return KODAMAGraphCluster_CUDA(graph, samples, options);
  if (options.backend == Backend::Metal) {
    throw std::runtime_error("Metal graph clustering is not implemented; use Backend::CPU explicitly.");
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
  if (opts.backend == Backend::Metal) {
    throw std::runtime_error("Metal graph clustering is not implemented; use Backend::CPU explicitly.");
  }
  EdgeList edges = edge_list_from_graph(graph, samples, opts);
  if (opts.target_clusters > 0) bridge_embedding_components(edges, embedding);
  GraphClusterResult out;
  if (opts.backend == Backend::CUDA) {
    if (opts.target_clusters > samples) throw std::invalid_argument("target_clusters cannot exceed number of graph vertices.");
#ifdef KODAMA_ENABLE_CUGRAPH
    auto run_at = [&](double resolution, double start) {
      GraphClusterOptions run_opts = opts;
      run_opts.resolution = resolution;
      return run_cugraph_cluster(edges, run_opts, start + timer.seconds());
    };
    out = search_target_clusters(opts, run_at, timer.seconds());
#else
    throw std::runtime_error("KODAMAEmbeddingCluster with CUDA clustering requires a CUDA build with RAPIDS libcugraph.");
#endif
  } else {
    opts.backend = Backend::CPU;
    out = run_cpu_cluster(edges, opts, timer.seconds());
  }
  out.runtime_seconds = timer.seconds();
  return out;
}

GraphClusterResult KODAMAEmbeddingCluster(MatrixView embedding, const GraphClusterOptions& options) {
  Timer timer;
  GraphClusterOptions graph_options = options;
  NeighborGraph graph = KODAMAKNNGraph(embedding, graph_options);
  GraphClusterResult out = KODAMAEmbeddingGraphCluster(embedding, graph, options);
  out.runtime_seconds = timer.seconds();
  return out;
}

}  // namespace kodama
