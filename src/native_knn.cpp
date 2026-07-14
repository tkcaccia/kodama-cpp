/*
 * Compact HNSW implementation distilled from the algorithmic organization in
 * FAISS 1.14.3, commit 0ca9df4792b173d573044ee14ca0704780176e82.
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2026 Stefano Caccia.
 *
 * Licensed under the MIT License. The FAISS copyright and MIT license must be
 * retained with redistributed derivatives of this file.
 */

#include "native_knn.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace kodama::detail {
namespace {

struct NodeDistance {
  float distance = 0.0f;
  int id = -1;
};

struct CloserFirst {
  bool operator()(const NodeDistance& a, const NodeDistance& b) const {
    if (a.distance != b.distance) return a.distance > b.distance;
    return a.id > b.id;
  }
};

struct FartherFirst {
  bool operator()(const NodeDistance& a, const NodeDistance& b) const {
    if (a.distance != b.distance) return a.distance < b.distance;
    return a.id < b.id;
  }
};

bool closer(const NodeDistance& a, const NodeDistance& b) {
  return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}

class CompactHNSW {
 public:
  CompactHNSW(
    std::vector<float> data,
    int n,
    int p,
    DistanceMetric metric,
    int m,
    int ef_construction,
    int ef_search
  ) :
      data_(std::move(data)),
      n_(n),
      p_(p),
      metric_(metric),
      m_(std::max(2, std::min(m, std::max(2, n - 1)))),
      ef_construction_(std::max(1, ef_construction)),
      ef_search_(std::max(1, ef_search)) {
    generate_levels();
    allocate_graph();
  }

  void build() {
    if (n_ == 0) return;
    entry_point_ = 0;
    current_max_level_ = levels_[0];
    for (int point_id = 1; point_id < n_; ++point_id) add_point(point_id);
  }

  NativeKNNResult search(
    const std::vector<float>& queries,
    int query_rows,
    int requested_k,
    int n_threads,
    const std::vector<int>& query_train_indices
  ) const {
    if (static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(p_) != queries.size()) {
      throw std::invalid_argument("native HNSW query matrix size mismatch.");
    }
    if (!query_train_indices.empty() && static_cast<int>(query_train_indices.size()) != query_rows) {
      throw std::invalid_argument("native HNSW query/train index size mismatch.");
    }

    const bool excludes_self = !query_train_indices.empty();
    const int available = n_ - (excludes_self ? 1 : 0);
    const int k = std::min(requested_k, std::max(0, available));
    NativeKNNResult output;
    output.queries = query_rows;
    output.neighbors = k;
    output.indices.assign(static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k), -1);
    output.distances.assign(
      static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k),
      std::numeric_limits<float>::infinity()
    );
    if (k == 0 || query_rows == 0) return output;

    std::atomic<int> next{0};
    n_threads = std::max(1, std::min(n_threads, query_rows));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int worker_id = 0; worker_id < n_threads; ++worker_id) {
      workers.emplace_back([&, worker_id]() {
        (void)worker_id;
        VisitTable visited(n_);
        for (;;) {
          const int query_id = next.fetch_add(1, std::memory_order_relaxed);
          if (query_id >= query_rows) break;
          const float* query = queries.data() + static_cast<std::size_t>(query_id) * static_cast<std::size_t>(p_);
          const int excluded = query_train_indices.empty() ? -1 : query_train_indices[static_cast<std::size_t>(query_id)];
          std::vector<NodeDistance> candidates = search_query(query, std::max(k + 1, ef_search_), visited);
          int used = 0;
          for (const NodeDistance& candidate : candidates) {
            if (candidate.id == excluded) continue;
            const std::size_t pos = static_cast<std::size_t>(query_id) * static_cast<std::size_t>(k) + static_cast<std::size_t>(used);
            output.indices[pos] = candidate.id;
            output.distances[pos] = candidate.distance;
            if (++used == k) break;
          }
          if (used < k) exact_fill(query, query_id, excluded, k, output);
        }
      });
    }
    for (std::thread& worker : workers) worker.join();
    return output;
  }

 private:
  struct VisitTable {
    explicit VisitTable(int n) : marks(static_cast<std::size_t>(n), 0), generation(1) {}

    void reset() {
      if (++generation == 0) {
        std::fill(marks.begin(), marks.end(), 0);
        generation = 1;
      }
    }

    bool set(int id) {
      if (marks[static_cast<std::size_t>(id)] == generation) return false;
      marks[static_cast<std::size_t>(id)] = generation;
      return true;
    }

    std::vector<std::uint32_t> marks;
    std::uint32_t generation;
  };

  std::vector<float> data_;
  int n_ = 0;
  int p_ = 0;
  DistanceMetric metric_ = DistanceMetric::Euclidean;
  int m_ = 16;
  int ef_construction_ = 80;
  int ef_search_ = 64;
  int entry_point_ = -1;
  int current_max_level_ = -1;
  std::vector<int> levels_;
  std::vector<std::size_t> node_offsets_;
  std::vector<std::size_t> level_offsets_;
  std::vector<std::uint16_t> counts_;
  std::vector<std::int32_t> neighbors_;

  const float* point(int id) const {
    return data_.data() + static_cast<std::size_t>(id) * static_cast<std::size_t>(p_);
  }

  float distance(const float* a, const float* b) const {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    if (metric_ == DistanceMetric::Euclidean) {
      int d = 0;
      for (; d + 3 < p_; d += 4) {
        const float d0 = a[d] - b[d];
        const float d1 = a[d + 1] - b[d + 1];
        const float d2 = a[d + 2] - b[d + 2];
        const float d3 = a[d + 3] - b[d + 3];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
      }
      float sum = (sum0 + sum1) + (sum2 + sum3);
      for (; d < p_; ++d) {
        const float delta = a[d] - b[d];
        sum += delta * delta;
      }
      return sum;
    }

    int d = 0;
    for (; d + 3 < p_; d += 4) {
      sum0 += a[d] * b[d];
      sum1 += a[d + 1] * b[d + 1];
      sum2 += a[d + 2] * b[d + 2];
      sum3 += a[d + 3] * b[d + 3];
    }
    float dot = (sum0 + sum1) + (sum2 + sum3);
    for (; d < p_; ++d) dot += a[d] * b[d];
    return -dot;
  }

  float distance(int a, int b) const {
    return distance(point(a), point(b));
  }

  int capacity(int level) const {
    return level == 0 ? 2 * m_ : m_;
  }

  std::size_t level_index(int node, int level) const {
    return level_offsets_[static_cast<std::size_t>(node)] + static_cast<std::size_t>(level);
  }

  std::size_t neighbor_offset(int node, int level) const {
    return node_offsets_[static_cast<std::size_t>(node)] +
      (level == 0 ? 0u : static_cast<std::size_t>(2 * m_ + (level - 1) * m_));
  }

  void generate_levels() {
    levels_.resize(static_cast<std::size_t>(n_));
    std::mt19937 generator(12345u);
    std::uniform_real_distribution<double> uniform(std::nextafter(0.0, 1.0), 1.0);
    const double level_multiplier = 1.0 / std::log(static_cast<double>(m_));
    for (int i = 0; i < n_; ++i) {
      levels_[static_cast<std::size_t>(i)] = static_cast<int>(-std::log(uniform(generator)) * level_multiplier);
    }
  }

  void allocate_graph() {
    node_offsets_.resize(static_cast<std::size_t>(n_) + 1u, 0u);
    level_offsets_.resize(static_cast<std::size_t>(n_) + 1u, 0u);
    for (int i = 0; i < n_; ++i) {
      node_offsets_[static_cast<std::size_t>(i + 1)] = node_offsets_[static_cast<std::size_t>(i)] +
        static_cast<std::size_t>(2 * m_ + levels_[static_cast<std::size_t>(i)] * m_);
      level_offsets_[static_cast<std::size_t>(i + 1)] = level_offsets_[static_cast<std::size_t>(i)] +
        static_cast<std::size_t>(levels_[static_cast<std::size_t>(i)] + 1);
    }
    neighbors_.assign(node_offsets_.back(), -1);
    counts_.assign(level_offsets_.back(), 0);
  }

  std::pair<const std::int32_t*, int> neighbor_range(int node, int level) const {
    const std::size_t offset = neighbor_offset(node, level);
    return {neighbors_.data() + offset, counts_[level_index(node, level)]};
  }

  std::pair<std::int32_t*, std::uint16_t*> mutable_neighbor_range(int node, int level) {
    const std::size_t offset = neighbor_offset(node, level);
    return {neighbors_.data() + offset, &counts_[level_index(node, level)]};
  }

  int greedy_search(const float* query, int entry, int level, float& entry_distance) const {
    bool changed = true;
    while (changed) {
      changed = false;
      const auto range = neighbor_range(entry, level);
      for (int j = 0; j < range.second; ++j) {
        const int candidate = range.first[j];
        const float candidate_distance = distance(query, point(candidate));
        if (closer({candidate_distance, candidate}, {entry_distance, entry})) {
          entry = candidate;
          entry_distance = candidate_distance;
          changed = true;
        }
      }
    }
    return entry;
  }

  std::vector<NodeDistance> search_layer(
    const float* query,
    int entry,
    int ef,
    int level,
    VisitTable& visited
  ) const {
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, CloserFirst> candidates;
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, FartherFirst> results;
    visited.reset();
    const NodeDistance initial{distance(query, point(entry)), entry};
    candidates.push(initial);
    results.push(initial);
    visited.set(entry);
    while (!candidates.empty()) {
      const NodeDistance current = candidates.top();
      if (results.size() >= static_cast<std::size_t>(ef) && closer(results.top(), current)) break;
      candidates.pop();
      const auto range = neighbor_range(current.id, level);
      for (int j = 0; j < range.second; ++j) {
        const int candidate_id = range.first[j];
        if (!visited.set(candidate_id)) continue;
        const NodeDistance candidate{distance(query, point(candidate_id)), candidate_id};
        if (results.size() < static_cast<std::size_t>(ef) || closer(candidate, results.top())) {
          candidates.push(candidate);
          results.push(candidate);
          if (results.size() > static_cast<std::size_t>(ef)) results.pop();
        }
      }
    }
    std::vector<NodeDistance> output;
    output.reserve(results.size());
    while (!results.empty()) {
      output.push_back(results.top());
      results.pop();
    }
    std::sort(output.begin(), output.end(), closer);
    return output;
  }

  std::vector<int> select_diverse(
    int query_id,
    const std::vector<NodeDistance>& candidates,
    int max_size
  ) const {
    std::vector<int> selected;
    std::vector<int> rejected;
    selected.reserve(static_cast<std::size_t>(max_size));
    rejected.reserve(static_cast<std::size_t>(max_size));
    for (const NodeDistance& candidate : candidates) {
      if (candidate.id == query_id) continue;
      bool good = true;
      for (int other : selected) {
        if (distance(candidate.id, other) < candidate.distance) {
          good = false;
          break;
        }
      }
      if (good) {
        selected.push_back(candidate.id);
        if (static_cast<int>(selected.size()) == max_size) break;
      } else {
        rejected.push_back(candidate.id);
      }
    }
    for (int candidate : rejected) {
      if (static_cast<int>(selected.size()) == max_size) break;
      selected.push_back(candidate);
    }
    return selected;
  }

  void replace_neighbors(int node, int level, const std::vector<int>& selected) {
    auto range = mutable_neighbor_range(node, level);
    const int cap = capacity(level);
    const int size = std::min(cap, static_cast<int>(selected.size()));
    for (int i = 0; i < size; ++i) range.first[i] = selected[static_cast<std::size_t>(i)];
    for (int i = size; i < cap; ++i) range.first[i] = -1;
    *range.second = static_cast<std::uint16_t>(size);
  }

  void add_reciprocal(int node, int other, int level) {
    auto range = mutable_neighbor_range(node, level);
    const int count = *range.second;
    for (int i = 0; i < count; ++i) {
      if (range.first[i] == other) return;
    }
    const int cap = capacity(level);
    if (count < cap) {
      range.first[count] = other;
      *range.second = static_cast<std::uint16_t>(count + 1);
      return;
    }
    std::vector<NodeDistance> candidates;
    candidates.reserve(static_cast<std::size_t>(count) + 1u);
    for (int i = 0; i < count; ++i) {
      candidates.push_back({distance(node, range.first[i]), range.first[i]});
    }
    candidates.push_back({distance(node, other), other});
    std::sort(candidates.begin(), candidates.end(), closer);
    replace_neighbors(node, level, select_diverse(node, candidates, cap));
  }

  void add_point(int point_id) {
    VisitTable visited(n_);
    int nearest = entry_point_;
    float nearest_distance = distance(point_id, nearest);
    for (int level = current_max_level_; level > levels_[static_cast<std::size_t>(point_id)]; --level) {
      nearest = greedy_search(point(point_id), nearest, level, nearest_distance);
    }
    for (int level = std::min(levels_[static_cast<std::size_t>(point_id)], current_max_level_); level >= 0; --level) {
      const std::vector<NodeDistance> candidates =
        search_layer(point(point_id), nearest, ef_construction_, level, visited);
      const std::vector<int> selected = select_diverse(point_id, candidates, capacity(level));
      replace_neighbors(point_id, level, selected);
      for (int other : selected) add_reciprocal(other, point_id, level);
      if (!candidates.empty()) {
        nearest = candidates.front().id;
        nearest_distance = candidates.front().distance;
      }
    }
    if (levels_[static_cast<std::size_t>(point_id)] > current_max_level_) {
      entry_point_ = point_id;
      current_max_level_ = levels_[static_cast<std::size_t>(point_id)];
    }
  }

  std::vector<NodeDistance> search_query(const float* query, int ef, VisitTable& visited) const {
    int nearest = entry_point_;
    float nearest_distance = distance(query, point(nearest));
    for (int level = current_max_level_; level >= 1; --level) {
      nearest = greedy_search(query, nearest, level, nearest_distance);
    }
    return search_layer(query, nearest, ef, 0, visited);
  }

  void exact_fill(
    const float* query,
    int query_id,
    int excluded,
    int k,
    NativeKNNResult& output
  ) const {
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, FartherFirst> heap;
    for (int candidate = 0; candidate < n_; ++candidate) {
      if (candidate == excluded) continue;
      const NodeDistance value{distance(query, point(candidate)), candidate};
      if (heap.size() < static_cast<std::size_t>(k) || closer(value, heap.top())) {
        heap.push(value);
        if (heap.size() > static_cast<std::size_t>(k)) heap.pop();
      }
    }
    std::vector<NodeDistance> exact;
    exact.reserve(heap.size());
    while (!heap.empty()) {
      exact.push_back(heap.top());
      heap.pop();
    }
    std::sort(exact.begin(), exact.end(), closer);
    for (int rank = 0; rank < k; ++rank) {
      const std::size_t pos = static_cast<std::size_t>(query_id) * static_cast<std::size_t>(k) + static_cast<std::size_t>(rank);
      output.indices[pos] = exact[static_cast<std::size_t>(rank)].id;
      output.distances[pos] = exact[static_cast<std::size_t>(rank)].distance;
    }
  }
};

}  // namespace

std::vector<float> prepare_native_matrix(
  MatrixView x,
  const std::vector<int>& rows,
  DistanceMetric metric
) {
  std::vector<float> output(rows.size() * x.cols, 0.0f);
  for (std::size_t row_pos = 0; row_pos < rows.size(); ++row_pos) {
    const int row = rows[row_pos];
    if (row < 0 || static_cast<std::size_t>(row) >= x.rows) {
      throw std::out_of_range("native KNN row index is outside the input matrix.");
    }
    float squared_norm = 0.0f;
    if (metric == DistanceMetric::Cosine) {
      for (std::size_t j = 0; j < x.cols; ++j) {
        const float value = x.value_float(static_cast<std::size_t>(row), j);
        squared_norm += value * value;
      }
    }
    const float scale = metric == DistanceMetric::Cosine && squared_norm > 0.0f && std::isfinite(squared_norm) ?
      1.0f / std::sqrt(squared_norm) : 1.0f;
    for (std::size_t j = 0; j < x.cols; ++j) {
      output[row_pos * x.cols + j] = x.value_float(static_cast<std::size_t>(row), j) * scale;
    }
  }
  return output;
}

std::vector<float> prepare_native_matrix(MatrixView x, DistanceMetric metric) {
  std::vector<int> rows(x.rows);
  std::iota(rows.begin(), rows.end(), 0);
  return prepare_native_matrix(x, rows, metric);
}

NativeKNNResult native_hnsw_search(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& query,
  int query_rows,
  int dimensions,
  int k,
  DistanceMetric metric,
  const NativeHNSWParameters& parameters,
  int n_threads,
  const std::vector<int>& query_train_indices
) {
  if (train_rows < 1 || query_rows < 0 || dimensions < 1 || k < 1) {
    throw std::invalid_argument("invalid native HNSW dimensions or neighbor count.");
  }
  if (train.size() != static_cast<std::size_t>(train_rows) * static_cast<std::size_t>(dimensions)) {
    throw std::invalid_argument("native HNSW training matrix size mismatch.");
  }
  if (query.size() != static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(dimensions)) {
    throw std::invalid_argument("native HNSW query matrix size mismatch.");
  }
  if (train_rows == 1) {
    NativeKNNResult output;
    output.queries = query_rows;
    output.neighbors = query_train_indices.empty() ? 1 : 0;
    if (output.neighbors == 1) {
      output.indices.assign(static_cast<std::size_t>(query_rows), 0);
      output.distances.resize(static_cast<std::size_t>(query_rows));
      CompactHNSW singleton(train, train_rows, dimensions, metric, 2, 2, 2);
      singleton.build();
      return singleton.search(query, query_rows, 1, n_threads, query_train_indices);
    }
    return output;
  }

  CompactHNSW index(
    train,
    train_rows,
    dimensions,
    metric,
    parameters.m,
    parameters.ef_construction,
    parameters.ef_search
  );
  index.build();
  return index.search(query, query_rows, k, n_threads, query_train_indices);
}

float native_knn_score(float internal_distance, DistanceMetric) {
  return -internal_distance;
}

float native_knn_output_distance(float internal_distance, DistanceMetric metric) {
  if (metric == DistanceMetric::Euclidean) {
    return std::sqrt(std::max(0.0f, internal_distance));
  }
  return metric == DistanceMetric::Cosine ?
    std::max(0.0f, 1.0f + internal_distance) :
    1.0f + internal_distance;
}

}  // namespace kodama::detail
