// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

double seconds_since(const std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

kodama::Backend parse_backend(const std::string& value) {
  if (value == "cuda") return kodama::Backend::CUDA;
  if (value == "metal") return kodama::Backend::Metal;
  throw std::invalid_argument("backend must be cuda or metal");
}

kodama::DistanceMetric parse_metric(const std::string& value) {
  if (value == "euclidean") return kodama::DistanceMetric::Euclidean;
  if (value == "cosine") return kodama::DistanceMetric::Cosine;
  if (value == "inner_product") return kodama::DistanceMetric::InnerProduct;
  throw std::invalid_argument("metric must be euclidean, cosine, or inner_product");
}

std::vector<float> synthetic_matrix(int rows, int columns) {
  std::vector<float> data(
    static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns)
  );
  std::mt19937 generator(41u);
  std::normal_distribution<float> noise(0.0f, 0.35f);
  constexpr int classes = 32;
  for (int row = 0; row < rows; ++row) {
    const int cls = row % classes;
    for (int column = 0; column < columns; ++column) {
      const float signal = column % classes == cls ? 2.5f : 0.0f;
      data[
        static_cast<std::size_t>(row) * static_cast<std::size_t>(columns) +
        static_cast<std::size_t>(column)
      ] = signal + noise(generator);
    }
  }
  return data;
}

std::vector<float> read_matrix(
  const std::string& path,
  int rows,
  int columns
) {
  if (path == "-") return synthetic_matrix(rows, columns);
  const std::size_t items =
    static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
  std::vector<float> data(items);
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open input matrix: " + path);
  input.read(
    reinterpret_cast<char*>(data.data()),
    static_cast<std::streamsize>(items * sizeof(float))
  );
  if (input.gcount() != static_cast<std::streamsize>(items * sizeof(float))) {
    throw std::runtime_error("input matrix size does not match rows and columns");
  }
  char extra = 0;
  if (input.read(&extra, 1)) {
    throw std::runtime_error("input matrix has trailing data");
  }
  return data;
}

double graph_overlap(
  const kodama::NeighborGraph& left,
  const kodama::NeighborGraph& right,
  int rows
) {
  const int k = std::min(left.neighbors, right.neighbors);
  if (rows < 1 || k < 1) return 1.0;
  std::size_t matches = 0;
  for (int row = 0; row < rows; ++row) {
    std::unordered_set<int> expected;
    expected.reserve(static_cast<std::size_t>(k));
    for (int neighbor = 0; neighbor < k; ++neighbor) {
      expected.insert(
        left.indices[
          static_cast<std::size_t>(row) * static_cast<std::size_t>(left.neighbors) +
          static_cast<std::size_t>(neighbor)
        ]
      );
    }
    for (int neighbor = 0; neighbor < k; ++neighbor) {
      const int observed =
        right.indices[
          static_cast<std::size_t>(row) * static_cast<std::size_t>(right.neighbors) +
          static_cast<std::size_t>(neighbor)
        ];
      if (expected.count(observed) != 0) ++matches;
    }
  }
  return static_cast<double>(matches) / static_cast<double>(rows * k);
}

double mean(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  return std::accumulate(values.begin(), values.end(), 0.0) /
    static_cast<double>(values.size());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 10 && argc != 11) {
    std::cerr
      << "Usage: " << argv[0]
      << " <cuda|metal> <rows> <columns> <k> <nlist> <nprobe> <repeats>"
      << " <euclidean|cosine|inner_product> <float32-row-major-file|->"
      << " [query-rows]\n";
    return 2;
  }

  try {
    const kodama::Backend backend = parse_backend(argv[1]);
    const int rows = std::stoi(argv[2]);
    const int columns = std::stoi(argv[3]);
    const int k = std::stoi(argv[4]);
    const int nlist = std::stoi(argv[5]);
    const int nprobe = std::stoi(argv[6]);
    const int repeats = std::stoi(argv[7]);
    const kodama::DistanceMetric metric = parse_metric(argv[8]);
    const int query_rows = argc == 11 ? std::stoi(argv[10]) : rows;
    if (rows < 2 || columns < 1 || k < 1 || repeats < 2) {
      throw std::invalid_argument("rows >= 2, columns >= 1, k >= 1, and repeats >= 2 are required");
    }
    if (query_rows < 1 || query_rows > rows) {
      throw std::invalid_argument("query-rows must be between 1 and rows");
    }

    std::vector<float> data = read_matrix(argv[9], rows, columns);
    const kodama::MatrixView view{
      data.data(),
      static_cast<std::size_t>(rows),
      static_cast<std::size_t>(columns)
    };
    const kodama::MatrixView query_view{
      data.data(),
      static_cast<std::size_t>(query_rows),
      static_cast<std::size_t>(columns)
    };
    kodama::KNNOptions options;
    options.backend = backend;
    options.metric = metric;
    options.ivf_nlist = nlist;
    options.ivf_nprobe = nprobe;
    options.hnsw_target_recall = 0.99;

    {
      auto warmup = kodama::BuildResidentIVFIndex(view, options);
      const kodama::MatrixView warmup_query{
        data.data(),
        1,
        static_cast<std::size_t>(columns)
      };
      (void)kodama::SearchResidentIVFIndex(warmup, warmup_query, k);
    }

    auto resident = kodama::BuildResidentIVFIndex(view, options);
    const double resident_build_seconds = resident.build_seconds();
    std::vector<double> resident_search_seconds;
    resident_search_seconds.reserve(static_cast<std::size_t>(repeats));
    kodama::NeighborGraph resident_reference;
    kodama::ResidentIVFSearchStats resident_stats;
    for (int iteration = 0; iteration < repeats; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      kodama::NeighborGraph graph = query_rows == rows
        ? kodama::SearchResidentIVFIndexSelf(
            resident,
            k,
            true,
            &resident_stats
          )
        : kodama::SearchResidentIVFIndex(
            resident,
            query_view,
            k,
            &resident_stats
          );
      resident_search_seconds.push_back(seconds_since(start));
      if (iteration == 0) resident_reference = std::move(graph);
    }

    std::vector<double> rebuilt_seconds;
    rebuilt_seconds.reserve(static_cast<std::size_t>(repeats));
    double minimum_overlap = 1.0;
    for (int iteration = 0; iteration < repeats; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      auto rebuilt = kodama::BuildResidentIVFIndex(view, options);
      const kodama::NeighborGraph graph = query_rows == rows
        ? kodama::SearchResidentIVFIndexSelf(rebuilt, k, true)
        : kodama::SearchResidentIVFIndex(rebuilt, query_view, k);
      rebuilt_seconds.push_back(seconds_since(start));
      minimum_overlap = std::min(
        minimum_overlap,
        graph_overlap(resident_reference, graph, query_rows)
      );
    }

    const double first_search_seconds = resident_search_seconds.front();
    const std::vector<double> reused_search_seconds(
      resident_search_seconds.begin() + 1,
      resident_search_seconds.end()
    );
    const double reused_mean_seconds = mean(reused_search_seconds);
    const double rebuild_mean_seconds = mean(rebuilt_seconds);
    const double speedup = reused_mean_seconds > 0.0
      ? rebuild_mean_seconds / reused_mean_seconds
      : 0.0;

    std::cout
      << "backend,rows,query_rows,columns,k,nlist,nprobe,pilot_recall,"
      << "build_seconds,first_search_seconds,reused_search_mean_seconds,"
      << "rebuild_each_mean_seconds,reuse_speedup,minimum_neighbor_overlap\n";
    std::cout
      << argv[1] << ',' << rows << ',' << query_rows << ',' << columns << ',' << k << ','
      << resident.nlist() << ',' << resident_stats.nprobe << ','
      << std::fixed << std::setprecision(6)
      << resident_stats.pilot_recall << ','
      << resident_build_seconds << ','
      << first_search_seconds << ','
      << reused_mean_seconds << ','
      << rebuild_mean_seconds << ','
      << speedup << ','
      << minimum_overlap << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
