// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

std::uint64_t storage_bytes(const kodama::NeighborGraph& graph) {
  return
    static_cast<std::uint64_t>(graph.indices.capacity()) * sizeof(int) +
    static_cast<std::uint64_t>(graph.distances.capacity()) * sizeof(float);
}

kodama::NeighborGraph make_graph(std::size_t samples, int neighbors) {
  kodama::NeighborGraph graph;
  graph.neighbors = neighbors;
  const std::size_t edges = samples * static_cast<std::size_t>(neighbors);
  graph.indices.resize(edges);
  graph.distances.resize(edges);
  for (std::size_t row = 0; row < samples; ++row) {
    const std::size_t offset = row * static_cast<std::size_t>(neighbors);
    for (int rank = 0; rank < neighbors; ++rank) {
      graph.indices[offset + static_cast<std::size_t>(rank)] =
        static_cast<int>((row + static_cast<std::size_t>(rank) + 1u) % samples);
      graph.distances[offset + static_cast<std::size_t>(rank)] =
        static_cast<float>(rank + 1);
    }
  }
  return graph;
}

kodama::NeighborGraph trim_copy(
  const kodama::NeighborGraph& graph,
  std::size_t samples,
  int neighbors
) {
  kodama::NeighborGraph out;
  out.neighbors = neighbors;
  out.indices.resize(samples * static_cast<std::size_t>(neighbors));
  out.distances.resize(samples * static_cast<std::size_t>(neighbors));
  for (std::size_t row = 0; row < samples; ++row) {
    const std::size_t source =
      row * static_cast<std::size_t>(graph.neighbors);
    const std::size_t destination =
      row * static_cast<std::size_t>(neighbors);
    for (int rank = 0; rank < neighbors; ++rank) {
      out.indices[destination + static_cast<std::size_t>(rank)] =
        graph.indices[source + static_cast<std::size_t>(rank)];
      out.distances[destination + static_cast<std::size_t>(rank)] =
        graph.distances[source + static_cast<std::size_t>(rank)];
    }
  }
  return out;
}

void trim_in_place(
  kodama::NeighborGraph& graph,
  std::size_t samples,
  int neighbors
) {
  const int input_neighbors = graph.neighbors;
  for (std::size_t row = 0; row < samples; ++row) {
    const std::size_t source =
      row * static_cast<std::size_t>(input_neighbors);
    const std::size_t destination =
      row * static_cast<std::size_t>(neighbors);
    for (int rank = 0; rank < neighbors; ++rank) {
      graph.indices[destination + static_cast<std::size_t>(rank)] =
        graph.indices[source + static_cast<std::size_t>(rank)];
      graph.distances[destination + static_cast<std::size_t>(rank)] =
        graph.distances[source + static_cast<std::size_t>(rank)];
    }
  }
  graph.neighbors = neighbors;
  graph.indices.resize(samples * static_cast<std::size_t>(neighbors));
  graph.distances.resize(samples * static_cast<std::size_t>(neighbors));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "single";
  const std::size_t samples =
    argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 1000021u;
  const int neighbors = argc > 3 ? std::stoi(argv[3]) : 100;
  if (samples < 2 || neighbors < 1) {
    throw std::invalid_argument("samples must be at least 2 and neighbors must be positive.");
  }

  kodama::NeighborGraph global_graph = make_graph(samples, neighbors + 1);
  const auto start = std::chrono::steady_clock::now();
  std::uint64_t retained_bytes = 0;
  std::uint64_t checksum = 0;
  double lifecycle_seconds = 0.0;

  if (mode == "legacy") {
    kodama::NeighborGraph result_knn =
      trim_copy(global_graph, samples, neighbors);
    kodama::NeighborGraph result_base_knn = result_knn;
    retained_bytes =
      storage_bytes(global_graph) +
      storage_bytes(result_knn) +
      storage_bytes(result_base_knn);
    checksum =
      static_cast<std::uint64_t>(global_graph.indices.back()) +
      static_cast<std::uint64_t>(result_knn.indices.back()) +
      static_cast<std::uint64_t>(result_base_knn.indices.back());
    lifecycle_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start
    ).count();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  } else if (mode == "single") {
    trim_in_place(global_graph, samples, neighbors);
    kodama::NeighborGraph result_knn = std::move(global_graph);
    retained_bytes = storage_bytes(result_knn);
    checksum = static_cast<std::uint64_t>(result_knn.indices.back());
    lifecycle_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start
    ).count();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  } else {
    throw std::invalid_argument("mode must be 'legacy' or 'single'.");
  }

  std::cout
    << "mode,samples,neighbors,retained_graph_bytes,lifecycle_seconds,checksum\n"
    << mode << ',' << samples << ',' << neighbors << ',' << retained_bytes
    << ',' << lifecycle_seconds << ',' << checksum << '\n';
  return 0;
}
