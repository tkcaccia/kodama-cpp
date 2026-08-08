// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint64_t hash_bytes(const void* data, std::size_t bytes, std::uint64_t hash) {
  const auto* values = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= values[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

kodama::Backend parse_backend(const std::string& value) {
  if (value == "cpu") return kodama::Backend::CPU;
  if (value == "cuda") return kodama::Backend::CUDA;
  if (value == "metal") return kodama::Backend::Metal;
  throw std::invalid_argument("backend must be cpu, cuda, or metal");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0] << " <samples> <neighbors> <runs> <threads> <backend>\n";
    return 2;
  }
  const int samples = std::stoi(argv[1]);
  const int neighbors = std::stoi(argv[2]);
  const int runs = std::stoi(argv[3]);
  const int threads = std::stoi(argv[4]);
  const kodama::Backend backend = parse_backend(argv[5]);
  if (samples <= neighbors || neighbors < 1 || runs < 1 || threads < 1) return 2;

  kodama::NeighborGraph graph;
  graph.neighbors = neighbors;
  const std::size_t edges = static_cast<std::size_t>(samples) * neighbors;
  graph.indices.resize(edges);
  graph.distances.resize(edges);
  for (int row = 0; row < samples; ++row) {
    for (int rank = 0; rank < neighbors; ++rank) {
      const std::size_t offset = static_cast<std::size_t>(row) * neighbors + rank;
      graph.indices[offset] = ((row + 1 + rank * 7919) % samples) + 1;
      graph.distances[offset] = 0.01f * static_cast<float>(rank + 1) +
        0.000001f * static_cast<float>(row % 997);
    }
  }
  std::vector<int> labels(static_cast<std::size_t>(runs) * samples);
  for (int run = 0; run < runs; ++run) {
    for (int row = 0; row < samples; ++row) {
      labels[static_cast<std::size_t>(run) * samples + row] =
        1 + ((row / 17 + run * 13 + (row % (run + 7))) % 101);
    }
  }

  const auto started = std::chrono::steady_clock::now();
  kodama::KODAMADissimilarityInPlace(
    graph, labels, runs, samples, backend, threads, 0
  );
  const double seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - started
  ).count();
  std::uint64_t hash = 1469598103934665603ULL;
  hash = hash_bytes(graph.indices.data(), graph.indices.size() * sizeof(int), hash);
  hash = hash_bytes(graph.distances.data(), graph.distances.size() * sizeof(float), hash);
  std::cout << "backend,samples,neighbors,runs,threads,seconds,hash\n"
            << kodama::to_string(backend) << ',' << samples << ',' << neighbors << ','
            << runs << ',' << threads << ',' << std::fixed << std::setprecision(6)
            << seconds << ',' << hash << '\n';
  return 0;
}
