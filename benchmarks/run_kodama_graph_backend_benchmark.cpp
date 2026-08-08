// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> synthetic_matrix(int rows, int columns) {
  std::vector<float> data(static_cast<std::size_t>(rows) * columns);
  std::mt19937 generator(41u);
  std::normal_distribution<float> noise(0.0f, 0.35f);
  constexpr int classes = 32;
  for (int row = 0; row < rows; ++row) {
    const int cls = row % classes;
    for (int column = 0; column < columns; ++column) {
      data[static_cast<std::size_t>(row) * columns + column] =
        (column % classes == cls ? 2.5f : 0.0f) + noise(generator);
    }
  }
  return data;
}

kodama::Backend parse_backend(const std::string& value) {
  if (value == "cuda") return kodama::Backend::CUDA;
  if (value == "metal") return kodama::Backend::Metal;
  throw std::invalid_argument("backend must be cuda or metal");
}

kodama::KNNIndexType parse_index(const std::string& backend, const std::string& value) {
  if (value == "auto") return kodama::KNNIndexType::NativeHNSW;
  if (backend == "cuda" && value == "exact") return kodama::KNNIndexType::CudaExact;
  if (backend == "cuda" && value == "ivf") return kodama::KNNIndexType::CudaIVFFlat;
  if (backend == "metal" && value == "exact") return kodama::KNNIndexType::MetalExact;
  if (backend == "metal" && value == "ivf") return kodama::KNNIndexType::MetalIVFFlat;
  throw std::invalid_argument("index must be auto, exact, or ivf");
}

std::vector<float> read_matrix(const std::string& path, int rows, int columns) {
  if (path == "-") return synthetic_matrix(rows, columns);
  const std::size_t count = static_cast<std::size_t>(rows) * columns;
  std::vector<float> data(count);
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open input matrix: " + path);
  input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
  if (input.gcount() != static_cast<std::streamsize>(count * sizeof(float)) || input.peek() != EOF) {
    throw std::runtime_error("input matrix size does not match rows and columns");
  }
  return data;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6 && argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <cuda|metal> <auto|exact|ivf> <rows> <columns> <k> [float32-row-major-file|-]\n";
    return 2;
  }
  try {
    const std::string backend_name = argv[1];
    const std::string index_name = argv[2];
    const int rows = std::stoi(argv[3]);
    const int columns = std::stoi(argv[4]);
    const int k = std::stoi(argv[5]);
    std::vector<float> data = read_matrix(argc == 7 ? argv[6] : "-", rows, columns);
    kodama::KODAMAGraphOptions options;
    options.backend = parse_backend(backend_name);
    options.index_type = parse_index(backend_name, index_name);
    options.metric = kodama::DistanceMetric::Euclidean;
    options.neighbors = k;
    options.n_threads = 4;
    options.seed = 41;
    const auto start = std::chrono::steady_clock::now();
    const kodama::KODAMAGraphResult result = kodama::KODAMAGraph(
      kodama::MatrixView{data.data(), static_cast<std::size_t>(rows), static_cast<std::size_t>(columns)},
      options
    );
    const double wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start
    ).count();
    std::cout
      << "backend,requested_index,used_index,rows,columns,k,graph_seconds,visual_init_seconds,wall_seconds,nlist,nprobe,pilot_recall\n"
      << backend_name << ',' << index_name << ',' << kodama::to_string(result.index_type) << ','
      << rows << ',' << columns << ',' << k << ','
      << std::fixed << std::setprecision(6)
      << result.graph_seconds << ',' << result.visual_init_seconds << ',' << wall_seconds << ','
      << result.ivf_nlist << ',' << result.ivf_nprobe << ',' << result.ivf_pilot_recall << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
