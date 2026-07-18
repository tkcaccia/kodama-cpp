// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_binary(const std::string& path, std::size_t expected) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open " + path);
  std::vector<T> values(expected);
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected * sizeof(T)));
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("Unexpected binary size for " + path);
  }
  return values;
}

int parse_int(const char* text, const char* name) {
  try {
    return std::stoi(text);
  } catch (...) {
    throw std::invalid_argument(std::string("Invalid ") + name + ": " + text);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 8 || argc > 10) {
    std::cerr << "Usage: " << argv[0]
              << " <x_double_rowmajor.bin> <labels_int32.bin> <n> <p> <k> <exact|ivf> <gpu_device> [nlist] [nprobe]\n";
    return 2;
  }
  try {
    const int n = parse_int(argv[3], "n");
    const int p = parse_int(argv[4], "p");
    const int k = parse_int(argv[5], "k");
    const std::string index = argv[6];
    const int gpu_device = parse_int(argv[7], "gpu_device");
    const int nlist = argc >= 9 ? parse_int(argv[8], "nlist") : 0;
    const int nprobe = argc >= 10 ? parse_int(argv[9], "nprobe") : 0;
    if (n < 2 || p < 1 || k < 1) throw std::invalid_argument("n, p, and k must be positive.");
    if (index != "exact" && index != "ivf") throw std::invalid_argument("index must be exact or ivf.");

    const std::vector<double> x = read_binary<double>(argv[1], static_cast<std::size_t>(n) * p);
    const std::vector<std::int32_t> labels32 = read_binary<std::int32_t>(argv[2], static_cast<std::size_t>(n));
    const std::vector<int> labels(labels32.begin(), labels32.end());
    kodama::KNNOptions options;
    options.backend = kodama::Backend::CUDA;
    options.index_type = index == "exact" ?
      kodama::KNNIndexType::CudaExact : kodama::KNNIndexType::CudaIVFFlat;
    options.metric = kodama::DistanceMetric::Cosine;
    options.k = k;
    options.ivf_nlist = nlist;
    options.ivf_nprobe = nprobe;
    options.gpu_device = gpu_device;
    options.cv.folds = 5;
    options.cv.stratified = true;
    options.cv.seed = 123;

    const kodama::KNNCVResult result = kodama::KNNCV_CUDA(
      kodama::MatrixView(x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)),
      labels,
      {},
      options
    );
    std::cout << "index,accuracy,seconds,nlist,nprobe,pilot_recall\n"
              << kodama::to_string(result.parameters.index_type) << ','
              << result.global_accuracy << ','
              << result.runtime_seconds << ','
              << result.parameters.ivf_nlist << ','
              << result.parameters.ivf_nprobe << ','
              << result.parameters.ivf_pilot_recall << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
