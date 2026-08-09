// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template<class T>
std::vector<T> read_binary(const std::string& path, std::size_t size) {
  std::vector<T> values(size);
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open " + path);
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!input) throw std::runtime_error("Cannot read " + path);
  return values;
}

template<class T>
void write_binary(const std::string& path, const std::vector<T>& values) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("Cannot create " + path);
  output.write(
    reinterpret_cast<const char*>(values.data()),
    static_cast<std::streamsize>(values.size() * sizeof(T))
  );
  if (!output) throw std::runtime_error("Cannot write " + path);
}

double choose_two(long long value) {
  return value < 2 ? 0.0 : static_cast<double>(value) * static_cast<double>(value - 1) * 0.5;
}

double adjusted_rand_index(const int* clusters, const std::vector<int>& truth, int rows) {
  std::map<int, std::map<int, long long>> cells;
  std::map<int, long long> cluster_counts;
  std::map<int, long long> truth_counts;
  for (int row = 0; row < rows; ++row) {
    ++cells[clusters[row]][truth[static_cast<std::size_t>(row)]];
    ++cluster_counts[clusters[row]];
    ++truth_counts[truth[static_cast<std::size_t>(row)]];
  }
  double agreements = 0.0;
  for (const auto& cluster : cells) {
    for (const auto& cell : cluster.second) agreements += choose_two(cell.second);
  }
  double cluster_pairs = 0.0;
  for (const auto& item : cluster_counts) cluster_pairs += choose_two(item.second);
  double truth_pairs = 0.0;
  for (const auto& item : truth_counts) truth_pairs += choose_two(item.second);
  const double all_pairs = choose_two(rows);
  const double expected = all_pairs > 0.0 ? cluster_pairs * truth_pairs / all_pairs : 0.0;
  const double denominator = 0.5 * (cluster_pairs + truth_pairs) - expected;
  return std::abs(denominator) < 1e-15 ? 0.0 : (agreements - expected) / denominator;
}

int unique_labels(const int* labels, int rows) {
  std::vector<int> values(labels, labels + rows);
  std::sort(values.begin(), values.end());
  return static_cast<int>(std::unique(values.begin(), values.end()) - values.begin());
}

double median_classes(const kodama::KODAMAMatrixResult& result) {
  std::vector<int> counts(static_cast<std::size_t>(result.runs));
  for (int run = 0; run < result.runs; ++run) {
    counts[static_cast<std::size_t>(run)] = unique_labels(
      result.res.data() + static_cast<std::size_t>(run) * result.samples,
      result.samples
    );
  }
  std::sort(counts.begin(), counts.end());
  const std::size_t middle = counts.size() / 2;
  return counts.size() % 2 == 0 ?
    0.5 * static_cast<double>(counts[middle - 1] + counts[middle]) :
    static_cast<double>(counts[middle]);
}

bool method_enabled(const std::string& name) {
  const char* filter = std::getenv("KODAMA_BENCH_METHODS");
  return filter == nullptr || std::string(filter).empty() || std::string(filter).find(name) != std::string::npos;
}

double sum_seconds(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0);
}

double median_seconds(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ?
    0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

void print_result(
  const char* method,
  const char* backend,
  const kodama::KODAMAMatrixResult& result,
  const std::vector<int>& truth,
  const kodama::MatrixView data
) {
  double best_ari = -1.0;
  int best_classes = 0;
  for (int run = 0; run < result.runs; ++run) {
    const int* labels = result.res.data() + static_cast<std::size_t>(run) * static_cast<std::size_t>(result.samples);
    const double ari = adjusted_rand_index(labels, truth, result.samples);
    if (ari > best_ari) {
      best_ari = ari;
      best_classes = unique_labels(labels, result.samples);
    }
  }
  const double best_cv = *std::max_element(result.acc.begin(), result.acc.end());
  std::cout << method << ',' << backend << ',' << std::fixed << std::setprecision(6)
            << result.runtime_seconds << ',' << result.graph_seconds << ','
            << best_cv << ',' << best_ari << ',' << best_classes << ','
            << result.graph_ivf_nlist << ',' << result.graph_ivf_nprobe << ','
            << result.graph_ivf_pilot_recall << ',' << median_classes(result) << ','
            << result.projection_sparse_uploads << ','
            << result.projection_full_downloads << ','
            << result.result_row_uploads << ','
            << result.result_matrix_downloads << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 9) {
    std::cerr << "Usage: " << argv[0]
              << " <x_float32_rowmajor.bin> <labels_int32.bin> <n> <p> <M> <Tcycle> <components> <threads>\n";
    return 2;
  }
  const std::string x_path = argv[1];
  const std::string labels_path = argv[2];
  const int rows = std::stoi(argv[3]);
  const int columns = std::stoi(argv[4]);
  const int runs = std::stoi(argv[5]);
  const int cycles = std::stoi(argv[6]);
  const int components = std::stoi(argv[7]);
  const int threads = std::stoi(argv[8]);
  const std::vector<float> data = read_binary<float>(x_path,
    static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns));
  const std::vector<std::int32_t> raw_labels = read_binary<std::int32_t>(labels_path, rows);
  const std::vector<int> labels(raw_labels.begin(), raw_labels.end());
  const kodama::MatrixView view{data.data(), static_cast<std::size_t>(rows), static_cast<std::size_t>(columns)};

  kodama::KODAMAMatrixOptions options;
  options.runs = runs;
  options.cycles = cycles;
  options.components = components;
  options.landmarks = 100000;
  if (const char* value = std::getenv("KODAMA_LANDMARKS")) {
    options.landmarks = std::max(1, std::stoi(value));
  }
  options.splitting = rows < 40000 ? 100 : 300;
  options.graph_neighbors = 100;
  options.n_threads = threads;
  options.knn.k = 30;
  options.knn.cv.folds = 5;
  options.knn.n_threads = threads;
  options.pls.cv.folds = 5;
  options.pls.n_threads = threads;
  options.seed = 1234;
  if (const char* path = std::getenv("KODAMA_SPATIAL_BIN")) {
    const char* cols_value = std::getenv("KODAMA_SPATIAL_COLS");
    if (cols_value == nullptr) {
      throw std::invalid_argument("KODAMA_SPATIAL_COLS is required with KODAMA_SPATIAL_BIN.");
    }
    options.spatial_cols = std::stoi(cols_value);
    options.spatial = read_binary<float>(
      path, static_cast<std::size_t>(rows) * options.spatial_cols);
    options.spatial_resolution = 0.3;
    options.spatial_constraint_mode = 0;
    options.spatial_graph_mix = true;
  }
  if (const char* value = std::getenv("KODAMA_RETURN_GRAPH")) {
    options.materialize_graph = std::string(value) != "0";
  }
  const char* metal_index = std::getenv("KODAMA_METAL_INDEX");
  if (metal_index != nullptr && std::string(metal_index) == "ivf") {
    options.knn.index_type = kodama::KNNIndexType::MetalIVFFlat;
  }
  const char* cuda_index = std::getenv("KODAMA_CUDA_INDEX");
  if (cuda_index != nullptr && std::string(cuda_index) == "exact") {
    options.knn.index_type = kodama::KNNIndexType::CudaExact;
  } else if (cuda_index != nullptr && std::string(cuda_index) == "ivf") {
    options.knn.index_type = kodama::KNNIndexType::CudaIVFFlat;
  }

  std::cout << "method,backend,seconds,graph_seconds,best_cv_accuracy,best_ari,best_classes,nlist,nprobe,pilot_recall,median_classes,sparse_projection_uploads,full_projection_downloads,result_row_uploads,result_matrix_downloads\n";
  options.classifier = kodama::CoreClassifier::KNN;
  if (method_enabled("KODAMA_KNN_CPU")) {
    options.backend = kodama::Backend::CPU;
    print_result("KODAMA_KNN", "cpu", kodama::KODAMAMatrix_CPU(view, {}, {}, {}, options), labels, view);
  }
  if (method_enabled("KODAMA_KNN_METAL") && kodama::MetalAvailable()) {
    options.backend = kodama::Backend::Metal;
    print_result("KODAMA_KNN", "metal", kodama::KODAMAMatrix_METAL(view, {}, {}, {}, options), labels, view);
  }
#if defined(KODAMA_ENABLE_CUDA)
  if (method_enabled("KODAMA_KNN_CUDA")) {
    options.backend = kodama::Backend::CUDA;
    print_result("KODAMA_KNN", "cuda", kodama::KODAMAMatrix_CUDA(view, {}, {}, {}, options), labels);
  }
#endif

  options.classifier = kodama::CoreClassifier::PLS_LDA;
  if (method_enabled("KODAMA_PLSLDA_CPU")) {
    options.backend = kodama::Backend::CPU;
    print_result("KODAMA_PLSLDA", "cpu", kodama::KODAMAMatrix_CPU(view, {}, {}, {}, options), labels, view);
  }
  if (method_enabled("KODAMA_PLSLDA_METAL") && kodama::MetalAvailable()) {
    options.backend = kodama::Backend::Metal;
    print_result("KODAMA_PLSLDA", "metal", kodama::KODAMAMatrix_METAL(view, {}, {}, {}, options), labels, view);
  }
#if defined(KODAMA_ENABLE_CUDA)
  if (method_enabled("KODAMA_PLSLDA_CUDA")) {
    options.backend = kodama::Backend::CUDA;
    print_result("KODAMA_PLSLDA", "cuda", kodama::KODAMAMatrix_CUDA(view, {}, {}, {}, options), labels);
  }
#endif
  return 0;
}
