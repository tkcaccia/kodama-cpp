// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kodama/kodama.hpp"

namespace {

struct BenchRow {
  std::string dataset;
  std::string function;
  std::string backend;
  std::string status;
  double accuracy = 0.0;
  double seconds = 0.0;
  double peak_memory_mb = 0.0;
  int selected_components = 0;
  std::string details;
};

std::vector<double> read_double_bin(const std::string& path, std::size_t n) {
  std::vector<double> out(n);
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open " + path);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(double)));
  if (!in) throw std::runtime_error("Could not read expected double count from " + path);
  return out;
}

std::vector<float> read_float_bin(const std::string& path, std::size_t n) {
  std::vector<float> out(n);
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open " + path);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
  if (!in) throw std::runtime_error("Could not read expected float count from " + path);
  return out;
}

std::vector<int> read_int_bin(const std::string& path, std::size_t n) {
  std::vector<int> out(n);
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open " + path);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(int)));
  if (!in) throw std::runtime_error("Could not read expected int count from " + path);
  return out;
}

bool use_float32_input(const std::string& path) {
  const char* dtype = std::getenv("KODAMA_INPUT_DTYPE");
  if (dtype != nullptr && std::string(dtype).empty() == false) {
    const std::string text(dtype);
    if (text == "float32" || text == "float" || text == "single") return true;
    if (text == "float64" || text == "double") return false;
    throw std::runtime_error("KODAMA_INPUT_DTYPE must be float32 or float64.");
  }
  return path.find("float32") != std::string::npos;
}

std::string csv_escape(const std::string& x) {
  std::string out = "\"";
  for (char c : x) out += c == '"' ? "\"\"" : std::string(1, c);
  out += "\"";
  return out;
}

template <class Fn>
BenchRow run_knn(const std::string& dataset, const std::string& name, const std::string& backend, Fn fn) {
  BenchRow row;
  row.dataset = dataset;
  row.function = name;
  row.backend = backend;
  const auto start = std::chrono::steady_clock::now();
  try {
    const kodama::KNNCVResult result = fn();
    row.status = "success";
    row.accuracy = result.global_accuracy;
    row.seconds = result.runtime_seconds;
    row.peak_memory_mb = result.peak_memory_mb;
    std::ostringstream os;
    os << "k=" << result.parameters.k
       << ";metric=" << kodama::to_string(result.parameters.metric)
       << ";index=" << kodama::to_string(result.parameters.index_type)
       << ";nlist=" << result.parameters.ivf_nlist
       << ";nprobe=" << result.parameters.ivf_nprobe
       << ";pilot_recall=" << result.parameters.ivf_pilot_recall
       << ";hnsw_m=" << result.parameters.hnsw_m
       << ";hnsw_ef_construction=" << result.parameters.hnsw_ef_construction
       << ";hnsw_ef_search=" << result.parameters.hnsw_ef_search
       << ";hnsw_tune_k=" << result.parameters.hnsw_tune_k
       << ";hnsw_target_recall=" << result.parameters.hnsw_target_recall;
    row.details = os.str();
  } catch (const std::exception& e) {
    row.status = "failed";
    row.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    row.details = e.what();
  }
  return row;
}

template <class Fn>
BenchRow run_pls(const std::string& dataset, const std::string& name, const std::string& backend, Fn fn) {
  BenchRow row;
  row.dataset = dataset;
  row.function = name;
  row.backend = backend;
  const auto start = std::chrono::steady_clock::now();
  try {
    const kodama::PLSCVResult result = fn();
    row.status = "success";
    row.accuracy = result.global_accuracy;
    row.seconds = result.runtime_seconds;
    row.peak_memory_mb = result.peak_memory_mb;
    row.selected_components = result.selected_components;
    std::ostringstream os;
    os << "mode=" << kodama::to_string(result.parameters.mode)
       << ";max_components=" << result.parameters.max_components
       << ";fixed_components=" << result.parameters.fixed_components
       << ";selected_components=" << result.parameters.selected_components;
    row.details = os.str();
  } catch (const std::exception& e) {
    row.status = "failed";
    row.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    row.details = e.what();
  }
  return row;
}

void write_csv(const std::string& path, const std::vector<BenchRow>& rows) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot write " + path);
  out << "dataset,function,backend,status,accuracy,seconds,peak_memory_mb,selected_components,details\n";
  out << std::fixed << std::setprecision(6);
  for (const BenchRow& row : rows) {
    out << csv_escape(row.dataset) << ','
        << csv_escape(row.function) << ','
        << csv_escape(row.backend) << ','
        << csv_escape(row.status) << ','
        << row.accuracy << ','
        << row.seconds << ','
        << row.peak_memory_mb << ','
        << row.selected_components << ','
        << csv_escape(row.details) << '\n';
  }
}

void write_csv_header(const std::string& path) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot write " + path);
  out << "dataset,function,backend,status,accuracy,seconds,peak_memory_mb,selected_components,details\n";
}

void append_csv_row(const std::string& path, const BenchRow& row) {
  std::ofstream out(path, std::ios::app);
  if (!out) throw std::runtime_error("Cannot append " + path);
  out << std::fixed << std::setprecision(6)
      << csv_escape(row.dataset) << ','
      << csv_escape(row.function) << ','
      << csv_escape(row.backend) << ','
      << csv_escape(row.status) << ','
      << row.accuracy << ','
      << row.seconds << ','
      << row.peak_memory_mb << ','
      << row.selected_components << ','
      << csv_escape(row.details) << '\n';
}

void print_row(const BenchRow& row) {
  std::cout << std::fixed << std::setprecision(6)
            << row.dataset << ','
            << row.function << ','
            << row.backend << ','
            << row.status << ','
            << row.accuracy << ','
            << row.seconds << ','
            << row.peak_memory_mb << ','
            << row.selected_components << ','
            << row.details << '\n'
            << std::flush;
}

bool method_enabled(const std::string& name) {
  const char* filter = std::getenv("KODAMA_BENCH_METHODS");
  if (filter == nullptr || std::string(filter).empty()) return true;
  std::stringstream ss(filter);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item == name) return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 9) {
    std::cerr << "Usage: " << argv[0]
              << " <dataset> <x_rowmajor.bin> <labels_int32.bin> <n> <p> <output.csv> <n_threads> <pls_max_components> [ivf_nlist] [ivf_nprobe]\n";
    return 2;
  }

  const std::string dataset = argv[1];
  const std::string x_path = argv[2];
  const std::string y_path = argv[3];
  const std::size_t n = static_cast<std::size_t>(std::stoull(argv[4]));
  const std::size_t p = static_cast<std::size_t>(std::stoull(argv[5]));
  const std::string out_path = argv[6];
  const int n_threads = std::stoi(argv[7]);
  const int pls_max_components = std::stoi(argv[8]);
  const int ivf_nlist = argc > 9 ? std::stoi(argv[9]) : 0;
  const int ivf_nprobe = argc > 10 ? std::stoi(argv[10]) : 0;

  std::vector<double> x64;
  std::vector<float> x32;
  kodama::MatrixView view;
  if (use_float32_input(x_path)) {
    x32 = read_float_bin(x_path, n * p);
    view = kodama::MatrixView{x32.data(), n, p};
  } else {
    x64 = read_double_bin(x_path, n * p);
    view = kodama::MatrixView{x64.data(), n, p};
  }
  std::vector<int> y = read_int_bin(y_path, n);
  std::vector<int> constrain;

  kodama::KNNOptions knn;
  knn.cv.folds = 5;
  knn.cv.seed = 7;
  knn.cv.stratified = true;
  knn.k = 10;
  knn.metric = kodama::DistanceMetric::Cosine;
  knn.ivf_nlist = ivf_nlist;
  knn.ivf_nprobe = ivf_nprobe;
  knn.n_threads = n_threads;

  kodama::PLSOptions pls;
  pls.cv.folds = 5;
  pls.cv.seed = 7;
  pls.cv.stratified = true;
  pls.max_components = pls_max_components;
  pls.fixed_components = pls_max_components;
  pls.center = true;
  pls.scale = true;
  pls.n_threads = n_threads;

  write_csv_header(out_path);
  std::cout << "dataset,function,backend,status,accuracy,seconds,peak_memory_mb,selected_components,details\n" << std::flush;
  auto record = [&](BenchRow row) {
    append_csv_row(out_path, row);
    print_row(row);
  };

  if (method_enabled("KNNCV_CPU")) {
    record(run_knn(dataset, "KNNCV_CPU", "cpu", [&] {
      kodama::KNNOptions opt = knn;
      opt.backend = kodama::Backend::CPU;
      return kodama::KNNCV_CPU(view, y, constrain, opt);
    }));
  }
  if (method_enabled("KNNCV_CUDA")) {
    record(run_knn(dataset, "KNNCV_CUDA", "cuda", [&] {
      kodama::KNNOptions opt = knn;
      opt.backend = kodama::Backend::CUDA;
      opt.gpu_device = 0;
      return kodama::KNNCV_CUDA(view, y, constrain, opt);
    }));
  }
  if (method_enabled("KNNCV_METAL") && kodama::MetalAvailable()) {
    record(run_knn(dataset, "KNNCV_METAL", "metal", [&] {
      kodama::KNNOptions opt = knn;
      opt.backend = kodama::Backend::Metal;
      opt.index_type = kodama::KNNIndexType::MetalExact;
      return kodama::KNNCV_METAL(view, y, constrain, opt);
    }));
  }
  if (method_enabled("KNNCV_METAL_IVF") && kodama::MetalAvailable()) {
    record(run_knn(dataset, "KNNCV_METAL_IVF", "metal", [&] {
      kodama::KNNOptions opt = knn;
      opt.backend = kodama::Backend::Metal;
      opt.index_type = kodama::KNNIndexType::MetalIVFFlat;
      return kodama::KNNCV_METAL(view, y, constrain, opt);
    }));
  }
  if (method_enabled("PLSDACV_CPU")) {
    record(run_pls(dataset, "PLSDACV_CPU", "cpu", [&] {
      kodama::PLSOptions opt = pls;
      opt.backend = kodama::Backend::CPU;
      return kodama::PLSDACV_CPU(view, y, constrain, opt);
    }));
  }
  if (method_enabled("PLSLDACV_CPU")) {
    record(run_pls(dataset, "PLSLDACV_CPU", "cpu", [&] {
      kodama::PLSOptions opt = pls;
      opt.backend = kodama::Backend::CPU;
      return kodama::PLSLDACV_CPU(view, y, constrain, opt);
    }));
  }
  if (method_enabled("PLSDACV_CUDA")) {
    record(run_pls(dataset, "PLSDACV_CUDA", "cuda", [&] {
      kodama::PLSOptions opt = pls;
      opt.backend = kodama::Backend::CUDA;
      opt.gpu_device = 0;
      return kodama::PLSDACV_CUDA(view, y, constrain, opt);
    }));
  }
  if (method_enabled("PLSLDACV_CUDA")) {
    record(run_pls(dataset, "PLSLDACV_CUDA", "cuda", [&] {
      kodama::PLSOptions opt = pls;
      opt.backend = kodama::Backend::CUDA;
      opt.gpu_device = 0;
      return kodama::PLSLDACV_CUDA(view, y, constrain, opt);
    }));
  }
  if (method_enabled("PLSLDACV_METAL") && kodama::MetalAvailable()) {
    record(run_pls(dataset, "PLSLDACV_METAL", "metal", [&] {
      kodama::PLSOptions opt = pls;
      opt.backend = kodama::Backend::Metal;
      return kodama::PLSLDACV_METAL(view, y, constrain, opt);
    }));
  }
  return 0;
}
