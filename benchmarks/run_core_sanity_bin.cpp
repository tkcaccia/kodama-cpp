#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "kodama/kodama.hpp"

namespace {

std::vector<double> read_double_bin(const std::string& path, std::size_t n) {
  std::vector<double> out(n);
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open " + path);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(double)));
  if (!in) throw std::runtime_error("Cannot read " + path);
  return out;
}

std::vector<float> read_float_bin(const std::string& path, std::size_t n) {
  std::vector<float> out(n);
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open " + path);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
  if (!in) throw std::runtime_error("Cannot read " + path);
  return out;
}

std::vector<int> read_int_bin(const std::string& path, std::size_t n) {
  std::vector<std::int32_t> tmp(n);
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open " + path);
  in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(tmp.size() * sizeof(std::int32_t)));
  if (!in) throw std::runtime_error("Cannot read " + path);
  return std::vector<int>(tmp.begin(), tmp.end());
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

double agreement(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.size() != b.size()) throw std::runtime_error("agreement size mismatch");
  int ok = 0;
  for (std::size_t i = 0; i < a.size(); ++i) ok += a[i] == b[i] ? 1 : 0;
  return static_cast<double>(ok) / static_cast<double>(a.size());
}

double mapped_agreement(const std::vector<int>& clusters, const std::vector<int>& truth) {
  if (clusters.size() != truth.size()) throw std::runtime_error("mapped agreement size mismatch");
  std::map<int, std::map<int, int>> counts;
  for (std::size_t i = 0; i < clusters.size(); ++i) counts[clusters[i]][truth[i]]++;
  std::map<int, int> mapping;
  for (const auto& kv : counts) {
    int best_label = kv.second.begin()->first;
    int best_count = kv.second.begin()->second;
    for (const auto& lc : kv.second) {
      if (lc.second > best_count) {
        best_label = lc.first;
        best_count = lc.second;
      }
    }
    mapping[kv.first] = best_label;
  }
  int ok = 0;
  for (std::size_t i = 0; i < clusters.size(); ++i) ok += mapping[clusters[i]] == truth[i] ? 1 : 0;
  return static_cast<double>(ok) / static_cast<double>(clusters.size());
}

double adjusted_rand_index(const std::vector<int>& clusters, const std::vector<int>& truth) {
  if (clusters.size() != truth.size()) throw std::runtime_error("ARI size mismatch");
  const double n = static_cast<double>(clusters.size());
  if (n < 2.0) return 1.0;

  std::map<int, std::map<int, long long>> contingency;
  std::map<int, long long> cluster_sum;
  std::map<int, long long> truth_sum;
  for (std::size_t i = 0; i < clusters.size(); ++i) {
    contingency[clusters[i]][truth[i]]++;
    cluster_sum[clusters[i]]++;
    truth_sum[truth[i]]++;
  }

  auto choose2 = [](long long x) -> double {
    return x < 2 ? 0.0 : static_cast<double>(x) * static_cast<double>(x - 1) * 0.5;
  };

  double sum_comb = 0.0;
  for (const auto& row : contingency) {
    for (const auto& cell : row.second) sum_comb += choose2(cell.second);
  }

  double sum_cluster = 0.0;
  for (const auto& kv : cluster_sum) sum_cluster += choose2(kv.second);

  double sum_truth = 0.0;
  for (const auto& kv : truth_sum) sum_truth += choose2(kv.second);

  const double total = choose2(static_cast<long long>(clusters.size()));
  if (total == 0.0) return 1.0;
  const double expected = (sum_cluster * sum_truth) / total;
  const double max_index = 0.5 * (sum_cluster + sum_truth);
  const double denom = max_index - expected;
  if (std::abs(denom) < 1e-15) return std::abs(sum_comb - expected) < 1e-15 ? 1.0 : 0.0;
  return (sum_comb - expected) / denom;
}

std::vector<int> noisy_labels(const std::vector<int>& labels) {
  std::vector<int> classes = labels;
  std::sort(classes.begin(), classes.end());
  classes.erase(std::unique(classes.begin(), classes.end()), classes.end());

  std::vector<int> out = labels;
  if (classes.size() < 2) return out;
  for (std::size_t i = 0; i < out.size(); i += 17) {
    const auto it = std::lower_bound(classes.begin(), classes.end(), out[i]);
    const std::size_t pos = it == classes.end() ? 0 : static_cast<std::size_t>(it - classes.begin());
    out[i] = classes[(pos + 1) % classes.size()];
  }
  return out;
}

template <class Real>
std::vector<int> kmeans_labels(const std::vector<Real>& x, int n, int p, int k) {
  if (k < 1) throw std::runtime_error("k-means k must be positive");
  std::vector<float> xf(static_cast<std::size_t>(n) * static_cast<std::size_t>(p));
  for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = static_cast<float>(x[i]);

  const int sample_n = std::min(n, std::max(k, 10000));
  std::vector<int> sample_rows(static_cast<std::size_t>(sample_n));
  for (int i = 0; i < sample_n; ++i) {
    sample_rows[static_cast<std::size_t>(i)] = static_cast<int>((static_cast<long long>(i) * n) / sample_n);
  }

  std::vector<float> centers(static_cast<std::size_t>(k) * static_cast<std::size_t>(p), 0.0f);
  for (int c = 0; c < k; ++c) {
    const int row = sample_rows[static_cast<std::size_t>((static_cast<long long>(c) * sample_n) / k)];
    std::copy_n(xf.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(p), p,
                centers.data() + static_cast<std::size_t>(c) * static_cast<std::size_t>(p));
  }

  std::vector<float> sample_x(static_cast<std::size_t>(sample_n) * static_cast<std::size_t>(p));
  for (int i = 0; i < sample_n; ++i) {
    const int row = sample_rows[static_cast<std::size_t>(i)];
    std::copy_n(xf.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(p), p,
                sample_x.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(p));
  }

  auto nearest_centroid = [&](const float* point) {
    int best = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for (int c = 0; c < k; ++c) {
      const float* center = centers.data() + static_cast<std::size_t>(c) * static_cast<std::size_t>(p);
      double distance = 0.0;
      for (int j = 0; j < p; ++j) {
        const double delta = static_cast<double>(point[j]) - static_cast<double>(center[j]);
        distance += delta * delta;
      }
      if (distance < best_distance || (distance == best_distance && c < best)) {
        best = c;
        best_distance = distance;
      }
    }
    return best;
  };

  std::vector<int> assign(static_cast<std::size_t>(sample_n), -1);
  for (int iter = 0; iter < 12; ++iter) {
    for (int i = 0; i < sample_n; ++i) {
      assign[static_cast<std::size_t>(i)] = nearest_centroid(
        sample_x.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(p)
      );
    }

    std::vector<float> next(centers.size(), 0.0f);
    std::vector<int> counts(static_cast<std::size_t>(k), 0);
    for (int i = 0; i < sample_n; ++i) {
      const int c = assign[static_cast<std::size_t>(i)];
      if (c < 0) continue;
      counts[static_cast<std::size_t>(c)]++;
      float* dst = next.data() + static_cast<std::size_t>(c) * static_cast<std::size_t>(p);
      const float* src = sample_x.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(p);
      for (int j = 0; j < p; ++j) dst[j] += src[j];
    }
    for (int c = 0; c < k; ++c) {
      float* dst = next.data() + static_cast<std::size_t>(c) * static_cast<std::size_t>(p);
      if (counts[static_cast<std::size_t>(c)] == 0) {
        std::copy_n(centers.data() + static_cast<std::size_t>(c) * static_cast<std::size_t>(p), p, dst);
      } else {
        const float inv = 1.0f / static_cast<float>(counts[static_cast<std::size_t>(c)]);
        for (int j = 0; j < p; ++j) dst[j] *= inv;
      }
    }
    centers.swap(next);
  }

  std::vector<int> out(static_cast<std::size_t>(n), 1);
  for (int i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i)] = nearest_centroid(
      xf.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(p)
    ) + 1;
  }
  return out;
}

bool method_enabled(const std::string& method) {
  const char* filter = std::getenv("KODAMA_CORE_METHODS");
  if (filter == nullptr || std::string(filter).empty()) return true;
  const std::string methods(filter);
  std::size_t start = 0;
  while (start <= methods.size()) {
    const std::size_t comma = methods.find(',', start);
    const std::string token = methods.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (token == method) return true;
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return false;
}

int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  return value == nullptr || std::string(value).empty() ? fallback : std::stoi(value);
}

bool env_bool(const char* name, bool fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) return fallback;
  const std::string text(value);
  return text == "1" || text == "true" || text == "TRUE" || text == "yes";
}

void apply_core_objective_env(kodama::CoreOptions& options) {
  options.auto_class_coarsening = env_bool("KODAMA_CORE_AUTO_COARSEN", options.auto_class_coarsening);
}

int unique_count(const std::vector<int>& labels) {
  std::vector<int> unique = labels;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  return static_cast<int>(unique.size());
}

void write_labels_if_requested(const std::string& method, const std::vector<int>& labels) {
  const char* prefix = std::getenv("KODAMA_CORE_WRITE_PREFIX");
  if (prefix == nullptr || std::string(prefix).empty()) return;
  std::ofstream out(std::string(prefix) + "_" + method + "_clbest.csv");
  if (!out) throw std::runtime_error("Cannot write core labels output.");
  out << "index,label\n";
  for (std::size_t i = 0; i < labels.size(); ++i) out << (i + 1) << ',' << labels[i] << '\n';
}

template <class Fn>
void run_one(
  const std::string& dataset,
  const std::string& method,
  const std::vector<int>& initial,
  const std::vector<int>& truth,
  Fn&& fn
) {
  const auto start = std::chrono::steady_clock::now();
  const kodama::CoreResult result = fn();
  const auto end = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(end - start).count();
  std::cout << dataset << ',' << method << ',' << std::fixed << std::setprecision(6)
            << result.accbest << ',' << result.scorebest << ',' << seconds << ','
            << result.cycles_completed << ',' << unique_count(result.clbest) << ','
            << (result.success ? 1 : 0) << ','
            << agreement(initial, truth) << ',' << agreement(result.clbest, truth) << ','
            << mapped_agreement(initial, truth) << ',' << mapped_agreement(result.clbest, truth) << ','
            << adjusted_rand_index(initial, truth) << ',' << adjusted_rand_index(result.clbest, truth) << '\n';
  write_labels_if_requested(method, result.clbest);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: kodama_core_sanity_bin DATASET X_BIN Y_BIN N P [CYCLES]\n";
    return 2;
  }

  const std::string dataset = argv[1];
  const std::string x_path = argv[2];
  const std::string y_path = argv[3];
  const int n = std::stoi(argv[4]);
  const int p = std::stoi(argv[5]);
  const int cycles = argc > 6 ? std::stoi(argv[6]) : 30;

  const std::size_t element_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(p);
  std::vector<double> x64;
  std::vector<float> x32;
  kodama::MatrixView view;
  if (use_float32_input(x_path)) {
    x32 = read_float_bin(x_path, element_count);
    view = kodama::MatrixView{x32.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)};
  } else {
    x64 = read_double_bin(x_path, element_count);
    view = kodama::MatrixView{x64.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)};
  }
  std::vector<int> truth = read_int_bin(y_path, static_cast<std::size_t>(n));
  const char* init_mode_env = std::getenv("KODAMA_CORE_INIT");
  const std::string init_mode = init_mode_env == nullptr ? "noisy" : init_mode_env;
  const int init_k = env_int("KODAMA_CORE_K", 100);
  const int core_seed = env_int("KODAMA_CORE_SEED", 123);
  const int core_knn_k = env_int("KODAMA_CORE_KNN_K", 10);
  std::vector<int> init;
  if (init_mode == "kmeans") {
    init = x32.empty() ? kmeans_labels(x64, n, p, init_k) : kmeans_labels(x32, n, p, init_k);
  } else {
    init = noisy_labels(truth);
  }
  std::vector<int> constrain(static_cast<std::size_t>(n));
  std::vector<int> fixed(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) constrain[static_cast<std::size_t>(i)] = i;

  std::cout << "dataset,method,accbest,scorebest,seconds,cycles_completed,final_classes,success,initial_agreement,final_agreement,initial_mapped_agreement,final_mapped_agreement,initial_ari,final_ari\n";

  kodama::CoreOptions pls;
  pls.cycles = cycles;
  pls.seed = static_cast<std::uint64_t>(core_seed);
  pls.classifier = kodama::CoreClassifier::PLS_LDA;
  pls.pls.max_components = 10;
  pls.pls.fixed_components = 10;
  pls.pls.cv.folds = 5;
  pls.pls.cv.seed = 123;
  pls.pls.backend = kodama::Backend::CPU;
  apply_core_objective_env(pls);
  if (method_enabled("core_pls_lda_cpu")) {
    run_one(dataset, "core_pls_lda_cpu", init, truth, [&] { return kodama::CorePLSLDA_CPU(view, init, constrain, fixed, pls); });
  }

  kodama::CoreOptions knn;
  knn.cycles = cycles;
  knn.seed = static_cast<std::uint64_t>(core_seed);
  knn.classifier = kodama::CoreClassifier::KNN;
  knn.knn.cv.folds = 5;
  knn.knn.cv.seed = 123;
  knn.knn.k = core_knn_k;
  knn.knn.backend = kodama::Backend::CPU;
  apply_core_objective_env(knn);
  if (method_enabled("core_knn_cpu")) {
    run_one(dataset, "core_knn_cpu", init, truth, [&] { return kodama::CoreKNN_CPU(view, init, constrain, fixed, knn); });
  }

#if defined(KODAMA_ENABLE_CUDA)
  kodama::CoreOptions cuda_pls = pls;
  cuda_pls.pls.backend = kodama::Backend::CUDA;
  if (method_enabled("core_pls_lda_cuda")) {
    run_one(dataset, "core_pls_lda_cuda", init, truth, [&] { return kodama::CorePLSLDA_CUDA(view, init, constrain, fixed, cuda_pls); });
  }

  kodama::CoreOptions cuda_knn = knn;
  cuda_knn.knn.backend = kodama::Backend::CUDA;
  cuda_knn.knn.ivf_nlist = std::max(1, std::min(64, n / 4));
  cuda_knn.knn.ivf_nprobe = std::max(1, std::min(8, cuda_knn.knn.ivf_nlist));
  if (method_enabled("core_knn_cuda")) {
    run_one(dataset, "core_knn_cuda", init, truth, [&] { return kodama::CoreKNN_CUDA(view, init, constrain, fixed, cuda_knn); });
  }
#endif

  return 0;
}
