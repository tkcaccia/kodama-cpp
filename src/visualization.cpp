#include "kodama/kodama.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace kodama {
namespace {

using Clock = std::chrono::steady_clock;

std::pair<double, double> find_ab_params(const double spread, const double min_dist) {
  if (std::abs(spread - 1.0) < 1e-12 && std::abs(min_dist - 0.1) < 1e-12) {
    return {1.5769434601962196, 0.8950608781227859};
  }

  std::vector<double> xs;
  std::vector<double> ys;
  xs.reserve(300);
  ys.reserve(300);
  for (int i = 0; i < 300; ++i) {
    const double x = (spread * 3.0) * static_cast<double>(i) / 299.0;
    xs.push_back(x);
    ys.push_back(x < min_dist ? 1.0 : std::exp(-(x - min_dist) / spread));
  }

  double best_a = 1.5769434601962196;
  double best_b = 0.8950608781227859;
  double best_loss = std::numeric_limits<double>::infinity();

  for (double loga = -4.0; loga <= 4.0001; loga += 0.2) {
    for (double b = 0.25; b <= 2.0001; b += 0.05) {
      const double a = std::exp(loga);
      double loss = 0.0;
      for (std::size_t i = 0; i < xs.size(); ++i) {
        const double x2b = std::pow(xs[i], 2.0 * b);
        const double yhat = 1.0 / (1.0 + a * x2b);
        const double e = yhat - ys[i];
        loss += e * e;
      }
      if (loss < best_loss) {
        best_loss = loss;
        best_a = a;
        best_b = b;
      }
    }
  }

  for (int iter = 0; iter < 80; ++iter) {
    double ga = 0.0;
    double gb = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
      const double x = std::max(xs[i], 1e-6);
      const double x2b = std::pow(x, 2.0 * best_b);
      const double denom = 1.0 + best_a * x2b;
      const double yhat = 1.0 / denom;
      const double e = yhat - ys[i];
      ga += e * (-x2b / (denom * denom));
      gb += e * (-(best_a * x2b * 2.0 * std::log(x)) / (denom * denom));
    }
    best_a = std::max(1e-4, best_a - 0.01 * ga);
    best_b = std::max(0.1, best_b - 0.01 * gb);
  }

  return {best_a, best_b};
}

std::vector<float> make_opentsne_random_init(const int n, const int n_components, const int seed) {
  std::vector<float> init(static_cast<std::size_t>(n) * static_cast<std::size_t>(n_components), 0.0f);
  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  std::normal_distribution<float> normal(0.0f, 1.0e-4f);
  for (int c = 0; c < n_components; ++c) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) {
      const float value = normal(rng);
      init[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_components) +
           static_cast<std::size_t>(c)] = value;
      mean += static_cast<double>(value);
    }
    mean /= static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
      init[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_components) +
           static_cast<std::size_t>(c)] -= static_cast<float>(mean);
    }
  }
  return init;
}

struct PreparedGraph {
  std::vector<int> indices;
  std::vector<float> distances;
  int samples = 0;
  int neighbors = 0;
  int index_offset = 0;
};

struct CsrGraph {
  std::vector<int> offsets;
  std::vector<int> neighbors;
  std::vector<float> weights;
  std::vector<float> epochs_per_sample;
  float max_weight = 1.0f;
};

PreparedGraph prepare_graph(const NeighborGraph& graph, int requested_neighbors) {
  if (graph.neighbors <= 0) {
    throw std::invalid_argument("NeighborGraph.neighbors must be positive.");
  }
  if (graph.indices.size() != graph.distances.size() ||
      graph.indices.size() % static_cast<std::size_t>(graph.neighbors) != 0) {
    throw std::invalid_argument("NeighborGraph indices/distances have inconsistent sizes.");
  }
  PreparedGraph out;
  out.samples = static_cast<int>(graph.indices.size() / static_cast<std::size_t>(graph.neighbors));
  out.neighbors = std::min(graph.neighbors, requested_neighbors > 0 ? requested_neighbors : graph.neighbors);
  if (out.samples < 2 || out.neighbors < 1) {
    throw std::invalid_argument("NeighborGraph must contain at least two samples and one neighbor.");
  }
  out.indices.assign(static_cast<std::size_t>(out.samples) * out.neighbors, 0);
  out.distances.assign(static_cast<std::size_t>(out.samples) * out.neighbors, 0.0f);

  int min_index = std::numeric_limits<int>::max();
  int max_index = std::numeric_limits<int>::min();
  for (int value : graph.indices) {
    min_index = std::min(min_index, value);
    max_index = std::max(max_index, value);
  }
  const bool one_based = min_index >= 1 && max_index <= out.samples;

  for (int i = 0; i < out.samples; ++i) {
    float row_max = 1.0f;
    for (int j = 0; j < graph.neighbors; ++j) {
      const float d = graph.distances[static_cast<std::size_t>(i) * graph.neighbors + j];
      if (std::isfinite(d) && d > row_max) row_max = d;
    }
    for (int j = 0; j < out.neighbors; ++j) {
      const std::size_t src = static_cast<std::size_t>(i) * graph.neighbors + j;
      const std::size_t dst = static_cast<std::size_t>(j) * out.samples + i;
      int index = graph.indices[src];
      if (one_based) --index;
      index = std::max(0, std::min(out.samples - 1, index));
      float distance = graph.distances[src];
      if (!std::isfinite(distance) || distance < 0.0f) distance = row_max;
      out.indices[dst] = index;
      out.distances[dst] = distance;
    }
  }
  return out;
}

float directed_umap_weight(const std::vector<float>& distances, int n, int k, int row, int col) {
  float rho = std::numeric_limits<float>::infinity();
  for (int j = 0; j < k; ++j) {
    const float d = distances[static_cast<std::size_t>(j) * n + row];
    if (std::isfinite(d) && d > 0.0f && d < rho) rho = d;
  }
  if (!std::isfinite(rho)) rho = 0.0f;

  const float target = std::log2(static_cast<float>(k < 2 ? 2 : k));
  float lo = 0.0f;
  float hi = std::numeric_limits<float>::infinity();
  float sigma = 1.0f;
  for (int iter = 0; iter < 48; ++iter) {
    float psum = 0.0f;
    for (int j = 0; j < k; ++j) {
      const float raw = distances[static_cast<std::size_t>(j) * n + row];
      if (!std::isfinite(raw)) continue;
      const float d = raw - rho;
      psum += d <= 0.0f ? 1.0f : std::exp(-d / sigma);
    }
    if (std::fabs(psum - target) < 1.0e-5f) break;
    if (psum > target) {
      hi = sigma;
      sigma = 0.5f * (lo + hi);
    } else {
      lo = sigma;
      sigma = std::isinf(hi) ? sigma * 2.0f : 0.5f * (lo + hi);
    }
  }
  sigma = std::max(sigma, 1.0e-6f);

  const float raw = distances[static_cast<std::size_t>(col) * n + row];
  if (!std::isfinite(raw)) return 0.0f;
  const float d = raw - rho;
  return d <= 0.0f ? 1.0f : std::exp(-d / sigma);
}

CsrGraph build_umap_csr_graph(const PreparedGraph& graph) {
  const int n = graph.samples;
  const int k = graph.neighbors;
  std::vector<std::vector<std::pair<int, float>>> directed(static_cast<std::size_t>(n));
  for (int row = 0; row < n; ++row) {
    directed[static_cast<std::size_t>(row)].reserve(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
      const int nb = graph.indices[static_cast<std::size_t>(j) * n + row];
      const float d = graph.distances[static_cast<std::size_t>(j) * n + row];
      if (nb < 0 || nb >= n || nb == row || !std::isfinite(d)) continue;
      const float w = directed_umap_weight(graph.distances, n, k, row, j);
      if (std::isfinite(w) && w > 1.0e-6f) directed[static_cast<std::size_t>(row)].push_back({nb, w});
    }
    std::sort(directed[static_cast<std::size_t>(row)].begin(), directed[static_cast<std::size_t>(row)].end());
  }

  std::vector<std::vector<std::pair<int, float>>> incoming(static_cast<std::size_t>(n));
  for (int row = 0; row < n; ++row) {
    for (const auto& edge : directed[static_cast<std::size_t>(row)]) {
      incoming[static_cast<std::size_t>(edge.first)].push_back({row, edge.second});
    }
  }
  for (auto& row : incoming) std::sort(row.begin(), row.end());

  CsrGraph out;
  out.offsets.resize(static_cast<std::size_t>(n) + 1u, 0);
  std::vector<std::vector<std::pair<int, float>>> rows(static_cast<std::size_t>(n));
  for (int row = 0; row < n; ++row) {
    const auto& fwd = directed[static_cast<std::size_t>(row)];
    const auto& rev = incoming[static_cast<std::size_t>(row)];
    std::size_t a = 0;
    std::size_t b = 0;
    while (a < fwd.size() || b < rev.size()) {
      const int out_nb = a < fwd.size() ? fwd[a].first : std::numeric_limits<int>::max();
      const int in_nb = b < rev.size() ? rev[b].first : std::numeric_limits<int>::max();
      int nb = 0;
      float forward = 0.0f;
      float reverse = 0.0f;
      if (out_nb == in_nb) {
        nb = out_nb;
        forward = fwd[a].second;
        reverse = rev[b].second;
        ++a;
        ++b;
      } else if (out_nb < in_nb) {
        nb = out_nb;
        forward = fwd[a].second;
        ++a;
      } else {
        nb = in_nb;
        reverse = rev[b].second;
        ++b;
      }
      if (nb < 0 || nb >= n || nb == row) continue;
      const float w = forward + reverse - forward * reverse;
      if (std::isfinite(w) && w > 1.0e-6f) rows[static_cast<std::size_t>(row)].push_back({nb, w});
    }
    out.offsets[static_cast<std::size_t>(row + 1)] =
      out.offsets[static_cast<std::size_t>(row)] + static_cast<int>(rows[static_cast<std::size_t>(row)].size());
  }

  out.neighbors.resize(static_cast<std::size_t>(out.offsets[static_cast<std::size_t>(n)]));
  out.weights.resize(out.neighbors.size());
  out.max_weight = 0.0f;
  for (int row = 0; row < n; ++row) {
    int pos = out.offsets[static_cast<std::size_t>(row)];
    for (const auto& edge : rows[static_cast<std::size_t>(row)]) {
      out.neighbors[static_cast<std::size_t>(pos)] = edge.first;
      out.weights[static_cast<std::size_t>(pos)] = edge.second;
      out.max_weight = std::max(out.max_weight, edge.second);
      ++pos;
    }
  }
  if (out.max_weight <= 0.0f) out.max_weight = 1.0f;
  out.epochs_per_sample.resize(out.weights.size());
  for (std::size_t i = 0; i < out.weights.size(); ++i) {
    out.epochs_per_sample[i] = out.max_weight / std::max(out.weights[i], 1.0e-6f);
  }
  return out;
}

void scale_embedding_max_abs_and_jitter(std::vector<double>& embedding, int n, double scale, double jitter, std::mt19937& rng) {
  double max_abs = 0.0;
  for (double v : embedding) max_abs = std::max(max_abs, std::abs(v));
  const double factor = max_abs > 0.0 ? scale / max_abs : 1.0;
  std::normal_distribution<double> noise(0.0, jitter);
  for (double& v : embedding) v = v * factor + noise(rng);
}

std::vector<float> initialize_layout_csr_spectral(
  const CsrGraph& graph,
  int n,
  int n_components,
  int spectral_n_iter,
  int seed
) {
  const int nnz = graph.offsets[static_cast<std::size_t>(n)];
  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  std::normal_distribution<double> normal(0.0, 0.0001);
  std::uniform_real_distribution<double> uniform(-10.0, 10.0);
  std::vector<double> embedding(static_cast<std::size_t>(n) * n_components, 0.0);
  if (nnz == 0) {
    for (double& v : embedding) v = uniform(rng);
    std::vector<float> out(embedding.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(embedding[i]);
    return out;
  }

  std::vector<int> spectral_rows;
  std::vector<int> spectral_pos;
  spectral_rows.reserve(static_cast<std::size_t>(nnz / 2));
  spectral_pos.reserve(static_cast<std::size_t>(nnz / 2));
  for (int row = 0; row < n; ++row) {
    for (int pos = graph.offsets[static_cast<std::size_t>(row)];
         pos < graph.offsets[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int nb = graph.neighbors[static_cast<std::size_t>(pos)];
      if (row < nb) {
        spectral_rows.push_back(row);
        spectral_pos.push_back(pos);
      }
    }
  }
  if (spectral_pos.empty()) {
    for (int row = 0; row < n; ++row) {
      for (int pos = graph.offsets[static_cast<std::size_t>(row)];
           pos < graph.offsets[static_cast<std::size_t>(row + 1)]; ++pos) {
        spectral_rows.push_back(row);
        spectral_pos.push_back(pos);
      }
    }
  }

  std::vector<float> degree(static_cast<std::size_t>(n), 0.0f);
  for (int row = 0; row < n; ++row) {
    double sum = 0.0;
    for (int pos = graph.offsets[static_cast<std::size_t>(row)];
         pos < graph.offsets[static_cast<std::size_t>(row + 1)]; ++pos) {
      sum += graph.weights[static_cast<std::size_t>(pos)];
    }
    degree[static_cast<std::size_t>(row)] = static_cast<float>(sum);
  }

  std::vector<std::vector<std::pair<int, float>>> spectral(static_cast<std::size_t>(n));
  for (std::size_t e = 0; e < spectral_pos.size(); ++e) {
    const int head = spectral_rows[e];
    const int pos = spectral_pos[e];
    const int tail = graph.neighbors[static_cast<std::size_t>(pos)];
    if (head < 0 || head >= n || tail < 0 || tail >= n || head == tail) continue;
    const double denom = std::sqrt(std::max(
      static_cast<double>(degree[static_cast<std::size_t>(head)]) *
      static_cast<double>(degree[static_cast<std::size_t>(tail)]),
      1e-24
    ));
    const float w = static_cast<float>(graph.weights[static_cast<std::size_t>(pos)] / denom);
    spectral[static_cast<std::size_t>(head)].push_back({tail, w});
    spectral[static_cast<std::size_t>(tail)].push_back({head, w});
  }

  std::vector<double> trivial(static_cast<std::size_t>(n), 0.0);
  double trivial_norm = 0.0;
  for (int i = 0; i < n; ++i) {
    trivial[static_cast<std::size_t>(i)] = std::sqrt(std::max(static_cast<double>(degree[static_cast<std::size_t>(i)]), 0.0));
    trivial_norm += trivial[static_cast<std::size_t>(i)] * trivial[static_cast<std::size_t>(i)];
  }
  trivial_norm = std::sqrt(std::max(trivial_norm, 1e-12));
  for (double& v : trivial) v /= trivial_norm;

  const int block_size = std::min(n - 1, std::max(n_components + 6, 8));
  if (block_size < n_components) {
    for (double& v : embedding) v = uniform(rng);
    std::vector<float> out(embedding.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(embedding[i]);
    return out;
  }

  auto at = [n](std::vector<double>& x, int row, int col) -> double& {
    return x[static_cast<std::size_t>(col) * n + row];
  };
  auto cat = [n](const std::vector<double>& x, int row, int col) -> double {
    return x[static_cast<std::size_t>(col) * n + row];
  };
  auto apply = [&](const std::vector<double>& x, std::vector<double>& y, int cols) {
    std::fill(y.begin(), y.end(), 0.0);
    for (int row = 0; row < n; ++row) {
      for (const auto& edge : spectral[static_cast<std::size_t>(row)]) {
        const int tail = edge.first;
        const double w = edge.second;
        for (int c = 0; c < cols; ++c) at(y, row, c) += w * cat(x, tail, c);
      }
    }
  };
  auto orthonormalize = [&](std::vector<double>& q, int cols) {
    for (int c = 0; c < cols; ++c) {
      double dot0 = 0.0;
      for (int i = 0; i < n; ++i) dot0 += cat(q, i, c) * trivial[static_cast<std::size_t>(i)];
      for (int i = 0; i < n; ++i) at(q, i, c) -= dot0 * trivial[static_cast<std::size_t>(i)];
      for (int prev = 0; prev < c; ++prev) {
        double dot = 0.0;
        for (int i = 0; i < n; ++i) dot += cat(q, i, c) * cat(q, i, prev);
        for (int i = 0; i < n; ++i) at(q, i, c) -= dot * cat(q, i, prev);
      }
      double norm = 0.0;
      for (int i = 0; i < n; ++i) norm += cat(q, i, c) * cat(q, i, c);
      norm = std::sqrt(norm);
      if (norm < 1e-10) {
        for (int i = 0; i < n; ++i) at(q, i, c) = normal(rng) * 10000.0;
        --c;
      } else {
        for (int i = 0; i < n; ++i) at(q, i, c) /= norm;
      }
    }
  };

  std::vector<double> q(static_cast<std::size_t>(n) * block_size);
  std::vector<double> z(static_cast<std::size_t>(n) * block_size);
  for (double& v : q) v = normal(rng) * 10000.0;
  orthonormalize(q, block_size);
  for (int iter = 0; iter < spectral_n_iter; ++iter) {
    apply(q, z, block_size);
    orthonormalize(z, block_size);
    q.swap(z);
  }

  apply(q, z, block_size);
  std::vector<double> projected(static_cast<std::size_t>(block_size) * block_size, 0.0);
  for (int row = 0; row < n; ++row) {
    for (int a = 0; a < block_size; ++a) {
      const double qv = cat(q, row, a);
      for (int b = a; b < block_size; ++b) {
        projected[static_cast<std::size_t>(a) * block_size + b] += qv * cat(z, row, b);
      }
    }
  }
  for (int a = 0; a < block_size; ++a) {
    for (int b = a + 1; b < block_size; ++b) projected[static_cast<std::size_t>(b) * block_size + a] = projected[static_cast<std::size_t>(a) * block_size + b];
  }

  std::vector<double> eigvec(static_cast<std::size_t>(block_size) * block_size, 0.0);
  for (int i = 0; i < block_size; ++i) eigvec[static_cast<std::size_t>(i) * block_size + i] = 1.0;
  for (int sweep = 0; sweep < 80; ++sweep) {
    int p = 0;
    int qidx = 1;
    double max_offdiag = 0.0;
    for (int i = 0; i < block_size; ++i) {
      for (int j = i + 1; j < block_size; ++j) {
        const double v = std::abs(projected[static_cast<std::size_t>(i) * block_size + j]);
        if (v > max_offdiag) {
          max_offdiag = v;
          p = i;
          qidx = j;
        }
      }
    }
    if (max_offdiag < 1e-10) break;
    const double app = projected[static_cast<std::size_t>(p) * block_size + p];
    const double aqq = projected[static_cast<std::size_t>(qidx) * block_size + qidx];
    const double apq = projected[static_cast<std::size_t>(p) * block_size + qidx];
    const double tau = (aqq - app) / (2.0 * apq);
    const double t = (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    const double cs = 1.0 / std::sqrt(1.0 + t * t);
    const double sn = t * cs;
    for (int k = 0; k < block_size; ++k) {
      if (k == p || k == qidx) continue;
      const double akp = projected[static_cast<std::size_t>(k) * block_size + p];
      const double akq = projected[static_cast<std::size_t>(k) * block_size + qidx];
      projected[static_cast<std::size_t>(k) * block_size + p] = cs * akp - sn * akq;
      projected[static_cast<std::size_t>(p) * block_size + k] = projected[static_cast<std::size_t>(k) * block_size + p];
      projected[static_cast<std::size_t>(k) * block_size + qidx] = sn * akp + cs * akq;
      projected[static_cast<std::size_t>(qidx) * block_size + k] = projected[static_cast<std::size_t>(k) * block_size + qidx];
    }
    projected[static_cast<std::size_t>(p) * block_size + p] = cs * cs * app - 2.0 * sn * cs * apq + sn * sn * aqq;
    projected[static_cast<std::size_t>(qidx) * block_size + qidx] = sn * sn * app + 2.0 * sn * cs * apq + cs * cs * aqq;
    projected[static_cast<std::size_t>(p) * block_size + qidx] = 0.0;
    projected[static_cast<std::size_t>(qidx) * block_size + p] = 0.0;
    for (int k = 0; k < block_size; ++k) {
      const double vip = eigvec[static_cast<std::size_t>(k) * block_size + p];
      const double viq = eigvec[static_cast<std::size_t>(k) * block_size + qidx];
      eigvec[static_cast<std::size_t>(k) * block_size + p] = cs * vip - sn * viq;
      eigvec[static_cast<std::size_t>(k) * block_size + qidx] = sn * vip + cs * viq;
    }
  }

  std::vector<int> order(block_size);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return projected[static_cast<std::size_t>(a) * block_size + a] >
           projected[static_cast<std::size_t>(b) * block_size + b];
  });
  for (int c = 0; c < n_components; ++c) {
    const int ritz_col = order[static_cast<std::size_t>(c)];
    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
      double value = 0.0;
      for (int basis = 0; basis < block_size; ++basis) {
        value += cat(q, i, basis) * eigvec[static_cast<std::size_t>(basis) * block_size + ritz_col];
      }
      values[static_cast<std::size_t>(i)] = value;
    }
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(n);
    for (int i = 0; i < n; ++i) embedding[static_cast<std::size_t>(i) * n_components + c] = values[static_cast<std::size_t>(i)] - mean;
  }
  scale_embedding_max_abs_and_jitter(embedding, n, 10.0, 1.0e-4, rng);

  std::vector<float> out(embedding.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(embedding[i]);
  return out;
}

const char* cuda_embedding_error();

}  // namespace

#if defined(KODAMA_ENABLE_CUDA)
extern "C" {
bool fastembedr_cuda_available();
const char* fastembedr_cuda_embedding_last_error();
int fastembedr_cuda_umap_from_knn_spectral_float(
  const int* indices,
  const float* distances,
  int n,
  int k,
  int n_epochs,
  int negative_sample_rate,
  float learning_rate,
  float a,
  float b,
  float repulsion_strength,
  int spectral_n_iter,
  unsigned int seed,
  int index_offset,
  int optimizer_mode,
  float* out
);
int fastembedr_cuda_umap_optimize_coo(
  const int* heads,
  const int* tails,
  const float* weights,
  const float* epochs_per_sample,
  const float* init,
  int n,
  int n_edges_capacity,
  int n_epochs,
  int negative_sample_rate,
  float learning_rate,
  float a,
  float b,
  float repulsion_strength,
  unsigned int seed,
  int optimizer_mode,
  float* out
);
int fastembedr_cuda_opentsne_fft_from_knn_float(
  const int* indices,
  const float* distances,
  const float* init,
  int has_init,
  int n,
  int k,
  int n_components,
  float perplexity,
  int early_exaggeration_iter,
  int n_iter,
  float early_exaggeration,
  float exaggeration,
  float learning_rate,
  int learning_rate_auto,
  float initial_momentum,
  float final_momentum,
  float min_gain,
  float max_step_norm,
  unsigned int seed,
  int index_offset,
  float* out
);
}

namespace {
const char* cuda_embedding_error() {
  const char* message = fastembedr_cuda_embedding_last_error();
  return message == nullptr || message[0] == '\0' ? "unknown CUDA embedding error" : message;
}
}  // namespace
#else
namespace {
const char* cuda_embedding_error() {
  return "kodama-cpp was built without CUDA visualization support";
}
}  // namespace
#endif

EmbeddingResult KODAMAUMAP_CUDA(
  const NeighborGraph& graph,
  const UMAPOptions& options
) {
#if !defined(KODAMA_ENABLE_CUDA)
  throw std::runtime_error(cuda_embedding_error());
#else
  if (options.n_components != 2) {
    throw std::invalid_argument("CUDA UMAP currently supports n_components=2.");
  }
  if (!fastembedr_cuda_available()) {
    throw std::runtime_error("No CUDA device is available for KODAMA UMAP.");
  }
  const auto started = Clock::now();
  PreparedGraph prepared = prepare_graph(graph, options.n_neighbors);
  CsrGraph csr = build_umap_csr_graph(prepared);
  if (csr.neighbors.empty()) {
    throw std::runtime_error("CUDA UMAP requires at least one graph edge.");
  }
  std::vector<int> heads(csr.neighbors.size());
  for (int row = 0; row < prepared.samples; ++row) {
    const int begin = csr.offsets[static_cast<std::size_t>(row)];
    const int end = csr.offsets[static_cast<std::size_t>(row + 1)];
    for (int pos = begin; pos < end; ++pos) {
      heads[static_cast<std::size_t>(pos)] = row;
    }
  }
  std::vector<float> init = options.init;
  if (init.size() != static_cast<std::size_t>(prepared.samples) * 2u) {
    init = initialize_layout_csr_spectral(
      csr,
      prepared.samples,
      2,
      options.spectral_n_iter,
      options.seed
    );
  }
  const auto ab = find_ab_params(1.0, options.min_dist);
  EmbeddingResult result;
  result.samples = prepared.samples;
  result.components = 2;
  result.backend = Backend::CUDA;
  result.embedding.assign(static_cast<std::size_t>(prepared.samples) * 2u, 0.0f);

  const int status = fastembedr_cuda_umap_optimize_coo(
    heads.data(),
    csr.neighbors.data(),
    csr.weights.data(),
    csr.epochs_per_sample.data(),
    init.data(),
    prepared.samples,
    static_cast<int>(csr.neighbors.size()),
    options.n_epochs,
    options.negative_sample_rate,
    static_cast<float>(options.learning_rate),
    static_cast<float>(ab.first),
    static_cast<float>(ab.second),
    static_cast<float>(options.repulsion_strength),
    static_cast<unsigned int>(options.seed),
    0,
    result.embedding.data()
  );
  if (status != 0) {
    throw std::runtime_error(std::string("CUDA UMAP failed: ") + cuda_embedding_error());
  }
  const auto finished = Clock::now();
  result.runtime_seconds = std::chrono::duration<double>(finished - started).count();
  return result;
#endif
}

EmbeddingResult KODAMAOpenTSNE_CUDA(
  const NeighborGraph& graph,
  const OpenTSNEOptions& options
) {
#if !defined(KODAMA_ENABLE_CUDA)
  throw std::runtime_error(cuda_embedding_error());
#else
  if (options.n_components != 2) {
    throw std::invalid_argument("CUDA openTSNE currently supports n_components=2.");
  }
  if (!fastembedr_cuda_available()) {
    throw std::runtime_error("No CUDA device is available for KODAMA openTSNE.");
  }
  const auto started = Clock::now();
  const int requested_neighbors = options.n_neighbors > 0 ?
    options.n_neighbors :
    static_cast<int>(std::ceil(options.perplexity));
  PreparedGraph prepared = prepare_graph(graph, requested_neighbors);
  std::vector<float> init = options.init;
  int has_init = static_cast<int>(init.size() == static_cast<std::size_t>(prepared.samples) * 2u);
  if (!has_init) {
    init = make_opentsne_random_init(prepared.samples, options.n_components, options.seed);
    has_init = 1;
  }

  EmbeddingResult result;
  result.samples = prepared.samples;
  result.components = 2;
  result.backend = Backend::CUDA;
  result.embedding.assign(static_cast<std::size_t>(prepared.samples) * 2u, 0.0f);
  const int status = fastembedr_cuda_opentsne_fft_from_knn_float(
    prepared.indices.data(),
    prepared.distances.data(),
    init.data(),
    has_init,
    prepared.samples,
    prepared.neighbors,
    options.n_components,
    static_cast<float>(options.perplexity),
    options.early_exaggeration_iter,
    options.n_iter,
    static_cast<float>(options.early_exaggeration),
    static_cast<float>(options.exaggeration),
    static_cast<float>(options.learning_rate),
    options.learning_rate_auto ? 1 : 0,
    static_cast<float>(options.initial_momentum),
    static_cast<float>(options.final_momentum),
    static_cast<float>(options.min_gain),
    static_cast<float>(options.max_step_norm),
    static_cast<unsigned int>(options.seed),
    prepared.index_offset,
    result.embedding.data()
  );
  if (status != 0) {
    throw std::runtime_error(std::string("CUDA openTSNE failed: ") + cuda_embedding_error());
  }
  const auto finished = Clock::now();
  result.runtime_seconds = std::chrono::duration<double>(finished - started).count();
  return result;
#endif
}

}  // namespace kodama
