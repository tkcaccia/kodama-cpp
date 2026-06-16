#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(KODAMA_ENABLE_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace kodama {

namespace {

#if defined(KODAMA_ENABLE_CUDA)
void check_cuda(cudaError_t code, const char* where) {
  if (code != cudaSuccess) {
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(code));
  }
}

void check_cublas(cublasStatus_t code, const char* where) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(where) + ": cuBLAS call failed");
  }
}

class CudaBlasContext {
 public:
  explicit CudaBlasContext(int device) {
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    check_cublas(cublasCreate(&handle_), "cublasCreate");
  }

  ~CudaBlasContext() {
    if (handle_ != nullptr) cublasDestroy(handle_);
  }

  CudaBlasContext(const CudaBlasContext&) = delete;
  CudaBlasContext& operator=(const CudaBlasContext&) = delete;

  cublasHandle_t handle() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) { reset(n); }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void reset(std::size_t n) {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = n;
    if (n > 0) check_cuda(cudaMalloc(&ptr_, n * sizeof(double)), "cudaMalloc");
  }

  double* data() { return ptr_; }
  const double* data() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  double* ptr_ = nullptr;
  std::size_t size_ = 0;
};
#endif

struct Dense {
  int rows = 0;
  int cols = 0;
  std::vector<double> data;

  Dense() = default;
  Dense(int r, int c) : rows(r), cols(c), data(static_cast<std::size_t>(r * c), 0.0) {}

  double& operator()(int i, int j) { return data[static_cast<std::size_t>(i * cols + j)]; }
  double operator()(int i, int j) const { return data[static_cast<std::size_t>(i * cols + j)]; }
};

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  long double out = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) out += static_cast<long double>(a[i]) * b[i];
  return static_cast<double>(out);
}

double norm2(const std::vector<double>& x) {
  return std::sqrt(std::max(0.0, dot(x, x)));
}

std::vector<double> mat_vec(const Dense& a, const std::vector<double>& x) {
  std::vector<double> out(static_cast<std::size_t>(a.rows), 0.0);
  for (int i = 0; i < a.rows; ++i) {
    long double s = 0.0;
    for (int j = 0; j < a.cols; ++j) s += static_cast<long double>(a(i, j)) * x[static_cast<std::size_t>(j)];
    out[static_cast<std::size_t>(i)] = static_cast<double>(s);
  }
  return out;
}

std::vector<double> t_mat_vec(const Dense& a, const std::vector<double>& x) {
  std::vector<double> out(static_cast<std::size_t>(a.cols), 0.0);
  for (int j = 0; j < a.cols; ++j) {
    long double s = 0.0;
    for (int i = 0; i < a.rows; ++i) s += static_cast<long double>(a(i, j)) * x[static_cast<std::size_t>(i)];
    out[static_cast<std::size_t>(j)] = static_cast<double>(s);
  }
  return out;
}

Dense subset_scale(
  MatrixView x,
  const std::vector<int>& rows,
  const std::vector<double>& mean,
  const std::vector<double>& scale
) {
  Dense out(static_cast<int>(rows.size()), static_cast<int>(x.cols));
  for (int i = 0; i < out.rows; ++i) {
    const int src = rows[static_cast<std::size_t>(i)];
    for (int j = 0; j < out.cols; ++j) {
      out(i, j) = (x(static_cast<std::size_t>(src), static_cast<std::size_t>(j)) - mean[static_cast<std::size_t>(j)]) /
                  scale[static_cast<std::size_t>(j)];
    }
  }
  return out;
}

void train_center_scale(
  MatrixView x,
  const std::vector<int>& rows,
  bool center,
  bool scale_columns,
  std::vector<double>& mean,
  std::vector<double>& scale
) {
  mean.assign(x.cols, 0.0);
  scale.assign(x.cols, 1.0);
  if (center) {
    for (int row : rows) {
      for (std::size_t j = 0; j < x.cols; ++j) mean[j] += x(static_cast<std::size_t>(row), j);
    }
    for (double& v : mean) v /= static_cast<double>(rows.size());
  }
  if (scale_columns) {
    for (int row : rows) {
      for (std::size_t j = 0; j < x.cols; ++j) {
        const double d = x(static_cast<std::size_t>(row), j) - mean[j];
        scale[j] += d * d;
      }
    }
    for (std::size_t j = 0; j < x.cols; ++j) {
      const double s = std::sqrt(scale[j] / std::max(1.0, static_cast<double>(rows.size() - 1)));
      scale[j] = s > 0.0 && std::isfinite(s) ? s : 1.0;
    }
  }
}

Dense one_hot_centered(const std::vector<int>& labels, const std::vector<int>& rows, const std::vector<int>& classes, std::vector<double>& y_mean) {
  Dense y(static_cast<int>(rows.size()), static_cast<int>(classes.size()));
  std::map<int, int> class_pos;
  for (int i = 0; i < static_cast<int>(classes.size()); ++i) class_pos[classes[static_cast<std::size_t>(i)]] = i;
  y_mean.assign(classes.size(), 0.0);
  for (int i = 0; i < y.rows; ++i) {
    const int cls = labels[static_cast<std::size_t>(rows[static_cast<std::size_t>(i)])];
    const int j = class_pos[cls];
    y(i, j) = 1.0;
    y_mean[static_cast<std::size_t>(j)] += 1.0;
  }
  for (double& v : y_mean) v /= static_cast<double>(rows.size());
  for (int i = 0; i < y.rows; ++i) {
    for (int j = 0; j < y.cols; ++j) y(i, j) -= y_mean[static_cast<std::size_t>(j)];
  }
  return y;
}

Dense crossprod(const Dense& x, const Dense& y) {
  Dense out(x.cols, y.cols);
  for (int i = 0; i < x.cols; ++i) {
    for (int j = 0; j < y.cols; ++j) {
      long double s = 0.0;
      for (int r = 0; r < x.rows; ++r) s += static_cast<long double>(x(r, i)) * y(r, j);
      out(i, j) = static_cast<double>(s);
    }
  }
  return out;
}

#if defined(KODAMA_ENABLE_CUDA)
Dense crossprod_cuda(const Dense& x, const Dense& y, int gpu_device) {
  if (x.rows != y.rows) throw std::invalid_argument("crossprod_cuda row mismatch.");
  if (x.rows == 0 || x.cols == 0 || y.cols == 0) return Dense(x.cols, y.cols);

  CudaBlasContext ctx(gpu_device);
  DeviceBuffer dx(x.data.size());
  DeviceBuffer dy(y.data.size());
  DeviceBuffer ds(static_cast<std::size_t>(x.cols) * static_cast<std::size_t>(y.cols));

  check_cuda(cudaMemcpy(dx.data(), x.data.data(), x.data.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy X to device");
  check_cuda(cudaMemcpy(dy.data(), y.data.data(), y.data.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy Y to device");

  const double alpha = 1.0;
  const double beta = 0.0;
  // Dense is row-major. The row-major n x p X buffer is column-major p x n (X').
  // The row-major n x c Y buffer is column-major c x n (Y'). Compute X'Y as
  // (p x n) * (c x n)' into a column-major p x c buffer.
  check_cublas(
    cublasDgemm(
      ctx.handle(),
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      x.cols,
      y.cols,
      x.rows,
      &alpha,
      dx.data(),
      x.cols,
      dy.data(),
      y.cols,
      &beta,
      ds.data(),
      x.cols
    ),
    "cublasDgemm X'Y"
  );

  std::vector<double> col_major(static_cast<std::size_t>(x.cols) * static_cast<std::size_t>(y.cols));
  check_cuda(cudaMemcpy(col_major.data(), ds.data(), col_major.size() * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy X'Y to host");
  Dense out(x.cols, y.cols);
  for (int j = 0; j < y.cols; ++j) {
    for (int i = 0; i < x.cols; ++i) {
      out(i, j) = col_major[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(x.cols)];
    }
  }
  return out;
}
#endif

std::vector<double> dominant_left_singular_vector(const Dense& s) {
  std::vector<double> u(static_cast<std::size_t>(s.cols), 1.0 / std::sqrt(std::max(1, s.cols)));
  std::vector<double> v(static_cast<std::size_t>(s.rows), 0.0);
  for (int iter = 0; iter < 80; ++iter) {
    v = mat_vec(s, u);
    double nv = norm2(v);
    if (nv <= 1e-12) break;
    for (double& x : v) x /= nv;
    u = t_mat_vec(s, v);
    double nu = norm2(u);
    if (nu <= 1e-12) break;
    for (double& x : u) x /= nu;
  }
  if (norm2(v) <= 1e-12) {
    v.assign(static_cast<std::size_t>(s.rows), 0.0);
    if (!v.empty()) v[0] = 1.0;
  }
  return v;
}

Dense first_columns(const Dense& x, int ncomp) {
  Dense out(x.rows, ncomp);
  for (int i = 0; i < x.rows; ++i) {
    for (int j = 0; j < ncomp; ++j) out(i, j) = x(i, j);
  }
  return out;
}

Dense solve_linear(Dense a, Dense b) {
  const int n = a.rows;
  const int m = b.cols;
  for (int i = 0; i < n; ++i) a(i, i) += 1e-9;
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double best = std::abs(a(col, col));
    for (int r = col + 1; r < n; ++r) {
      const double val = std::abs(a(r, col));
      if (val > best) {
        best = val;
        pivot = r;
      }
    }
    if (best < 1e-14) continue;
    if (pivot != col) {
      for (int j = 0; j < n; ++j) std::swap(a(col, j), a(pivot, j));
      for (int j = 0; j < m; ++j) std::swap(b(col, j), b(pivot, j));
    }
    const double div = a(col, col);
    for (int j = col; j < n; ++j) a(col, j) /= div;
    for (int j = 0; j < m; ++j) b(col, j) /= div;
    for (int r = 0; r < n; ++r) {
      if (r == col) continue;
      const double f = a(r, col);
      if (f == 0.0) continue;
      for (int j = col; j < n; ++j) a(r, j) -= f * a(col, j);
      for (int j = 0; j < m; ++j) b(r, j) -= f * b(col, j);
    }
  }
  return b;
}

struct PLSFit {
  Dense weights;
  Dense loadings;
  Dense y_weights;
  Dense train_scores;
};

PLSFit fit_pls_components(const Dense& x, const Dense& y, int max_components) {
  Dense s = crossprod(x, y);
  Dense gram(s.rows, s.rows);
  for (int i = 0; i < s.rows; ++i) {
    for (int j = 0; j < s.rows; ++j) {
      long double val = 0.0;
      for (int c = 0; c < s.cols; ++c) val += static_cast<long double>(s(i, c)) * s(j, c);
      gram(i, j) = static_cast<double>(val);
    }
  }
  const int max_rank = std::min({max_components, s.rows, s.cols, std::max(1, x.rows - 1)});
  Dense w(x.cols, max_rank);
  Dense pmat(x.cols, max_rank);
  Dense qmat(y.cols, max_rank);
  Dense tmat(x.rows, max_rank);
  for (int a = 0; a < max_rank; ++a) {
    std::vector<double> wa(static_cast<std::size_t>(x.cols), 0.0);
    for (int j = 0; j < x.cols; ++j) {
      wa[static_cast<std::size_t>(j)] = std::sin(static_cast<double>((a + 1) * (j + 1)));
    }
    for (int prev = 0; prev < a; ++prev) {
      double proj = 0.0;
      for (int j = 0; j < x.cols; ++j) proj += wa[static_cast<std::size_t>(j)] * w(j, prev);
      for (int j = 0; j < x.cols; ++j) wa[static_cast<std::size_t>(j)] -= proj * w(j, prev);
    }
    double nwa = norm2(wa);
    if (nwa <= 1e-12) {
      std::fill(wa.begin(), wa.end(), 0.0);
      wa[static_cast<std::size_t>(a % x.cols)] = 1.0;
    } else {
      for (double& v : wa) v /= nwa;
    }
    for (int iter = 0; iter < 120; ++iter) {
      std::vector<double> next(static_cast<std::size_t>(x.cols), 0.0);
      for (int i = 0; i < x.cols; ++i) {
        long double val = 0.0;
        for (int j = 0; j < x.cols; ++j) val += static_cast<long double>(gram(i, j)) * wa[static_cast<std::size_t>(j)];
        next[static_cast<std::size_t>(i)] = static_cast<double>(val);
      }
      for (int prev = 0; prev < a; ++prev) {
        double proj = 0.0;
        for (int j = 0; j < x.cols; ++j) proj += next[static_cast<std::size_t>(j)] * w(j, prev);
        for (int j = 0; j < x.cols; ++j) next[static_cast<std::size_t>(j)] -= proj * w(j, prev);
      }
      double nn = norm2(next);
      if (nn <= 1e-12) break;
      for (double& v : next) v /= nn;
      wa.swap(next);
    }
    std::vector<double> right = t_mat_vec(s, wa);
    const double sigma = std::max(1e-12, norm2(right));
    for (double& v : right) v /= sigma;

    std::vector<double> t(static_cast<std::size_t>(x.rows), 0.0);
    for (int i = 0; i < x.rows; ++i) {
      long double val = 0.0;
      for (int j = 0; j < x.cols; ++j) val += static_cast<long double>(x(i, j)) * wa[static_cast<std::size_t>(j)];
      t[static_cast<std::size_t>(i)] = static_cast<double>(val);
    }
    for (int j = 0; j < x.cols; ++j) {
      w(j, a) = wa[static_cast<std::size_t>(j)];
      pmat(j, a) = 0.0;
    }
    for (int j = 0; j < y.cols; ++j) qmat(j, a) = right[static_cast<std::size_t>(j)];
    for (int i = 0; i < x.rows; ++i) tmat(i, a) = t[static_cast<std::size_t>(i)];

  }
  return PLSFit{w, pmat, qmat, tmat};
}

Dense transform_pls_scores(const Dense& x, const PLSFit& fit, int ncomp) {
  Dense out(x.rows, ncomp);
  for (int a = 0; a < ncomp; ++a) {
    for (int i = 0; i < x.rows; ++i) {
      long double val = 0.0;
      for (int j = 0; j < x.cols; ++j) val += static_cast<long double>(x(i, j)) * fit.weights(j, a);
      out(i, a) = static_cast<double>(val);
    }
  }
  return out;
}

#if defined(KODAMA_ENABLE_CUDA)
Dense transform_pls_scores_cuda(const Dense& x, const PLSFit& fit, int ncomp, int gpu_device) {
  if (ncomp < 1) return Dense(x.rows, 0);
  if (ncomp > fit.weights.cols) throw std::invalid_argument("transform_pls_scores_cuda ncomp exceeds fit rank.");
  if (x.cols != fit.weights.rows) throw std::invalid_argument("transform_pls_scores_cuda column mismatch.");

  CudaBlasContext ctx(gpu_device);
  DeviceBuffer dx(x.data.size());
  DeviceBuffer dw(static_cast<std::size_t>(fit.weights.rows) * static_cast<std::size_t>(ncomp));
  DeviceBuffer dt(static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(ncomp));

  std::vector<double> w_prefix(static_cast<std::size_t>(fit.weights.rows) * static_cast<std::size_t>(ncomp));
  for (int i = 0; i < fit.weights.rows; ++i) {
    for (int j = 0; j < ncomp; ++j) {
      w_prefix[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncomp) + static_cast<std::size_t>(j)] = fit.weights(i, j);
    }
  }

  check_cuda(cudaMemcpy(dx.data(), x.data.data(), x.data.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy score X to device");
  check_cuda(cudaMemcpy(dw.data(), w_prefix.data(), w_prefix.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy score W to device");

  const double alpha = 1.0;
  const double beta = 0.0;
  // Row-major X (n x p) is column-major X' (p x n). Row-major W (p x k)
  // is column-major W' (k x p). Compute T' = W' X' into a k x n
  // column-major buffer, which is byte-identical to row-major T (n x k).
  check_cublas(
    cublasDgemm(
      ctx.handle(),
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      ncomp,
      x.rows,
      x.cols,
      &alpha,
      dw.data(),
      ncomp,
      dx.data(),
      x.cols,
      &beta,
      dt.data(),
      ncomp
    ),
    "cublasDgemm XW"
  );

  Dense out(x.rows, ncomp);
  check_cuda(cudaMemcpy(out.data.data(), dt.data(), out.data.size() * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy scores to host");
  return out;
}

PLSFit fit_pls_components_cuda(const Dense& x, const Dense& y, int max_components, int gpu_device) {
  Dense s = crossprod_cuda(x, y, gpu_device);
  Dense gram(s.rows, s.rows);
  for (int i = 0; i < s.rows; ++i) {
    for (int j = 0; j < s.rows; ++j) {
      long double val = 0.0;
      for (int c = 0; c < s.cols; ++c) val += static_cast<long double>(s(i, c)) * s(j, c);
      gram(i, j) = static_cast<double>(val);
    }
  }
  const int max_rank = std::min({max_components, s.rows, s.cols, std::max(1, x.rows - 1)});
  Dense w(x.cols, max_rank);
  Dense pmat(x.cols, max_rank);
  Dense qmat(y.cols, max_rank);
  for (int a = 0; a < max_rank; ++a) {
    std::vector<double> wa(static_cast<std::size_t>(x.cols), 0.0);
    for (int j = 0; j < x.cols; ++j) {
      wa[static_cast<std::size_t>(j)] = std::sin(static_cast<double>((a + 1) * (j + 1)));
    }
    for (int prev = 0; prev < a; ++prev) {
      double proj = 0.0;
      for (int j = 0; j < x.cols; ++j) proj += wa[static_cast<std::size_t>(j)] * w(j, prev);
      for (int j = 0; j < x.cols; ++j) wa[static_cast<std::size_t>(j)] -= proj * w(j, prev);
    }
    double nwa = norm2(wa);
    if (nwa <= 1e-12) {
      std::fill(wa.begin(), wa.end(), 0.0);
      wa[static_cast<std::size_t>(a % x.cols)] = 1.0;
    } else {
      for (double& v : wa) v /= nwa;
    }
    for (int iter = 0; iter < 120; ++iter) {
      std::vector<double> next(static_cast<std::size_t>(x.cols), 0.0);
      for (int i = 0; i < x.cols; ++i) {
        long double val = 0.0;
        for (int j = 0; j < x.cols; ++j) val += static_cast<long double>(gram(i, j)) * wa[static_cast<std::size_t>(j)];
        next[static_cast<std::size_t>(i)] = static_cast<double>(val);
      }
      for (int prev = 0; prev < a; ++prev) {
        double proj = 0.0;
        for (int j = 0; j < x.cols; ++j) proj += next[static_cast<std::size_t>(j)] * w(j, prev);
        for (int j = 0; j < x.cols; ++j) next[static_cast<std::size_t>(j)] -= proj * w(j, prev);
      }
      double nn = norm2(next);
      if (nn <= 1e-12) break;
      for (double& v : next) v /= nn;
      wa.swap(next);
    }
    std::vector<double> right = t_mat_vec(s, wa);
    const double sigma = std::max(1e-12, norm2(right));
    for (double& v : right) v /= sigma;

    for (int j = 0; j < x.cols; ++j) {
      w(j, a) = wa[static_cast<std::size_t>(j)];
      pmat(j, a) = 0.0;
    }
    for (int j = 0; j < y.cols; ++j) qmat(j, a) = right[static_cast<std::size_t>(j)];
  }

  PLSFit fit{w, pmat, qmat, Dense()};
  fit.train_scores = transform_pls_scores_cuda(x, fit, max_rank, gpu_device);
  return fit;
}
#endif

Dense regression_coefficients(const Dense& t, const Dense& y) {
  Dense lhs(t.cols, t.cols);
  Dense rhs(t.cols, y.cols);
  for (int i = 0; i < t.cols; ++i) {
    for (int j = 0; j < t.cols; ++j) {
      long double s = 0.0;
      for (int r = 0; r < t.rows; ++r) s += static_cast<long double>(t(r, i)) * t(r, j);
      lhs(i, j) = static_cast<double>(s);
    }
    for (int j = 0; j < y.cols; ++j) {
      long double s = 0.0;
      for (int r = 0; r < t.rows; ++r) s += static_cast<long double>(t(r, i)) * y(r, j);
      rhs(i, j) = static_cast<double>(s);
    }
  }
  return solve_linear(lhs, rhs);
}

Dense y_scores_from_q(const Dense& y, const Dense& q, int ncomp) {
  Dense out(y.rows, ncomp);
  for (int i = 0; i < y.rows; ++i) {
    for (int a = 0; a < ncomp; ++a) {
      long double s = 0.0;
      for (int j = 0; j < y.cols; ++j) s += static_cast<long double>(y(i, j)) * q(j, a);
      out(i, a) = static_cast<double>(s);
    }
  }
  return out;
}

std::vector<int> predict_pls_da(
  const Dense& t_train,
  const std::vector<int>& y_train_labels,
  const Dense& t_val,
  const std::vector<int>& classes
) {
  const int cnum = static_cast<int>(classes.size());
  std::map<int, int> cpos;
  for (int i = 0; i < cnum; ++i) cpos[classes[static_cast<std::size_t>(i)]] = i;
  Dense cent(cnum, t_train.cols);
  std::vector<int> counts(static_cast<std::size_t>(cnum), 0);
  for (int i = 0; i < t_train.rows; ++i) {
    const int c = cpos[y_train_labels[static_cast<std::size_t>(i)]];
    counts[static_cast<std::size_t>(c)]++;
    for (int a = 0; a < t_train.cols; ++a) cent(c, a) += t_train(i, a);
  }
  for (int c = 0; c < cnum; ++c) {
    const double inv = counts[static_cast<std::size_t>(c)] > 0 ? 1.0 / counts[static_cast<std::size_t>(c)] : 0.0;
    for (int a = 0; a < t_train.cols; ++a) cent(c, a) *= inv;
  }
  std::vector<int> pred(static_cast<std::size_t>(t_val.rows), classes.front());
  for (int i = 0; i < t_val.rows; ++i) {
    int best = 0;
    double best_dist = std::numeric_limits<double>::infinity();
    for (int c = 0; c < cnum; ++c) {
      long double d2 = 0.0;
      for (int a = 0; a < t_val.cols; ++a) {
        const double d = t_val(i, a) - cent(c, a);
        d2 += static_cast<long double>(d) * d;
      }
      if (d2 < best_dist) {
        best_dist = static_cast<double>(d2);
        best = c;
      }
    }
    pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(best)];
  }
  return pred;
}

std::vector<int> predict_pls_lda(
  const Dense& t_train,
  const std::vector<int>& y_train,
  const Dense& t_val,
  const std::vector<int>& classes
) {
  const int cnum = static_cast<int>(classes.size());
  std::map<int, int> cpos;
  for (int i = 0; i < cnum; ++i) cpos[classes[static_cast<std::size_t>(i)]] = i;
  Dense cent(cnum, t_train.cols);
  std::vector<int> counts(static_cast<std::size_t>(cnum), 0);
  for (int i = 0; i < t_train.rows; ++i) {
    const int c = cpos[y_train[static_cast<std::size_t>(i)]];
    counts[static_cast<std::size_t>(c)]++;
    for (int a = 0; a < t_train.cols; ++a) cent(c, a) += t_train(i, a);
  }
  for (int c = 0; c < cnum; ++c) {
    const double inv = counts[static_cast<std::size_t>(c)] > 0 ? 1.0 / counts[static_cast<std::size_t>(c)] : 0.0;
    for (int a = 0; a < t_train.cols; ++a) cent(c, a) *= inv;
  }
  std::vector<double> var(static_cast<std::size_t>(t_train.cols), 1e-9);
  for (int i = 0; i < t_train.rows; ++i) {
    const int c = cpos[y_train[static_cast<std::size_t>(i)]];
    for (int a = 0; a < t_train.cols; ++a) {
      const double d = t_train(i, a) - cent(c, a);
      var[static_cast<std::size_t>(a)] += d * d;
    }
  }
  for (double& v : var) v /= std::max(1, t_train.rows - cnum);

  std::vector<int> pred(static_cast<std::size_t>(t_val.rows), classes.front());
  for (int i = 0; i < t_val.rows; ++i) {
    int best = 0;
    double best_dist = std::numeric_limits<double>::infinity();
    for (int c = 0; c < cnum; ++c) {
      long double d2 = 0.0;
      for (int a = 0; a < t_val.cols; ++a) {
        const double d = t_val(i, a) - cent(c, a);
        d2 += static_cast<long double>(d) * d / var[static_cast<std::size_t>(a)];
      }
      if (d2 < best_dist) {
        best_dist = static_cast<double>(d2);
        best = c;
      }
    }
    pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(best)];
  }
  return pred;
}

}  // namespace

namespace {

PLSCVResult run_plscv_cpu(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options,
  PLSMode mode
) {
  detail::validate_inputs(x, labels, constrain);
  if (options.max_components < 1) throw std::invalid_argument("PLSOptions::max_components must be positive.");

  detail::Timer timer;
  PLSCVResult result;
  result.true_labels = labels;
  result.fold_assignments = detail::make_folds(labels, constrain, options.cv);
  result.accuracy_by_components.assign(static_cast<std::size_t>(options.max_components), 0.0);
  std::vector<std::vector<int>> pred_by_comp(
    static_cast<std::size_t>(options.max_components),
    std::vector<int>(labels.size(), labels.empty() ? 0 : labels.front())
  );
  const std::vector<int> classes = detail::unique_labels(labels);
  const std::vector<int> fold_ids = detail::sorted_unique_folds(result.fold_assignments);

  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(result.fold_assignments, fold, true);
    const std::vector<int> train = detail::indices_where_fold(result.fold_assignments, fold, false);
    std::vector<double> x_mean;
    std::vector<double> x_scale;
    train_center_scale(x, train, options.center, options.scale, x_mean, x_scale);
    Dense x_train = subset_scale(x, train, x_mean, x_scale);
    Dense x_val = subset_scale(x, validation, x_mean, x_scale);
    std::vector<double> y_mean;
    Dense y_train = one_hot_centered(labels, train, classes, y_mean);
    PLSFit fit = fit_pls_components(x_train, y_train, options.max_components);
    std::vector<int> y_train_labels(train.size(), 0);
    for (std::size_t i = 0; i < train.size(); ++i) y_train_labels[i] = labels[static_cast<std::size_t>(train[i])];

    for (int a = 1; a <= fit.weights.cols; ++a) {
      Dense t_train = first_columns(fit.train_scores, a);
      Dense t_val = transform_pls_scores(x_val, fit, a);
      std::vector<int> fold_pred = mode == PLSMode::PLS_LDA ?
        predict_pls_lda(t_train, y_train_labels, t_val, classes) :
        predict_pls_da(t_train, y_train_labels, t_val, classes);
      for (std::size_t i = 0; i < validation.size(); ++i) {
        pred_by_comp[static_cast<std::size_t>(a - 1)][static_cast<std::size_t>(validation[i])] = fold_pred[i];
      }
    }
  }

  int best_comp = 1;
  double best_acc = -1.0;
  for (int a = 1; a <= options.max_components; ++a) {
    result.accuracy_by_components[static_cast<std::size_t>(a - 1)] =
      detail::accuracy(labels, pred_by_comp[static_cast<std::size_t>(a - 1)]);
    if (result.accuracy_by_components[static_cast<std::size_t>(a - 1)] > best_acc) {
      best_acc = result.accuracy_by_components[static_cast<std::size_t>(a - 1)];
      best_comp = a;
    }
  }

  result.selected_components = best_comp;
  result.predicted_labels = pred_by_comp[static_cast<std::size_t>(best_comp - 1)];
  result.global_accuracy = detail::accuracy(labels, result.predicted_labels);
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(result.fold_assignments, fold, true);
    const std::vector<int> train = detail::indices_where_fold(result.fold_assignments, fold, false);
    result.folds.push_back(FoldResult{
      fold,
      static_cast<int>(train.size()),
      static_cast<int>(validation.size()),
      detail::accuracy_on_indices(labels, result.predicted_labels, validation)
    });
  }
  result.confusion = detail::make_confusion(labels, result.predicted_labels);
  result.runtime_seconds = timer.seconds();
  result.peak_memory_mb = detail::peak_memory_mb();
  result.parameters.backend = Backend::CPU;
  result.parameters.mode = mode;
  result.parameters.max_components = options.max_components;
  result.parameters.selected_components = best_comp;
  result.parameters.center = options.center;
  result.parameters.scale = options.scale;
  result.parameters.gpu_device = options.gpu_device;
  result.parameters.n_threads = options.n_threads;
  return result;
}

#if defined(KODAMA_ENABLE_CUDA)
PLSCVResult run_plscv_cuda(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options,
  PLSMode mode
) {
  detail::validate_inputs(x, labels, constrain);
  if (options.max_components < 1) throw std::invalid_argument("PLSOptions::max_components must be positive.");

  detail::Timer timer;
  check_cuda(cudaSetDevice(options.gpu_device), "cudaSetDevice(run_plscv_cuda)");
  PLSCVResult result;
  result.true_labels = labels;
  result.fold_assignments = detail::make_folds(labels, constrain, options.cv);
  result.accuracy_by_components.assign(static_cast<std::size_t>(options.max_components), 0.0);
  std::vector<std::vector<int>> pred_by_comp(
    static_cast<std::size_t>(options.max_components),
    std::vector<int>(labels.size(), labels.empty() ? 0 : labels.front())
  );
  const std::vector<int> classes = detail::unique_labels(labels);
  const std::vector<int> fold_ids = detail::sorted_unique_folds(result.fold_assignments);

  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(result.fold_assignments, fold, true);
    const std::vector<int> train = detail::indices_where_fold(result.fold_assignments, fold, false);
    std::vector<double> x_mean;
    std::vector<double> x_scale;
    train_center_scale(x, train, options.center, options.scale, x_mean, x_scale);
    Dense x_train = subset_scale(x, train, x_mean, x_scale);
    Dense x_val = subset_scale(x, validation, x_mean, x_scale);
    std::vector<double> y_mean;
    Dense y_train = one_hot_centered(labels, train, classes, y_mean);
    PLSFit fit = fit_pls_components_cuda(x_train, y_train, options.max_components, options.gpu_device);
    std::vector<int> y_train_labels(train.size(), 0);
    for (std::size_t i = 0; i < train.size(); ++i) y_train_labels[i] = labels[static_cast<std::size_t>(train[i])];

    for (int a = 1; a <= fit.weights.cols; ++a) {
      Dense t_train = first_columns(fit.train_scores, a);
      Dense t_val = transform_pls_scores_cuda(x_val, fit, a, options.gpu_device);
      std::vector<int> fold_pred = mode == PLSMode::PLS_LDA ?
        predict_pls_lda(t_train, y_train_labels, t_val, classes) :
        predict_pls_da(t_train, y_train_labels, t_val, classes);
      for (std::size_t i = 0; i < validation.size(); ++i) {
        pred_by_comp[static_cast<std::size_t>(a - 1)][static_cast<std::size_t>(validation[i])] = fold_pred[i];
      }
    }
  }

  int best_comp = 1;
  double best_acc = -1.0;
  for (int a = 1; a <= options.max_components; ++a) {
    result.accuracy_by_components[static_cast<std::size_t>(a - 1)] =
      detail::accuracy(labels, pred_by_comp[static_cast<std::size_t>(a - 1)]);
    if (result.accuracy_by_components[static_cast<std::size_t>(a - 1)] > best_acc) {
      best_acc = result.accuracy_by_components[static_cast<std::size_t>(a - 1)];
      best_comp = a;
    }
  }

  result.selected_components = best_comp;
  result.predicted_labels = pred_by_comp[static_cast<std::size_t>(best_comp - 1)];
  result.global_accuracy = detail::accuracy(labels, result.predicted_labels);
  for (int fold : fold_ids) {
    const std::vector<int> validation = detail::indices_where_fold(result.fold_assignments, fold, true);
    const std::vector<int> train = detail::indices_where_fold(result.fold_assignments, fold, false);
    result.folds.push_back(FoldResult{
      fold,
      static_cast<int>(train.size()),
      static_cast<int>(validation.size()),
      detail::accuracy_on_indices(labels, result.predicted_labels, validation)
    });
  }
  result.confusion = detail::make_confusion(labels, result.predicted_labels);
  result.runtime_seconds = timer.seconds();
  result.peak_memory_mb = detail::peak_memory_mb();
  result.parameters.backend = Backend::CUDA;
  result.parameters.mode = mode;
  result.parameters.max_components = options.max_components;
  result.parameters.selected_components = best_comp;
  result.parameters.center = options.center;
  result.parameters.scale = options.scale;
  result.parameters.gpu_device = options.gpu_device;
  result.parameters.n_threads = options.n_threads;
  return result;
}
#endif

}  // namespace

PLSCVResult PLSDACV(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  if (options.backend == Backend::CUDA) return PLSDACV_CUDA(x, labels, constrain, options);
  return PLSDACV_CPU(x, labels, constrain, options);
}

PLSCVResult PLSLDACV(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  if (options.backend == Backend::CUDA) return PLSLDACV_CUDA(x, labels, constrain, options);
  return PLSLDACV_CPU(x, labels, constrain, options);
}

PLSCVResult PLSDACV_CPU(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  return run_plscv_cpu(x, labels, constrain, options, PLSMode::PLS_DA);
}

PLSCVResult PLSLDACV_CPU(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  return run_plscv_cpu(x, labels, constrain, options, PLSMode::PLS_LDA);
}

PLSCVResult PLSDACV_CUDA(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
#if defined(KODAMA_ENABLE_CUDA)
  PLSOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  return run_plscv_cuda(x, labels, constrain, cuda_options, PLSMode::PLS_DA);
#else
  (void)x;
  (void)labels;
  (void)constrain;
  (void)options;
  throw std::runtime_error("PLSDACV_CUDA requires a CUDA/cuBLAS build.");
#endif
}

PLSCVResult PLSLDACV_CUDA(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
#if defined(KODAMA_ENABLE_CUDA)
  PLSOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  return run_plscv_cuda(x, labels, constrain, cuda_options, PLSMode::PLS_LDA);
#else
  (void)x;
  (void)labels;
  (void)constrain;
  (void)options;
  throw std::runtime_error("PLSLDACV_CUDA requires a CUDA/cuBLAS build.");
#endif
}

}  // namespace kodama
