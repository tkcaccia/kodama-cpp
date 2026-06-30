#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(KODAMA_ENABLE_CUDA)
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

extern "C" void kodama_cuda_candidate_knn_scores_prefix(
  const double* t_test,
  const double* t_train,
  const double* test_norm2,
  const double* train_norm2,
  const int* ncomp,
  const int* class_offsets,
  const int* class_indices,
  const int* candidates,
  const double* candidate_base,
  const double* bias,
  int ntest,
  int ntrain,
  int kdim,
  int n_classes,
  int nslice,
  int top_m,
  int knn_k,
  double tau,
  double alpha,
  double* out_scores,
  cudaStream_t stream
);

extern "C" void kodama_cuda_lda_label_sums_row(const double*, const int*, int, int, int, double*, cudaStream_t);
extern "C" void kodama_cuda_lda_means_row(double*, const double*, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_pooled_col(double*, const double*, const double*, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_copy_cov(const double*, double*, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_add_ridge(double*, int, double, double*, cudaStream_t);
extern "C" void kodama_cuda_lda_means_to_rhs(const double*, double*, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_finalize_linear_row(const double*, const double*, const double*, double*, double*, int, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_score_argmax_row(const double*, const double*, const double*, int*, int, int, int, cudaStream_t);
extern "C" bool kodama_fastpls_simpls_fit_cuda(const double*, int, int, const double*, int, int, int, double*, double*);
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

void check_cusolver(cusolverStatus_t code, const char* where) {
  if (code != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(where) + ": cuSolver call failed");
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

class CudaLDAContext {
 public:
  explicit CudaLDAContext(int device) : device_(device) {
    check_cuda(cudaSetDevice(device_), "cudaSetDevice(CudaLDAContext)");
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate(CudaLDAContext)");
    check_cublas(cublasCreate(&blas_), "cublasCreate(CudaLDAContext)");
    check_cublas(cublasSetStream(blas_, stream_), "cublasSetStream(CudaLDAContext)");
    check_cusolver(cusolverDnCreate(&solver_), "cusolverDnCreate(CudaLDAContext)");
    check_cusolver(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream(CudaLDAContext)");
  }

  ~CudaLDAContext() {
    if (solver_ != nullptr) cusolverDnDestroy(solver_);
    if (blas_ != nullptr) cublasDestroy(blas_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  CudaLDAContext(const CudaLDAContext&) = delete;
  CudaLDAContext& operator=(const CudaLDAContext&) = delete;

  int device() const { return device_; }
  cudaStream_t stream() const { return stream_; }
  cublasHandle_t blas() const { return blas_; }
  cusolverDnHandle_t solver() const { return solver_; }

 private:
  int device_ = 0;
  cudaStream_t stream_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  cusolverDnHandle_t solver_ = nullptr;
};

CudaLDAContext& cuda_lda_context(int device) {
  thread_local std::unique_ptr<CudaLDAContext> context;
  if (!context || context->device() != device) {
    context = std::make_unique<CudaLDAContext>(device);
  } else {
    check_cuda(cudaSetDevice(device), "cudaSetDevice(cuda_lda_context)");
  }
  return *context;
}

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

class DeviceIntBuffer {
 public:
  DeviceIntBuffer() = default;
  explicit DeviceIntBuffer(std::size_t n) { reset(n); }

  ~DeviceIntBuffer() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  DeviceIntBuffer(const DeviceIntBuffer&) = delete;
  DeviceIntBuffer& operator=(const DeviceIntBuffer&) = delete;

  void reset(std::size_t n) {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = n;
    if (n > 0) check_cuda(cudaMalloc(&ptr_, n * sizeof(int)), "cudaMalloc(int)");
  }

  int* data() { return ptr_; }
  const int* data() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  int* ptr_ = nullptr;
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
);

struct PLSFoldData {
  int fold = 0;
  std::vector<int> train;
  std::vector<int> validation;
  std::vector<double> mean;
  std::vector<double> scale;
  Dense x_train;
  Dense x_val;
};

struct PLSFoldXCache {
  bool valid = false;
  const void* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;
  MatrixValueType value_type = MatrixValueType::Float64;
  bool center = true;
  bool scale_columns = false;
  int folds = 0;
  std::uint64_t seed = 0;
  std::size_t constrain_hash = 0;
  std::vector<int> fold_assignments;
  std::vector<int> fold_ids;
  std::vector<PLSFoldData> folds_data;
};

std::size_t hash_int_vector(const std::vector<int>& values) {
  std::size_t h = 1469598103934665603ull;
  for (int value : values) {
    h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(value));
    h *= 1099511628211ull;
  }
  return h;
}

bool cache_matches(
  const PLSFoldXCache& cache,
  MatrixView x,
  const std::vector<int>& constrain,
  const PLSOptions& options
) {
  return cache.valid &&
         cache.data == x.data &&
         cache.rows == x.rows &&
         cache.cols == x.cols &&
         cache.value_type == x.value_type &&
         cache.center == options.center &&
         cache.scale_columns == options.scale &&
         cache.folds == options.cv.folds &&
         cache.seed == options.cv.seed &&
         cache.constrain_hash == hash_int_vector(constrain);
}

const PLSFoldXCache& get_pls_fold_x_cache(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options
) {
  thread_local PLSFoldXCache cache;
  if (cache_matches(cache, x, constrain, options)) return cache;

  cache = PLSFoldXCache{};
  cache.valid = true;
  cache.data = x.data;
  cache.rows = x.rows;
  cache.cols = x.cols;
  cache.value_type = x.value_type;
  cache.center = options.center;
  cache.scale_columns = options.scale;
  cache.folds = options.cv.folds;
  cache.seed = options.cv.seed;
  cache.constrain_hash = hash_int_vector(constrain);
  cache.fold_assignments = detail::make_folds(labels, constrain, options.cv);
  cache.fold_ids = detail::sorted_unique_folds(cache.fold_assignments);
  cache.folds_data.reserve(cache.fold_ids.size());

  for (int fold : cache.fold_ids) {
    PLSFoldData fold_data;
    fold_data.fold = fold;
    fold_data.validation = detail::indices_where_fold(cache.fold_assignments, fold, true);
    fold_data.train = detail::indices_where_fold(cache.fold_assignments, fold, false);
    train_center_scale(x, fold_data.train, options.center, options.scale, fold_data.mean, fold_data.scale);
    fold_data.x_train = subset_scale(x, fold_data.train, fold_data.mean, fold_data.scale);
    fold_data.x_val = subset_scale(x, fold_data.validation, fold_data.mean, fold_data.scale);
    cache.folds_data.push_back(std::move(fold_data));
  }
  return cache;
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
  std::vector<int> encoded(static_cast<std::size_t>(y.rows), 0);
  for (int i = 0; i < y.rows; ++i) {
    const int cls = labels[static_cast<std::size_t>(rows[static_cast<std::size_t>(i)])];
    const int j = class_pos[cls];
    encoded[static_cast<std::size_t>(i)] = j;
    y_mean[static_cast<std::size_t>(j)] += 1.0;
  }
  for (double& v : y_mean) v /= static_cast<double>(rows.size());
  for (int i = 0; i < y.rows; ++i) {
    double* row = y.data.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(y.cols);
    for (int j = 0; j < y.cols; ++j) row[j] = -y_mean[static_cast<std::size_t>(j)];
    row[encoded[static_cast<std::size_t>(i)]] += 1.0;
  }
  return y;
}

std::vector<double> one_hot_centered_colmajor(
  const std::vector<int>& labels,
  const std::vector<int>& rows,
  const std::vector<int>& classes,
  std::vector<double>& y_mean
) {
  std::map<int, int> class_pos;
  for (int i = 0; i < static_cast<int>(classes.size()); ++i) class_pos[classes[static_cast<std::size_t>(i)]] = i;

  const int n = static_cast<int>(rows.size());
  const int m = static_cast<int>(classes.size());
  y_mean.assign(classes.size(), 0.0);
  std::vector<int> encoded(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    const int cls = labels[static_cast<std::size_t>(rows[static_cast<std::size_t>(i)])];
    const int j = class_pos[cls];
    encoded[static_cast<std::size_t>(i)] = j;
    y_mean[static_cast<std::size_t>(j)] += 1.0;
  }
  for (double& v : y_mean) v /= static_cast<double>(rows.size());

  std::vector<double> y_colmajor(static_cast<std::size_t>(n) * static_cast<std::size_t>(m), 0.0);
  for (int j = 0; j < m; ++j) {
    double* col = y_colmajor.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
    std::fill(col, col + n, -y_mean[static_cast<std::size_t>(j)]);
  }
  for (int i = 0; i < n; ++i) {
    const int j = encoded[static_cast<std::size_t>(i)];
    y_colmajor[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(n)] += 1.0;
  }
  return y_colmajor;
}

Dense crossprod(const Dense& x, const Dense& y) {
  Dense out(x.cols, y.cols);
  std::vector<long double> accum(static_cast<std::size_t>(x.cols) * static_cast<std::size_t>(y.cols), 0.0L);
  for (int r = 0; r < x.rows; ++r) {
    const double* x_row = x.data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(x.cols);
    const double* y_row = y.data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(y.cols);
    for (int i = 0; i < x.cols; ++i) {
      const long double xv = x_row[i];
      for (int j = 0; j < y.cols; ++j) {
        accum[static_cast<std::size_t>(i) * static_cast<std::size_t>(y.cols) + static_cast<std::size_t>(j)] +=
          xv * y_row[j];
      }
    }
  }
  for (int i = 0; i < x.cols; ++i) {
    for (int j = 0; j < y.cols; ++j) {
      out(i, j) = static_cast<double>(
        accum[static_cast<std::size_t>(i) * static_cast<std::size_t>(y.cols) + static_cast<std::size_t>(j)]
      );
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

std::vector<int> components_to_evaluate(const PLSOptions& options, int available_components) {
  if (available_components < 1) return {};
  const int requested = options.fixed_components > 0 ? options.fixed_components : options.max_components;
  return {std::min(requested, available_components)};
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

PLSFit fit_pls_components_from_crossprod(const Dense& x, const Dense& y, Dense s, int max_components) {
  const int max_rank = std::min({max_components, x.cols, std::max(1, x.rows - 1)});
  Dense w(x.cols, max_rank);
  Dense pmat(x.cols, max_rank);
  Dense qmat(y.cols, max_rank);
  Dense tmat(x.rows, max_rank);
  for (int a = 0; a < max_rank; ++a) {
    Dense gram(s.rows, s.rows);
    for (int i = 0; i < s.rows; ++i) {
      for (int j = 0; j < s.rows; ++j) {
        long double val = 0.0;
        for (int c = 0; c < s.cols; ++c) val += static_cast<long double>(s(i, c)) * s(j, c);
        gram(i, j) = static_cast<double>(val);
      }
    }
    std::vector<double> wa = dominant_left_singular_vector(s);
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
    long double tnorm2 = 0.0;
    for (double tv : t) tnorm2 += static_cast<long double>(tv) * tv;
    const double inv_tnorm2 = tnorm2 > 1e-20 ? 1.0 / static_cast<double>(tnorm2) : 0.0;
    std::vector<double> vvec(static_cast<std::size_t>(x.cols), 0.0);
    if (inv_tnorm2 > 0.0) {
      for (int j = 0; j < x.cols; ++j) {
        long double val = 0.0;
        for (int i = 0; i < x.rows; ++i) val += static_cast<long double>(x(i, j)) * t[static_cast<std::size_t>(i)];
        vvec[static_cast<std::size_t>(j)] = static_cast<double>(val) * inv_tnorm2;
      }
    } else {
      vvec = wa;
    }
    for (int prev = 0; prev < a; ++prev) {
      double proj = 0.0;
      for (int j = 0; j < x.cols; ++j) proj += vvec[static_cast<std::size_t>(j)] * pmat(j, prev);
      for (int j = 0; j < x.cols; ++j) vvec[static_cast<std::size_t>(j)] -= proj * pmat(j, prev);
    }
    double nv = norm2(vvec);
    if (nv <= 1e-12) {
      vvec = wa;
      for (int prev = 0; prev < a; ++prev) {
        double proj = 0.0;
        for (int j = 0; j < x.cols; ++j) proj += vvec[static_cast<std::size_t>(j)] * pmat(j, prev);
        for (int j = 0; j < x.cols; ++j) vvec[static_cast<std::size_t>(j)] -= proj * pmat(j, prev);
      }
      nv = norm2(vvec);
    }
    if (nv <= 1e-12) {
      std::fill(vvec.begin(), vvec.end(), 0.0);
      vvec[static_cast<std::size_t>(a % x.cols)] = 1.0;
      for (int prev = 0; prev < a; ++prev) {
        double proj = 0.0;
        for (int j = 0; j < x.cols; ++j) proj += vvec[static_cast<std::size_t>(j)] * pmat(j, prev);
        for (int j = 0; j < x.cols; ++j) vvec[static_cast<std::size_t>(j)] -= proj * pmat(j, prev);
      }
      nv = std::max(norm2(vvec), 1e-12);
    }
    for (double& vv : vvec) vv /= nv;

    for (int j = 0; j < x.cols; ++j) {
      w(j, a) = wa[static_cast<std::size_t>(j)];
      pmat(j, a) = vvec[static_cast<std::size_t>(j)];
    }
    for (int j = 0; j < y.cols; ++j) qmat(j, a) = right[static_cast<std::size_t>(j)];
    for (int i = 0; i < x.rows; ++i) tmat(i, a) = t[static_cast<std::size_t>(i)];

    std::vector<double> vs(static_cast<std::size_t>(s.cols), 0.0);
    for (int c = 0; c < s.cols; ++c) {
      long double val = 0.0;
      for (int j = 0; j < s.rows; ++j) val += static_cast<long double>(vvec[static_cast<std::size_t>(j)]) * s(j, c);
      vs[static_cast<std::size_t>(c)] = static_cast<double>(val);
    }
    for (int j = 0; j < s.rows; ++j) {
      for (int c = 0; c < s.cols; ++c) {
        s(j, c) -= vvec[static_cast<std::size_t>(j)] * vs[static_cast<std::size_t>(c)];
      }
    }
  }
  return PLSFit{w, pmat, qmat, tmat};
}

PLSFit fit_pls_components(const Dense& x, const Dense& y, int max_components) {
  return fit_pls_components_from_crossprod(x, y, crossprod(x, y), max_components);
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

PLSFit fit_pls_components_cuda_colmajor_y(
  const Dense& x,
  const double* y_colmajor,
  int y_cols,
  int max_components,
  int gpu_device
) {
  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice(fit_pls_components_cuda)");
  const int max_rank = std::min({max_components, x.cols, std::max(1, x.rows - 1)});
  std::vector<double> x_colmajor(static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols));
  for (int j = 0; j < x.cols; ++j) {
    for (int i = 0; i < x.rows; ++i) {
      x_colmajor[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(x.rows)] = x(i, j);
    }
  }

  std::vector<double> rr_colmajor(static_cast<std::size_t>(x.cols) * static_cast<std::size_t>(max_rank), 0.0);
  std::vector<double> qq_colmajor(static_cast<std::size_t>(y_cols) * static_cast<std::size_t>(max_rank), 0.0);
  const bool ok = kodama_fastpls_simpls_fit_cuda(
    x_colmajor.data(),
    x.rows,
    x.cols,
    y_colmajor,
    y_cols,
    max_rank,
    1,
    rr_colmajor.data(),
    qq_colmajor.data()
  );
  if (!ok) throw std::runtime_error("fastPLS CUDA SIMPLS fit failed.");

  PLSFit fit{Dense(x.cols, max_rank), Dense(x.cols, max_rank), Dense(y_cols, max_rank), Dense()};
  for (int a = 0; a < max_rank; ++a) {
    for (int j = 0; j < x.cols; ++j) {
      fit.weights(j, a) = rr_colmajor[static_cast<std::size_t>(j) + static_cast<std::size_t>(a) * static_cast<std::size_t>(x.cols)];
    }
    for (int j = 0; j < y_cols; ++j) {
      fit.y_weights(j, a) = qq_colmajor[static_cast<std::size_t>(j) + static_cast<std::size_t>(a) * static_cast<std::size_t>(y_cols)];
    }
  }
  fit.train_scores = transform_pls_scores_cuda(x, fit, fit.weights.cols, gpu_device);
  return fit;
}

PLSFit fit_pls_components_cuda(const Dense& x, const Dense& y, int max_components, int gpu_device) {
  std::vector<double> y_colmajor(static_cast<std::size_t>(y.rows) * static_cast<std::size_t>(y.cols));
  for (int j = 0; j < y.cols; ++j) {
    for (int i = 0; i < y.rows; ++i) {
      y_colmajor[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(y.rows)] = y(i, j);
    }
  }
  return fit_pls_components_cuda_colmajor_y(x, y_colmajor.data(), y.cols, max_components, gpu_device);
}
#endif

Dense regression_coefficients(const Dense& t, const Dense& y, int ncomp) {
  Dense lhs(ncomp, ncomp);
  Dense rhs(ncomp, y.cols);
  for (int i = 0; i < ncomp; ++i) {
    for (int j = 0; j < ncomp; ++j) {
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

Dense regression_coefficients(const Dense& t, const Dense& y) {
  return regression_coefficients(t, y, t.cols);
}

Dense regression_coefficients_centered_one_hot(
  const Dense& t,
  const std::vector<int>& labels,
  const std::vector<int>& classes,
  const std::vector<double>& y_mean,
  int ncomp
) {
  if (static_cast<int>(labels.size()) != t.rows) throw std::invalid_argument("PLS-DA label size mismatch.");
  const int cnum = static_cast<int>(classes.size());
  Dense lhs(ncomp, ncomp);
  Dense rhs(ncomp, cnum);
  std::map<int, int> class_pos;
  for (int c = 0; c < cnum; ++c) class_pos[classes[static_cast<std::size_t>(c)]] = c;
  std::vector<int> encoded(labels.size(), 0);
  for (std::size_t i = 0; i < labels.size(); ++i) encoded[i] = class_pos.at(labels[i]);
  std::vector<double> score_sums(static_cast<std::size_t>(ncomp), 0.0);
  std::vector<double> lhs_accum(static_cast<std::size_t>(ncomp) * static_cast<std::size_t>(ncomp), 0.0);

  for (int r = 0; r < t.rows; ++r) {
    const int c = encoded[static_cast<std::size_t>(r)];
    const double* t_row = t.data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(t.cols);
    for (int a = 0; a < ncomp; ++a) {
      const double value = t_row[a];
      rhs(a, c) += value;
      score_sums[static_cast<std::size_t>(a)] += value;
      for (int b = a; b < ncomp; ++b) {
        lhs_accum[static_cast<std::size_t>(a) * static_cast<std::size_t>(ncomp) + static_cast<std::size_t>(b)] +=
          value * t_row[b];
      }
    }
  }
  for (int a = 0; a < ncomp; ++a) {
    for (int b = a; b < ncomp; ++b) {
      const double value = lhs_accum[static_cast<std::size_t>(a) * static_cast<std::size_t>(ncomp) + static_cast<std::size_t>(b)];
      lhs(a, b) = value;
      lhs(b, a) = value;
    }
  }
  for (int a = 0; a < ncomp; ++a) {
    const double sum = score_sums[static_cast<std::size_t>(a)];
    for (int c = 0; c < cnum; ++c) rhs(a, c) -= sum * y_mean[static_cast<std::size_t>(c)];
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
  const Dense& y_train,
  const Dense& t_val,
  const std::vector<int>& classes,
  const std::vector<double>& y_mean,
  int ncomp
) {
  const Dense coef = regression_coefficients(t_train, y_train, ncomp);
  std::vector<int> pred(static_cast<std::size_t>(t_val.rows), classes.front());
  for (int i = 0; i < t_val.rows; ++i) {
    int best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int c = 0; c < static_cast<int>(classes.size()); ++c) {
      long double score = c < static_cast<int>(y_mean.size()) ? y_mean[static_cast<std::size_t>(c)] : 0.0;
      for (int a = 0; a < ncomp; ++a) {
        score += static_cast<long double>(t_val(i, a)) * coef(a, c);
      }
      if (score > best_score) {
        best_score = static_cast<double>(score);
        best = c;
      }
    }
    pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(best)];
  }
  return pred;
}

std::vector<int> predict_pls_da_labels(
  const Dense& t_train,
  const std::vector<int>& y_train_labels,
  const Dense& t_val,
  const std::vector<int>& classes,
  const std::vector<double>& y_mean,
  int ncomp
) {
  const Dense coef = regression_coefficients_centered_one_hot(t_train, y_train_labels, classes, y_mean, ncomp);
  std::vector<int> pred(static_cast<std::size_t>(t_val.rows), classes.front());
  for (int i = 0; i < t_val.rows; ++i) {
    int best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int c = 0; c < static_cast<int>(classes.size()); ++c) {
      long double score = c < static_cast<int>(y_mean.size()) ? y_mean[static_cast<std::size_t>(c)] : 0.0;
      for (int a = 0; a < ncomp; ++a) {
        score += static_cast<long double>(t_val(i, a)) * coef(a, c);
      }
      if (score > best_score) {
        best_score = static_cast<double>(score);
        best = c;
      }
    }
    pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(best)];
  }
  return pred;
}

std::vector<int> predict_pls_da(
  const Dense& t_train,
  const Dense& y_train,
  const Dense& t_val,
  const std::vector<int>& classes,
  const std::vector<double>& y_mean
) {
  return predict_pls_da(t_train, y_train, t_val, classes, y_mean, t_val.cols);
}

#if defined(KODAMA_ENABLE_CUDA)
std::vector<int> predict_pls_da_cuda(
  const Dense& t_train,
  const std::vector<int>& y_train_labels,
  const Dense& t_val,
  const std::vector<int>& classes,
  const std::vector<double>& y_mean,
  int gpu_device
) {
  if (t_val.rows < 1) return {};
  const int n = t_val.rows;
  const int kk = t_val.cols;
  const int cnum = static_cast<int>(classes.size());
  const Dense coef = regression_coefficients_centered_one_hot(t_train, y_train_labels, classes, y_mean, kk);

  std::vector<double> linear(static_cast<std::size_t>(cnum) * static_cast<std::size_t>(kk), 0.0);
  std::vector<double> constants(static_cast<std::size_t>(cnum), 0.0);
  for (int c = 0; c < cnum; ++c) {
    constants[static_cast<std::size_t>(c)] =
      c < static_cast<int>(y_mean.size()) ? y_mean[static_cast<std::size_t>(c)] : 0.0;
    for (int a = 0; a < kk; ++a) {
      linear[static_cast<std::size_t>(c) * static_cast<std::size_t>(kk) + static_cast<std::size_t>(a)] = coef(a, c);
    }
  }

  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice(predict_pls_da_cuda)");
  cudaStream_t stream = nullptr;
  DeviceBuffer d_t(t_val.data.size());
  DeviceBuffer d_linear(linear.size());
  DeviceBuffer d_constants(constants.size());
  DeviceIntBuffer d_pred(static_cast<std::size_t>(n));
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate(predict_pls_da_cuda)");
  check_cuda(cudaMemcpyAsync(d_t.data(), t_val.data.data(), t_val.data.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA PLS-DA T");
  check_cuda(cudaMemcpyAsync(d_linear.data(), linear.data(), linear.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA PLS-DA linear");
  check_cuda(cudaMemcpyAsync(d_constants.data(), constants.data(), constants.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA PLS-DA constants");
  kodama_cuda_lda_score_argmax_row(d_t.data(), d_linear.data(), d_constants.data(), d_pred.data(), n, kk, cnum, stream);
  check_cuda(cudaGetLastError(), "kodama_cuda_lda_score_argmax_row PLS-DA");
  std::vector<int> codes(static_cast<std::size_t>(n), 1);
  check_cuda(cudaMemcpyAsync(codes.data(), d_pred.data(), codes.size() * sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync CUDA PLS-DA labels");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CUDA PLS-DA predict");
  cudaStreamDestroy(stream);

  std::vector<int> pred(static_cast<std::size_t>(n), classes.front());
  for (int i = 0; i < n; ++i) {
    const int cls = std::max(1, std::min(cnum, codes[static_cast<std::size_t>(i)])) - 1;
    pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(cls)];
  }
  return pred;
}
#endif

std::vector<int> predict_pls_lda(
  const Dense& t_train,
  const std::vector<int>& y_train,
  const Dense& t_val,
  const std::vector<int>& classes,
  int ncomp
) {
  const int cnum = static_cast<int>(classes.size());
  std::map<int, int> cpos;
  for (int i = 0; i < cnum; ++i) cpos[classes[static_cast<std::size_t>(i)]] = i;
  Dense cent(cnum, ncomp);
  std::vector<int> counts(static_cast<std::size_t>(cnum), 0);
  for (int i = 0; i < t_train.rows; ++i) {
    const int c = cpos[y_train[static_cast<std::size_t>(i)]];
    counts[static_cast<std::size_t>(c)]++;
    for (int a = 0; a < ncomp; ++a) cent(c, a) += t_train(i, a);
  }
  for (int c = 0; c < cnum; ++c) {
    const double inv = counts[static_cast<std::size_t>(c)] > 0 ? 1.0 / counts[static_cast<std::size_t>(c)] : 0.0;
    for (int a = 0; a < ncomp; ++a) cent(c, a) *= inv;
  }

  Dense pooled(ncomp, ncomp);
  for (int i = 0; i < t_train.rows; ++i) {
    const int c = cpos[y_train[static_cast<std::size_t>(i)]];
    for (int r = 0; r < ncomp; ++r) {
      const double dr = t_train(i, r) - cent(c, r);
      for (int col = 0; col < ncomp; ++col) {
        const double dc = t_train(i, col) - cent(c, col);
        pooled(r, col) += dr * dc;
      }
    }
  }
  const double df = static_cast<double>(std::max(1, t_train.rows - cnum));
  double trace = 0.0;
  for (int r = 0; r < ncomp; ++r) {
    for (int col = 0; col < ncomp; ++col) pooled(r, col) /= df;
    trace += pooled(r, r);
  }
  const double ridge = 1e-8 * (std::isfinite(trace) && trace > 0.0 ? trace / std::max(1, ncomp) : 1.0);
  for (int r = 0; r < ncomp; ++r) pooled(r, r) += ridge;

  Dense rhs(ncomp, cnum);
  for (int c = 0; c < cnum; ++c) {
    for (int a = 0; a < ncomp; ++a) rhs(a, c) = cent(c, a);
  }
  Dense solved = solve_linear(pooled, rhs);

  Dense linear(cnum, ncomp);
  std::vector<double> constants(static_cast<std::size_t>(cnum), 0.0);
  for (int c = 0; c < cnum; ++c) {
    double dot_mu = 0.0;
    for (int a = 0; a < ncomp; ++a) {
      linear(c, a) = solved(a, c);
      dot_mu += cent(c, a) * linear(c, a);
    }
    const double prior = std::max(
      static_cast<double>(counts[static_cast<std::size_t>(c)]) / std::max(1, t_train.rows),
      std::numeric_limits<double>::min()
    );
    constants[static_cast<std::size_t>(c)] = -0.5 * dot_mu + std::log(prior);
  }

  std::vector<int> pred(static_cast<std::size_t>(t_val.rows), classes.front());
  for (int i = 0; i < t_val.rows; ++i) {
    int best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int c = 0; c < cnum; ++c) {
      long double score = constants[static_cast<std::size_t>(c)];
      for (int a = 0; a < ncomp; ++a) {
        score += static_cast<long double>(t_val(i, a)) * linear(c, a);
      }
      if (score > best_score) {
        best_score = static_cast<double>(score);
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
  return predict_pls_lda(t_train, y_train, t_val, classes, t_val.cols);
}

#if defined(KODAMA_ENABLE_CUDA)
struct CudaLDAModel {
  Dense linear;
  std::vector<double> constants;
};

CudaLDAModel train_pls_lda_cuda(
  const Dense& t_train,
  const std::vector<int>& y_train,
  const std::vector<int>& classes,
  int gpu_device
) {
  if (t_train.rows < 1 || t_train.cols < 1) throw std::invalid_argument("CUDA LDA requires a non-empty score matrix.");
  if (static_cast<int>(y_train.size()) != t_train.rows) throw std::invalid_argument("CUDA LDA label size mismatch.");
  const int n = t_train.rows;
  const int k = t_train.cols;
  const int cnum = static_cast<int>(classes.size());
  CudaLDAContext& context = cuda_lda_context(gpu_device);
  cudaStream_t stream = context.stream();
  cublasHandle_t blas = context.blas();
  cusolverDnHandle_t solver = context.solver();

  std::map<int, int> cpos;
  for (int c = 0; c < cnum; ++c) cpos[classes[static_cast<std::size_t>(c)]] = c;
  std::vector<int> encoded(static_cast<std::size_t>(n), 1);
  std::vector<double> counts(static_cast<std::size_t>(cnum), 0.0);
  for (int i = 0; i < n; ++i) {
    const int cls = cpos.at(y_train[static_cast<std::size_t>(i)]);
    encoded[static_cast<std::size_t>(i)] = cls + 1;
    counts[static_cast<std::size_t>(cls)] += 1.0;
  }

  DeviceBuffer d_t(t_train.data.size());
  DeviceIntBuffer d_labels(encoded.size());
  DeviceBuffer d_counts(counts.size());
  DeviceBuffer d_means(static_cast<std::size_t>(cnum) * static_cast<std::size_t>(k));
  DeviceBuffer d_pooled(static_cast<std::size_t>(k) * static_cast<std::size_t>(k));
  DeviceBuffer d_cov(static_cast<std::size_t>(k) * static_cast<std::size_t>(k));
  DeviceBuffer d_rhs(static_cast<std::size_t>(k) * static_cast<std::size_t>(cnum));
  DeviceBuffer d_linear(static_cast<std::size_t>(cnum) * static_cast<std::size_t>(k));
  DeviceBuffer d_constants(static_cast<std::size_t>(cnum));
  DeviceBuffer d_lambda(1);
  DeviceIntBuffer d_info(1);
  double* d_work = nullptr;

  auto cleanup = [&]() {
    if (d_work != nullptr) cudaFree(d_work);
  };

  try {
    check_cuda(cudaMemcpyAsync(d_t.data(), t_train.data.data(), t_train.data.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync LDA T");
    check_cuda(cudaMemcpyAsync(d_labels.data(), encoded.data(), encoded.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync LDA labels");
    check_cuda(cudaMemcpyAsync(d_counts.data(), counts.data(), counts.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync LDA counts");

    kodama_cuda_lda_label_sums_row(d_t.data(), d_labels.data(), n, k, cnum, d_means.data(), stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_label_sums_row");
    kodama_cuda_lda_means_row(d_means.data(), d_counts.data(), k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_means_row");

    const double one = 1.0;
    const double zero = 0.0;
    check_cublas(
      cublasDgemm(
        blas,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        k,
        k,
        n,
        &one,
        d_t.data(),
        k,
        d_t.data(),
        k,
        &zero,
        d_pooled.data(),
        k
      ),
      "cublasDgemm CUDA LDA TtT"
    );
    kodama_cuda_lda_pooled_col(d_pooled.data(), d_means.data(), d_counts.data(), n, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_pooled_col");
    kodama_cuda_lda_copy_cov(d_pooled.data(), d_cov.data(), k, k, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_copy_cov");
    kodama_cuda_lda_add_ridge(d_cov.data(), k, 1e-8, d_lambda.data(), stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_add_ridge");
    kodama_cuda_lda_means_to_rhs(d_means.data(), d_rhs.data(), k, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_means_to_rhs");

    int lwork = 0;
    check_cusolver(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, k, d_cov.data(), k, &lwork), "cusolverDnDpotrf_bufferSize CUDA LDA");
    check_cuda(cudaMalloc(&d_work, sizeof(double) * static_cast<std::size_t>(std::max(lwork, 1))), "cudaMalloc CUDA LDA work");
    check_cusolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, k, d_cov.data(), k, d_work, lwork, d_info.data()), "cusolverDnDpotrf CUDA LDA");
    int info = 0;
    check_cuda(cudaMemcpyAsync(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync CUDA LDA potrf info");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CUDA LDA potrf");
    if (info != 0) throw std::runtime_error("cusolverDnDpotrf CUDA LDA returned non-zero info.");
    check_cusolver(cusolverDnDpotrs(solver, CUBLAS_FILL_MODE_LOWER, k, cnum, d_cov.data(), k, d_rhs.data(), k, d_info.data()), "cusolverDnDpotrs CUDA LDA");
    check_cuda(cudaMemcpyAsync(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync CUDA LDA potrs info");
    kodama_cuda_lda_finalize_linear_row(d_rhs.data(), d_means.data(), d_counts.data(), d_linear.data(), d_constants.data(), n, k, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_finalize_linear_row");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CUDA LDA solve");
    if (info != 0) throw std::runtime_error("cusolverDnDpotrs CUDA LDA returned non-zero info.");

    CudaLDAModel model{Dense(cnum, k), std::vector<double>(static_cast<std::size_t>(cnum), 0.0)};
    check_cuda(cudaMemcpy(model.linear.data.data(), d_linear.data(), model.linear.data.size() * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy CUDA LDA linear");
    check_cuda(cudaMemcpy(model.constants.data(), d_constants.data(), model.constants.size() * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy CUDA LDA constants");
    cleanup();
    return model;
  } catch (...) {
    cleanup();
    throw;
  }
}

std::vector<int> train_predict_pls_lda_cuda(
  const Dense& t_train,
  const std::vector<int>& y_train,
  const Dense& t_val,
  const std::vector<int>& classes,
  int gpu_device
) {
  if (t_train.rows < 1 || t_train.cols < 1) throw std::invalid_argument("CUDA LDA requires a non-empty score matrix.");
  if (static_cast<int>(y_train.size()) != t_train.rows) throw std::invalid_argument("CUDA LDA label size mismatch.");
  if (t_val.rows < 1) return {};
  if (t_val.cols != t_train.cols) throw std::invalid_argument("CUDA LDA train/validation score column mismatch.");
  const int n = t_train.rows;
  const int k = t_train.cols;
  const int n_val = t_val.rows;
  const int cnum = static_cast<int>(classes.size());
  CudaLDAContext& context = cuda_lda_context(gpu_device);
  cudaStream_t stream = context.stream();
  cublasHandle_t blas = context.blas();
  cusolverDnHandle_t solver = context.solver();

  std::map<int, int> cpos;
  for (int c = 0; c < cnum; ++c) cpos[classes[static_cast<std::size_t>(c)]] = c;
  std::vector<int> encoded(static_cast<std::size_t>(n), 1);
  std::vector<double> counts(static_cast<std::size_t>(cnum), 0.0);
  for (int i = 0; i < n; ++i) {
    const int cls = cpos.at(y_train[static_cast<std::size_t>(i)]);
    encoded[static_cast<std::size_t>(i)] = cls + 1;
    counts[static_cast<std::size_t>(cls)] += 1.0;
  }

  DeviceBuffer d_t(t_train.data.size());
  DeviceIntBuffer d_labels(encoded.size());
  DeviceBuffer d_counts(counts.size());
  DeviceBuffer d_means(static_cast<std::size_t>(cnum) * static_cast<std::size_t>(k));
  DeviceBuffer d_pooled(static_cast<std::size_t>(k) * static_cast<std::size_t>(k));
  DeviceBuffer d_cov(static_cast<std::size_t>(k) * static_cast<std::size_t>(k));
  DeviceBuffer d_rhs(static_cast<std::size_t>(k) * static_cast<std::size_t>(cnum));
  DeviceBuffer d_linear(static_cast<std::size_t>(cnum) * static_cast<std::size_t>(k));
  DeviceBuffer d_constants(static_cast<std::size_t>(cnum));
  DeviceBuffer d_lambda(1);
  DeviceIntBuffer d_info(1);
  DeviceBuffer d_t_val(t_val.data.size());
  DeviceIntBuffer d_pred(static_cast<std::size_t>(n_val));
  double* d_work = nullptr;

  auto cleanup = [&]() {
    if (d_work != nullptr) cudaFree(d_work);
  };

  try {
    check_cuda(cudaMemcpyAsync(d_t.data(), t_train.data.data(), t_train.data.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync LDA T");
    check_cuda(cudaMemcpyAsync(d_labels.data(), encoded.data(), encoded.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync LDA labels");
    check_cuda(cudaMemcpyAsync(d_counts.data(), counts.data(), counts.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync LDA counts");

    kodama_cuda_lda_label_sums_row(d_t.data(), d_labels.data(), n, k, cnum, d_means.data(), stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_label_sums_row");
    kodama_cuda_lda_means_row(d_means.data(), d_counts.data(), k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_means_row");

    const double one = 1.0;
    const double zero = 0.0;
    check_cublas(
      cublasDgemm(
        blas,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        k,
        k,
        n,
        &one,
        d_t.data(),
        k,
        d_t.data(),
        k,
        &zero,
        d_pooled.data(),
        k
      ),
      "cublasDgemm CUDA LDA TtT"
    );
    kodama_cuda_lda_pooled_col(d_pooled.data(), d_means.data(), d_counts.data(), n, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_pooled_col");
    kodama_cuda_lda_copy_cov(d_pooled.data(), d_cov.data(), k, k, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_copy_cov");
    kodama_cuda_lda_add_ridge(d_cov.data(), k, 1e-8, d_lambda.data(), stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_add_ridge");
    kodama_cuda_lda_means_to_rhs(d_means.data(), d_rhs.data(), k, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_means_to_rhs");

    int lwork = 0;
    check_cusolver(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, k, d_cov.data(), k, &lwork), "cusolverDnDpotrf_bufferSize CUDA LDA");
    check_cuda(cudaMalloc(&d_work, sizeof(double) * static_cast<std::size_t>(std::max(lwork, 1))), "cudaMalloc CUDA LDA work");
    check_cusolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, k, d_cov.data(), k, d_work, lwork, d_info.data()), "cusolverDnDpotrf CUDA LDA");
    int info = 0;
    check_cuda(cudaMemcpyAsync(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync CUDA LDA potrf info");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CUDA LDA potrf");
    if (info != 0) throw std::runtime_error("cusolverDnDpotrf CUDA LDA returned non-zero info.");
    check_cusolver(cusolverDnDpotrs(solver, CUBLAS_FILL_MODE_LOWER, k, cnum, d_cov.data(), k, d_rhs.data(), k, d_info.data()), "cusolverDnDpotrs CUDA LDA");
    check_cuda(cudaMemcpyAsync(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync CUDA LDA potrs info");
    kodama_cuda_lda_finalize_linear_row(d_rhs.data(), d_means.data(), d_counts.data(), d_linear.data(), d_constants.data(), n, k, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_finalize_linear_row");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CUDA LDA solve");
    if (info != 0) throw std::runtime_error("cusolverDnDpotrs CUDA LDA returned non-zero info.");

    check_cuda(cudaMemcpyAsync(d_t_val.data(), t_val.data.data(), t_val.data.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA LDA predict T");
    kodama_cuda_lda_score_argmax_row(d_t_val.data(), d_linear.data(), d_constants.data(), d_pred.data(), n_val, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_score_argmax_row");
    std::vector<int> codes(static_cast<std::size_t>(n_val), 1);
    check_cuda(cudaMemcpyAsync(codes.data(), d_pred.data(), codes.size() * sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync CUDA LDA predict labels");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CUDA LDA predict");
    cleanup();

    std::vector<int> pred(static_cast<std::size_t>(n_val), classes.front());
    for (int i = 0; i < n_val; ++i) {
      const int cls = std::max(1, std::min(cnum, codes[static_cast<std::size_t>(i)])) - 1;
      pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(cls)];
    }
    return pred;
  } catch (...) {
    cleanup();
    throw;
  }
}

std::vector<int> predict_pls_lda_cuda(
  const Dense& t_val,
  const CudaLDAModel& model,
  const std::vector<int>& classes,
  int gpu_device
) {
  if (t_val.rows < 1) return {};
  const int n = t_val.rows;
  const int k = t_val.cols;
  const int cnum = static_cast<int>(classes.size());
  CudaLDAContext& context = cuda_lda_context(gpu_device);
  cudaStream_t stream = context.stream();
  DeviceBuffer d_t(t_val.data.size());
  DeviceBuffer d_linear(model.linear.data.size());
  DeviceBuffer d_constants(model.constants.size());
  DeviceIntBuffer d_pred(static_cast<std::size_t>(n));
  check_cuda(cudaMemcpyAsync(d_t.data(), t_val.data.data(), t_val.data.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA LDA predict T");
  check_cuda(cudaMemcpyAsync(d_linear.data(), model.linear.data.data(), model.linear.data.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA LDA predict linear");
  check_cuda(cudaMemcpyAsync(d_constants.data(), model.constants.data(), model.constants.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA LDA predict constants");
  kodama_cuda_lda_score_argmax_row(d_t.data(), d_linear.data(), d_constants.data(), d_pred.data(), n, k, cnum, stream);
  check_cuda(cudaGetLastError(), "kodama_cuda_lda_score_argmax_row");
  std::vector<int> codes(static_cast<std::size_t>(n), 1);
  check_cuda(cudaMemcpyAsync(codes.data(), d_pred.data(), codes.size() * sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync CUDA LDA predict labels");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CUDA LDA predict");
  std::vector<int> pred(static_cast<std::size_t>(n), classes.front());
  for (int i = 0; i < n; ++i) {
    const int cls = std::max(1, std::min(cnum, codes[static_cast<std::size_t>(i)])) - 1;
    pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(cls)];
  }
  return pred;
}

#endif

// Candidate-kNN classifier ported from fastPLS (GPL-3): it scores class
// centroids in normalized PLS score space, reranks only the best candidate
// classes, and reuses prefix dot products across component counts.
void candidate_insert_top(
  std::vector<double>& scores,
  std::vector<int>& index,
  double value,
  int cls
) {
  const int k = static_cast<int>(scores.size());
  for (int j = 0; j < k; ++j) {
    if (value > scores[static_cast<std::size_t>(j)]) {
      for (int h = k - 1; h > j; --h) {
        scores[static_cast<std::size_t>(h)] = scores[static_cast<std::size_t>(h - 1)];
        index[static_cast<std::size_t>(h)] = index[static_cast<std::size_t>(h - 1)];
      }
      scores[static_cast<std::size_t>(j)] = value;
      index[static_cast<std::size_t>(j)] = cls;
      return;
    }
  }
}

Dense cumulative_norm2(const Dense& x) {
  Dense out(x.rows, x.cols);
  for (int i = 0; i < x.rows; ++i) {
    double acc = 0.0;
    for (int j = 0; j < x.cols; ++j) {
      acc += x(i, j) * x(i, j);
      out(i, j) = acc;
    }
  }
  return out;
}

double prefix_row_norm(const Dense& norm2, int row, int ncomp) {
  if (row < 0 || row >= norm2.rows || ncomp < 1 || ncomp > norm2.cols) return 0.0;
  const double val = norm2(row, ncomp - 1);
  return std::isfinite(val) && val > 0.0 ? std::sqrt(val) : 0.0;
}

Dense candidate_prefix_centroids(
  const Dense& t_train,
  const Dense& train_norm2,
  const std::vector<int>& y_codes,
  int n_classes,
  int ncomp
) {
  Dense cent(n_classes, ncomp);
  std::vector<double> counts(static_cast<std::size_t>(n_classes), 0.0);
  for (int i = 0; i < t_train.rows; ++i) {
    const int cls = y_codes[static_cast<std::size_t>(i)];
    if (cls < 1 || cls > n_classes) continue;
    const double nrm = prefix_row_norm(train_norm2, i, ncomp);
    if (nrm <= 0.0) continue;
    const int c = cls - 1;
    counts[static_cast<std::size_t>(c)] += 1.0;
    for (int d = 0; d < ncomp; ++d) cent(c, d) += t_train(i, d) / nrm;
  }
  for (int c = 0; c < n_classes; ++c) {
    if (counts[static_cast<std::size_t>(c)] > 0.0) {
      const double inv = 1.0 / counts[static_cast<std::size_t>(c)];
      for (int d = 0; d < ncomp; ++d) cent(c, d) *= inv;
    }
    long double n2 = 0.0;
    for (int d = 0; d < ncomp; ++d) n2 += static_cast<long double>(cent(c, d)) * cent(c, d);
    const double nrm = std::sqrt(static_cast<double>(n2));
    if (std::isfinite(nrm) && nrm > 0.0) {
      for (int d = 0; d < ncomp; ++d) cent(c, d) /= nrm;
    }
  }
  return cent;
}

std::vector<int> labels_to_codes(
  const std::vector<int>& y,
  const std::vector<int>& classes
) {
  std::map<int, int> pos;
  for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
    pos[classes[static_cast<std::size_t>(i)]] = i + 1;
  }
  std::vector<int> out(y.size(), 0);
  for (std::size_t i = 0; i < y.size(); ++i) out[i] = pos.at(y[i]);
  return out;
}

std::vector<std::vector<int>> predict_pls_cknn_prefix(
  const Dense& t_train,
  const std::vector<int>& y_train_labels,
  const Dense& t_val,
  const std::vector<int>& classes,
  const std::vector<int>& components,
  int top_m_in,
  int knn_k_in,
  double tau_in,
  double alpha_in,
  int n_threads
) {
  if (t_train.cols != t_val.cols) throw std::invalid_argument("PLS-cKNN train/test score dimensions do not match.");
  if (t_train.rows != static_cast<int>(y_train_labels.size())) throw std::invalid_argument("PLS-cKNN label length mismatch.");
  if (components.empty()) throw std::invalid_argument("PLS-cKNN requires at least one component.");
  const int ncomp_total = *std::max_element(components.begin(), components.end());
  if (ncomp_total < 1 || ncomp_total > t_train.cols || ncomp_total > t_val.cols) {
    throw std::invalid_argument("PLS-cKNN component count is out of range.");
  }
  const int n_classes = static_cast<int>(classes.size());
  const int top_m = std::max(1, std::min(top_m_in, n_classes));
  const int knn_k = std::max(1, knn_k_in);
  const double tau = (std::isfinite(tau_in) && tau_in > 0.0) ? tau_in : 0.2;
  const double alpha = std::isfinite(alpha_in) ? alpha_in : 0.5;
  const std::vector<int> y_codes = labels_to_codes(y_train_labels, classes);

  std::vector<std::vector<int>> class_rows(static_cast<std::size_t>(n_classes));
  for (int i = 0; i < static_cast<int>(y_codes.size()); ++i) {
    const int cls = y_codes[static_cast<std::size_t>(i)];
    if (cls >= 1 && cls <= n_classes) class_rows[static_cast<std::size_t>(cls - 1)].push_back(i);
  }

  const Dense train_norm2 = cumulative_norm2(t_train);
  const Dense test_norm2 = cumulative_norm2(t_val);
  std::vector<char> should_score(static_cast<std::size_t>(ncomp_total + 1), 0);
  std::vector<Dense> centroids_by_comp(static_cast<std::size_t>(ncomp_total + 1));
  for (int a : components) {
    should_score[static_cast<std::size_t>(a)] = 1;
    centroids_by_comp[static_cast<std::size_t>(a)] =
      candidate_prefix_centroids(t_train, train_norm2, y_codes, n_classes, a);
  }

  std::vector<std::vector<int>> pred_by_comp(
    static_cast<std::size_t>(ncomp_total),
    std::vector<int>(static_cast<std::size_t>(t_val.rows), classes.front())
  );

  const int workers = std::max(1, std::min(n_threads, t_val.rows));
  auto worker = [&](int begin, int end) {
    std::vector<double> row_scores(static_cast<std::size_t>(top_m), -std::numeric_limits<double>::infinity());
    std::vector<int> row_index(static_cast<std::size_t>(top_m), -1);
    std::vector<double> rerank_scores(static_cast<std::size_t>(top_m), -std::numeric_limits<double>::infinity());
    std::vector<int> rerank_index(static_cast<std::size_t>(top_m), -1);
    std::vector<double> train_dot(static_cast<std::size_t>(t_train.rows), 0.0);

    for (int i = begin; i < end; ++i) {
      std::fill(train_dot.begin(), train_dot.end(), 0.0);
      for (int a = 1; a <= ncomp_total; ++a) {
        const int d = a - 1;
        const double test_val = t_val(i, d);
        for (int r = 0; r < t_train.rows; ++r) train_dot[static_cast<std::size_t>(r)] += t_train(r, d) * test_val;
        if (!should_score[static_cast<std::size_t>(a)]) continue;

        const Dense& cent = centroids_by_comp[static_cast<std::size_t>(a)];
        std::fill(row_scores.begin(), row_scores.end(), -std::numeric_limits<double>::infinity());
        std::fill(row_index.begin(), row_index.end(), -1);
        const double test_norm = prefix_row_norm(test_norm2, i, a);
        for (int cls = 0; cls < n_classes; ++cls) {
          double base = 0.0;
          if (test_norm > 0.0) {
            for (int j = 0; j < a; ++j) base += t_val(i, j) * cent(cls, j);
            base /= test_norm;
          }
          candidate_insert_top(row_scores, row_index, base, cls + 1);
        }

        std::fill(rerank_scores.begin(), rerank_scores.end(), -std::numeric_limits<double>::infinity());
        std::fill(rerank_index.begin(), rerank_index.end(), -1);
        for (int slot = 0; slot < top_m; ++slot) {
          const int cls = row_index[static_cast<std::size_t>(slot)];
          if (cls < 1 || cls > n_classes) continue;
          const std::vector<int>& rows = class_rows[static_cast<std::size_t>(cls - 1)];
          double local = -std::numeric_limits<double>::infinity();
          if (!rows.empty()) {
            const int use_k = std::max(1, std::min(knn_k, static_cast<int>(rows.size())));
            std::vector<double> top_vals(static_cast<std::size_t>(use_k), -std::numeric_limits<double>::infinity());
            std::vector<int> top_dummy(static_cast<std::size_t>(use_k), -1);
            for (int train_row : rows) {
              const double train_norm = prefix_row_norm(train_norm2, train_row, a);
              const double sim = (test_norm > 0.0 && train_norm > 0.0) ?
                train_dot[static_cast<std::size_t>(train_row)] / (test_norm * train_norm) :
                0.0;
              candidate_insert_top(top_vals, top_dummy, sim, 0);
            }
            if (!std::isfinite(tau) || tau <= 0.0) {
              local = 0.0;
              for (int h = 0; h < use_k; ++h) local += top_vals[static_cast<std::size_t>(h)];
              local /= static_cast<double>(use_k);
            } else {
              const double mx = top_vals[0];
              if (std::isfinite(mx)) {
                double acc = 0.0;
                for (int h = 0; h < use_k; ++h) acc += std::exp((top_vals[static_cast<std::size_t>(h)] - mx) / tau);
                local = mx + tau * std::log(acc / static_cast<double>(use_k));
              }
            }
          }
          const double score = local + alpha * row_scores[static_cast<std::size_t>(slot)];
          candidate_insert_top(rerank_scores, rerank_index, score, cls);
        }
        const int best_code = rerank_index[0] > 0 ? rerank_index[0] : row_index[0];
        const int best_pos = std::max(1, best_code) - 1;
        pred_by_comp[static_cast<std::size_t>(d)][static_cast<std::size_t>(i)] =
          classes[static_cast<std::size_t>(std::min(best_pos, n_classes - 1))];
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(workers - 1));
  const int chunk = (t_val.rows + workers - 1) / workers;
  int begin = 0;
  for (int w = 0; w < workers; ++w) {
    const int end = std::min(t_val.rows, begin + chunk);
    if (begin >= end) break;
    if (w + 1 == workers) {
      worker(begin, end);
    } else {
      threads.emplace_back(worker, begin, end);
    }
    begin = end;
  }
  for (std::thread& th : threads) th.join();
  return pred_by_comp;
}

#if defined(KODAMA_ENABLE_CUDA)
std::vector<std::vector<int>> predict_pls_cknn_prefix_cuda(
  const Dense& t_train,
  const std::vector<int>& y_train_labels,
  const Dense& t_val,
  const std::vector<int>& classes,
  const std::vector<int>& components,
  int top_m_in,
  int knn_k_in,
  double tau_in,
  double alpha_in,
  int gpu_device
) {
  if (t_train.cols != t_val.cols) throw std::invalid_argument("CUDA PLS-cKNN train/test score dimensions do not match.");
  if (t_train.rows != static_cast<int>(y_train_labels.size())) throw std::invalid_argument("CUDA PLS-cKNN label length mismatch.");
  if (components.empty()) throw std::invalid_argument("CUDA PLS-cKNN requires at least one component.");
  const int max_component = *std::max_element(components.begin(), components.end());
  if (max_component < 1 || max_component > t_train.cols || max_component > t_val.cols) {
    throw std::invalid_argument("CUDA PLS-cKNN component count is out of range.");
  }
  const int n_classes = static_cast<int>(classes.size());
  const int top_m = std::max(1, std::min(top_m_in, n_classes));
  const int knn_k = std::max(1, std::min(knn_k_in, 32));
  const double tau = (std::isfinite(tau_in) && tau_in > 0.0) ? tau_in : 0.2;
  const double alpha = std::isfinite(alpha_in) ? alpha_in : 0.5;
  const std::vector<int> y_codes = labels_to_codes(y_train_labels, classes);

  const Dense train_norm2 = cumulative_norm2(t_train);
  const Dense test_norm2 = cumulative_norm2(t_val);

  std::vector<std::vector<int>> class_rows(static_cast<std::size_t>(n_classes));
  for (int i = 0; i < static_cast<int>(y_codes.size()); ++i) {
    const int cls = y_codes[static_cast<std::size_t>(i)];
    if (cls >= 1 && cls <= n_classes) class_rows[static_cast<std::size_t>(cls - 1)].push_back(i);
  }
  std::vector<int> class_offsets(static_cast<std::size_t>(n_classes + 1), 0);
  std::vector<int> class_indices;
  class_indices.reserve(y_codes.size());
  for (int cls = 0; cls < n_classes; ++cls) {
    class_offsets[static_cast<std::size_t>(cls)] = static_cast<int>(class_indices.size());
    class_indices.insert(class_indices.end(), class_rows[static_cast<std::size_t>(cls)].begin(), class_rows[static_cast<std::size_t>(cls)].end());
  }
  class_offsets[static_cast<std::size_t>(n_classes)] = static_cast<int>(class_indices.size());

  std::vector<int> ncomp_values = components;

  const std::size_t grid_cols = static_cast<std::size_t>(ncomp_values.size()) * static_cast<std::size_t>(top_m);
  std::vector<int> candidates(grid_cols * static_cast<std::size_t>(t_val.rows), 0);
  std::vector<double> candidate_base(grid_cols * static_cast<std::size_t>(t_val.rows), -std::numeric_limits<double>::infinity());
  std::vector<double> row_scores(static_cast<std::size_t>(top_m), -std::numeric_limits<double>::infinity());
  std::vector<int> row_index(static_cast<std::size_t>(top_m), -1);
  for (std::size_t s = 0; s < ncomp_values.size(); ++s) {
    const int a = ncomp_values[s];
    const Dense cent = candidate_prefix_centroids(t_train, train_norm2, y_codes, n_classes, a);
    for (int i = 0; i < t_val.rows; ++i) {
      std::fill(row_scores.begin(), row_scores.end(), -std::numeric_limits<double>::infinity());
      std::fill(row_index.begin(), row_index.end(), -1);
      const double test_norm = prefix_row_norm(test_norm2, i, a);
      for (int cls = 0; cls < n_classes; ++cls) {
        double base = 0.0;
        if (test_norm > 0.0) {
          for (int d = 0; d < a; ++d) base += t_val(i, d) * cent(cls, d);
          base /= test_norm;
        }
        candidate_insert_top(row_scores, row_index, base, cls + 1);
      }
      for (int slot = 0; slot < top_m; ++slot) {
        const std::size_t col = s * static_cast<std::size_t>(top_m) + static_cast<std::size_t>(slot);
        const std::size_t offset = col * static_cast<std::size_t>(t_val.rows) + static_cast<std::size_t>(i);
        candidates[offset] = row_index[static_cast<std::size_t>(slot)];
        candidate_base[offset] = row_scores[static_cast<std::size_t>(slot)];
      }
    }
  }

  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice(predict_pls_cknn_prefix_cuda)");
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate(predict_pls_cknn_prefix_cuda)");

  const std::size_t n_test_values = t_val.data.size();
  const std::size_t n_train_values = t_train.data.size();
  const std::size_t n_norm_test = test_norm2.data.size();
  const std::size_t n_norm_train = train_norm2.data.size();
  const std::size_t n_grid = candidate_base.size();
  DeviceBuffer d_test(n_test_values);
  DeviceBuffer d_train(n_train_values);
  DeviceBuffer d_test_norm(n_norm_test);
  DeviceBuffer d_train_norm(n_norm_train);
  DeviceIntBuffer d_ncomp(ncomp_values.size());
  DeviceIntBuffer d_offsets(class_offsets.size());
  DeviceIntBuffer d_indices(class_indices.size());
  DeviceIntBuffer d_candidates(candidates.size());
  DeviceBuffer d_base(n_grid);
  DeviceBuffer d_bias(static_cast<std::size_t>(n_classes));
  DeviceBuffer d_scores(n_grid);
  std::vector<double> bias(static_cast<std::size_t>(n_classes), 0.0);

  check_cuda(cudaMemcpyAsync(d_test.data(), t_val.data.data(), n_test_values * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync cKNN Ttest");
  check_cuda(cudaMemcpyAsync(d_train.data(), t_train.data.data(), n_train_values * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync cKNN Ttrain");
  check_cuda(cudaMemcpyAsync(d_test_norm.data(), test_norm2.data.data(), n_norm_test * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync cKNN test_norm2");
  check_cuda(cudaMemcpyAsync(d_train_norm.data(), train_norm2.data.data(), n_norm_train * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync cKNN train_norm2");
  check_cuda(cudaMemcpyAsync(d_ncomp.data(), ncomp_values.data(), ncomp_values.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync cKNN ncomp");
  check_cuda(cudaMemcpyAsync(d_offsets.data(), class_offsets.data(), class_offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync cKNN offsets");
  check_cuda(cudaMemcpyAsync(d_indices.data(), class_indices.data(), class_indices.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync cKNN indices");
  check_cuda(cudaMemcpyAsync(d_candidates.data(), candidates.data(), candidates.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync cKNN candidates");
  check_cuda(cudaMemcpyAsync(d_base.data(), candidate_base.data(), n_grid * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync cKNN base");
  check_cuda(cudaMemcpyAsync(d_bias.data(), bias.data(), bias.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync cKNN bias");

  kodama_cuda_candidate_knn_scores_prefix(
    d_test.data(),
    d_train.data(),
    d_test_norm.data(),
    d_train_norm.data(),
    d_ncomp.data(),
    d_offsets.data(),
    d_indices.data(),
    d_candidates.data(),
    d_base.data(),
    d_bias.data(),
    t_val.rows,
    t_train.rows,
    t_val.cols,
    n_classes,
    static_cast<int>(ncomp_values.size()),
    top_m,
    knn_k,
    tau,
    alpha,
    d_scores.data(),
    stream
  );
  check_cuda(cudaGetLastError(), "kodama_cuda_candidate_knn_scores_prefix");
  std::vector<double> scores(n_grid, -std::numeric_limits<double>::infinity());
  check_cuda(cudaMemcpyAsync(scores.data(), d_scores.data(), n_grid * sizeof(double), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync cKNN scores");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize cKNN scores");
  cudaStreamDestroy(stream);

  std::vector<std::vector<int>> pred_by_comp(
    static_cast<std::size_t>(max_component),
    std::vector<int>(static_cast<std::size_t>(t_val.rows), classes.front())
  );
  for (std::size_t s = 0; s < ncomp_values.size(); ++s) {
    const int a = ncomp_values[s];
    for (int i = 0; i < t_val.rows; ++i) {
      int best_code = 1;
      double best_score = -std::numeric_limits<double>::infinity();
      for (int slot = 0; slot < top_m; ++slot) {
        const std::size_t col = s * static_cast<std::size_t>(top_m) + static_cast<std::size_t>(slot);
        const std::size_t offset = col * static_cast<std::size_t>(t_val.rows) + static_cast<std::size_t>(i);
        if (scores[offset] > best_score) {
          best_score = scores[offset];
          best_code = candidates[offset];
        }
      }
      const int best_pos = std::max(1, best_code) - 1;
      pred_by_comp[static_cast<std::size_t>(a - 1)][static_cast<std::size_t>(i)] =
        classes[static_cast<std::size_t>(std::min(best_pos, n_classes - 1))];
    }
  }
  return pred_by_comp;
}
#endif

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
  if (options.fixed_components < 0) throw std::invalid_argument("PLSOptions::fixed_components must be non-negative.");
  if (options.fixed_components > options.max_components) throw std::invalid_argument("PLSOptions::fixed_components cannot exceed max_components.");
  if (options.cknn_k < 1) throw std::invalid_argument("PLSOptions::cknn_k must be positive.");
  if (options.cknn_top_m < 1) throw std::invalid_argument("PLSOptions::cknn_top_m must be positive.");

  detail::Timer timer;
  PLSCVResult result;
  result.true_labels = labels;
  const bool use_fold_cache = !options.cv.stratified;
  const PLSFoldXCache* fold_cache = use_fold_cache ?
    &get_pls_fold_x_cache(x, labels, constrain, options) : nullptr;
  result.fold_assignments = fold_cache != nullptr ?
    fold_cache->fold_assignments : detail::make_folds(labels, constrain, options.cv);
  result.accuracy_by_components.assign(static_cast<std::size_t>(options.max_components), 0.0);
  std::vector<std::vector<int>> pred_by_comp(
    static_cast<std::size_t>(options.max_components),
    std::vector<int>(labels.size(), labels.empty() ? 0 : labels.front())
  );
  const std::vector<int> classes = detail::unique_labels(labels);
  const std::vector<int> fold_ids = fold_cache != nullptr ?
    fold_cache->fold_ids : detail::sorted_unique_folds(result.fold_assignments);
  int evaluated_component = options.fixed_components > 0 ? options.fixed_components : options.max_components;

  for (std::size_t fold_pos = 0; fold_pos < fold_ids.size(); ++fold_pos) {
    const int fold = fold_ids[fold_pos];
    std::vector<int> validation_storage;
    std::vector<int> train_storage;
    Dense x_train_storage;
    Dense x_val_storage;
    const std::vector<int>* validation = nullptr;
    const std::vector<int>* train = nullptr;
    const Dense* x_train = nullptr;
    const Dense* x_val = nullptr;
    if (fold_cache != nullptr) {
      const PLSFoldData& fold_data = fold_cache->folds_data[fold_pos];
      validation = &fold_data.validation;
      train = &fold_data.train;
      x_train = &fold_data.x_train;
      x_val = &fold_data.x_val;
    } else {
      validation_storage = detail::indices_where_fold(result.fold_assignments, fold, true);
      train_storage = detail::indices_where_fold(result.fold_assignments, fold, false);
      std::vector<double> x_mean;
      std::vector<double> x_scale;
      train_center_scale(x, train_storage, options.center, options.scale, x_mean, x_scale);
      x_train_storage = subset_scale(x, train_storage, x_mean, x_scale);
      x_val_storage = subset_scale(x, validation_storage, x_mean, x_scale);
      validation = &validation_storage;
      train = &train_storage;
      x_train = &x_train_storage;
      x_val = &x_val_storage;
    }
    std::vector<double> y_mean;
    Dense y_train = one_hot_centered(labels, *train, classes, y_mean);
    std::vector<int> y_train_labels(train->size(), 0);
    for (std::size_t i = 0; i < train->size(); ++i) y_train_labels[i] = labels[static_cast<std::size_t>((*train)[i])];
    PLSFit fit = fit_pls_components(*x_train, y_train, options.max_components);
    const std::vector<int> eval_components = components_to_evaluate(options, fit.weights.cols);
    evaluated_component = std::min(evaluated_component, eval_components.front());

    if (mode == PLSMode::PLS_CKNN) {
      Dense t_val_full = transform_pls_scores(*x_val, fit, fit.weights.cols);
      std::vector<std::vector<int>> fold_pred_by_comp = predict_pls_cknn_prefix(
        fit.train_scores,
        y_train_labels,
        t_val_full,
        classes,
        eval_components,
        options.cknn_top_m,
        options.cknn_k,
        options.cknn_tau,
        options.cknn_alpha,
        options.n_threads
      );
      for (int a : eval_components) {
        const std::vector<int>& fold_pred = fold_pred_by_comp[static_cast<std::size_t>(a - 1)];
        for (std::size_t i = 0; i < validation->size(); ++i) {
          pred_by_comp[static_cast<std::size_t>(a - 1)][static_cast<std::size_t>((*validation)[i])] = fold_pred[i];
        }
      }
    } else {
      Dense t_val_full = transform_pls_scores(*x_val, fit, fit.weights.cols);
      for (int a : eval_components) {
        std::vector<int> fold_pred = mode == PLSMode::PLS_LDA ?
          predict_pls_lda(fit.train_scores, y_train_labels, t_val_full, classes, a) :
          predict_pls_da_labels(fit.train_scores, y_train_labels, t_val_full, classes, y_mean, a);
        for (std::size_t i = 0; i < validation->size(); ++i) {
          pred_by_comp[static_cast<std::size_t>(a - 1)][static_cast<std::size_t>((*validation)[i])] = fold_pred[i];
        }
      }
    }
  }

  const int best_comp = std::min(evaluated_component, static_cast<int>(pred_by_comp.size()));
  result.accuracy_by_components[static_cast<std::size_t>(best_comp - 1)] =
    detail::accuracy(labels, pred_by_comp[static_cast<std::size_t>(best_comp - 1)]);
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
  result.parameters.fixed_components = options.fixed_components;
  result.parameters.cknn_k = options.cknn_k;
  result.parameters.cknn_top_m = options.cknn_top_m;
  result.parameters.cknn_tau = options.cknn_tau;
  result.parameters.cknn_alpha = options.cknn_alpha;
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
  if (options.fixed_components < 0) throw std::invalid_argument("PLSOptions::fixed_components must be non-negative.");
  if (options.fixed_components > options.max_components) throw std::invalid_argument("PLSOptions::fixed_components cannot exceed max_components.");
  if (options.cknn_k < 1) throw std::invalid_argument("PLSOptions::cknn_k must be positive.");
  if (options.cknn_top_m < 1) throw std::invalid_argument("PLSOptions::cknn_top_m must be positive.");

  detail::Timer timer;
  check_cuda(cudaSetDevice(options.gpu_device), "cudaSetDevice(run_plscv_cuda)");
  PLSCVResult result;
  result.true_labels = labels;
  const bool use_fold_cache = !options.cv.stratified;
  const PLSFoldXCache* fold_cache = use_fold_cache ?
    &get_pls_fold_x_cache(x, labels, constrain, options) : nullptr;
  result.fold_assignments = fold_cache != nullptr ?
    fold_cache->fold_assignments : detail::make_folds(labels, constrain, options.cv);
  result.accuracy_by_components.assign(static_cast<std::size_t>(options.max_components), 0.0);
  std::vector<std::vector<int>> pred_by_comp(
    static_cast<std::size_t>(options.max_components),
    std::vector<int>(labels.size(), labels.empty() ? 0 : labels.front())
  );
  const std::vector<int> classes = detail::unique_labels(labels);
  const std::vector<int> fold_ids = fold_cache != nullptr ?
    fold_cache->fold_ids : detail::sorted_unique_folds(result.fold_assignments);
  int evaluated_component = options.fixed_components > 0 ? options.fixed_components : options.max_components;

  for (std::size_t fold_pos = 0; fold_pos < fold_ids.size(); ++fold_pos) {
    const int fold = fold_ids[fold_pos];
    std::vector<int> validation_storage;
    std::vector<int> train_storage;
    Dense x_train_storage;
    Dense x_val_storage;
    const std::vector<int>* validation = nullptr;
    const std::vector<int>* train = nullptr;
    const Dense* x_train = nullptr;
    const Dense* x_val = nullptr;
    if (fold_cache != nullptr) {
      const PLSFoldData& fold_data = fold_cache->folds_data[fold_pos];
      validation = &fold_data.validation;
      train = &fold_data.train;
      x_train = &fold_data.x_train;
      x_val = &fold_data.x_val;
    } else {
      validation_storage = detail::indices_where_fold(result.fold_assignments, fold, true);
      train_storage = detail::indices_where_fold(result.fold_assignments, fold, false);
      std::vector<double> x_mean;
      std::vector<double> x_scale;
      train_center_scale(x, train_storage, options.center, options.scale, x_mean, x_scale);
      x_train_storage = subset_scale(x, train_storage, x_mean, x_scale);
      x_val_storage = subset_scale(x, validation_storage, x_mean, x_scale);
      validation = &validation_storage;
      train = &train_storage;
      x_train = &x_train_storage;
      x_val = &x_val_storage;
    }
    std::vector<int> y_train_labels(train->size(), 0);
    for (std::size_t i = 0; i < train->size(); ++i) y_train_labels[i] = labels[static_cast<std::size_t>((*train)[i])];
    std::vector<double> y_mean;
    const std::vector<double> y_train_colmajor = one_hot_centered_colmajor(labels, *train, classes, y_mean);
    PLSFit fit = fit_pls_components_cuda_colmajor_y(
      *x_train,
      y_train_colmajor.data(),
      static_cast<int>(classes.size()),
      options.max_components,
      options.gpu_device
    );
    const std::vector<int> eval_components = components_to_evaluate(options, fit.weights.cols);
    evaluated_component = std::min(evaluated_component, eval_components.front());

    if (mode == PLSMode::PLS_CKNN) {
      Dense t_val_full = transform_pls_scores_cuda(*x_val, fit, fit.weights.cols, options.gpu_device);
      std::vector<std::vector<int>> fold_pred_by_comp = predict_pls_cknn_prefix_cuda(
        fit.train_scores,
        y_train_labels,
        t_val_full,
        classes,
        eval_components,
        options.cknn_top_m,
        options.cknn_k,
        options.cknn_tau,
        options.cknn_alpha,
        options.gpu_device
      );
      for (int a : eval_components) {
        const std::vector<int>& fold_pred = fold_pred_by_comp[static_cast<std::size_t>(a - 1)];
        for (std::size_t i = 0; i < validation->size(); ++i) {
          pred_by_comp[static_cast<std::size_t>(a - 1)][static_cast<std::size_t>((*validation)[i])] = fold_pred[i];
        }
      }
    } else {
      Dense t_val_full = transform_pls_scores_cuda(*x_val, fit, fit.weights.cols, options.gpu_device);
      for (int a : eval_components) {
        std::vector<int> fold_pred;
        if (mode == PLSMode::PLS_LDA) {
          fold_pred = train_predict_pls_lda_cuda(fit.train_scores, y_train_labels, t_val_full, classes, options.gpu_device);
        } else {
          fold_pred = predict_pls_da_cuda(fit.train_scores, y_train_labels, t_val_full, classes, y_mean, options.gpu_device);
        }
        for (std::size_t i = 0; i < validation->size(); ++i) {
          pred_by_comp[static_cast<std::size_t>(a - 1)][static_cast<std::size_t>((*validation)[i])] = fold_pred[i];
        }
      }
    }
  }

  const int best_comp = std::min(evaluated_component, static_cast<int>(pred_by_comp.size()));
  result.accuracy_by_components[static_cast<std::size_t>(best_comp - 1)] =
    detail::accuracy(labels, pred_by_comp[static_cast<std::size_t>(best_comp - 1)]);
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
  result.parameters.fixed_components = options.fixed_components;
  result.parameters.cknn_k = options.cknn_k;
  result.parameters.cknn_top_m = options.cknn_top_m;
  result.parameters.cknn_tau = options.cknn_tau;
  result.parameters.cknn_alpha = options.cknn_alpha;
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

PLSCVResult PLSCKNNCV(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  if (options.backend == Backend::CUDA) return PLSCKNNCV_CUDA(x, labels, constrain, options);
  return PLSCKNNCV_CPU(x, labels, constrain, options);
}

PLSCVResult PLSDACV_CPU(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  return run_plscv_cpu(x, labels, constrain, options, PLSMode::PLS_DA);
}

PLSCVResult PLSLDACV_CPU(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  return run_plscv_cpu(x, labels, constrain, options, PLSMode::PLS_LDA);
}

PLSCVResult PLSCKNNCV_CPU(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  return run_plscv_cpu(x, labels, constrain, options, PLSMode::PLS_CKNN);
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

PLSCVResult PLSCKNNCV_CUDA(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
#if defined(KODAMA_ENABLE_CUDA)
  PLSOptions cuda_options = options;
  cuda_options.backend = Backend::CUDA;
  return run_plscv_cuda(x, labels, constrain, cuda_options, PLSMode::PLS_CKNN);
#else
  (void)x;
  (void)labels;
  (void)constrain;
  (void)options;
  throw std::runtime_error("PLSCKNNCV_CUDA requires a CUDA/cuBLAS build.");
#endif
}

std::vector<int> PLSLDAPredict_CPU(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options
) {
  if (train.data == nullptr || test.data == nullptr) throw std::invalid_argument("PLSLDAPredict input matrix pointer is null.");
  if (train.rows != labels.size()) throw std::invalid_argument("PLSLDAPredict labels size must match training rows.");
  if (train.cols != test.cols) throw std::invalid_argument("PLSLDAPredict train/test column mismatch.");
  if (train.rows == 0 || test.rows == 0 || train.cols == 0) return {};
  const std::vector<int> classes = detail::unique_labels(labels);
  if (classes.empty()) return {};
  if (classes.size() == 1) return std::vector<int>(test.rows, classes.front());

  std::vector<int> train_rows(train.rows);
  std::iota(train_rows.begin(), train_rows.end(), 0);
  std::vector<int> test_rows(test.rows);
  std::iota(test_rows.begin(), test_rows.end(), 0);
  std::vector<double> mean;
  std::vector<double> scale;
  train_center_scale(train, train_rows, options.center, options.scale, mean, scale);
  Dense x_train = subset_scale(train, train_rows, mean, scale);
  Dense x_test = subset_scale(test, test_rows, mean, scale);
  std::vector<double> y_mean;
  Dense y_train = one_hot_centered(labels, train_rows, classes, y_mean);
  const int requested = options.fixed_components > 0 ? options.fixed_components : options.max_components;
  const int ncomp = std::max(1, std::min({requested, x_train.cols, std::max(1, x_train.rows - 1)}));
  PLSFit fit = fit_pls_components(x_train, y_train, ncomp);
  fit.train_scores = transform_pls_scores(x_train, fit, fit.weights.cols);
  Dense t_test = transform_pls_scores(x_test, fit, fit.weights.cols);
  return predict_pls_lda(fit.train_scores, labels, t_test, classes, fit.weights.cols);
}

std::vector<int> PLSLDAPredict_CUDA(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  if (train.data == nullptr || test.data == nullptr) throw std::invalid_argument("PLSLDAPredict input matrix pointer is null.");
  if (train.rows != labels.size()) throw std::invalid_argument("PLSLDAPredict labels size must match training rows.");
  if (train.cols != test.cols) throw std::invalid_argument("PLSLDAPredict train/test column mismatch.");
  if (train.rows == 0 || test.rows == 0 || train.cols == 0) return {};
  const std::vector<int> classes = detail::unique_labels(labels);
  if (classes.empty()) return {};
  if (classes.size() == 1) return std::vector<int>(test.rows, classes.front());

  std::vector<int> train_rows(train.rows);
  std::iota(train_rows.begin(), train_rows.end(), 0);
  std::vector<int> test_rows(test.rows);
  std::iota(test_rows.begin(), test_rows.end(), 0);
  std::vector<double> mean;
  std::vector<double> scale;
  train_center_scale(train, train_rows, options.center, options.scale, mean, scale);
  Dense x_train = subset_scale(train, train_rows, mean, scale);
  Dense x_test = subset_scale(test, test_rows, mean, scale);
  std::vector<double> y_mean;
  Dense y_train = one_hot_centered(labels, train_rows, classes, y_mean);
  const int requested = options.fixed_components > 0 ? options.fixed_components : options.max_components;
  const int ncomp = std::max(1, std::min({requested, x_train.cols, std::max(1, x_train.rows - 1)}));
  PLSFit fit = fit_pls_components_cuda(x_train, y_train, ncomp, options.gpu_device);
  Dense t_test = transform_pls_scores_cuda(x_test, fit, fit.weights.cols, options.gpu_device);
  CudaLDAModel lda = train_pls_lda_cuda(fit.train_scores, labels, classes, options.gpu_device);
  return predict_pls_lda_cuda(t_test, lda, classes, options.gpu_device);
#else
  (void)train;
  (void)labels;
  (void)test;
  (void)options;
  throw std::runtime_error("PLSLDAPredict_CUDA requires a CUDA/cuBLAS build.");
#endif
}

std::vector<int> PLSLDAPredict(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options
) {
  if (options.backend == Backend::CUDA) return PLSLDAPredict_CUDA(train, labels, test, options);
  return PLSLDAPredict_CPU(train, labels, test, options);
}

}  // namespace kodama
