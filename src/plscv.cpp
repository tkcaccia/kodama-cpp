#include "common.hpp"
#include "metal_backend.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(KODAMA_ENABLE_CUDA)
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

extern "C" void kodama_cuda_lda_label_sums_row(const double*, const int*, int, int, int, double*, cudaStream_t);
extern "C" void kodama_cuda_lda_means_row(double*, const double*, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_pooled_col(double*, const double*, const double*, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_copy_cov(const double*, double*, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_add_ridge(double*, int, double, double*, cudaStream_t);
extern "C" void kodama_cuda_lda_means_to_rhs(const double*, double*, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_finalize_linear_row(const double*, const double*, const double*, double*, double*, int, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_score_argmax_row(const double*, const double*, const double*, int*, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_label_sums_row_float(const float*, const int*, int, int, int, float*, cudaStream_t);
extern "C" void kodama_cuda_lda_means_row_float(float*, const float*, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_pooled_col_float(float*, const float*, const float*, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_copy_cov_float(const float*, float*, int, int, cudaStream_t);
extern "C" void kodama_cuda_symmetrize_lower_float(float*, int, cudaStream_t);
extern "C" void kodama_cuda_lda_add_ridge_float(float*, int, float, float*, cudaStream_t);
extern "C" void kodama_cuda_lda_means_to_rhs_float(const float*, float*, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_finalize_linear_row_float(const float*, const float*, const float*, float*, float*, int, int, int, int, cudaStream_t);
extern "C" void kodama_cuda_lda_score_argmax_row_float(const float*, const float*, const float*, int*, int, int, int, cudaStream_t);
extern "C" bool kodama_fastpls_simpls_fit_cuda_float(const float*, int, int, const float*, int, int, int, float*, float*);
extern "C" bool kodama_fastpls_simpls_fit_cuda_float_crossprod(const float*, int, int, const float*, int, int, int, float*, float*);
extern "C" bool kodama_fastpls_simpls_fit_cuda_float_labels(const float*, int, int, const int*, const float*, int, int, int, float*, float*);
extern "C" bool kodama_fastpls_simpls_fit_cuda_float_labels_gram(const float*, int, int, const float*, const int*, const float*, int, int, int, float*, float*);
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
  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void reset(std::size_t n) {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = n;
    if (n > 0) check_cuda(cudaMalloc(&ptr_, n * sizeof(double)), "cudaMalloc");
  }

  void ensure(std::size_t n) {
    if (n > size_) reset(n);
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
  DeviceIntBuffer(DeviceIntBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  DeviceIntBuffer& operator=(DeviceIntBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void reset(std::size_t n) {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = n;
    if (n > 0) check_cuda(cudaMalloc(&ptr_, n * sizeof(int)), "cudaMalloc(int)");
  }

  void ensure(std::size_t n) {
    if (n > size_) reset(n);
  }

  int* data() { return ptr_; }
  const int* data() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  int* ptr_ = nullptr;
  std::size_t size_ = 0;
};

class DeviceFloatBuffer {
 public:
  DeviceFloatBuffer() = default;
  explicit DeviceFloatBuffer(std::size_t n) { reset(n); }

  ~DeviceFloatBuffer() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  DeviceFloatBuffer(const DeviceFloatBuffer&) = delete;
  DeviceFloatBuffer& operator=(const DeviceFloatBuffer&) = delete;
  DeviceFloatBuffer(DeviceFloatBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  DeviceFloatBuffer& operator=(DeviceFloatBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void reset(std::size_t n) {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = n;
    if (n > 0) check_cuda(cudaMalloc(&ptr_, n * sizeof(float)), "cudaMalloc(float)");
  }

  void ensure(std::size_t n) {
    if (n > size_) reset(n);
  }

  float* data() { return ptr_; }
  const float* data() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  float* ptr_ = nullptr;
  std::size_t size_ = 0;
};

struct CudaPLSLDAFloatWorkspace {
  explicit CudaPLSLDAFloatWorkspace(int dev) : device(dev) {}

  int device = 0;
  DeviceFloatBuffer x_train;
  DeviceFloatBuffer x_val;
  DeviceFloatBuffer weights;
  DeviceFloatBuffer train_scores;
  DeviceFloatBuffer val_scores;
  DeviceIntBuffer labels;
  DeviceFloatBuffer counts;
  DeviceFloatBuffer means;
  DeviceFloatBuffer pooled;
  DeviceFloatBuffer cov;
  DeviceFloatBuffer rhs;
  DeviceFloatBuffer linear;
  DeviceFloatBuffer constants;
  DeviceFloatBuffer lambda;
  DeviceIntBuffer info;
  DeviceIntBuffer pred;
  DeviceFloatBuffer solver_work;
  std::vector<int> encoded;
  std::vector<float> class_counts;
  std::vector<float> weights_prefix;
  std::vector<int> pred_codes;
};

CudaPLSLDAFloatWorkspace& cuda_pls_lda_float_workspace(int device) {
  thread_local std::unique_ptr<CudaPLSLDAFloatWorkspace> workspace;
  if (!workspace || workspace->device != device) {
    workspace = std::make_unique<CudaPLSLDAFloatWorkspace>(device);
  }
  return *workspace;
}

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

struct DenseF {
  int rows = 0;
  int cols = 0;
  std::vector<float> data;

  DenseF() = default;
  DenseF(int r, int c) : rows(r), cols(c), data(static_cast<std::size_t>(r * c), 0.0f) {}

  float& operator()(int i, int j) { return data[static_cast<std::size_t>(i * cols + j)]; }
  float operator()(int i, int j) const { return data[static_cast<std::size_t>(i * cols + j)]; }
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

float matrix_value_float(MatrixView x, std::size_t i, std::size_t j) {
  return x.value_float(i, j);
}

DenseF subset_scale_float(
  MatrixView x,
  const std::vector<int>& rows,
  const std::vector<float>& mean,
  const std::vector<float>& scale
) {
  DenseF out(static_cast<int>(rows.size()), static_cast<int>(x.cols));
  for (int i = 0; i < out.rows; ++i) {
    const int src = rows[static_cast<std::size_t>(i)];
    for (int j = 0; j < out.cols; ++j) {
      out(i, j) = (matrix_value_float(x, static_cast<std::size_t>(src), static_cast<std::size_t>(j)) - mean[static_cast<std::size_t>(j)]) /
                  scale[static_cast<std::size_t>(j)];
    }
  }
  return out;
}

std::vector<float> densef_to_colmajor(const DenseF& x) {
  std::vector<float> out(static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols));
  for (int j = 0; j < x.cols; ++j) {
    for (int i = 0; i < x.rows; ++i) {
      out[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(x.rows)] = x(i, j);
    }
  }
  return out;
}

int sorted_class_index(const std::vector<int>& classes, int label) {
  const auto it = std::lower_bound(classes.begin(), classes.end(), label);
  if (it == classes.end() || *it != label) {
    throw std::invalid_argument("label is not present in the active class set.");
  }
  return static_cast<int>(it - classes.begin());
}

template <typename Count>
void encode_labels_from_sorted_classes(
  const std::vector<int>& labels,
  const std::vector<int>& classes,
  std::vector<int>& encoded,
  std::vector<Count>& counts,
  int code_offset
) {
  encoded.resize(labels.size());
  counts.assign(classes.size(), Count{});
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const int cls = sorted_class_index(classes, labels[i]);
    encoded[i] = cls + code_offset;
    counts[static_cast<std::size_t>(cls)] += Count{1};
  }
}

#if defined(KODAMA_ENABLE_CUDA)
std::vector<float> densef_gram_colmajor_cuda(
  const std::vector<float>& x_colmajor,
  int rows,
  int cols,
  int gpu_device
) {
  if (rows < 1 || cols < 1) return {};
  if (x_colmajor.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) {
    throw std::invalid_argument("densef_gram_colmajor_cuda input size mismatch.");
  }

  CudaBlasContext ctx(gpu_device);
  DeviceFloatBuffer dx(x_colmajor.size());
  DeviceFloatBuffer dg(static_cast<std::size_t>(cols) * static_cast<std::size_t>(cols));
  check_cuda(cudaMemcpy(dx.data(), x_colmajor.data(), x_colmajor.size() * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy raw float32 X for fold Gram");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(
    cublasSgemm(
      ctx.handle(),
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      cols,
      cols,
      rows,
      &alpha,
      dx.data(),
      rows,
      dx.data(),
      rows,
      &beta,
      dg.data(),
      cols
    ),
    "cublasSgemm raw float32 fold Gram"
  );

  std::vector<float> gram(static_cast<std::size_t>(cols) * static_cast<std::size_t>(cols));
  check_cuda(cudaMemcpy(gram.data(), dg.data(), gram.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy raw float32 fold Gram to host");
  return gram;
}
#endif

void train_center_scale(
  MatrixView x,
  const std::vector<int>& rows,
  bool center,
  bool scale_columns,
  std::vector<double>& mean,
  std::vector<double>& scale
);

void train_center_scale_float(
  MatrixView x,
  const std::vector<int>& rows,
  bool center,
  bool scale_columns,
  std::vector<float>& mean,
  std::vector<float>& scale
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

struct PLSFoldDataF {
  int fold = 0;
  std::vector<int> train;
  std::vector<int> validation;
  std::vector<float> mean;
  std::vector<float> scale;
  DenseF x_train;
  DenseF x_val;
  std::vector<float> x_train_colmajor;
  std::vector<float> x_train_gram_colmajor;
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

struct PLSFoldXCacheF {
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
  bool train_colmajor = false;
  std::vector<int> fold_assignments;
  std::vector<int> fold_ids;
  std::vector<PLSFoldDataF> folds_data;
};

std::size_t hash_int_vector(const std::vector<int>& values) {
  std::size_t h = 1469598103934665603ull;
  for (int value : values) {
    h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(value));
    h *= 1099511628211ull;
  }
  return h;
}

bool env_flag_enabled(const char* key) {
  const char* raw = std::getenv(key);
  if (raw == nullptr) return false;
  const std::string value(raw);
  return value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON";
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

bool cache_matches_float(
  const PLSFoldXCacheF& cache,
  MatrixView x,
  const std::vector<int>& constrain,
  const PLSOptions& options,
  bool require_train_colmajor
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
         cache.constrain_hash == hash_int_vector(constrain) &&
         (!require_train_colmajor || cache.train_colmajor);
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

const PLSFoldXCacheF& get_pls_fold_x_cache_float(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options,
  bool cache_train_colmajor
) {
  thread_local PLSFoldXCacheF cache;
  if (cache_matches_float(cache, x, constrain, options, cache_train_colmajor)) return cache;

  cache = PLSFoldXCacheF{};
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
  cache.train_colmajor = cache_train_colmajor;
  cache.fold_assignments = detail::make_folds(labels, constrain, options.cv);
  cache.fold_ids = detail::sorted_unique_folds(cache.fold_assignments);
  cache.folds_data.reserve(cache.fold_ids.size());

  for (int fold : cache.fold_ids) {
    PLSFoldDataF fold_data;
    fold_data.fold = fold;
    fold_data.validation = detail::indices_where_fold(cache.fold_assignments, fold, true);
    fold_data.train = detail::indices_where_fold(cache.fold_assignments, fold, false);
    train_center_scale_float(x, fold_data.train, options.center, options.scale, fold_data.mean, fold_data.scale);
    fold_data.x_train = subset_scale_float(x, fold_data.train, fold_data.mean, fold_data.scale);
    fold_data.x_val = subset_scale_float(x, fold_data.validation, fold_data.mean, fold_data.scale);
    if (cache_train_colmajor) {
      fold_data.x_train_colmajor = densef_to_colmajor(fold_data.x_train);
#if defined(KODAMA_ENABLE_CUDA)
      fold_data.x_train_gram_colmajor = densef_gram_colmajor_cuda(
        fold_data.x_train_colmajor,
        fold_data.x_train.rows,
        fold_data.x_train.cols,
        options.gpu_device
      );
#endif
    }
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

void train_center_scale_float(
  MatrixView x,
  const std::vector<int>& rows,
  bool center,
  bool scale_columns,
  std::vector<float>& mean,
  std::vector<float>& scale
) {
  mean.assign(x.cols, 0.0f);
  scale.assign(x.cols, 1.0f);
  if (center) {
    for (int row : rows) {
      for (std::size_t j = 0; j < x.cols; ++j) mean[j] += matrix_value_float(x, static_cast<std::size_t>(row), j);
    }
    const float inv_n = rows.empty() ? 0.0f : 1.0f / static_cast<float>(rows.size());
    for (float& v : mean) v *= inv_n;
  }
  if (scale_columns) {
    for (int row : rows) {
      for (std::size_t j = 0; j < x.cols; ++j) {
        const float d = matrix_value_float(x, static_cast<std::size_t>(row), j) - mean[j];
        scale[j] += d * d;
      }
    }
    for (std::size_t j = 0; j < x.cols; ++j) {
      const float s = std::sqrt(scale[j] / std::max(1.0f, static_cast<float>(rows.size() - 1)));
      scale[j] = s > 0.0f && std::isfinite(s) ? s : 1.0f;
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

std::vector<float> one_hot_centered_colmajor_float(
  const std::vector<int>& labels,
  const std::vector<int>& rows,
  const std::vector<int>& classes,
  std::vector<float>& y_mean
) {
  std::map<int, int> class_pos;
  for (int i = 0; i < static_cast<int>(classes.size()); ++i) class_pos[classes[static_cast<std::size_t>(i)]] = i;

  const int n = static_cast<int>(rows.size());
  const int m = static_cast<int>(classes.size());
  y_mean.assign(classes.size(), 0.0f);
  std::vector<int> encoded(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    const int cls = labels[static_cast<std::size_t>(rows[static_cast<std::size_t>(i)])];
    const int j = class_pos[cls];
    encoded[static_cast<std::size_t>(i)] = j;
    y_mean[static_cast<std::size_t>(j)] += 1.0f;
  }
  const float inv_n = n > 0 ? 1.0f / static_cast<float>(n) : 0.0f;
  for (float& v : y_mean) v *= inv_n;

  std::vector<float> y_colmajor(static_cast<std::size_t>(n) * static_cast<std::size_t>(m), 0.0f);
  for (int j = 0; j < m; ++j) {
    float* col = y_colmajor.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
    std::fill(col, col + n, -y_mean[static_cast<std::size_t>(j)]);
  }
  for (int i = 0; i < n; ++i) {
    const int j = encoded[static_cast<std::size_t>(i)];
    y_colmajor[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(n)] += 1.0f;
  }
  return y_colmajor;
}

std::vector<float> centered_label_crossprod_colmajor_float(
  const DenseF& x,
  const std::vector<int>& y_train,
  const std::vector<int>& classes
) {
  if (static_cast<int>(y_train.size()) != x.rows) throw std::invalid_argument("label-aware float SIMPLS label size mismatch.");
  const int n = x.rows;
  const int p = x.cols;
  const int m = static_cast<int>(classes.size());
  std::vector<float> s(static_cast<std::size_t>(p) * static_cast<std::size_t>(m), 0.0f);
  std::vector<float> x_sum(static_cast<std::size_t>(p), 0.0f);
  std::vector<int> encoded;
  std::vector<float> counts;
  encode_labels_from_sorted_classes(y_train, classes, encoded, counts, 0);

  for (int i = 0; i < n; ++i) {
    const int cls = encoded[static_cast<std::size_t>(i)];
    for (int j = 0; j < p; ++j) {
      const float value = x(i, j);
      x_sum[static_cast<std::size_t>(j)] += value;
      s[static_cast<std::size_t>(j) + static_cast<std::size_t>(cls) * static_cast<std::size_t>(p)] += value;
    }
  }

  const float inv_n = n > 0 ? 1.0f / static_cast<float>(n) : 0.0f;
  for (int cls = 0; cls < m; ++cls) {
    const float y_mean = counts[static_cast<std::size_t>(cls)] * inv_n;
    float* col = s.data() + static_cast<std::size_t>(cls) * static_cast<std::size_t>(p);
    for (int j = 0; j < p; ++j) col[j] -= y_mean * x_sum[static_cast<std::size_t>(j)];
  }
  return s;
}

DenseF centered_label_crossprod_float(
  const DenseF& x,
  const std::vector<int>& y_train,
  const std::vector<int>& classes
) {
  const std::vector<float> colmajor = centered_label_crossprod_colmajor_float(x, y_train, classes);
  DenseF out(x.cols, static_cast<int>(classes.size()));
  for (int c = 0; c < out.cols; ++c) {
    for (int j = 0; j < out.rows; ++j) {
      out(j, c) = colmajor[static_cast<std::size_t>(j) + static_cast<std::size_t>(c) * static_cast<std::size_t>(out.rows)];
    }
  }
  return out;
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

int pls_component_limit(int requested, int rows, int cols) {
  return std::max(1, std::min({
    requested,
    cols,
    std::max(1, rows - 1)
  }));
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

DenseF solve_linear_float(DenseF a, DenseF b);

struct PLSFit {
  Dense weights;
  Dense loadings;
  Dense y_weights;
  Dense train_scores;
};

struct PLSFitF {
  DenseF weights;
  DenseF y_weights;
};

float dot_float(const std::vector<float>& a, const std::vector<float>& b) {
  double out = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    out += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(out);
}

float norm2_float(const std::vector<float>& x) {
  return std::sqrt(std::max(0.0f, dot_float(x, x)));
}

std::vector<float> mat_vec_float(const DenseF& a, const std::vector<float>& x) {
  std::vector<float> out(static_cast<std::size_t>(a.rows), 0.0f);
  for (int i = 0; i < a.rows; ++i) {
    double s = 0.0;
    for (int j = 0; j < a.cols; ++j) s += static_cast<double>(a(i, j)) * static_cast<double>(x[static_cast<std::size_t>(j)]);
    out[static_cast<std::size_t>(i)] = static_cast<float>(s);
  }
  return out;
}

std::vector<float> t_mat_vec_float(const DenseF& a, const std::vector<float>& x) {
  std::vector<float> out(static_cast<std::size_t>(a.cols), 0.0f);
  for (int j = 0; j < a.cols; ++j) {
    double s = 0.0;
    for (int i = 0; i < a.rows; ++i) s += static_cast<double>(a(i, j)) * static_cast<double>(x[static_cast<std::size_t>(i)]);
    out[static_cast<std::size_t>(j)] = static_cast<float>(s);
  }
  return out;
}

std::vector<float> dominant_left_singular_vector_float(const DenseF& s) {
  std::vector<float> u(static_cast<std::size_t>(s.cols), 1.0f / std::sqrt(static_cast<float>(std::max(1, s.cols))));
  std::vector<float> v(static_cast<std::size_t>(s.rows), 0.0f);
  for (int iter = 0; iter < 80; ++iter) {
    v = mat_vec_float(s, u);
    float nv = norm2_float(v);
    if (nv <= 1e-6f) break;
    for (float& x : v) x /= nv;
    u = t_mat_vec_float(s, v);
    float nu = norm2_float(u);
    if (nu <= 1e-6f) break;
    for (float& x : u) x /= nu;
  }
  if (norm2_float(v) <= 1e-6f) {
    v.assign(static_cast<std::size_t>(s.rows), 0.0f);
    if (!v.empty()) v[0] = 1.0f;
  }
  return v;
}

PLSFitF fit_pls_components_from_crossprod_float(const DenseF& x, DenseF s, int max_components) {
  const int max_rank = std::min({max_components, x.cols, std::max(1, x.rows - 1)});
  DenseF w(x.cols, max_rank);
  DenseF qmat(s.cols, max_rank);
  DenseF pmat(x.cols, max_rank);
  for (int a = 0; a < max_rank; ++a) {
    DenseF gram(s.rows, s.rows);
    for (int i = 0; i < s.rows; ++i) {
      for (int j = 0; j < s.rows; ++j) {
        double val = 0.0;
        for (int c = 0; c < s.cols; ++c) val += static_cast<double>(s(i, c)) * static_cast<double>(s(j, c));
        gram(i, j) = static_cast<float>(val);
      }
    }
    std::vector<float> wa = dominant_left_singular_vector_float(s);
    for (int prev = 0; prev < a; ++prev) {
      double proj = 0.0;
      for (int j = 0; j < x.cols; ++j) proj += static_cast<double>(wa[static_cast<std::size_t>(j)]) * static_cast<double>(w(j, prev));
      for (int j = 0; j < x.cols; ++j) wa[static_cast<std::size_t>(j)] -= static_cast<float>(proj * w(j, prev));
    }
    float nwa = norm2_float(wa);
    if (nwa <= 1e-6f) {
      std::fill(wa.begin(), wa.end(), 0.0f);
      wa[static_cast<std::size_t>(a % x.cols)] = 1.0f;
    } else {
      for (float& v : wa) v /= nwa;
    }
    for (int iter = 0; iter < 120; ++iter) {
      std::vector<float> next(static_cast<std::size_t>(x.cols), 0.0f);
      for (int i = 0; i < x.cols; ++i) {
        double val = 0.0;
        for (int j = 0; j < x.cols; ++j) val += static_cast<double>(gram(i, j)) * static_cast<double>(wa[static_cast<std::size_t>(j)]);
        next[static_cast<std::size_t>(i)] = static_cast<float>(val);
      }
      for (int prev = 0; prev < a; ++prev) {
        double proj = 0.0;
        for (int j = 0; j < x.cols; ++j) proj += static_cast<double>(next[static_cast<std::size_t>(j)]) * static_cast<double>(w(j, prev));
        for (int j = 0; j < x.cols; ++j) next[static_cast<std::size_t>(j)] -= static_cast<float>(proj * w(j, prev));
      }
      float nn = norm2_float(next);
      if (nn <= 1e-6f) break;
      for (float& v : next) v /= nn;
      wa.swap(next);
    }
    std::vector<float> right = t_mat_vec_float(s, wa);
    const float sigma = std::max(1e-6f, norm2_float(right));
    for (float& v : right) v /= sigma;

    std::vector<float> t(static_cast<std::size_t>(x.rows), 0.0f);
    for (int i = 0; i < x.rows; ++i) {
      double val = 0.0;
      for (int j = 0; j < x.cols; ++j) val += static_cast<double>(x(i, j)) * static_cast<double>(wa[static_cast<std::size_t>(j)]);
      t[static_cast<std::size_t>(i)] = static_cast<float>(val);
    }
    double tnorm2 = 0.0;
    for (float tv : t) tnorm2 += static_cast<double>(tv) * static_cast<double>(tv);
    const double inv_tnorm2 = tnorm2 > 1e-20 ? 1.0 / tnorm2 : 0.0;
    std::vector<float> vvec(static_cast<std::size_t>(x.cols), 0.0f);
    if (inv_tnorm2 > 0.0) {
      for (int j = 0; j < x.cols; ++j) {
        double val = 0.0;
        for (int i = 0; i < x.rows; ++i) val += static_cast<double>(x(i, j)) * static_cast<double>(t[static_cast<std::size_t>(i)]);
        vvec[static_cast<std::size_t>(j)] = static_cast<float>(val * inv_tnorm2);
      }
    } else {
      vvec = wa;
    }
    for (int prev = 0; prev < a; ++prev) {
      double proj = 0.0;
      for (int j = 0; j < x.cols; ++j) proj += static_cast<double>(vvec[static_cast<std::size_t>(j)]) * static_cast<double>(pmat(j, prev));
      for (int j = 0; j < x.cols; ++j) vvec[static_cast<std::size_t>(j)] -= static_cast<float>(proj * pmat(j, prev));
    }
    float nv = norm2_float(vvec);
    if (nv <= 1e-6f) {
      vvec = wa;
      for (int prev = 0; prev < a; ++prev) {
        double proj = 0.0;
        for (int j = 0; j < x.cols; ++j) proj += static_cast<double>(vvec[static_cast<std::size_t>(j)]) * static_cast<double>(pmat(j, prev));
        for (int j = 0; j < x.cols; ++j) vvec[static_cast<std::size_t>(j)] -= static_cast<float>(proj * pmat(j, prev));
      }
      nv = norm2_float(vvec);
    }
    if (nv <= 1e-6f) {
      std::fill(vvec.begin(), vvec.end(), 0.0f);
      vvec[static_cast<std::size_t>(a % x.cols)] = 1.0f;
      for (int prev = 0; prev < a; ++prev) {
        double proj = 0.0;
        for (int j = 0; j < x.cols; ++j) proj += static_cast<double>(vvec[static_cast<std::size_t>(j)]) * static_cast<double>(pmat(j, prev));
        for (int j = 0; j < x.cols; ++j) vvec[static_cast<std::size_t>(j)] -= static_cast<float>(proj * pmat(j, prev));
      }
      nv = std::max(norm2_float(vvec), 1e-6f);
    }
    for (float& vv : vvec) vv /= nv;

    for (int j = 0; j < x.cols; ++j) {
      w(j, a) = wa[static_cast<std::size_t>(j)];
      pmat(j, a) = vvec[static_cast<std::size_t>(j)];
    }
    for (int j = 0; j < s.cols; ++j) qmat(j, a) = right[static_cast<std::size_t>(j)];

    std::vector<float> vs(static_cast<std::size_t>(s.cols), 0.0f);
    for (int c = 0; c < s.cols; ++c) {
      double val = 0.0;
      for (int j = 0; j < s.rows; ++j) val += static_cast<double>(vvec[static_cast<std::size_t>(j)]) * static_cast<double>(s(j, c));
      vs[static_cast<std::size_t>(c)] = static_cast<float>(val);
    }
    for (int j = 0; j < s.rows; ++j) {
      for (int c = 0; c < s.cols; ++c) {
        s(j, c) -= vvec[static_cast<std::size_t>(j)] * vs[static_cast<std::size_t>(c)];
      }
    }
  }
  return PLSFitF{w, qmat};
}

PLSFitF fit_pls_components_labels_float(
  const DenseF& x,
  const std::vector<int>& y_train,
  const std::vector<int>& classes,
  int max_components
) {
  const int max_rank = pls_component_limit(max_components, x.rows, x.cols);
  return fit_pls_components_from_crossprod_float(x, centered_label_crossprod_float(x, y_train, classes), max_rank);
}

PLSFitF fit_pls_components_labels_metal_float(
  const DenseF& x,
  const std::vector<int>& y_train,
  const std::vector<int>& classes,
  int max_components
) {
  const DenseF cross_product = centered_label_crossprod_float(x, y_train, classes);
  const detail::MetalSIMPLSResult fit = detail::metal_simpls_fit(
    x.data,
    x.rows,
    x.cols,
    cross_product.data,
    cross_product.cols,
    max_components
  );
  DenseF weights(fit.predictors, fit.components);
  weights.data = fit.weights;
  DenseF y_weights(fit.responses, fit.components);
  y_weights.data = fit.y_weights;
  return PLSFitF{std::move(weights), std::move(y_weights)};
}

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

DenseF transform_pls_scores_float(const DenseF& x, const PLSFitF& fit, int ncomp) {
  DenseF out(x.rows, ncomp);
  for (int a = 0; a < ncomp; ++a) {
    for (int i = 0; i < x.rows; ++i) {
      double val = 0.0;
      for (int j = 0; j < x.cols; ++j) val += static_cast<double>(x(i, j)) * static_cast<double>(fit.weights(j, a));
      out(i, a) = static_cast<float>(val);
    }
  }
  return out;
}

DenseF transform_pls_scores_metal_float(const DenseF& x, const PLSFitF& fit, int ncomp) {
  if (ncomp < 1 || ncomp > fit.weights.cols || x.cols != fit.weights.rows) {
    throw std::invalid_argument("Metal PLS score projection dimension mismatch.");
  }
  std::vector<float> weights(static_cast<std::size_t>(fit.weights.rows) * static_cast<std::size_t>(ncomp));
  for (int row = 0; row < fit.weights.rows; ++row) {
    std::copy_n(
      fit.weights.data.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(fit.weights.cols),
      ncomp,
      weights.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(ncomp)
    );
  }
  DenseF out(x.rows, ncomp);
  out.data = detail::metal_matrix_multiply(
    x.data,
    x.rows,
    x.cols,
    weights,
    fit.weights.rows,
    ncomp
  );
  return out;
}

#if defined(KODAMA_ENABLE_CUDA)
DenseF transform_pls_scores_cuda_float(const DenseF& x, const PLSFitF& fit, int ncomp, int gpu_device) {
  if (ncomp < 1) return DenseF(x.rows, 0);
  if (ncomp > fit.weights.cols) throw std::invalid_argument("transform_pls_scores_cuda_float ncomp exceeds fit rank.");
  if (x.cols != fit.weights.rows) throw std::invalid_argument("transform_pls_scores_cuda_float column mismatch.");

  CudaBlasContext ctx(gpu_device);
  DeviceFloatBuffer dx(x.data.size());
  DeviceFloatBuffer dw(static_cast<std::size_t>(fit.weights.rows) * static_cast<std::size_t>(ncomp));
  DeviceFloatBuffer dt(static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(ncomp));

  std::vector<float> w_prefix(static_cast<std::size_t>(fit.weights.rows) * static_cast<std::size_t>(ncomp));
  for (int i = 0; i < fit.weights.rows; ++i) {
    for (int j = 0; j < ncomp; ++j) {
      w_prefix[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncomp) + static_cast<std::size_t>(j)] = fit.weights(i, j);
    }
  }

  check_cuda(cudaMemcpy(dx.data(), x.data.data(), x.data.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy float score X to device");
  check_cuda(cudaMemcpy(dw.data(), w_prefix.data(), w_prefix.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy float score W to device");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(
    cublasSgemm(
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
    "cublasSgemm float32 XW"
  );

  DenseF out(x.rows, ncomp);
  check_cuda(cudaMemcpy(out.data.data(), dt.data(), out.data.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy float scores to host");
  return out;
}

PLSFitF fit_pls_components_cuda_colmajor_y_float(
  const DenseF& x,
  const float* y_colmajor,
  int y_cols,
  int max_components,
  int gpu_device
) {
  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice(fit_pls_components_cuda_float)");
  const int max_rank = std::min({max_components, x.cols, std::max(1, x.rows - 1)});
  std::vector<float> x_colmajor(static_cast<std::size_t>(x.rows) * static_cast<std::size_t>(x.cols));
  for (int j = 0; j < x.cols; ++j) {
    for (int i = 0; i < x.rows; ++i) {
      x_colmajor[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(x.rows)] = x(i, j);
    }
  }

  std::vector<float> rr_colmajor(static_cast<std::size_t>(x.cols) * static_cast<std::size_t>(max_rank), 0.0f);
  std::vector<float> qq_colmajor(static_cast<std::size_t>(y_cols) * static_cast<std::size_t>(max_rank), 0.0f);
  const bool ok = kodama_fastpls_simpls_fit_cuda_float(
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
  if (!ok) throw std::runtime_error("fastPLS CUDA float32 SIMPLS fit failed.");

  PLSFitF fit{DenseF(x.cols, max_rank), DenseF(y_cols, max_rank)};
  for (int a = 0; a < max_rank; ++a) {
    for (int j = 0; j < x.cols; ++j) {
      fit.weights(j, a) = rr_colmajor[static_cast<std::size_t>(j) + static_cast<std::size_t>(a) * static_cast<std::size_t>(x.cols)];
    }
    for (int j = 0; j < y_cols; ++j) {
      fit.y_weights(j, a) = qq_colmajor[static_cast<std::size_t>(j) + static_cast<std::size_t>(a) * static_cast<std::size_t>(y_cols)];
    }
  }
  return fit;
}

PLSFitF fit_pls_components_cuda_labels_float(
  const DenseF& x,
  const std::vector<int>& y_train,
  const std::vector<int>& classes,
  int max_components,
  int gpu_device,
  const float* x_colmajor_precomputed = nullptr,
  const float* x_gram_colmajor_precomputed = nullptr
) {
  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice(fit_pls_components_cuda_labels_float)");
  const int y_cols = static_cast<int>(classes.size());
  const int max_rank = pls_component_limit(max_components, x.rows, x.cols);
  std::vector<float> x_colmajor_storage;
  const float* x_colmajor = x_colmajor_precomputed;
  if (x_colmajor == nullptr) {
    x_colmajor_storage = densef_to_colmajor(x);
    x_colmajor = x_colmajor_storage.data();
  }
  if (static_cast<int>(y_train.size()) != x.rows) throw std::invalid_argument("CUDA label-aware float32 SIMPLS label size mismatch.");
  std::vector<int> encoded;
  std::vector<float> class_counts;
  encode_labels_from_sorted_classes(y_train, classes, encoded, class_counts, 1);

  int fitted_rank = 0;
  std::vector<float> rr_colmajor;
  std::vector<float> qq_colmajor;
  for (int trial_rank = max_rank; trial_rank >= 1; --trial_rank) {
    rr_colmajor.assign(static_cast<std::size_t>(x.cols) * static_cast<std::size_t>(trial_rank), 0.0f);
    qq_colmajor.assign(static_cast<std::size_t>(y_cols) * static_cast<std::size_t>(trial_rank), 0.0f);
    const bool ok = x_gram_colmajor_precomputed != nullptr ?
      kodama_fastpls_simpls_fit_cuda_float_labels_gram(
        x_colmajor,
        x.rows,
        x.cols,
        x_gram_colmajor_precomputed,
        encoded.data(),
        class_counts.data(),
        y_cols,
        trial_rank,
        1,
        rr_colmajor.data(),
        qq_colmajor.data()
      ) :
      kodama_fastpls_simpls_fit_cuda_float_labels(
        x_colmajor,
        x.rows,
        x.cols,
        encoded.data(),
        class_counts.data(),
        y_cols,
        trial_rank,
        1,
        rr_colmajor.data(),
        qq_colmajor.data()
      );
    if (ok) {
      fitted_rank = trial_rank;
      break;
    }
  }
  if (fitted_rank < 1) throw std::runtime_error("fastPLS CUDA label-aware float32 SIMPLS fit failed.");

  PLSFitF fit{DenseF(x.cols, fitted_rank), DenseF(y_cols, fitted_rank)};
  for (int a = 0; a < fitted_rank; ++a) {
    for (int j = 0; j < x.cols; ++j) {
      fit.weights(j, a) = rr_colmajor[static_cast<std::size_t>(j) + static_cast<std::size_t>(a) * static_cast<std::size_t>(x.cols)];
    }
    for (int j = 0; j < y_cols; ++j) {
      fit.y_weights(j, a) = qq_colmajor[static_cast<std::size_t>(j) + static_cast<std::size_t>(a) * static_cast<std::size_t>(y_cols)];
    }
  }
  return fit;
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

std::vector<float> label_means_float(
  const std::vector<int>& labels,
  const std::vector<int>& classes
) {
  std::vector<int> encoded;
  std::vector<float> y_mean;
  encode_labels_from_sorted_classes(labels, classes, encoded, y_mean, 0);
  const float inv_n = labels.empty() ? 0.0f : 1.0f / static_cast<float>(labels.size());
  for (float& value : y_mean) value *= inv_n;
  return y_mean;
}

DenseF regression_coefficients_centered_one_hot_float(
  const DenseF& t,
  const std::vector<int>& labels,
  const std::vector<int>& classes,
  const std::vector<float>& y_mean,
  int ncomp
) {
  if (static_cast<int>(labels.size()) != t.rows) throw std::invalid_argument("float32 PLS-DA label size mismatch.");
  const int cnum = static_cast<int>(classes.size());
  DenseF lhs(ncomp, ncomp);
  DenseF rhs(ncomp, cnum);
  std::vector<int> encoded;
  std::vector<float> class_counts;
  encode_labels_from_sorted_classes(labels, classes, encoded, class_counts, 0);
  std::vector<double> score_sums(static_cast<std::size_t>(ncomp), 0.0);
  std::vector<double> lhs_accum(static_cast<std::size_t>(ncomp) * static_cast<std::size_t>(ncomp), 0.0);

  for (int r = 0; r < t.rows; ++r) {
    const int c = encoded[static_cast<std::size_t>(r)];
    const float* t_row = t.data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(t.cols);
    for (int a = 0; a < ncomp; ++a) {
      const double value = t_row[a];
      rhs(a, c) += static_cast<float>(value);
      score_sums[static_cast<std::size_t>(a)] += value;
      for (int b = a; b < ncomp; ++b) {
        lhs_accum[static_cast<std::size_t>(a) * static_cast<std::size_t>(ncomp) + static_cast<std::size_t>(b)] +=
          value * static_cast<double>(t_row[b]);
      }
    }
  }
  for (int a = 0; a < ncomp; ++a) {
    for (int b = a; b < ncomp; ++b) {
      const float value = static_cast<float>(lhs_accum[static_cast<std::size_t>(a) * static_cast<std::size_t>(ncomp) + static_cast<std::size_t>(b)]);
      lhs(a, b) = value;
      lhs(b, a) = value;
    }
  }
  for (int a = 0; a < ncomp; ++a) {
    const double sum = score_sums[static_cast<std::size_t>(a)];
    for (int c = 0; c < cnum; ++c) rhs(a, c) -= static_cast<float>(sum * static_cast<double>(y_mean[static_cast<std::size_t>(c)]));
  }

  return solve_linear_float(lhs, rhs);
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

std::vector<int> predict_pls_da_labels_float(
  const DenseF& t_train,
  const std::vector<int>& y_train_labels,
  const DenseF& t_val,
  const std::vector<int>& classes,
  const std::vector<float>& y_mean,
  int ncomp
) {
  const DenseF coef = regression_coefficients_centered_one_hot_float(t_train, y_train_labels, classes, y_mean, ncomp);
  std::vector<int> pred(static_cast<std::size_t>(t_val.rows), classes.front());
  for (int i = 0; i < t_val.rows; ++i) {
    int best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < static_cast<int>(classes.size()); ++c) {
      float score = c < static_cast<int>(y_mean.size()) ? y_mean[static_cast<std::size_t>(c)] : 0.0f;
      for (int a = 0; a < ncomp; ++a) {
        score += t_val(i, a) * coef(a, c);
      }
      if (score > best_score) {
        best_score = score;
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

std::vector<int> predict_pls_da_cuda_float(
  const DenseF& t_train,
  const std::vector<int>& y_train_labels,
  const DenseF& t_val,
  const std::vector<int>& classes,
  const std::vector<float>& y_mean,
  int gpu_device
) {
  if (t_val.rows < 1) return {};
  const int n = t_val.rows;
  const int kk = t_val.cols;
  const int cnum = static_cast<int>(classes.size());
  const DenseF coef = regression_coefficients_centered_one_hot_float(t_train, y_train_labels, classes, y_mean, kk);

  std::vector<float> linear(static_cast<std::size_t>(cnum) * static_cast<std::size_t>(kk), 0.0f);
  std::vector<float> constants(static_cast<std::size_t>(cnum), 0.0f);
  for (int c = 0; c < cnum; ++c) {
    constants[static_cast<std::size_t>(c)] =
      c < static_cast<int>(y_mean.size()) ? y_mean[static_cast<std::size_t>(c)] : 0.0f;
    for (int a = 0; a < kk; ++a) {
      linear[static_cast<std::size_t>(c) * static_cast<std::size_t>(kk) + static_cast<std::size_t>(a)] = coef(a, c);
    }
  }

  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice(predict_pls_da_cuda_float)");
  cudaStream_t stream = nullptr;
  DeviceFloatBuffer d_t(t_val.data.size());
  DeviceFloatBuffer d_linear(linear.size());
  DeviceFloatBuffer d_constants(constants.size());
  DeviceIntBuffer d_pred(static_cast<std::size_t>(n));
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate(predict_pls_da_cuda_float)");
  check_cuda(cudaMemcpyAsync(d_t.data(), t_val.data.data(), t_val.data.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA float32 PLS-DA T");
  check_cuda(cudaMemcpyAsync(d_linear.data(), linear.data(), linear.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA float32 PLS-DA linear");
  check_cuda(cudaMemcpyAsync(d_constants.data(), constants.data(), constants.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync CUDA float32 PLS-DA constants");
  kodama_cuda_lda_score_argmax_row_float(d_t.data(), d_linear.data(), d_constants.data(), d_pred.data(), n, kk, cnum, stream);
  check_cuda(cudaGetLastError(), "kodama_cuda_lda_score_argmax_row_float PLS-DA");
  std::vector<int> codes(static_cast<std::size_t>(n), 1);
  check_cuda(cudaMemcpyAsync(codes.data(), d_pred.data(), codes.size() * sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync CUDA float32 PLS-DA labels");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CUDA float32 PLS-DA predict");
  cudaStreamDestroy(stream);

  std::vector<int> pred(static_cast<std::size_t>(n), classes.front());
  for (int i = 0; i < n; ++i) {
    const int cls = std::max(1, std::min(cnum, codes[static_cast<std::size_t>(i)])) - 1;
    pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(cls)];
  }
  return pred;
}
#endif

DenseF solve_linear_float(DenseF a, DenseF b) {
  const int n = a.rows;
  const int m = b.cols;
  for (int i = 0; i < n; ++i) a(i, i) += 1.0e-9f;
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    float best = std::abs(a(col, col));
    for (int r = col + 1; r < n; ++r) {
      const float val = std::abs(a(r, col));
      if (val > best) {
        best = val;
        pivot = r;
      }
    }
    if (best < 1.0e-12f) continue;
    if (pivot != col) {
      for (int j = 0; j < n; ++j) std::swap(a(col, j), a(pivot, j));
      for (int j = 0; j < m; ++j) std::swap(b(col, j), b(pivot, j));
    }
    const float div = a(col, col);
    for (int j = col; j < n; ++j) a(col, j) /= div;
    for (int j = 0; j < m; ++j) b(col, j) /= div;
    for (int r = 0; r < n; ++r) {
      if (r == col) continue;
      const float f = a(r, col);
      if (f == 0.0f) continue;
      for (int j = col; j < n; ++j) a(r, j) -= f * a(col, j);
      for (int j = 0; j < m; ++j) b(r, j) -= f * b(col, j);
    }
  }
  return b;
}

std::vector<int> predict_pls_lda_float(
  const DenseF& t_train,
  const std::vector<int>& y_train,
  const DenseF& t_val,
  const std::vector<int>& classes,
  int ncomp
) {
  const int cnum = static_cast<int>(classes.size());
  std::vector<int> encoded;
  std::vector<int> counts;
  encode_labels_from_sorted_classes(y_train, classes, encoded, counts, 0);
  DenseF cent(cnum, ncomp);
  for (int i = 0; i < t_train.rows; ++i) {
    const int c = encoded[static_cast<std::size_t>(i)];
    for (int a = 0; a < ncomp; ++a) cent(c, a) += t_train(i, a);
  }
  for (int c = 0; c < cnum; ++c) {
    const float inv = counts[static_cast<std::size_t>(c)] > 0 ? 1.0f / static_cast<float>(counts[static_cast<std::size_t>(c)]) : 0.0f;
    for (int a = 0; a < ncomp; ++a) cent(c, a) *= inv;
  }

  DenseF pooled(ncomp, ncomp);
  for (int i = 0; i < t_train.rows; ++i) {
    const int c = encoded[static_cast<std::size_t>(i)];
    for (int r = 0; r < ncomp; ++r) {
      const float dr = t_train(i, r) - cent(c, r);
      for (int col = 0; col < ncomp; ++col) {
        const float dc = t_train(i, col) - cent(c, col);
        pooled(r, col) += dr * dc;
      }
    }
  }
  const float df = static_cast<float>(std::max(1, t_train.rows - cnum));
  float trace = 0.0f;
  for (int r = 0; r < ncomp; ++r) {
    for (int col = 0; col < ncomp; ++col) pooled(r, col) /= df;
    trace += pooled(r, r);
  }
  const float ridge = 1e-8f * (std::isfinite(trace) && trace > 0.0f ? trace / static_cast<float>(std::max(1, ncomp)) : 1.0f);
  for (int r = 0; r < ncomp; ++r) pooled(r, r) += ridge;

  DenseF rhs(ncomp, cnum);
  for (int c = 0; c < cnum; ++c) {
    for (int a = 0; a < ncomp; ++a) rhs(a, c) = cent(c, a);
  }
  DenseF solved = solve_linear_float(pooled, rhs);

  DenseF linear(cnum, ncomp);
  std::vector<float> constants(static_cast<std::size_t>(cnum), 0.0f);
  for (int c = 0; c < cnum; ++c) {
    float dot_mu = 0.0f;
    for (int a = 0; a < ncomp; ++a) {
      linear(c, a) = solved(a, c);
      dot_mu += cent(c, a) * linear(c, a);
    }
    const float prior = std::max(
      static_cast<float>(counts[static_cast<std::size_t>(c)]) / static_cast<float>(std::max(1, t_train.rows)),
      std::numeric_limits<float>::min()
    );
    constants[static_cast<std::size_t>(c)] = -0.5f * dot_mu + std::log(prior);
  }

  std::vector<int> pred(static_cast<std::size_t>(t_val.rows), classes.front());
  for (int i = 0; i < t_val.rows; ++i) {
    int best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < cnum; ++c) {
      float score = constants[static_cast<std::size_t>(c)];
      for (int a = 0; a < ncomp; ++a) {
        score += t_val(i, a) * linear(c, a);
      }
      if (score > best_score) {
        best_score = score;
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

std::vector<int> train_predict_pls_lda_projected_cuda(
  const Dense& x_train,
  const std::vector<int>& y_train,
  const Dense& x_val,
  const PLSFit& fit,
  int ncomp,
  const std::vector<int>& classes,
  int gpu_device
) {
  if (x_train.rows < 1 || x_train.cols < 1) throw std::invalid_argument("CUDA PLS-LDA requires a non-empty training matrix.");
  if (static_cast<int>(y_train.size()) != x_train.rows) throw std::invalid_argument("CUDA PLS-LDA label size mismatch.");
  if (x_val.rows < 1) return {};
  if (x_val.cols != x_train.cols) throw std::invalid_argument("CUDA PLS-LDA train/validation column mismatch.");
  if (fit.weights.rows != x_train.cols) throw std::invalid_argument("CUDA PLS-LDA projection column mismatch.");
  if (ncomp < 1 || ncomp > fit.weights.cols) throw std::invalid_argument("CUDA PLS-LDA component count exceeds fit rank.");

  const int n = x_train.rows;
  const int p = x_train.cols;
  const int n_val = x_val.rows;
  const int k = ncomp;
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

  std::vector<double> w_prefix(static_cast<std::size_t>(p) * static_cast<std::size_t>(k));
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < k; ++j) {
      w_prefix[static_cast<std::size_t>(i) * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)] = fit.weights(i, j);
    }
  }

  DeviceBuffer d_x_train(x_train.data.size());
  DeviceBuffer d_x_val(x_val.data.size());
  DeviceBuffer d_w(w_prefix.size());
  DeviceBuffer d_t(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
  DeviceBuffer d_t_val(static_cast<std::size_t>(n_val) * static_cast<std::size_t>(k));
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
  DeviceIntBuffer d_pred(static_cast<std::size_t>(n_val));
  double* d_work = nullptr;

  auto cleanup = [&]() {
    if (d_work != nullptr) cudaFree(d_work);
  };

  try {
    check_cuda(cudaMemcpyAsync(d_x_train.data(), x_train.data.data(), x_train.data.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync PLS-LDA Xtrain");
    check_cuda(cudaMemcpyAsync(d_x_val.data(), x_val.data.data(), x_val.data.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync PLS-LDA Xval");
    check_cuda(cudaMemcpyAsync(d_w.data(), w_prefix.data(), w_prefix.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync PLS-LDA weights");
    check_cuda(cudaMemcpyAsync(d_labels.data(), encoded.data(), encoded.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync PLS-LDA labels");
    check_cuda(cudaMemcpyAsync(d_counts.data(), counts.data(), counts.size() * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync PLS-LDA counts");

    const double one = 1.0;
    const double zero = 0.0;
    check_cublas(
      cublasDgemm(
        blas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        k,
        n,
        p,
        &one,
        d_w.data(),
        k,
        d_x_train.data(),
        p,
        &zero,
        d_t.data(),
        k
      ),
      "cublasDgemm PLS-LDA train scores"
    );
    check_cublas(
      cublasDgemm(
        blas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        k,
        n_val,
        p,
        &one,
        d_w.data(),
        k,
        d_x_val.data(),
        p,
        &zero,
        d_t_val.data(),
        k
      ),
      "cublasDgemm PLS-LDA validation scores"
    );

    kodama_cuda_lda_label_sums_row(d_t.data(), d_labels.data(), n, k, cnum, d_means.data(), stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_label_sums_row fused PLS-LDA");
    kodama_cuda_lda_means_row(d_means.data(), d_counts.data(), k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_means_row fused PLS-LDA");
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
      "cublasDgemm fused CUDA LDA TtT"
    );
    kodama_cuda_lda_pooled_col(d_pooled.data(), d_means.data(), d_counts.data(), n, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_pooled_col fused PLS-LDA");
    kodama_cuda_lda_copy_cov(d_pooled.data(), d_cov.data(), k, k, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_copy_cov fused PLS-LDA");
    kodama_cuda_lda_add_ridge(d_cov.data(), k, 1e-8, d_lambda.data(), stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_add_ridge fused PLS-LDA");
    kodama_cuda_lda_means_to_rhs(d_means.data(), d_rhs.data(), k, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_means_to_rhs fused PLS-LDA");

    int lwork = 0;
    check_cusolver(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, k, d_cov.data(), k, &lwork), "cusolverDnDpotrf_bufferSize fused PLS-LDA");
    check_cuda(cudaMalloc(&d_work, sizeof(double) * static_cast<std::size_t>(std::max(lwork, 1))), "cudaMalloc fused PLS-LDA work");
    check_cusolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, k, d_cov.data(), k, d_work, lwork, d_info.data()), "cusolverDnDpotrf fused PLS-LDA");
    int info = 0;
    check_cuda(cudaMemcpyAsync(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync fused PLS-LDA potrf info");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize fused PLS-LDA potrf");
    if (info != 0) throw std::runtime_error("cusolverDnDpotrf fused PLS-LDA returned non-zero info.");
    check_cusolver(cusolverDnDpotrs(solver, CUBLAS_FILL_MODE_LOWER, k, cnum, d_cov.data(), k, d_rhs.data(), k, d_info.data()), "cusolverDnDpotrs fused PLS-LDA");
    check_cuda(cudaMemcpyAsync(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync fused PLS-LDA potrs info");
    kodama_cuda_lda_finalize_linear_row(d_rhs.data(), d_means.data(), d_counts.data(), d_linear.data(), d_constants.data(), n, k, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_finalize_linear_row fused PLS-LDA");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize fused PLS-LDA solve");
    if (info != 0) throw std::runtime_error("cusolverDnDpotrs fused PLS-LDA returned non-zero info.");

    kodama_cuda_lda_score_argmax_row(d_t_val.data(), d_linear.data(), d_constants.data(), d_pred.data(), n_val, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_score_argmax_row fused PLS-LDA");
    std::vector<int> codes(static_cast<std::size_t>(n_val), 1);
    check_cuda(cudaMemcpyAsync(codes.data(), d_pred.data(), codes.size() * sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync fused PLS-LDA labels");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize fused PLS-LDA predict");
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

std::vector<int> train_predict_pls_lda_projected_cuda_float(
  const DenseF& x_train,
  const std::vector<int>& y_train,
  const DenseF& x_val,
  const PLSFitF& fit,
  int ncomp,
  const std::vector<int>& classes,
  int gpu_device
) {
  if (x_train.rows < 1 || x_train.cols < 1) throw std::invalid_argument("CUDA float32 PLS-LDA requires a non-empty training matrix.");
  if (static_cast<int>(y_train.size()) != x_train.rows) throw std::invalid_argument("CUDA float32 PLS-LDA label size mismatch.");
  if (x_val.rows < 1) return {};
  if (x_val.cols != x_train.cols) throw std::invalid_argument("CUDA float32 PLS-LDA train/validation column mismatch.");
  if (fit.weights.rows != x_train.cols) throw std::invalid_argument("CUDA float32 PLS-LDA projection column mismatch.");
  if (ncomp < 1 || ncomp > fit.weights.cols) throw std::invalid_argument("CUDA float32 PLS-LDA component count exceeds fit rank.");

  const int n = x_train.rows;
  const int p = x_train.cols;
  const int n_val = x_val.rows;
  const int k = ncomp;
  const int cnum = static_cast<int>(classes.size());
  CudaLDAContext& context = cuda_lda_context(gpu_device);
  cudaStream_t stream = context.stream();
  cublasHandle_t blas = context.blas();
  cusolverDnHandle_t solver = context.solver();
  CudaPLSLDAFloatWorkspace& workspace = cuda_pls_lda_float_workspace(gpu_device);

  encode_labels_from_sorted_classes(y_train, classes, workspace.encoded, workspace.class_counts, 1);

  workspace.weights_prefix.resize(static_cast<std::size_t>(p) * static_cast<std::size_t>(k));
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < k; ++j) {
      workspace.weights_prefix[static_cast<std::size_t>(i) * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)] = fit.weights(i, j);
    }
  }

  workspace.x_train.ensure(x_train.data.size());
  workspace.x_val.ensure(x_val.data.size());
  workspace.weights.ensure(workspace.weights_prefix.size());
  workspace.train_scores.ensure(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
  workspace.val_scores.ensure(static_cast<std::size_t>(n_val) * static_cast<std::size_t>(k));
  workspace.labels.ensure(workspace.encoded.size());
  workspace.counts.ensure(workspace.class_counts.size());
  workspace.means.ensure(static_cast<std::size_t>(cnum) * static_cast<std::size_t>(k));
  workspace.pooled.ensure(static_cast<std::size_t>(k) * static_cast<std::size_t>(k));
  workspace.cov.ensure(static_cast<std::size_t>(k) * static_cast<std::size_t>(k));
  workspace.rhs.ensure(static_cast<std::size_t>(k) * static_cast<std::size_t>(cnum));
  workspace.linear.ensure(static_cast<std::size_t>(cnum) * static_cast<std::size_t>(k));
  workspace.constants.ensure(static_cast<std::size_t>(cnum));
  workspace.lambda.ensure(1);
  workspace.info.ensure(1);
  workspace.pred.ensure(static_cast<std::size_t>(n_val));

  try {
    check_cuda(cudaMemcpyAsync(workspace.x_train.data(), x_train.data.data(), x_train.data.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync float32 PLS-LDA Xtrain");
    check_cuda(cudaMemcpyAsync(workspace.x_val.data(), x_val.data.data(), x_val.data.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync float32 PLS-LDA Xval");
    check_cuda(cudaMemcpyAsync(workspace.weights.data(), workspace.weights_prefix.data(), workspace.weights_prefix.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync float32 PLS-LDA weights");
    check_cuda(cudaMemcpyAsync(workspace.labels.data(), workspace.encoded.data(), workspace.encoded.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync float32 PLS-LDA labels");
    check_cuda(cudaMemcpyAsync(workspace.counts.data(), workspace.class_counts.data(), workspace.class_counts.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync float32 PLS-LDA counts");

    const float one = 1.0f;
    const float zero = 0.0f;
    check_cublas(
      cublasSgemm(
        blas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        k,
        n,
        p,
        &one,
        workspace.weights.data(),
        k,
        workspace.x_train.data(),
        p,
        &zero,
        workspace.train_scores.data(),
        k
      ),
      "cublasSgemm float32 PLS-LDA train scores"
    );
    check_cublas(
      cublasSgemm(
        blas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        k,
        n_val,
        p,
        &one,
        workspace.weights.data(),
        k,
        workspace.x_val.data(),
        p,
        &zero,
        workspace.val_scores.data(),
        k
      ),
      "cublasSgemm float32 PLS-LDA validation scores"
    );

    kodama_cuda_lda_label_sums_row_float(workspace.train_scores.data(), workspace.labels.data(), n, k, cnum, workspace.means.data(), stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_label_sums_row_float");
    kodama_cuda_lda_means_row_float(workspace.means.data(), workspace.counts.data(), k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_means_row_float");
    check_cublas(
      cublasSsyrk(
        blas,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        k,
        n,
        &one,
        workspace.train_scores.data(),
        k,
        &zero,
        workspace.pooled.data(),
        k
      ),
      "cublasSsyrk float32 CUDA LDA TtT"
    );
    kodama_cuda_symmetrize_lower_float(workspace.pooled.data(), k, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_symmetrize_lower_float");
    kodama_cuda_lda_pooled_col_float(workspace.pooled.data(), workspace.means.data(), workspace.counts.data(), n, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_pooled_col_float");
    kodama_cuda_lda_means_to_rhs_float(workspace.means.data(), workspace.rhs.data(), k, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_means_to_rhs_float");

    int lwork = 0;
    check_cusolver(cusolverDnSpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, k, workspace.cov.data(), k, &lwork), "cusolverDnSpotrf_bufferSize float32 PLS-LDA");
    workspace.solver_work.ensure(static_cast<std::size_t>(std::max(lwork, 1)));
    int info = 0;
    bool factorized = false;
    const float ridge_grid[] = {1e-8f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f};
    for (float ridge : ridge_grid) {
      kodama_cuda_lda_copy_cov_float(workspace.pooled.data(), workspace.cov.data(), k, k, stream);
      check_cuda(cudaGetLastError(), "kodama_cuda_lda_copy_cov_float");
      kodama_cuda_lda_add_ridge_float(workspace.cov.data(), k, ridge, workspace.lambda.data(), stream);
      check_cuda(cudaGetLastError(), "kodama_cuda_lda_add_ridge_float");
      check_cusolver(cusolverDnSpotrf(solver, CUBLAS_FILL_MODE_LOWER, k, workspace.cov.data(), k, workspace.solver_work.data(), lwork, workspace.info.data()), "cusolverDnSpotrf float32 PLS-LDA");
      check_cuda(cudaMemcpyAsync(&info, workspace.info.data(), sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync float32 PLS-LDA potrf info");
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize float32 PLS-LDA potrf");
      if (info == 0) {
        factorized = true;
        break;
      }
    }
    if (!factorized) throw std::runtime_error("cusolverDnSpotrf float32 PLS-LDA returned non-zero info.");
    check_cusolver(cusolverDnSpotrs(solver, CUBLAS_FILL_MODE_LOWER, k, cnum, workspace.cov.data(), k, workspace.rhs.data(), k, workspace.info.data()), "cusolverDnSpotrs float32 PLS-LDA");
    check_cuda(cudaMemcpyAsync(&info, workspace.info.data(), sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync float32 PLS-LDA potrs info");
    kodama_cuda_lda_finalize_linear_row_float(workspace.rhs.data(), workspace.means.data(), workspace.counts.data(), workspace.linear.data(), workspace.constants.data(), n, k, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_finalize_linear_row_float");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize float32 PLS-LDA solve");
    if (info != 0) throw std::runtime_error("cusolverDnSpotrs float32 PLS-LDA returned non-zero info.");

    kodama_cuda_lda_score_argmax_row_float(workspace.val_scores.data(), workspace.linear.data(), workspace.constants.data(), workspace.pred.data(), n_val, k, cnum, stream);
    check_cuda(cudaGetLastError(), "kodama_cuda_lda_score_argmax_row_float");
    workspace.pred_codes.assign(static_cast<std::size_t>(n_val), 1);
    check_cuda(cudaMemcpyAsync(workspace.pred_codes.data(), workspace.pred.data(), workspace.pred_codes.size() * sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync float32 PLS-LDA labels");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize float32 PLS-LDA predict");

    std::vector<int> pred(static_cast<std::size_t>(n_val), classes.front());
    for (int i = 0; i < n_val; ++i) {
      const int cls = std::max(1, std::min(cnum, workspace.pred_codes[static_cast<std::size_t>(i)])) - 1;
      pred[static_cast<std::size_t>(i)] = classes[static_cast<std::size_t>(cls)];
    }
    return pred;
  } catch (...) {
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


}  // namespace

namespace {

PLSCVResult run_plscv_host(
  MatrixView x,
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const PLSOptions& options,
  PLSMode mode,
  Backend backend
) {
  if (backend != Backend::CPU && backend != Backend::Metal) {
    throw std::invalid_argument("Host PLS-CV backend must be CPU or Metal.");
  }
  detail::validate_inputs(x, labels, constrain);
  if (options.max_components < 1) throw std::invalid_argument("PLSOptions::max_components must be positive.");
  if (options.fixed_components < 0) throw std::invalid_argument("PLSOptions::fixed_components must be non-negative.");
  if (options.fixed_components > options.max_components) throw std::invalid_argument("PLSOptions::fixed_components cannot exceed max_components.");

  detail::Timer timer;
  PLSCVResult result;
  result.true_labels = labels;
  const bool use_fold_cache = !options.cv.stratified;
  const PLSFoldXCacheF* fold_cache = use_fold_cache ?
    &get_pls_fold_x_cache_float(x, labels, constrain, options, false) : nullptr;
  result.fold_assignments = fold_cache != nullptr ?
    fold_cache->fold_assignments :
    detail::make_folds(labels, constrain, options.cv);
  result.accuracy_by_components.assign(static_cast<std::size_t>(options.max_components), 0.0);
  std::vector<int> selected_pred(labels.size(), labels.empty() ? 0 : labels.front());
  const std::vector<int> fold_ids = fold_cache != nullptr ?
    fold_cache->fold_ids :
    detail::sorted_unique_folds(result.fold_assignments);
  int evaluated_component = options.fixed_components > 0 ? options.fixed_components : options.max_components;

  std::vector<int> fold_evaluated_components(fold_ids.size(), evaluated_component);
  auto process_fold = [&](std::size_t fold_pos) {
    const int fold = fold_ids[fold_pos];
    std::vector<int> validation_storage;
    std::vector<int> train_storage;
    DenseF x_train_storage;
    DenseF x_val_storage;
    const std::vector<int>* validation = nullptr;
    const std::vector<int>* train = nullptr;
    const DenseF* x_train = nullptr;
    const DenseF* x_val = nullptr;
    if (fold_cache != nullptr) {
      const PLSFoldDataF& fold_data = fold_cache->folds_data[fold_pos];
      validation = &fold_data.validation;
      train = &fold_data.train;
      x_train = &fold_data.x_train;
      x_val = &fold_data.x_val;
    } else {
      validation_storage = detail::indices_where_fold(result.fold_assignments, fold, true);
      train_storage = detail::indices_where_fold(result.fold_assignments, fold, false);
      std::vector<float> x_mean;
      std::vector<float> x_scale;
      train_center_scale_float(x, train_storage, options.center, options.scale, x_mean, x_scale);
      x_train_storage = subset_scale_float(x, train_storage, x_mean, x_scale);
      x_val_storage = subset_scale_float(x, validation_storage, x_mean, x_scale);
      x_train = &x_train_storage;
      x_val = &x_val_storage;
      validation = &validation_storage;
      train = &train_storage;
    }
    std::vector<int> y_train_labels(train->size(), 0);
    for (std::size_t i = 0; i < train->size(); ++i) y_train_labels[i] = labels[static_cast<std::size_t>((*train)[i])];
    const std::vector<int> fold_classes = detail::unique_labels(y_train_labels);
    if (fold_classes.size() <= 1U) {
      const int pred_label = fold_classes.empty() ? (labels.empty() ? 0 : labels.front()) : fold_classes.front();
      fold_evaluated_components[fold_pos] = 1;
      for (std::size_t i = 0; i < validation->size(); ++i) {
        selected_pred[static_cast<std::size_t>((*validation)[i])] = pred_label;
      }
      return;
    }
    PLSFitF fit = backend == Backend::Metal ?
      fit_pls_components_labels_metal_float(*x_train, y_train_labels, fold_classes, options.max_components) :
      fit_pls_components_labels_float(*x_train, y_train_labels, fold_classes, options.max_components);
    const std::vector<int> eval_components = components_to_evaluate(options, fit.weights.cols);
    fold_evaluated_components[fold_pos] = eval_components.front();
    DenseF t_train_full = backend == Backend::Metal ?
      transform_pls_scores_metal_float(*x_train, fit, fit.weights.cols) :
      transform_pls_scores_float(*x_train, fit, fit.weights.cols);
    DenseF t_val_full = backend == Backend::Metal ?
      transform_pls_scores_metal_float(*x_val, fit, fit.weights.cols) :
      transform_pls_scores_float(*x_val, fit, fit.weights.cols);
    const std::vector<float> y_mean = label_means_float(y_train_labels, fold_classes);
    for (int a : eval_components) {
      std::vector<int> fold_pred = mode == PLSMode::PLS_LDA ?
        predict_pls_lda_float(t_train_full, y_train_labels, t_val_full, fold_classes, a) :
        predict_pls_da_labels_float(t_train_full, y_train_labels, t_val_full, fold_classes, y_mean, a);
      for (std::size_t i = 0; i < validation->size(); ++i) {
        selected_pred[static_cast<std::size_t>((*validation)[i])] = fold_pred[i];
      }
    }
  };

  const int fold_workers = std::max(1, std::min(options.n_threads, static_cast<int>(fold_ids.size())));
  if (fold_workers <= 1) {
    for (std::size_t fold_pos = 0; fold_pos < fold_ids.size(); ++fold_pos) process_fold(fold_pos);
  } else {
    std::atomic<std::size_t> next_fold{0};
    std::exception_ptr worker_error;
    std::mutex worker_error_mutex;
    auto worker = [&]() {
      for (;;) {
        const std::size_t fold_pos = next_fold.fetch_add(1);
        if (fold_pos >= fold_ids.size()) break;
        try {
          process_fold(fold_pos);
        } catch (...) {
          std::lock_guard<std::mutex> lock(worker_error_mutex);
          if (!worker_error) worker_error = std::current_exception();
          break;
        }
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(fold_workers - 1));
    for (int w = 1; w < fold_workers; ++w) workers.emplace_back(worker);
    worker();
    for (std::thread& th : workers) th.join();
    if (worker_error) std::rethrow_exception(worker_error);
  }
  for (int comp : fold_evaluated_components) evaluated_component = std::min(evaluated_component, comp);

  const int best_comp = std::min(evaluated_component, options.max_components);
  result.accuracy_by_components[static_cast<std::size_t>(best_comp - 1)] =
    detail::accuracy(labels, selected_pred);
  result.selected_components = best_comp;
  result.predicted_labels = std::move(selected_pred);
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
  result.parameters.backend = backend;
  result.parameters.mode = mode;
  result.parameters.max_components = options.max_components;
  result.parameters.selected_components = best_comp;
  result.parameters.fixed_components = options.fixed_components;
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

  detail::Timer timer;
  check_cuda(cudaSetDevice(options.gpu_device), "cudaSetDevice(run_plscv_cuda)");
  PLSCVResult result;
  result.true_labels = labels;
  const bool use_fold_cache = !options.cv.stratified;
  const PLSFoldXCacheF* fold_cache = use_fold_cache ?
    &get_pls_fold_x_cache_float(x, labels, constrain, options, true) : nullptr;
  result.fold_assignments = fold_cache != nullptr ?
    fold_cache->fold_assignments :
    detail::make_folds(labels, constrain, options.cv);
  result.accuracy_by_components.assign(static_cast<std::size_t>(options.max_components), 0.0);
  std::vector<int> selected_pred(labels.size(), labels.empty() ? 0 : labels.front());
  const std::vector<int> fold_ids = fold_cache != nullptr ?
    fold_cache->fold_ids :
    detail::sorted_unique_folds(result.fold_assignments);
  int evaluated_component = options.fixed_components > 0 ? options.fixed_components : options.max_components;

  for (std::size_t fold_pos = 0; fold_pos < fold_ids.size(); ++fold_pos) {
    const int fold = fold_ids[fold_pos];
    std::vector<int> validation_storage;
    std::vector<int> train_storage;
    DenseF x_train_storage;
    DenseF x_val_storage;
    const std::vector<int>* validation = nullptr;
    const std::vector<int>* train = nullptr;
    const DenseF* x_train = nullptr;
    const DenseF* x_val = nullptr;
    const std::vector<float>* x_train_colmajor = nullptr;
    const std::vector<float>* x_train_gram_colmajor = nullptr;
    if (fold_cache != nullptr) {
      const PLSFoldDataF& fold_data = fold_cache->folds_data[fold_pos];
      validation = &fold_data.validation;
      train = &fold_data.train;
      x_train = &fold_data.x_train;
      x_val = &fold_data.x_val;
      x_train_colmajor = &fold_data.x_train_colmajor;
      x_train_gram_colmajor = &fold_data.x_train_gram_colmajor;
    } else {
      validation_storage = detail::indices_where_fold(result.fold_assignments, fold, true);
      train_storage = detail::indices_where_fold(result.fold_assignments, fold, false);
      std::vector<float> x_mean;
      std::vector<float> x_scale;
      train_center_scale_float(x, train_storage, options.center, options.scale, x_mean, x_scale);
      x_train_storage = subset_scale_float(x, train_storage, x_mean, x_scale);
      x_val_storage = subset_scale_float(x, validation_storage, x_mean, x_scale);
      x_train = &x_train_storage;
      x_val = &x_val_storage;
      validation = &validation_storage;
      train = &train_storage;
    }
    std::vector<int> y_train_labels(train->size(), 0);
    for (std::size_t i = 0; i < train->size(); ++i) y_train_labels[i] = labels[static_cast<std::size_t>((*train)[i])];
    const std::vector<int> fold_classes = detail::unique_labels(y_train_labels);
    if (fold_classes.size() <= 1U) {
      const int pred_label = fold_classes.empty() ? (labels.empty() ? 0 : labels.front()) : fold_classes.front();
      evaluated_component = std::min(evaluated_component, 1);
      for (std::size_t i = 0; i < validation->size(); ++i) {
        selected_pred[static_cast<std::size_t>((*validation)[i])] = pred_label;
      }
      continue;
    }
    PLSFitF fit = fit_pls_components_cuda_labels_float(
        *x_train,
        y_train_labels,
        fold_classes,
        options.max_components,
        options.gpu_device,
        x_train_colmajor != nullptr ? x_train_colmajor->data() : nullptr,
        (x_train_gram_colmajor != nullptr && !x_train_gram_colmajor->empty()) ? x_train_gram_colmajor->data() : nullptr
      );
    const std::vector<int> eval_components = components_to_evaluate(options, fit.weights.cols);
    evaluated_component = std::min(evaluated_component, eval_components.front());
    if (mode == PLSMode::PLS_LDA) {
      for (int a : eval_components) {
        const std::vector<int> fold_pred = train_predict_pls_lda_projected_cuda_float(
            *x_train,
            y_train_labels,
            *x_val,
            fit,
            a,
            fold_classes,
            options.gpu_device
          );
        for (std::size_t i = 0; i < validation->size(); ++i) {
          selected_pred[static_cast<std::size_t>((*validation)[i])] = fold_pred[i];
        }
      }
    } else {
      const std::vector<float> y_mean = label_means_float(y_train_labels, fold_classes);
      for (int a : eval_components) {
        DenseF t_train_prefix = transform_pls_scores_cuda_float(*x_train, fit, a, options.gpu_device);
        DenseF t_val_prefix = transform_pls_scores_cuda_float(*x_val, fit, a, options.gpu_device);
        const std::vector<int> fold_pred = predict_pls_da_cuda_float(
            t_train_prefix,
            y_train_labels,
            t_val_prefix,
            fold_classes,
            y_mean,
            options.gpu_device
          );
        for (std::size_t i = 0; i < validation->size(); ++i) {
          selected_pred[static_cast<std::size_t>((*validation)[i])] = fold_pred[i];
        }
      }
    }
  }

  const int best_comp = std::min(evaluated_component, options.max_components);
  result.accuracy_by_components[static_cast<std::size_t>(best_comp - 1)] =
    detail::accuracy(labels, selected_pred);
  result.selected_components = best_comp;
  result.predicted_labels = std::move(selected_pred);
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
  result.parameters.center = options.center;
  result.parameters.scale = options.scale;
  result.parameters.gpu_device = options.gpu_device;
  result.parameters.n_threads = options.n_threads;
  return result;
}

std::vector<int> plslda_predict_cuda(
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
  std::vector<float> mean;
  std::vector<float> scale;
  train_center_scale_float(train, train_rows, options.center, options.scale, mean, scale);
  DenseF x_train = subset_scale_float(train, train_rows, mean, scale);
  DenseF x_test = subset_scale_float(test, test_rows, mean, scale);
  const int requested = options.fixed_components > 0 ? options.fixed_components : options.max_components;
  const int ncomp = std::max(1, std::min({requested, x_train.cols, std::max(1, x_train.rows - 1)}));
  PLSFitF fit = fit_pls_components_cuda_labels_float(
    x_train,
    labels,
    classes,
    ncomp,
    options.gpu_device,
    nullptr,
    nullptr
  );
  return train_predict_pls_lda_projected_cuda_float(
    x_train,
    labels,
    x_test,
    fit,
    fit.weights.cols,
    classes,
    options.gpu_device
  );
}
#endif

}  // namespace

PLSCVResult PLSDACV(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  if (options.backend == Backend::CUDA) return PLSDACV_CUDA(x, labels, constrain, options);
  return PLSDACV_CPU(x, labels, constrain, options);
}

PLSCVResult PLSLDACV(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  if (options.backend == Backend::CUDA) return PLSLDACV_CUDA(x, labels, constrain, options);
  if (options.backend == Backend::Metal) return PLSLDACV_METAL(x, labels, constrain, options);
  return PLSLDACV_CPU(x, labels, constrain, options);
}

PLSCVResult PLSDACV_CPU(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  return run_plscv_host(x, labels, constrain, options, PLSMode::PLS_DA, Backend::CPU);
}

PLSCVResult PLSLDACV_CPU(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
  return run_plscv_host(x, labels, constrain, options, PLSMode::PLS_LDA, Backend::CPU);
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

PLSCVResult PLSLDACV_METAL(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain, const PLSOptions& options) {
#if defined(KODAMA_ENABLE_METAL)
  PLSOptions metal_options = options;
  metal_options.backend = Backend::Metal;
  return run_plscv_host(x, labels, constrain, metal_options, PLSMode::PLS_LDA, Backend::Metal);
#else
  (void)x;
  (void)labels;
  (void)constrain;
  (void)options;
  throw std::runtime_error("PLSLDACV_METAL requires an Apple Metal build.");
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
  std::vector<float> mean;
  std::vector<float> scale;
  train_center_scale_float(train, train_rows, options.center, options.scale, mean, scale);
  DenseF x_train = subset_scale_float(train, train_rows, mean, scale);
  DenseF x_test = subset_scale_float(test, test_rows, mean, scale);
  const int requested = options.fixed_components > 0 ? options.fixed_components : options.max_components;
  const int ncomp = std::max(1, std::min({requested, x_train.cols, std::max(1, x_train.rows - 1)}));
  PLSFitF fit = fit_pls_components_labels_float(x_train, labels, classes, ncomp);
  DenseF t_train = transform_pls_scores_float(x_train, fit, fit.weights.cols);
  DenseF t_test = transform_pls_scores_float(x_test, fit, fit.weights.cols);
  return predict_pls_lda_float(t_train, labels, t_test, classes, fit.weights.cols);
}

std::vector<int> PLSLDAPredict_CUDA(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options
) {
#if defined(KODAMA_ENABLE_CUDA)
  return plslda_predict_cuda(train, labels, test, options);
#else
  (void)train;
  (void)labels;
  (void)test;
  (void)options;
  throw std::runtime_error("PLSLDAPredict_CUDA requires a CUDA/cuBLAS build.");
#endif
}

std::vector<int> PLSLDAPredict_METAL(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options
) {
#if defined(KODAMA_ENABLE_METAL)
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
  std::vector<float> mean;
  std::vector<float> scale;
  train_center_scale_float(train, train_rows, options.center, options.scale, mean, scale);
  DenseF x_train = subset_scale_float(train, train_rows, mean, scale);
  DenseF x_test = subset_scale_float(test, test_rows, mean, scale);
  const int requested = options.fixed_components > 0 ? options.fixed_components : options.max_components;
  const int ncomp = std::max(1, std::min({requested, x_train.cols, std::max(1, x_train.rows - 1)}));
  PLSFitF fit = fit_pls_components_labels_metal_float(x_train, labels, classes, ncomp);
  DenseF t_train = transform_pls_scores_metal_float(x_train, fit, fit.weights.cols);
  DenseF t_test = transform_pls_scores_metal_float(x_test, fit, fit.weights.cols);
  return predict_pls_lda_float(t_train, labels, t_test, classes, fit.weights.cols);
#else
  (void)train;
  (void)labels;
  (void)test;
  (void)options;
  throw std::runtime_error("PLSLDAPredict_METAL requires an Apple Metal build.");
#endif
}

std::vector<int> PLSLDAPredict(
  MatrixView train,
  const std::vector<int>& labels,
  MatrixView test,
  const PLSOptions& options
) {
  if (options.backend == Backend::CUDA) return PLSLDAPredict_CUDA(train, labels, test, options);
  if (options.backend == Backend::Metal) return PLSLDAPredict_METAL(train, labels, test, options);
  return PLSLDAPredict_CPU(train, labels, test, options);
}

}  // namespace kodama
