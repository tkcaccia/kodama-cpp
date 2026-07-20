// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT
//
// Adapted from tkcaccia/fastPLS CUDA SIMPLS and relicensed for kodama-cpp
// under MIT by Stefano Cacciatore, the developer and sole code committer of
// the inspected fastPLS source history.
// Float32 label-aware CUDA SIMPLS for KODAMA PLS cross-validation.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cuda_runtime.h>

extern "C" void kodama_cuda_centered_label_crossprod_colmajor_float(
  const float* x_colmajor,
  const int* labels,
  const float* counts,
  int n,
  int p,
  int n_classes,
  float* s_colmajor,
  float* x_sums,
  cudaStream_t stream
);
extern "C" std::size_t kodama_cuda_centered_label_crossprod_colmajor_float_scratch_size(int n, int p, int n_classes);

namespace kodama_fastpls_cuda {

void check_cuda(cudaError_t code, const char* where) {
  if (code != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(code));
}

void check_cublas(cublasStatus_t code, const char* where) {
  if (code != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(where) + ": cublas call failed");
}

void check_curand(curandStatus_t code, const char* where) {
  if (code != CURAND_STATUS_SUCCESS) throw std::runtime_error(std::string(where) + ": curand call failed");
}

void check_cusolver(cusolverStatus_t code, const char* where) {
  if (code != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(std::string(where) + ": cusolver call failed");
}
int env_int_or(const char* key, int fallback, int lo, int hi) {
  const char* raw = std::getenv(key);
  if (raw == nullptr) return fallback;
  char* endptr = nullptr;
  long v = std::strtol(raw, &endptr, 10);
  if (endptr == raw) return fallback;
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return static_cast<int>(v);
}
bool runtime_available() {
  int n_devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&n_devices);
  return status == cudaSuccess && n_devices > 0;
}

class FloatDeviceArray {
 public:
  FloatDeviceArray() = default;
  explicit FloatDeviceArray(std::size_t n) { reset(n); }
  ~FloatDeviceArray() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  FloatDeviceArray(const FloatDeviceArray&) = delete;
  FloatDeviceArray& operator=(const FloatDeviceArray&) = delete;

  void reset(std::size_t n) {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = n;
    if (n > 0) check_cuda(cudaMalloc(&ptr_, n * sizeof(float)), "cudaMalloc(float SIMPLS)");
  }

  void ensure(std::size_t n) {
    if (n > size_) reset(n);
  }

  float* data() { return ptr_; }
  const float* data() const { return ptr_; }

 private:
  float* ptr_ = nullptr;
  std::size_t size_ = 0;
};

class IntDeviceArray {
 public:
  IntDeviceArray() = default;
  explicit IntDeviceArray(std::size_t n) { reset(n); }
  ~IntDeviceArray() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  IntDeviceArray(const IntDeviceArray&) = delete;
  IntDeviceArray& operator=(const IntDeviceArray&) = delete;

  void reset(std::size_t n) {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = n;
    if (n > 0) check_cuda(cudaMalloc(&ptr_, n * sizeof(int)), "cudaMalloc(int SIMPLS)");
  }

  void ensure(std::size_t n) {
    if (n > size_) reset(n);
  }

  int* data() { return ptr_; }
  const int* data() const { return ptr_; }

 private:
  int* ptr_ = nullptr;
  std::size_t size_ = 0;
};

struct FloatSimplsWorkspace {
  ~FloatSimplsWorkspace() {
    if (handle != nullptr) cublasDestroy(handle);
    if (solver_handle != nullptr) cusolverDnDestroy(solver_handle);
    if (rng_handle != nullptr) curandDestroyGenerator(rng_handle);
    if (stream_handle != nullptr) cudaStreamDestroy(stream_handle);
  }

  cudaStream_t stream() {
    if (stream_handle == nullptr) {
      check_cuda(cudaStreamCreateWithFlags(&stream_handle, cudaStreamNonBlocking), "cudaStreamCreateWithFlags(float SIMPLS)");
    }
    return stream_handle;
  }

  cublasHandle_t blas() {
    if (handle == nullptr) {
      check_cublas(cublasCreate(&handle), "cublasCreate(float SIMPLS)");
      check_cublas(cublasSetStream(handle, stream()), "cublasSetStream(float SIMPLS)");
    }
    return handle;
  }

  cusolverDnHandle_t solver() {
    if (solver_handle == nullptr) {
      check_cusolver(cusolverDnCreate(&solver_handle), "cusolverDnCreate(float SIMPLS)");
      check_cusolver(cusolverDnSetStream(solver_handle, stream()), "cusolverDnSetStream(float SIMPLS)");
    }
    return solver_handle;
  }

  curandGenerator_t rng() {
    if (rng_handle == nullptr) {
      check_curand(curandCreateGenerator(&rng_handle, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator(float SIMPLS)");
      check_curand(curandSetStream(rng_handle, stream()), "curandSetStream(float SIMPLS)");
    }
    return rng_handle;
  }

  cublasHandle_t handle = nullptr;
  cusolverDnHandle_t solver_handle = nullptr;
  curandGenerator_t rng_handle = nullptr;
  cudaStream_t stream_handle = nullptr;
  FloatDeviceArray dX;
  FloatDeviceArray dY;
  FloatDeviceArray dS;
  FloatDeviceArray dS0;
  FloatDeviceArray dA;
  FloatDeviceArray dA0;
  FloatDeviceArray dW;
  FloatDeviceArray dP;
  FloatDeviceArray dQ;
  FloatDeviceArray dRR;
  FloatDeviceArray dQQ;
  FloatDeviceArray dVV;
  FloatDeviceArray dUblock;
  FloatDeviceArray dBsmall;
  FloatDeviceArray dGram;
  FloatDeviceArray dOmega;
  FloatDeviceArray dTau;
  FloatDeviceArray dEvals;
  FloatDeviceArray dWork;
  FloatDeviceArray dCoeff;
  FloatDeviceArray dCur;
  FloatDeviceArray dNext;
  FloatDeviceArray dT;
  FloatDeviceArray dPvec;
  FloatDeviceArray dRvec;
  FloatDeviceArray dQvec;
  FloatDeviceArray dVS;
  FloatDeviceArray dZmat;
  FloatDeviceArray dCounts;
  FloatDeviceArray dXsum;
  FloatDeviceArray dXtX;
  IntDeviceArray dLabels;
  IntDeviceArray dInfo;
};

thread_local FloatSimplsWorkspace g_float_workspace;

void orthogonalize_float(cublasHandle_t handle, int n, float* vec, const FloatDeviceArray& basis, int n_basis) {
  const float minus_one = -1.0f;
  for (int j = 0; j < n_basis; ++j) {
    float coeff = 0.0f;
    check_cublas(cublasSdot(handle, n, vec, 1, basis.data() + static_cast<std::size_t>(j) * n, 1, &coeff), "cublasSdot(float SIMPLS orthogonalize)");
    const float alpha = minus_one * coeff;
    check_cublas(cublasSaxpy(handle, n, &alpha, basis.data() + static_cast<std::size_t>(j) * n, 1, vec, 1), "cublasSaxpy(float SIMPLS orthogonalize)");
  }
}

bool normalize_float_vector(cublasHandle_t handle, int n, float* vec) {
  float nrm = 0.0f;
  check_cublas(cublasSnrm2(handle, n, vec, 1, &nrm), "cublasSnrm2(float SIMPLS)");
  if (!std::isfinite(nrm) || nrm <= 1.0e-20f) return false;
  const float inv = 1.0f / nrm;
  check_cublas(cublasSscal(handle, n, &inv, vec, 1), "cublasSscal(float SIMPLS)");
  return true;
}

void set_unit_axis_float(float* d_vec, int n, int axis) {
  std::vector<float> host(static_cast<std::size_t>(n), 0.0f);
  host[static_cast<std::size_t>(std::max(0, std::min(axis, n - 1)))] = 1.0f;
  check_cuda(cudaMemcpy(d_vec, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(float SIMPLS unit axis)");
}

std::size_t padded_random_elems_float(int rows, int cols) {
  std::size_t elems = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if ((elems % 2U) != 0U) ++elems;
  return elems;
}

void orthonormalize_float_qr_inplace(FloatSimplsWorkspace& ws, int rows, int cols, float* d_mat) {
  cublasHandle_t blas = ws.blas();
  if (cols == 1) {
    if (!normalize_float_vector(blas, rows, d_mat)) set_unit_axis_float(d_mat, rows, 0);
    return;
  }

  cusolverDnHandle_t solver = ws.solver();
  ws.dTau.ensure(static_cast<std::size_t>(cols));
  ws.dInfo.ensure(1);

  int lwork_geqrf = 0;
  int lwork_orgqr = 0;
  check_cusolver(cusolverDnSgeqrf_bufferSize(solver, rows, cols, d_mat, rows, &lwork_geqrf), "cusolverDnSgeqrf_bufferSize(float SIMPLS)");
  check_cusolver(cusolverDnSorgqr_bufferSize(solver, rows, cols, cols, d_mat, rows, ws.dTau.data(), &lwork_orgqr), "cusolverDnSorgqr_bufferSize(float SIMPLS)");
  const int lwork = std::max(lwork_geqrf, lwork_orgqr);
  ws.dWork.ensure(static_cast<std::size_t>(std::max(lwork, 1)));

  check_cusolver(cusolverDnSgeqrf(solver, rows, cols, d_mat, rows, ws.dTau.data(), ws.dWork.data(), lwork_geqrf, ws.dInfo.data()), "cusolverDnSgeqrf(float SIMPLS)");
  int info = 0;
  check_cuda(cudaMemcpyAsync(&info, ws.dInfo.data(), sizeof(int), cudaMemcpyDeviceToHost, ws.stream()),
             "cudaMemcpyAsync(float SIMPLS geqrf info)");
  check_cuda(cudaStreamSynchronize(ws.stream()), "cudaStreamSynchronize(float SIMPLS geqrf info)");
  if (info != 0) throw std::runtime_error("cusolverDnSgeqrf returned non-zero info in float SIMPLS");

  check_cusolver(cusolverDnSorgqr(solver, rows, cols, cols, d_mat, rows, ws.dTau.data(), ws.dWork.data(), lwork_orgqr, ws.dInfo.data()), "cusolverDnSorgqr(float SIMPLS)");
  check_cuda(cudaMemcpyAsync(&info, ws.dInfo.data(), sizeof(int), cudaMemcpyDeviceToHost, ws.stream()),
             "cudaMemcpyAsync(float SIMPLS orgqr info)");
  check_cuda(cudaStreamSynchronize(ws.stream()), "cudaStreamSynchronize(float SIMPLS orgqr info)");
  if (info != 0) throw std::runtime_error("cusolverDnSorgqr returned non-zero info in float SIMPLS");
}

void finalize_float_left_block(FloatSimplsWorkspace& ws, int l, int k) {
  if (l == 1) {
    const float one = 1.0f;
    check_cuda(cudaMemcpyAsync(ws.dOmega.data(), &one, sizeof(float), cudaMemcpyHostToDevice, ws.stream()),
               "cudaMemcpyAsync(float SIMPLS omega scalar)");
    return;
  }

  cusolverDnHandle_t solver = ws.solver();
  ws.dEvals.ensure(static_cast<std::size_t>(l));
  ws.dInfo.ensure(1);

  int lwork = 0;
  check_cusolver(
    cusolverDnSsyevd_bufferSize(
      solver,
      CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_UPPER,
      l,
      ws.dGram.data(),
      l,
      ws.dEvals.data(),
      &lwork
    ),
    "cusolverDnSsyevd_bufferSize(float SIMPLS)"
  );
  ws.dWork.ensure(static_cast<std::size_t>(std::max(lwork, 1)));
  check_cusolver(
    cusolverDnSsyevd(
      solver,
      CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_UPPER,
      l,
      ws.dGram.data(),
      l,
      ws.dEvals.data(),
      ws.dWork.data(),
      lwork,
      ws.dInfo.data()
    ),
    "cusolverDnSsyevd(float SIMPLS)"
  );
  int info = 0;
  check_cuda(cudaMemcpyAsync(&info, ws.dInfo.data(), sizeof(int), cudaMemcpyDeviceToHost, ws.stream()),
             "cudaMemcpyAsync(float SIMPLS syevd info)");
  check_cuda(cudaStreamSynchronize(ws.stream()), "cudaStreamSynchronize(float SIMPLS syevd info)");
  if (info != 0) throw std::runtime_error("cusolverDnSsyevd returned non-zero info in float SIMPLS");

  cublasHandle_t blas = ws.blas();
  for (int j = 0; j < k; ++j) {
    const int src_col = l - 1 - j;
    check_cublas(
      cublasScopy(blas, l, ws.dGram.data() + static_cast<std::size_t>(src_col) * static_cast<std::size_t>(l), 1,
                  ws.dOmega.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(l), 1),
      "cublasScopy(float SIMPLS Uhat)"
    );
  }
}

bool simpls_fit_cuda_float_materialized(
  const float* x_colmajor,
  int n,
  int p,
  const float* y_colmajor,
  int m,
  int max_ncomp,
  float* rr_colmajor,
  float* qq_colmajor
) {
  try {
    FloatSimplsWorkspace& ws = g_float_workspace;
    cublasHandle_t handle = ws.blas();
    ws.dX.ensure(static_cast<std::size_t>(n) * static_cast<std::size_t>(p));
    ws.dY.ensure(static_cast<std::size_t>(n) * static_cast<std::size_t>(m));
    ws.dS.ensure(static_cast<std::size_t>(p) * static_cast<std::size_t>(m));
    ws.dW.ensure(static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp));
    ws.dP.ensure(static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp));
    ws.dQ.ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(max_ncomp));
    ws.dCur.ensure(static_cast<std::size_t>(p));
    ws.dNext.ensure(static_cast<std::size_t>(p));
    ws.dT.ensure(static_cast<std::size_t>(n));
    ws.dQvec.ensure(static_cast<std::size_t>(m));
    ws.dVS.ensure(static_cast<std::size_t>(m));

    FloatDeviceArray& dX = ws.dX;
    FloatDeviceArray& dY = ws.dY;
    FloatDeviceArray& dS = ws.dS;
    FloatDeviceArray& dW = ws.dW;
    FloatDeviceArray& dP = ws.dP;
    FloatDeviceArray& dQ = ws.dQ;
    FloatDeviceArray& dCur = ws.dCur;
    FloatDeviceArray& dNext = ws.dNext;
    FloatDeviceArray& dT = ws.dT;
    FloatDeviceArray& dQvec = ws.dQvec;
    FloatDeviceArray& dVS = ws.dVS;

    check_cuda(cudaMemcpy(dX.data(), x_colmajor, sizeof(float) * static_cast<std::size_t>(n) * static_cast<std::size_t>(p), cudaMemcpyHostToDevice), "cudaMemcpy(float SIMPLS X)");
    check_cuda(cudaMemcpy(dY.data(), y_colmajor, sizeof(float) * static_cast<std::size_t>(n) * static_cast<std::size_t>(m), cudaMemcpyHostToDevice), "cudaMemcpy(float SIMPLS Y)");
    check_cuda(cudaMemset(dW.data(), 0, sizeof(float) * static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp)), "cudaMemset(float SIMPLS W)");
    check_cuda(cudaMemset(dP.data(), 0, sizeof(float) * static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp)), "cudaMemset(float SIMPLS P)");
    check_cuda(cudaMemset(dQ.data(), 0, sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(max_ncomp)), "cudaMemset(float SIMPLS Q)");

    const float one = 1.0f;
    const float zero = 0.0f;
    const float minus_one = -1.0f;
    check_cublas(
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, m, n, &one, dX.data(), n, dY.data(), n, &zero, dS.data(), p),
      "cublasSgemm(float SIMPLS X'Y)"
    );

    const int power_iterations = env_int_or("KODAMA_FLOAT_SIMPLS_ITERS", 2, 1, 64);
    std::vector<float> init(static_cast<std::size_t>(p), 1.0f / std::sqrt(static_cast<float>(std::max(1, p))));

    for (int a = 0; a < max_ncomp; ++a) {
      check_cuda(cudaMemcpy(dCur.data(), init.data(), init.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(float SIMPLS init)");
      orthogonalize_float(handle, p, dCur.data(), dP, a);
      if (!normalize_float_vector(handle, p, dCur.data())) set_unit_axis_float(dCur.data(), p, a % p);

      for (int iter = 0; iter < power_iterations; ++iter) {
        check_cublas(
          cublasSgemv(handle, CUBLAS_OP_T, p, m, &one, dS.data(), p, dCur.data(), 1, &zero, dVS.data(), 1),
          "cublasSgemv(float SIMPLS power S't)"
        );
        check_cublas(
          cublasSgemv(handle, CUBLAS_OP_N, p, m, &one, dS.data(), p, dVS.data(), 1, &zero, dNext.data(), 1),
          "cublasSgemv(float SIMPLS power SSt)"
        );
        orthogonalize_float(handle, p, dNext.data(), dP, a);
        if (!normalize_float_vector(handle, p, dNext.data())) break;
        check_cuda(cudaMemcpy(dCur.data(), dNext.data(), sizeof(float) * static_cast<std::size_t>(p), cudaMemcpyDeviceToDevice), "cudaMemcpy(float SIMPLS next)");
      }

      check_cublas(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, p, &one, dX.data(), n, dCur.data(), p, &zero, dT.data(), n),
        "cublasSgemm(float SIMPLS t=Xr)"
      );
      float tnorm = 0.0f;
      check_cublas(cublasSnrm2(handle, n, dT.data(), 1, &tnorm), "cublasSnrm2(float SIMPLS t)");
      if (!std::isfinite(tnorm) || tnorm <= 1.0e-20f) {
        if (handle != nullptr) cublasDestroy(handle);
        return false;
      }
      const float inv_tnorm = 1.0f / tnorm;
      check_cublas(cublasSscal(handle, n, &inv_tnorm, dT.data(), 1), "cublasSscal(float SIMPLS t)");
      check_cublas(cublasSscal(handle, p, &inv_tnorm, dCur.data(), 1), "cublasSscal(float SIMPLS r)");

      check_cublas(
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, 1, n, &one, dX.data(), n, dT.data(), n, &zero, dNext.data(), p),
        "cublasSgemm(float SIMPLS p=X't)"
      );
      check_cublas(
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, 1, n, &one, dY.data(), n, dT.data(), n, &zero, dQvec.data(), m),
        "cublasSgemm(float SIMPLS q=Y't)"
      );

      orthogonalize_float(handle, p, dNext.data(), dP, a);
      orthogonalize_float(handle, p, dNext.data(), dP, a);
      if (!normalize_float_vector(handle, p, dNext.data())) {
        set_unit_axis_float(dNext.data(), p, a % p);
        orthogonalize_float(handle, p, dNext.data(), dP, a);
        if (!normalize_float_vector(handle, p, dNext.data())) {
          if (handle != nullptr) cublasDestroy(handle);
          return false;
        }
      }

      check_cublas(cublasScopy(handle, p, dCur.data(), 1, dW.data() + static_cast<std::size_t>(a) * p, 1), "cublasScopy(float SIMPLS W)");
      check_cublas(cublasScopy(handle, p, dNext.data(), 1, dP.data() + static_cast<std::size_t>(a) * p, 1), "cublasScopy(float SIMPLS P)");
      check_cublas(cublasScopy(handle, m, dQvec.data(), 1, dQ.data() + static_cast<std::size_t>(a) * m, 1), "cublasScopy(float SIMPLS Q)");

      check_cublas(
        cublasSgemv(handle, CUBLAS_OP_T, p, m, &one, dS.data(), p, dNext.data(), 1, &zero, dVS.data(), 1),
        "cublasSgemv(float SIMPLS vS)"
      );
      check_cublas(
        cublasSger(handle, p, m, &minus_one, dNext.data(), 1, dVS.data(), 1, dS.data(), p),
        "cublasSger(float SIMPLS deflate)"
      );
    }

    check_cuda(cudaMemcpy(rr_colmajor, dW.data(), sizeof(float) * static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp), cudaMemcpyDeviceToHost), "cudaMemcpy(float SIMPLS RR)");
    check_cuda(cudaMemcpy(qq_colmajor, dQ.data(), sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(max_ncomp), cudaMemcpyDeviceToHost), "cudaMemcpy(float SIMPLS QQ)");
    return true;
  } catch (...) {
    return false;
  }
}

bool simpls_fit_cuda_float_crossprod(
  const float* x_colmajor,
  int n,
  int p,
  const float* s_colmajor,
  int m,
  int max_ncomp,
  float* rr_colmajor,
  float* qq_colmajor
) {
  try {
    FloatSimplsWorkspace& ws = g_float_workspace;
    cublasHandle_t handle = ws.blas();
    ws.dX.ensure(static_cast<std::size_t>(n) * static_cast<std::size_t>(p));
    ws.dS.ensure(static_cast<std::size_t>(p) * static_cast<std::size_t>(m));
    ws.dS0.ensure(static_cast<std::size_t>(p) * static_cast<std::size_t>(m));
    ws.dW.ensure(static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp));
    ws.dP.ensure(static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp));
    ws.dQ.ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(max_ncomp));
    ws.dCur.ensure(static_cast<std::size_t>(p));
    ws.dNext.ensure(static_cast<std::size_t>(p));
    ws.dT.ensure(static_cast<std::size_t>(n));
    ws.dQvec.ensure(static_cast<std::size_t>(m));
    ws.dVS.ensure(static_cast<std::size_t>(m));

    FloatDeviceArray& dX = ws.dX;
    FloatDeviceArray& dS = ws.dS;
    FloatDeviceArray& dS0 = ws.dS0;
    FloatDeviceArray& dW = ws.dW;
    FloatDeviceArray& dP = ws.dP;
    FloatDeviceArray& dQ = ws.dQ;
    FloatDeviceArray& dCur = ws.dCur;
    FloatDeviceArray& dNext = ws.dNext;
    FloatDeviceArray& dT = ws.dT;
    FloatDeviceArray& dQvec = ws.dQvec;
    FloatDeviceArray& dVS = ws.dVS;

    const std::size_t x_size = static_cast<std::size_t>(n) * static_cast<std::size_t>(p);
    const std::size_t s_size = static_cast<std::size_t>(p) * static_cast<std::size_t>(m);
    check_cuda(cudaMemcpy(dX.data(), x_colmajor, sizeof(float) * x_size, cudaMemcpyHostToDevice), "cudaMemcpy(float SIMPLS X)");
    check_cuda(cudaMemcpy(dS.data(), s_colmajor, sizeof(float) * s_size, cudaMemcpyHostToDevice), "cudaMemcpy(float SIMPLS S)");
    check_cuda(cudaMemcpy(dS0.data(), s_colmajor, sizeof(float) * s_size, cudaMemcpyHostToDevice), "cudaMemcpy(float SIMPLS S0)");
    check_cuda(cudaMemset(dW.data(), 0, sizeof(float) * static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp)), "cudaMemset(float SIMPLS W)");
    check_cuda(cudaMemset(dP.data(), 0, sizeof(float) * static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp)), "cudaMemset(float SIMPLS P)");
    check_cuda(cudaMemset(dQ.data(), 0, sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(max_ncomp)), "cudaMemset(float SIMPLS Q)");

    const float one = 1.0f;
    const float zero = 0.0f;
    const float minus_one = -1.0f;
    const int power_iterations = env_int_or("KODAMA_FLOAT_SIMPLS_ITERS", 2, 1, 64);
    std::vector<float> init(static_cast<std::size_t>(p), 1.0f / std::sqrt(static_cast<float>(std::max(1, p))));

    for (int a = 0; a < max_ncomp; ++a) {
      check_cuda(cudaMemcpy(dCur.data(), init.data(), init.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(float SIMPLS init)");
      orthogonalize_float(handle, p, dCur.data(), dP, a);
      if (!normalize_float_vector(handle, p, dCur.data())) set_unit_axis_float(dCur.data(), p, a % p);

      for (int iter = 0; iter < power_iterations; ++iter) {
        check_cublas(
          cublasSgemv(handle, CUBLAS_OP_T, p, m, &one, dS.data(), p, dCur.data(), 1, &zero, dVS.data(), 1),
          "cublasSgemv(float SIMPLS power S't)"
        );
        check_cublas(
          cublasSgemv(handle, CUBLAS_OP_N, p, m, &one, dS.data(), p, dVS.data(), 1, &zero, dNext.data(), 1),
          "cublasSgemv(float SIMPLS power SSt)"
        );
        orthogonalize_float(handle, p, dNext.data(), dP, a);
        if (!normalize_float_vector(handle, p, dNext.data())) break;
        check_cuda(cudaMemcpy(dCur.data(), dNext.data(), sizeof(float) * static_cast<std::size_t>(p), cudaMemcpyDeviceToDevice), "cudaMemcpy(float SIMPLS next)");
      }

      check_cublas(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, p, &one, dX.data(), n, dCur.data(), p, &zero, dT.data(), n),
        "cublasSgemm(float SIMPLS t=Xr)"
      );
      float tnorm = 0.0f;
      check_cublas(cublasSnrm2(handle, n, dT.data(), 1, &tnorm), "cublasSnrm2(float SIMPLS t)");
      if (!std::isfinite(tnorm) || tnorm <= 1.0e-20f) return false;
      const float inv_tnorm = 1.0f / tnorm;
      check_cublas(cublasSscal(handle, n, &inv_tnorm, dT.data(), 1), "cublasSscal(float SIMPLS t)");
      check_cublas(cublasSscal(handle, p, &inv_tnorm, dCur.data(), 1), "cublasSscal(float SIMPLS r)");

      check_cublas(
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, 1, n, &one, dX.data(), n, dT.data(), n, &zero, dNext.data(), p),
        "cublasSgemm(float SIMPLS p=X't)"
      );
      check_cublas(
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, 1, p, &one, dS0.data(), p, dCur.data(), p, &zero, dQvec.data(), m),
        "cublasSgemm(float SIMPLS q=S0'r)"
      );

      orthogonalize_float(handle, p, dNext.data(), dP, a);
      orthogonalize_float(handle, p, dNext.data(), dP, a);
      if (!normalize_float_vector(handle, p, dNext.data())) {
        set_unit_axis_float(dNext.data(), p, a % p);
        orthogonalize_float(handle, p, dNext.data(), dP, a);
        if (!normalize_float_vector(handle, p, dNext.data())) return false;
      }

      check_cublas(cublasScopy(handle, p, dCur.data(), 1, dW.data() + static_cast<std::size_t>(a) * p, 1), "cublasScopy(float SIMPLS W)");
      check_cublas(cublasScopy(handle, p, dNext.data(), 1, dP.data() + static_cast<std::size_t>(a) * p, 1), "cublasScopy(float SIMPLS P)");
      check_cublas(cublasScopy(handle, m, dQvec.data(), 1, dQ.data() + static_cast<std::size_t>(a) * m, 1), "cublasScopy(float SIMPLS Q)");

      check_cublas(
        cublasSgemv(handle, CUBLAS_OP_T, p, m, &one, dS.data(), p, dNext.data(), 1, &zero, dVS.data(), 1),
        "cublasSgemv(float SIMPLS vS)"
      );
      check_cublas(
        cublasSger(handle, p, m, &minus_one, dNext.data(), 1, dVS.data(), 1, dS.data(), p),
        "cublasSger(float SIMPLS deflate)"
      );
    }

    check_cuda(cudaMemcpy(rr_colmajor, dW.data(), sizeof(float) * static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp), cudaMemcpyDeviceToHost), "cudaMemcpy(float SIMPLS RR)");
    check_cuda(cudaMemcpy(qq_colmajor, dQ.data(), sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(max_ncomp), cudaMemcpyDeviceToHost), "cudaMemcpy(float SIMPLS QQ)");
    return true;
  } catch (...) {
    return false;
  }
}

bool simpls_fit_cuda_float_fast_crossprod(
  const float* x_colmajor,
  int n,
  int p,
  const float* s_colmajor,
  int m,
  int max_ncomp,
  float* rr_colmajor,
  float* qq_colmajor,
  const int* labels_1based = nullptr,
  const float* class_counts = nullptr,
  const float* xtx_colmajor = nullptr
) {
  try {
    FloatSimplsWorkspace& ws = g_float_workspace;
    cudaStream_t stream = ws.stream();
    cublasHandle_t blas = ws.blas();
    curandGenerator_t rng = ws.rng();
    const bool use_label_crossprod = s_colmajor == nullptr;

    const std::size_t x_size = static_cast<std::size_t>(n) * static_cast<std::size_t>(p);
    const std::size_t s_size = static_cast<std::size_t>(p) * static_cast<std::size_t>(m);
    ws.dX.ensure(x_size);
    ws.dA.ensure(s_size);
    ws.dA0.ensure(s_size);
    ws.dRR.ensure(static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp));
    ws.dQQ.ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(max_ncomp));
    ws.dVV.ensure(static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp));
    ws.dT.ensure(static_cast<std::size_t>(n));
    ws.dPvec.ensure(static_cast<std::size_t>(p));
    ws.dQvec.ensure(static_cast<std::size_t>(m));
    ws.dRvec.ensure(static_cast<std::size_t>(p));
    ws.dVS.ensure(static_cast<std::size_t>(m));
    ws.dCoeff.ensure(static_cast<std::size_t>(std::max(max_ncomp, 1)));
    if (xtx_colmajor != nullptr) {
      ws.dXtX.ensure(static_cast<std::size_t>(p) * static_cast<std::size_t>(p));
    }
    if (use_label_crossprod) {
      if (labels_1based == nullptr || class_counts == nullptr) return false;
      ws.dLabels.ensure(static_cast<std::size_t>(n));
      ws.dCounts.ensure(static_cast<std::size_t>(m));
      ws.dXsum.ensure(kodama_cuda_centered_label_crossprod_colmajor_float_scratch_size(n, p, m));
    }

    check_cuda(cudaMemcpyAsync(ws.dX.data(), x_colmajor, sizeof(float) * x_size, cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(float fast SIMPLS X)");
    if (xtx_colmajor != nullptr) {
      check_cuda(cudaMemcpyAsync(ws.dXtX.data(), xtx_colmajor, sizeof(float) * static_cast<std::size_t>(p) * static_cast<std::size_t>(p), cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(float fast SIMPLS X'X)");
    }
    if (use_label_crossprod) {
      check_cuda(cudaMemcpyAsync(ws.dLabels.data(), labels_1based, sizeof(int) * static_cast<std::size_t>(n), cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(float fast SIMPLS labels)");
      check_cuda(cudaMemcpyAsync(ws.dCounts.data(), class_counts, sizeof(float) * static_cast<std::size_t>(m), cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(float fast SIMPLS class counts)");
      kodama_cuda_centered_label_crossprod_colmajor_float(
        ws.dX.data(),
        ws.dLabels.data(),
        ws.dCounts.data(),
        n,
        p,
        m,
        ws.dA.data(),
        ws.dXsum.data(),
        stream
      );
      check_cuda(cudaGetLastError(), "kodama_cuda_centered_label_crossprod_colmajor_float");
      check_cuda(cudaMemcpyAsync(ws.dA0.data(), ws.dA.data(), sizeof(float) * s_size, cudaMemcpyDeviceToDevice, stream),
                 "cudaMemcpyAsync(float fast SIMPLS label S0)");
    } else {
      check_cuda(cudaMemcpyAsync(ws.dA.data(), s_colmajor, sizeof(float) * s_size, cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(float fast SIMPLS S)");
      check_cuda(cudaMemcpyAsync(ws.dA0.data(), s_colmajor, sizeof(float) * s_size, cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(float fast SIMPLS S0)");
    }
    check_cuda(cudaMemsetAsync(ws.dRR.data(), 0, sizeof(float) * static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp), stream),
               "cudaMemsetAsync(float fast SIMPLS RR)");
    check_cuda(cudaMemsetAsync(ws.dQQ.data(), 0, sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(max_ncomp), stream),
               "cudaMemsetAsync(float fast SIMPLS QQ)");
    check_cuda(cudaMemsetAsync(ws.dVV.data(), 0, sizeof(float) * static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp), stream),
               "cudaMemsetAsync(float fast SIMPLS VV)");

    const float one = 1.0f;
    const float zero = 0.0f;
    const float minus_one = -1.0f;
    const int refresh_block = env_int_or("FASTPLS_FAST_BLOCK", 1, 1, 16);
    const int power_iters = env_int_or("FASTPLS_FAST_INC_ITERS", 2, 1, 6);
    const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);

    bool has_rr_prev = false;
    int a = 0;
    while (a < max_ncomp) {
      const int remaining = max_ncomp - a;
      const int k_block = std::min(refresh_block, remaining);
      const int l = k_block;

      ws.dY.ensure(padded_random_elems_float(p, l));
      ws.dZmat.ensure(static_cast<std::size_t>(m) * static_cast<std::size_t>(l));
      ws.dUblock.ensure(static_cast<std::size_t>(p) * static_cast<std::size_t>(k_block));
      ws.dBsmall.ensure(static_cast<std::size_t>(l) * static_cast<std::size_t>(m));
      ws.dGram.ensure(static_cast<std::size_t>(l) * static_cast<std::size_t>(l));
      ws.dOmega.ensure(static_cast<std::size_t>(l) * static_cast<std::size_t>(k_block));

      if (has_rr_prev && l == 1) {
        check_cublas(cublasScopy(blas, p, ws.dRvec.data(), 1, ws.dY.data(), 1), "cublasScopy(float fast SIMPLS rr warm start)");
      } else {
        check_curand(curandSetPseudoRandomGeneratorSeed(rng, static_cast<unsigned long long>(1 + a)), "curandSetPseudoRandomGeneratorSeed(float fast SIMPLS)");
        check_curand(curandGenerateNormal(rng, ws.dY.data(), padded_random_elems_float(p, l), 0.0f, 1.0f), "curandGenerateNormal(float fast SIMPLS Y0)");
        if (has_rr_prev) {
          check_cublas(cublasScopy(blas, p, ws.dRvec.data(), 1, ws.dY.data(), 1), "cublasScopy(float fast SIMPLS rr warm start)");
        }
      }

      for (int iter = 0; iter < power_iters; ++iter) {
        if (l == 1) {
          check_cublas(
            cublasSgemv(blas, CUBLAS_OP_T, p, m, &one, ws.dA.data(), p, ws.dY.data(), 1, &zero, ws.dZmat.data(), 1),
            "cublasSgemv(float fast SIMPLS S'Y)"
          );
          check_cublas(
            cublasSgemv(blas, CUBLAS_OP_N, p, m, &one, ws.dA.data(), p, ws.dZmat.data(), 1, &zero, ws.dY.data(), 1),
            "cublasSgemv(float fast SIMPLS SZ)"
          );
        } else {
          check_cublas(
            cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, m, l, p, &one, ws.dA.data(), p, ws.dY.data(), p, &zero, ws.dZmat.data(), m),
            "cublasSgemm(float fast SIMPLS S'Y)"
          );
          check_cublas(
            cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, p, l, m, &one, ws.dA.data(), p, ws.dZmat.data(), m, &zero, ws.dY.data(), p),
            "cublasSgemm(float fast SIMPLS SZ)"
          );
        }
      }

      orthonormalize_float_qr_inplace(ws, p, l, ws.dY.data());
      if (l == 1) {
        check_cublas(
          cublasScopy(blas, p, ws.dY.data(), 1, ws.dUblock.data(), 1),
          "cublasScopy(float fast SIMPLS rank-one Ublock)"
        );
      } else {
        check_cublas(
          cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, l, m, p, &one, ws.dY.data(), p, ws.dA.data(), p, &zero, ws.dBsmall.data(), l),
          "cublasSgemm(float fast SIMPLS Bsmall)"
        );
        check_cublas(
          cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, l, l, m, &one, ws.dBsmall.data(), l, ws.dBsmall.data(), l, &zero, ws.dGram.data(), l),
          "cublasSgemm(float fast SIMPLS Gram)"
        );
        finalize_float_left_block(ws, l, k_block);
        check_cublas(
          cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, p, k_block, l, &one, ws.dY.data(), p, ws.dOmega.data(), l, &zero, ws.dUblock.data(), p),
          "cublasSgemm(float fast SIMPLS Ublock)"
        );
      }

      for (int j = 0; j < k_block && a < max_ncomp; ++j) {
        check_cublas(cublasScopy(blas, p, ws.dUblock.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(p), 1, ws.dRvec.data(), 1),
                     "cublasScopy(float fast SIMPLS r)");
      if (xtx_colmajor != nullptr) {
        check_cublas(
          cublasSgemv(blas, CUBLAS_OP_N, p, p, &one, ws.dXtX.data(), p, ws.dRvec.data(), 1, &zero, ws.dPvec.data(), 1),
          "cublasSgemv(float fast SIMPLS p=X'Xr)"
        );
          float tnorm2 = 0.0f;
          check_cublas(cublasSdot(blas, p, ws.dRvec.data(), 1, ws.dPvec.data(), 1, &tnorm2), "cublasSdot(float fast SIMPLS r'X'Xr)");
          if (!std::isfinite(tnorm2) || tnorm2 <= 1.0e-20f) return false;
          const float inv_tnorm = 1.0f / std::sqrt(tnorm2);
          check_cublas(cublasSscal(blas, p, &inv_tnorm, ws.dRvec.data(), 1), "cublasSscal(float fast SIMPLS gram r)");
          check_cublas(cublasSscal(blas, p, &inv_tnorm, ws.dPvec.data(), 1), "cublasSscal(float fast SIMPLS gram p)");
        } else {
          check_cublas(
            cublasSgemv(blas, CUBLAS_OP_N, n, p, &one, ws.dX.data(), n, ws.dRvec.data(), 1, &zero, ws.dT.data(), 1),
            "cublasSgemv(float fast SIMPLS t=Xr)"
          );
          float tnorm = 0.0f;
          check_cublas(cublasSnrm2(blas, n, ws.dT.data(), 1, &tnorm), "cublasSnrm2(float fast SIMPLS t)");
          if (!std::isfinite(tnorm) || tnorm <= 1.0e-20f) return false;
          const float inv_tnorm = 1.0f / tnorm;
          check_cublas(cublasSscal(blas, n, &inv_tnorm, ws.dT.data(), 1), "cublasSscal(float fast SIMPLS t)");
          check_cublas(cublasSscal(blas, p, &inv_tnorm, ws.dRvec.data(), 1), "cublasSscal(float fast SIMPLS r)");

          check_cublas(
            cublasSgemv(blas, CUBLAS_OP_T, n, p, &one, ws.dX.data(), n, ws.dT.data(), 1, &zero, ws.dPvec.data(), 1),
            "cublasSgemv(float fast SIMPLS p=X't)"
          );
        }
        check_cublas(
          cublasSgemv(blas, CUBLAS_OP_T, p, m, &one, ws.dA0.data(), p, ws.dRvec.data(), 1, &zero, ws.dQvec.data(), 1),
          "cublasSgemv(float fast SIMPLS q=S0'r)"
        );

        if (a > 0) {
          check_cublas(
            cublasSgemv(blas, CUBLAS_OP_T, p, a, &one, ws.dVV.data(), p, ws.dPvec.data(), 1, &zero, ws.dCoeff.data(), 1),
            "cublasSgemv(float fast SIMPLS V'p)"
          );
          check_cublas(
            cublasSgemv(blas, CUBLAS_OP_N, p, a, &minus_one, ws.dVV.data(), p, ws.dCoeff.data(), 1, &one, ws.dPvec.data(), 1),
            "cublasSgemv(float fast SIMPLS p-=Vcoeff)"
          );
          if (reorth_v == 1) {
            check_cublas(
              cublasSgemv(blas, CUBLAS_OP_T, p, a, &one, ws.dVV.data(), p, ws.dPvec.data(), 1, &zero, ws.dCoeff.data(), 1),
              "cublasSgemv(float fast SIMPLS V'p reorth)"
            );
            check_cublas(
              cublasSgemv(blas, CUBLAS_OP_N, p, a, &minus_one, ws.dVV.data(), p, ws.dCoeff.data(), 1, &one, ws.dPvec.data(), 1),
              "cublasSgemv(float fast SIMPLS p-=Vcoeff reorth)"
            );
          }
        }

        float vnorm = 0.0f;
        check_cublas(cublasSnrm2(blas, p, ws.dPvec.data(), 1, &vnorm), "cublasSnrm2(float fast SIMPLS v)");
        if (!std::isfinite(vnorm) || vnorm <= 1.0e-20f) return false;
        const float inv_vnorm = 1.0f / vnorm;
        check_cublas(cublasSscal(blas, p, &inv_vnorm, ws.dPvec.data(), 1), "cublasSscal(float fast SIMPLS v)");

        check_cublas(
          cublasSgemv(blas, CUBLAS_OP_T, p, m, &one, ws.dA.data(), p, ws.dPvec.data(), 1, &zero, ws.dVS.data(), 1),
          "cublasSgemv(float fast SIMPLS v'S)"
        );
        check_cublas(
          cublasSger(blas, p, m, &minus_one, ws.dPvec.data(), 1, ws.dVS.data(), 1, ws.dA.data(), p),
          "cublasSger(float fast SIMPLS deflate)"
        );

        check_cublas(cublasScopy(blas, p, ws.dRvec.data(), 1, ws.dRR.data() + static_cast<std::size_t>(a) * static_cast<std::size_t>(p), 1),
                     "cublasScopy(float fast SIMPLS RR)");
        check_cublas(cublasScopy(blas, m, ws.dQvec.data(), 1, ws.dQQ.data() + static_cast<std::size_t>(a) * static_cast<std::size_t>(m), 1),
                     "cublasScopy(float fast SIMPLS QQ)");
        check_cublas(cublasScopy(blas, p, ws.dPvec.data(), 1, ws.dVV.data() + static_cast<std::size_t>(a) * static_cast<std::size_t>(p), 1),
                     "cublasScopy(float fast SIMPLS VV)");
        has_rr_prev = true;
        ++a;
      }
    }

    if (rr_colmajor != nullptr) {
      check_cuda(cudaMemcpyAsync(rr_colmajor, ws.dRR.data(), sizeof(float) * static_cast<std::size_t>(p) * static_cast<std::size_t>(max_ncomp), cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync(float fast SIMPLS RR)");
    }
    if (qq_colmajor != nullptr) {
      check_cuda(cudaMemcpyAsync(qq_colmajor, ws.dQQ.data(), sizeof(float) * static_cast<std::size_t>(m) * static_cast<std::size_t>(max_ncomp), cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync(float fast SIMPLS QQ)");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(float fast SIMPLS output)");
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace kodama_fastpls_cuda

extern "C" bool kodama_fastpls_simpls_fit_cuda_float_crossprod(
  const float* x_colmajor,
  int n,
  int p,
  const float* s_colmajor,
  int m,
  int max_components,
  int seed,
  float* rr_colmajor,
  float* qq_colmajor
) {
  using namespace kodama_fastpls_cuda;
  (void)seed;
  if (!runtime_available()) return false;
  if (n < 1 || p < 1 || m < 1 || max_components < 1) return false;

  const int max_ncomp = std::min(max_components, std::min(p, std::max(1, n - 1)));
  if (env_int_or("KODAMA_FLOAT_SIMPLS_SIMPLE", 0, 0, 1) == 1) {
    return simpls_fit_cuda_float_crossprod(
      x_colmajor, n, p, s_colmajor, m, max_ncomp, rr_colmajor, qq_colmajor);
  }
  return simpls_fit_cuda_float_fast_crossprod(
    x_colmajor, n, p, s_colmajor, m, max_ncomp, rr_colmajor, qq_colmajor);
}

extern "C" bool kodama_fastpls_simpls_fit_cuda_float_labels(
  const float* x_colmajor,
  int n,
  int p,
  const int* labels_1based,
  const float* class_counts,
  int m,
  int max_components,
  int seed,
  float* rr_colmajor,
  float* qq_colmajor
) {
  using namespace kodama_fastpls_cuda;
  (void)seed;
  if (!runtime_available()) return false;
  if (n < 1 || p < 1 || m < 1 || max_components < 1) return false;
  if (labels_1based == nullptr || class_counts == nullptr) return false;

  const int max_ncomp = std::min(max_components, std::min(p, std::max(1, n - 1)));
  return simpls_fit_cuda_float_fast_crossprod(
    x_colmajor,
    n,
    p,
    nullptr,
    m,
    max_ncomp,
    rr_colmajor,
    qq_colmajor,
    labels_1based,
    class_counts
  );
}

extern "C" bool kodama_fastpls_simpls_fit_cuda_float_labels_gram(
  const float* x_colmajor,
  int n,
  int p,
  const float* xtx_colmajor,
  const int* labels_1based,
  const float* class_counts,
  int m,
  int max_components,
  int seed,
  float* rr_colmajor,
  float* qq_colmajor
) {
  using namespace kodama_fastpls_cuda;
  (void)seed;
  if (!runtime_available()) return false;
  if (n < 1 || p < 1 || m < 1 || max_components < 1) return false;
  if (x_colmajor == nullptr || xtx_colmajor == nullptr) return false;
  if (labels_1based == nullptr || class_counts == nullptr) return false;

  const int max_ncomp = std::min(max_components, std::min(p, std::max(1, n - 1)));
  return simpls_fit_cuda_float_fast_crossprod(
    x_colmajor,
    n,
    p,
    nullptr,
    m,
    max_ncomp,
    rr_colmajor,
    qq_colmajor,
    labels_1based,
    class_counts,
    xtx_colmajor
  );
}

extern "C" bool kodama_fastpls_simpls_fit_cuda_float(
  const float* x_colmajor,
  int n,
  int p,
  const float* y_colmajor,
  int m,
  int max_components,
  int seed,
  float* rr_colmajor,
  float* qq_colmajor
) {
  using namespace kodama_fastpls_cuda;
  (void)seed;
  if (!runtime_available()) return false;
  if (n < 1 || p < 1 || m < 1 || max_components < 1) return false;

  const int max_ncomp = std::min(max_components, std::min(p, std::max(1, n - 1)));
  return simpls_fit_cuda_float_materialized(
    x_colmajor, n, p, y_colmajor, m, max_ncomp, rr_colmajor, qq_colmajor);
}
