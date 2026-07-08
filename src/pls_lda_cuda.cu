#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

extern "C" {

__global__ void kodama_lda_label_sums_row_kernel(
  const double* t,
  const int* labels,
  int n,
  int kmax,
  int n_classes,
  double* sums
) {
  const int cls = blockIdx.x;
  const int j = blockIdx.y;
  if (cls >= n_classes || j >= kmax) return;

  extern __shared__ double partial[];
  double value = 0.0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (labels[i] == cls + 1) value += t[i * kmax + j];
  }
  partial[threadIdx.x] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) sums[cls * kmax + j] = partial[0];
}

__global__ void kodama_lda_means_row_kernel(double* means, const double* counts, int kmax, int n_classes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kmax * n_classes;
  if (idx >= total) return;
  const int cls = idx / kmax;
  const double cnt = counts[cls];
  means[idx] = (cnt > 0.0) ? means[idx] / cnt : 0.0;
}

__global__ void kodama_lda_pooled_col_kernel(
  double* pooled,
  const double* means,
  const double* counts,
  int n,
  int kmax,
  int n_classes
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kmax * kmax;
  if (idx >= total) return;
  const int r = idx % kmax;
  const int c = idx / kmax;
  double between = 0.0;
  for (int cls = 0; cls < n_classes; ++cls) {
    between += counts[cls] * means[cls * kmax + r] * means[cls * kmax + c];
  }
  const double df = fmax(1.0, static_cast<double>(n - n_classes));
  pooled[idx] = (pooled[idx] - between) / df;
}

__global__ void kodama_lda_copy_cov_kernel(const double* pooled, double* cov, int kmax, int kk) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kk * kk;
  if (idx >= total) return;
  const int r = idx % kk;
  const int c = idx / kk;
  cov[idx] = pooled[r + c * kmax];
}

__global__ void kodama_lda_add_ridge_kernel(double* cov, int kk, double ridge, double* lambda_out) {
  __shared__ double trace;
  if (threadIdx.x == 0) trace = 0.0;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < kk; ++i) trace += cov[i + i * kk];
  }
  __syncthreads();
  const double scale = isfinite(trace) && trace > 0.0 ? trace / static_cast<double>(kk) : 1.0;
  const double lambda = (isfinite(ridge) && ridge >= 0.0 ? ridge : 1e-8) * scale;
  if (threadIdx.x == 0) *lambda_out = lambda;
  for (int i = threadIdx.x; i < kk; i += blockDim.x) cov[i + i * kk] += lambda;
}

__global__ void kodama_lda_means_to_rhs_kernel(
  const double* means,
  double* rhs,
  int kmax,
  int kk,
  int n_classes
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kk * n_classes;
  if (idx >= total) return;
  const int j = idx % kk;
  const int cls = idx / kk;
  rhs[j + cls * kk] = means[cls * kmax + j];
}

__global__ void kodama_lda_finalize_linear_row_kernel(
  const double* rhs,
  const double* means,
  const double* counts,
  double* linear,
  double* constants,
  int n,
  int kmax,
  int kk,
  int n_classes
) {
  const int cls = blockIdx.x * blockDim.x + threadIdx.x;
  if (cls >= n_classes) return;
  double dot = 0.0;
  for (int j = 0; j < kk; ++j) {
    const double value = rhs[j + cls * kk];
    linear[cls * kk + j] = value;
    dot += means[cls * kmax + j] * value;
  }
  const double prior = fmax(counts[cls] / static_cast<double>(n), 2.2250738585072014e-308);
  constants[cls] = -0.5 * dot + log(prior);
}

__global__ void kodama_lda_score_argmax_row_kernel(
  const double* t,
  const double* linear,
  const double* constants,
  int* pred,
  int n,
  int kk,
  int n_classes
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  int best = 0;
  double best_value = -1.0 / 0.0;
  for (int cls = 0; cls < n_classes; ++cls) {
    double score = constants[cls];
    for (int j = 0; j < kk; ++j) score += t[row * kk + j] * linear[cls * kk + j];
    if (score > best_value) {
      best_value = score;
      best = cls;
    }
  }
  pred[row] = best + 1;
}

void kodama_cuda_lda_label_sums_row(const double* t, const int* labels, int n, int kmax, int n_classes, double* sums, cudaStream_t stream) {
  const int threads = 256;
  const dim3 blocks(n_classes, kmax);
  const size_t shared = sizeof(double) * static_cast<size_t>(threads);
  kodama_lda_label_sums_row_kernel<<<blocks, threads, shared, stream>>>(t, labels, n, kmax, n_classes, sums);
}

void kodama_cuda_lda_means_row(double* means, const double* counts, int kmax, int n_classes, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kmax * n_classes + threads - 1) / threads;
  kodama_lda_means_row_kernel<<<blocks, threads, 0, stream>>>(means, counts, kmax, n_classes);
}

void kodama_cuda_lda_pooled_col(double* pooled, const double* means, const double* counts, int n, int kmax, int n_classes, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kmax * kmax + threads - 1) / threads;
  kodama_lda_pooled_col_kernel<<<blocks, threads, 0, stream>>>(pooled, means, counts, n, kmax, n_classes);
}

void kodama_cuda_lda_copy_cov(const double* pooled, double* cov, int kmax, int kk, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kk * kk + threads - 1) / threads;
  kodama_lda_copy_cov_kernel<<<blocks, threads, 0, stream>>>(pooled, cov, kmax, kk);
}

void kodama_cuda_lda_add_ridge(double* cov, int kk, double ridge, double* lambda_out, cudaStream_t stream) {
  kodama_lda_add_ridge_kernel<<<1, 256, 0, stream>>>(cov, kk, ridge, lambda_out);
}

void kodama_cuda_lda_means_to_rhs(const double* means, double* rhs, int kmax, int kk, int n_classes, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kk * n_classes + threads - 1) / threads;
  kodama_lda_means_to_rhs_kernel<<<blocks, threads, 0, stream>>>(means, rhs, kmax, kk, n_classes);
}

void kodama_cuda_lda_finalize_linear_row(
  const double* rhs,
  const double* means,
  const double* counts,
  double* linear,
  double* constants,
  int n,
  int kmax,
  int kk,
  int n_classes,
  cudaStream_t stream
) {
  const int threads = 256;
  const int blocks = (n_classes + threads - 1) / threads;
  kodama_lda_finalize_linear_row_kernel<<<blocks, threads, 0, stream>>>(rhs, means, counts, linear, constants, n, kmax, kk, n_classes);
}

void kodama_cuda_lda_score_argmax_row(
  const double* t,
  const double* linear,
  const double* constants,
  int* pred,
  int n,
  int kk,
  int n_classes,
  cudaStream_t stream
) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  kodama_lda_score_argmax_row_kernel<<<blocks, threads, 0, stream>>>(t, linear, constants, pred, n, kk, n_classes);
}

__global__ void kodama_label_crossprod_accum_colmajor_float_kernel(
  const float* x,
  const int* labels,
  int n,
  int p,
  int n_classes,
  int row_tile,
  float* partial
) {
  const int col = static_cast<int>(blockIdx.x);
  const int tile = static_cast<int>(blockIdx.y);
  if (col >= p) return;

  extern __shared__ float class_sums[];
  for (int cls = static_cast<int>(threadIdx.x); cls < n_classes; cls += static_cast<int>(blockDim.x)) {
    class_sums[cls] = 0.0f;
  }
  __syncthreads();

  const int start = tile * row_tile;
  const int end = min(n, start + row_tile);
  for (int row = start + static_cast<int>(threadIdx.x); row < end; row += static_cast<int>(blockDim.x)) {
    const int cls = labels[row] - 1;
    if (cls >= 0 && cls < n_classes) {
      atomicAdd(class_sums + cls, x[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(n)]);
    }
  }
  __syncthreads();

  const size_t tile_base =
    (static_cast<size_t>(tile) * static_cast<size_t>(p) + static_cast<size_t>(col)) *
    static_cast<size_t>(n_classes);
  for (int cls = static_cast<int>(threadIdx.x); cls < n_classes; cls += static_cast<int>(blockDim.x)) {
    partial[tile_base + static_cast<size_t>(cls)] = class_sums[cls];
  }
}

__global__ void kodama_label_crossprod_reduce_colmajor_float_kernel(
  const float* partial,
  float* s,
  int row_tiles,
  int p,
  int n_classes
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = p * n_classes;
  if (idx >= total) return;
  const int col = idx % p;
  const int cls = idx / p;

  float sum = 0.0f;
  for (int tile = 0; tile < row_tiles; ++tile) {
    sum += partial[
      (static_cast<size_t>(tile) * static_cast<size_t>(p) + static_cast<size_t>(col)) *
      static_cast<size_t>(n_classes) + static_cast<size_t>(cls)
    ];
  }
  s[static_cast<size_t>(col) + static_cast<size_t>(cls) * static_cast<size_t>(p)] = sum;
}

__global__ void kodama_label_crossprod_center_colmajor_float_kernel(
  float* s,
  const float* counts,
  int n,
  int p,
  int n_classes
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = p * n_classes;
  if (idx >= total) return;
  const int col = idx % p;
  const int cls = idx / p;
  float x_sum = 0.0f;
  for (int c = 0; c < n_classes; ++c) {
    x_sum += s[col + c * p];
  }
  const float y_mean = n > 0 ? counts[cls] / static_cast<float>(n) : 0.0f;
  s[idx] -= y_mean * x_sum;
}

size_t kodama_cuda_centered_label_crossprod_colmajor_float_scratch_size(
  int n,
  int p,
  int n_classes
) {
  if (n < 1 || p < 1 || n_classes < 1) return 0;
  const int row_tile = 256;
  const int row_tiles = (n + row_tile - 1) / row_tile;
  return static_cast<size_t>(row_tiles) * static_cast<size_t>(p) * static_cast<size_t>(n_classes);
}

void kodama_cuda_centered_label_crossprod_colmajor_float(
  const float* x_colmajor,
  const int* labels,
  const float* counts,
  int n,
  int p,
  int n_classes,
  float* s_colmajor,
  float* x_sums,
  cudaStream_t stream
) {
  const int threads = 256;
  const int s_total = p * n_classes;
  const int row_tile = 256;
  const int row_tiles = (n + row_tile - 1) / row_tile;
  const dim3 sum_blocks(static_cast<unsigned int>(p), static_cast<unsigned int>(row_tiles));
  const size_t shared = sizeof(float) * static_cast<size_t>(n_classes);
  kodama_label_crossprod_accum_colmajor_float_kernel<<<sum_blocks, threads, shared, stream>>>(
    x_colmajor,
    labels,
    n,
    p,
    n_classes,
    row_tile,
    x_sums
  );
  kodama_label_crossprod_reduce_colmajor_float_kernel<<<(s_total + threads - 1) / threads, threads, 0, stream>>>(
    x_sums,
    s_colmajor,
    row_tiles,
    p,
    n_classes
  );
  kodama_label_crossprod_center_colmajor_float_kernel<<<(s_total + threads - 1) / threads, threads, 0, stream>>>(
    s_colmajor,
    counts,
    n,
    p,
    n_classes
  );
}

__global__ void kodama_lda_label_sums_row_float_kernel(
  const float* t,
  const int* labels,
  int n,
  int kmax,
  int n_classes,
  float* sums
) {
  const int cls = blockIdx.x;
  const int j = blockIdx.y;
  if (cls >= n_classes || j >= kmax) return;

  extern __shared__ float partial_f[];
  float value = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (labels[i] == cls + 1) value += t[i * kmax + j];
  }
  partial_f[threadIdx.x] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial_f[threadIdx.x] += partial_f[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) sums[cls * kmax + j] = partial_f[0];
}

__global__ void kodama_lda_means_row_float_kernel(float* means, const float* counts, int kmax, int n_classes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kmax * n_classes;
  if (idx >= total) return;
  const int cls = idx / kmax;
  const float cnt = counts[cls];
  means[idx] = (cnt > 0.0f) ? means[idx] / cnt : 0.0f;
}

__global__ void kodama_lda_pooled_col_float_kernel(
  float* pooled,
  const float* means,
  const float* counts,
  int n,
  int kmax,
  int n_classes
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kmax * kmax;
  if (idx >= total) return;
  const int r = idx % kmax;
  const int c = idx / kmax;
  float between = 0.0f;
  for (int cls = 0; cls < n_classes; ++cls) {
    between += counts[cls] * means[cls * kmax + r] * means[cls * kmax + c];
  }
  const float df = fmaxf(1.0f, static_cast<float>(n - n_classes));
  pooled[idx] = (pooled[idx] - between) / df;
}

__global__ void kodama_lda_copy_cov_float_kernel(const float* pooled, float* cov, int kmax, int kk) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kk * kk;
  if (idx >= total) return;
  const int r = idx % kk;
  const int c = idx / kk;
  cov[idx] = pooled[r + c * kmax];
}

__global__ void kodama_symmetrize_lower_float_kernel(float* matrix, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * n;
  if (idx >= total) return;
  const int r = idx % n;
  const int c = idx / n;
  if (r < c) matrix[idx] = matrix[c + r * n];
}

__global__ void kodama_lda_add_ridge_float_kernel(float* cov, int kk, float ridge, float* lambda_out) {
  __shared__ float trace;
  if (threadIdx.x == 0) trace = 0.0f;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < kk; ++i) trace += cov[i + i * kk];
  }
  __syncthreads();
  const float scale = isfinite(trace) && trace > 0.0f ? trace / static_cast<float>(kk) : 1.0f;
  const float lambda = (isfinite(ridge) && ridge >= 0.0f ? ridge : 1e-8f) * scale;
  if (threadIdx.x == 0) *lambda_out = lambda;
  for (int i = threadIdx.x; i < kk; i += blockDim.x) cov[i + i * kk] += lambda;
}

__global__ void kodama_lda_means_to_rhs_float_kernel(
  const float* means,
  float* rhs,
  int kmax,
  int kk,
  int n_classes
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kk * n_classes;
  if (idx >= total) return;
  const int j = idx % kk;
  const int cls = idx / kk;
  rhs[j + cls * kk] = means[cls * kmax + j];
}

__global__ void kodama_lda_finalize_linear_row_float_kernel(
  const float* rhs,
  const float* means,
  const float* counts,
  float* linear,
  float* constants,
  int n,
  int kmax,
  int kk,
  int n_classes
) {
  const int cls = blockIdx.x * blockDim.x + threadIdx.x;
  if (cls >= n_classes) return;
  float dot = 0.0f;
  for (int j = 0; j < kk; ++j) {
    const float value = rhs[j + cls * kk];
    linear[cls * kk + j] = value;
    dot += means[cls * kmax + j] * value;
  }
  const float prior = fmaxf(counts[cls] / static_cast<float>(n), FLT_MIN);
  constants[cls] = -0.5f * dot + logf(prior);
}

__global__ void kodama_lda_score_argmax_row_float_kernel(
  const float* t,
  const float* linear,
  const float* constants,
  int* pred,
  int n,
  int kk,
  int n_classes
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  int best = 0;
  float best_value = -FLT_MAX;
  for (int cls = 0; cls < n_classes; ++cls) {
    float score = constants[cls];
    for (int j = 0; j < kk; ++j) score += t[row * kk + j] * linear[cls * kk + j];
    if (score > best_value) {
      best_value = score;
      best = cls;
    }
  }
  pred[row] = best + 1;
}

void kodama_cuda_lda_label_sums_row_float(const float* t, const int* labels, int n, int kmax, int n_classes, float* sums, cudaStream_t stream) {
  const int threads = 256;
  const dim3 blocks(n_classes, kmax);
  const size_t shared = sizeof(float) * static_cast<size_t>(threads);
  kodama_lda_label_sums_row_float_kernel<<<blocks, threads, shared, stream>>>(t, labels, n, kmax, n_classes, sums);
}

void kodama_cuda_lda_means_row_float(float* means, const float* counts, int kmax, int n_classes, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kmax * n_classes + threads - 1) / threads;
  kodama_lda_means_row_float_kernel<<<blocks, threads, 0, stream>>>(means, counts, kmax, n_classes);
}

void kodama_cuda_lda_pooled_col_float(float* pooled, const float* means, const float* counts, int n, int kmax, int n_classes, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kmax * kmax + threads - 1) / threads;
  kodama_lda_pooled_col_float_kernel<<<blocks, threads, 0, stream>>>(pooled, means, counts, n, kmax, n_classes);
}

void kodama_cuda_lda_copy_cov_float(const float* pooled, float* cov, int kmax, int kk, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kk * kk + threads - 1) / threads;
  kodama_lda_copy_cov_float_kernel<<<blocks, threads, 0, stream>>>(pooled, cov, kmax, kk);
}

void kodama_cuda_symmetrize_lower_float(float* matrix, int n, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (n * n + threads - 1) / threads;
  kodama_symmetrize_lower_float_kernel<<<blocks, threads, 0, stream>>>(matrix, n);
}

void kodama_cuda_lda_add_ridge_float(float* cov, int kk, float ridge, float* lambda_out, cudaStream_t stream) {
  kodama_lda_add_ridge_float_kernel<<<1, 256, 0, stream>>>(cov, kk, ridge, lambda_out);
}

void kodama_cuda_lda_means_to_rhs_float(const float* means, float* rhs, int kmax, int kk, int n_classes, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kk * n_classes + threads - 1) / threads;
  kodama_lda_means_to_rhs_float_kernel<<<blocks, threads, 0, stream>>>(means, rhs, kmax, kk, n_classes);
}

void kodama_cuda_lda_finalize_linear_row_float(
  const float* rhs,
  const float* means,
  const float* counts,
  float* linear,
  float* constants,
  int n,
  int kmax,
  int kk,
  int n_classes,
  cudaStream_t stream
) {
  const int threads = 256;
  const int blocks = (n_classes + threads - 1) / threads;
  kodama_lda_finalize_linear_row_float_kernel<<<blocks, threads, 0, stream>>>(rhs, means, counts, linear, constants, n, kmax, kk, n_classes);
}

void kodama_cuda_lda_score_argmax_row_float(
  const float* t,
  const float* linear,
  const float* constants,
  int* pred,
  int n,
  int kk,
  int n_classes,
  cudaStream_t stream
) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  kodama_lda_score_argmax_row_float_kernel<<<blocks, threads, 0, stream>>>(t, linear, constants, pred, n, kk, n_classes);
}

}
