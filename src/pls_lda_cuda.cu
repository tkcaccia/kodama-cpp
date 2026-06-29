#include <cuda_runtime.h>
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

}
