#include <cuda_runtime.h>

#include <cmath>

namespace {

__device__ void insert_top(double value, double* top_vals, int top_k) {
  for (int j = 0; j < top_k; ++j) {
    if (value > top_vals[j]) {
      for (int h = top_k - 1; h > j; --h) top_vals[h] = top_vals[h - 1];
      top_vals[j] = value;
      return;
    }
  }
}

__global__ void candidate_scores_prefix_kernel(
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
  double* out_scores
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = ntest * top_m * nslice;
  if (idx >= total) return;

  const int row = idx % ntest;
  const int tmp = idx / ntest;
  const int cand_slot = tmp % top_m;
  const int slice = tmp / top_m;
  const int col = slice * top_m + cand_slot;
  const int cls = candidates[col * ntest + row];
  if (cls < 1 || cls > n_classes) {
    out_scores[idx] = -INFINITY;
    return;
  }

  int kk = ncomp[slice];
  if (kk < 1) kk = 1;
  if (kk > kdim) kk = kdim;

  const double test_n2 = test_norm2[row * kdim + (kk - 1)];
  const double test_norm = (isfinite(test_n2) && test_n2 > 0.0) ? sqrt(test_n2) : 0.0;
  const int use_k = max(1, min(knn_k, 32));
  double top_vals[32];
  for (int j = 0; j < use_k; ++j) top_vals[j] = -INFINITY;

  const int cls0 = cls - 1;
  const int start = class_offsets[cls0];
  const int end = class_offsets[cls0 + 1];
  const int found = max(0, end - start);
  for (int pos = start; pos < end; ++pos) {
    const int tr = class_indices[pos];
    if (tr < 0 || tr >= ntrain) continue;
    const double train_n2 = train_norm2[tr * kdim + (kk - 1)];
    const double train_norm = (isfinite(train_n2) && train_n2 > 0.0) ? sqrt(train_n2) : 0.0;
    double sim = 0.0;
    if (test_norm > 0.0 && train_norm > 0.0) {
      double dot = 0.0;
      const int test_base = row * kdim;
      const int train_base = tr * kdim;
      for (int d = 0; d < kk; ++d) dot += t_test[test_base + d] * t_train[train_base + d];
      sim = dot / (test_norm * train_norm);
    }
    insert_top(sim, top_vals, use_k);
  }

  if (found < 1 || !isfinite(top_vals[0])) {
    out_scores[idx] = -INFINITY;
    return;
  }

  const int denom = max(1, min(use_k, found));
  double local = 0.0;
  if (!isfinite(tau) || tau <= 0.0) {
    for (int j = 0; j < denom; ++j) local += top_vals[j];
    local /= static_cast<double>(denom);
  } else {
    const double mx = top_vals[0];
    double acc = 0.0;
    for (int j = 0; j < denom; ++j) acc += exp((top_vals[j] - mx) / tau);
    local = mx + tau * log(acc / static_cast<double>(denom));
  }
  out_scores[idx] = local + alpha * candidate_base[idx] + bias[cls - 1];
}

}  // namespace

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
) {
  const int threads = 256;
  const int total = ntest * top_m * nslice;
  const int blocks = (total + threads - 1) / threads;
  candidate_scores_prefix_kernel<<<blocks, threads, 0, stream>>>(
    t_test,
    t_train,
    test_norm2,
    train_norm2,
    ncomp,
    class_offsets,
    class_indices,
    candidates,
    candidate_base,
    bias,
    ntest,
    ntrain,
    kdim,
    n_classes,
    nslice,
    top_m,
    knn_k,
    tau,
    alpha,
    out_scores
  );
}
