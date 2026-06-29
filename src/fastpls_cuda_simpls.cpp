// Adapted from tkcaccia/fastPLS src/svd_cuda_rsvd.cpp (GPL-3).
// CUDA-resident SIMPLS workspace for kodama-cpp PLS cross-validation kernels.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cuda_runtime.h>

namespace kodama_fastpls_cuda {

struct SVDResult {
  arma::mat U;
  arma::vec s;
  arma::mat Vt;
};

struct SVDOptions {
  int oversample = 10;
  int power_iters = 1;
  unsigned int seed = 1;
  bool left_only = false;
};

struct PLSSVDGPUResult {
  arma::mat R;
  arma::mat Q;
  arma::mat Ttrain;
  arma::cube C_latent;
  arma::cube W_latent;
  arma::cube B;
  arma::cube Yfit;
  arma::vec R2Y;
};

SVDResult finalize_from_q_bsmall(const arma::mat& Q, const arma::mat& B, int k, bool left_only) {
  SVDResult out;
  const arma::uword max_rank = std::min(B.n_rows, B.n_cols);
  const arma::uword rank = std::min<arma::uword>(max_rank, static_cast<arma::uword>(std::max(k, 1)));

  if (B.n_rows <= B.n_cols) {
    arma::vec evals;
    arma::mat eigvec;
    if (arma::eig_sym(evals, eigvec, B * B.t())) {
      arma::uvec order = arma::sort_index(evals, "descend");
      arma::vec evals_desc = evals.elem(order);
      arma::mat Uhat_desc = eigvec.cols(order);
      const double tol = std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(B.n_rows, B.n_cols)) *
        (evals_desc.n_elem > 0 ? std::max(evals_desc(0), 1.0) : 1.0);
      arma::uword usable = 0;
      while (usable < evals_desc.n_elem && evals_desc(usable) > tol) ++usable;
      usable = std::min<arma::uword>(usable, rank);
      if (usable > 0) {
        arma::vec s = arma::sqrt(arma::clamp(evals_desc.subvec(0, usable - 1), 0.0, arma::datum::inf));
        arma::mat Uhat = Uhat_desc.cols(0, usable - 1);
        out.U = Q * Uhat;
        out.s = s;
        if (!left_only) {
          arma::mat Vt = Uhat.t() * B;
          Vt.each_col() /= s;
          out.Vt = Vt;
        }
        return out;
      }
    }
  }

  arma::mat Uhat;
  arma::vec s;
  arma::mat V;
  if (left_only) {
    arma::svd_econ(Uhat, s, V, B, "left");
    arma::mat U = Q * Uhat;
    out.U = U.cols(0, rank - 1);
    out.s = s.subvec(0, rank - 1);
    out.Vt.reset();
    return out;
  }

  arma::svd_econ(Uhat, s, V, B, "both");
  arma::mat U = Q * Uhat;
  out.U = U.cols(0, rank - 1);
  out.s = s.subvec(0, rank - 1);
  out.Vt = V.cols(0, rank - 1).t();
  return out;
}

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

arma::mat inv_sqrt_psd(const arma::mat& G) {
  arma::vec evals;
  arma::mat evecs;
  if (!arma::eig_sym(evals, evecs, G)) throw std::runtime_error("eig_sym failed in inv_sqrt_psd");
  if (evals.n_elem < 1) throw std::runtime_error("empty eigenspectrum in inv_sqrt_psd");
  const double max_eval = std::max(evals.max(), 0.0);
  const double tol = std::max(max_eval, 1.0) * 1e-10;
  arma::vec scale = arma::zeros<arma::vec>(evals.n_elem);
  for (arma::uword i = 0; i < evals.n_elem; ++i) {
    if (evals(i) > tol) scale(i) = 1.0 / std::sqrt(evals(i));
  }
  return evecs * arma::diagmat(scale) * evecs.t();
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

double rq_value_local(const arma::mat& yData, const arma::mat& yPred) {
  double tss = 0.0;
  double press = 0.0;
  for (arma::uword i = 0; i < yData.n_cols; ++i) {
    const double my = arma::mean(yData.col(i));
    for (arma::uword j = 0; j < yData.n_rows; ++j) {
      const double diff = yData(j, i) - yPred(j, i);
      const double centered = yData(j, i) - my;
      press += diff * diff;
      tss += centered * centered;
    }
  }
  if (!std::isfinite(tss) || tss <= 0.0) return std::numeric_limits<double>::quiet_NaN();
  return 1.0 - (press / tss);
}

class CudaRSVDWorkspace {
 public:
  CudaRSVDWorkspace() = default;

  ~CudaRSVDWorkspace() {
    release();
  }

  void reset() {
    release();
  }

  void ensure_capacity(int m, int n, int l) {
    ensure_stream();
    if (!handle_ready_) {
      check_cublas(cublasCreate(&handle_), "cublasCreate");
      if (stream_ready_) {
        check_cublas(cublasSetStream(handle_, stream_), "cublasSetStream");
      }
      handle_ready_ = true;
    }
    if (!solver_ready_) {
      check_cusolver(cusolverDnCreate(&solver_), "cusolverDnCreate");
      if (stream_ready_) {
        check_cusolver(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream");
      }
      solver_ready_ = true;
    }

    ensure_buffer(dA_, bytes_for(m, n), bytes_A_, "cudaMalloc(dA)");
    ensure_buffer(dOmega_, bytes_for(n, l), bytes_Omega_, "cudaMalloc(dOmega)");
    ensure_buffer(dY_, bytes_for_padded_random(m, l), bytes_Y_, "cudaMalloc(dY)");
    ensure_buffer(dZ_, bytes_for(n, l), bytes_Z_, "cudaMalloc(dZ)");
    ensure_buffer(dQ_, bytes_for(m, l), bytes_Q_, "cudaMalloc(dQ)");
    ensure_buffer(dBsmall_, bytes_for(l, n), bytes_Bsmall_, "cudaMalloc(dBsmall)");
    ensure_buffer(dGram_, bytes_for(l, l), bytes_Gram_, "cudaMalloc(dGram)");
    ensure_buffer(dTau_, bytes_for(l, 1), bytes_Tau_, "cudaMalloc(dTau)");
    ensure_buffer(dEvals_, bytes_for(l, 1), bytes_Evals_, "cudaMalloc(dEvals)");
    ensure_int_buffer(dInfo_, sizeof(int), bytes_Info_, "cudaMalloc(dInfo)");

    if (!rng_ready_) {
      check_curand(curandCreateGenerator(&rng_, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator");
      if (stream_ready_) {
        check_curand(curandSetStream(rng_, stream_), "curandSetStream");
      }
      rng_ready_ = true;
    }
  }

  void ensure_matrix_free_capacity(int m, int n, int l) {
    ensure_stream();
    if (!handle_ready_) {
      check_cublas(cublasCreate(&handle_), "cublasCreate");
      if (stream_ready_) {
        check_cublas(cublasSetStream(handle_, stream_), "cublasSetStream");
      }
      handle_ready_ = true;
    }
    if (!solver_ready_) {
      check_cusolver(cusolverDnCreate(&solver_), "cusolverDnCreate");
      if (stream_ready_) {
        check_cusolver(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream");
      }
      solver_ready_ = true;
    }

    ensure_buffer(dOmega_, bytes_for_padded_random(n, l), bytes_Omega_, "cudaMalloc(dOmega)");
    ensure_buffer(dY_, bytes_for_padded_random(m, l), bytes_Y_, "cudaMalloc(dY)");
    ensure_buffer(dZ_, bytes_for(n, l), bytes_Z_, "cudaMalloc(dZ)");
    ensure_buffer(dQ_, bytes_for(m, l), bytes_Q_, "cudaMalloc(dQ)");
    ensure_buffer(dBsmall_, bytes_for(l, n), bytes_Bsmall_, "cudaMalloc(dBsmall)");
    ensure_buffer(dGram_, bytes_for(l, l), bytes_Gram_, "cudaMalloc(dGram)");
    ensure_buffer(dTau_, bytes_for(l, 1), bytes_Tau_, "cudaMalloc(dTau)");
    ensure_buffer(dEvals_, bytes_for(l, 1), bytes_Evals_, "cudaMalloc(dEvals)");
    ensure_int_buffer(dInfo_, sizeof(int), bytes_Info_, "cudaMalloc(dInfo)");

    if (!rng_ready_) {
      check_curand(curandCreateGenerator(&rng_, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator");
      if (stream_ready_) {
        check_curand(curandSetStream(rng_, stream_), "curandSetStream");
      }
      rng_ready_ = true;
    }
  }

  void set_pls_training_matrices(
    const double* hX,
    int n,
    int p,
    const double* hY,
    int m,
    bool fit,
    bool form_crossprod = true
  ) {
    if (form_crossprod) {
      ensure_capacity(p, m, 1);
    } else {
      ensure_matrix_free_capacity(p, m, 1);
    }
    ensure_buffer(dX_, bytes_for(n, p), bytes_X_, "cudaMalloc(dX)");
    ensure_buffer(dYtrain_, bytes_for(n, m), bytes_Ytrain_, "cudaMalloc(dYtrain)");
    check_cuda(cudaMemcpy(dX_, hX, bytes_for(n, p), cudaMemcpyHostToDevice), "cudaMemcpy(Xtrain)");
    check_cuda(cudaMemcpy(dYtrain_, hY, bytes_for(n, m), cudaMemcpyHostToDevice), "cudaMemcpy(Ytrain)");
    ensure_buffer(dTvec_, bytes_for(n, 1), bytes_Tvec_, "cudaMalloc(dTvec)");
    ensure_buffer(dPvec_, bytes_for(p, 1), bytes_Pvec_, "cudaMalloc(dPvec)");
    ensure_buffer(dQvec_, bytes_for(m, 1), bytes_Qvec_, "cudaMalloc(dQvec)");
    ensure_buffer(dRvec_, bytes_for(p, 1), bytes_Rvec_, "cudaMalloc(dRvec)");
    if (fit) {
      ensure_buffer(dYfit_, bytes_for(n, m), bytes_Yfit_, "cudaMalloc(dYfit)");
      check_cuda(cudaMemset(dYfit_, 0, bytes_for(n, m)), "cudaMemset(dYfit)");
    }
    if (form_crossprod) {
      crossprod_training_xy(n, p, m);
    }
  }

  void release_pls_ytrain_buffer() {
    if (dYtrain_ != nullptr) {
      cudaFree(dYtrain_);
      dYtrain_ = nullptr;
      bytes_Ytrain_ = 0;
    }
  }

  void simpls_fast_begin_device_loop(
    int n,
    int p,
    int m,
    int max_ncomp,
    bool fit
  ) {
    ensure_buffer(dRRmat_, bytes_for(p, max_ncomp), bytes_RRmat_, "cudaMalloc(dRRmat)");
    ensure_buffer(dQQmat_, bytes_for(m, max_ncomp), bytes_QQmat_, "cudaMalloc(dQQmat)");
    ensure_buffer(dVVmat_, bytes_for(p, max_ncomp), bytes_VVmat_, "cudaMalloc(dVVmat)");
    ensure_buffer(dBcur_, bytes_for(p, m), bytes_Bcur_, "cudaMalloc(dBcur)");
    ensure_buffer(dCoeff_, bytes_for(std::max(max_ncomp, 1), 1), bytes_Coeff_, "cudaMalloc(dCoeff)");
    check_cuda(cudaMemset(dRRmat_, 0, bytes_for(p, max_ncomp)), "cudaMemset(dRRmat)");
    check_cuda(cudaMemset(dQQmat_, 0, bytes_for(m, max_ncomp)), "cudaMemset(dQQmat)");
    check_cuda(cudaMemset(dVVmat_, 0, bytes_for(p, max_ncomp)), "cudaMemset(dVVmat)");
    check_cuda(cudaMemset(dBcur_, 0, bytes_for(p, m)), "cudaMemset(dBcur)");
    if (fit) {
      ensure_buffer(dYfit_, bytes_for(n, m), bytes_Yfit_, "cudaMalloc(dYfit)");
      check_cuda(cudaMemset(dYfit_, 0, bytes_for(n, m)), "cudaMemset(dYfit)");
    }
  }

  void simpls_fast_refresh_block_resident(
    int p,
    int m,
    int l,
    int k,
    bool use_rr_warm_start,
    unsigned int seed,
    int power_iters,
    double* hSvals
  ) {
    ensure_capacity(p, m, l);
    check_curand(
      curandSetPseudoRandomGeneratorSeed(rng_, static_cast<unsigned long long>(seed)),
      "curandSetPseudoRandomGeneratorSeed"
    );
    check_curand(
      curandGenerateNormalDouble(rng_, dY_, padded_random_elems(p, l), 0.0, 1.0),
      "curandGenerateNormalDouble(Y0)"
    );
    if (use_rr_warm_start) {
      check_cublas(
        cublasDcopy(handle_, p, dRvec_, 1, dY_, 1),
        "cublasDcopy(rr->Y0)"
      );
    }

    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < power_iters; ++i) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, l, p, &alpha, dA_, p, dY_, p, &beta, dZ_, m),
        "cublasDgemm(S^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, l, m, &alpha, dA_, p, dZ_, m, &beta, dY_, p),
        "cublasDgemm(S*Z)"
      );
    }

    orthonormalize_qr_inplace(p, l);
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, m, p, &alpha, dY_, p, dA_, p, &beta, dBsmall_, l),
      "cublasDgemm(Bsmall=Q^T*S)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, m, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );

    finalize_left_block_from_gram_inplace(l, k);
    copy_top_singular_values(l, k, hSvals);

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, k, l, &alpha, dY_, p, dOmega_, l, &beta, dQ_, p),
      "cublasDgemm(Ublock=Q*Uhat)"
    );
  }

  void simpls_fast_refresh_block_implicit_resident(
    int n,
    int p,
    int m,
    int l,
    int k,
    int prev_v_cols,
    bool use_rr_warm_start,
    unsigned int seed,
    int power_iters,
    double* hSvals
  ) {
    ensure_matrix_free_capacity(p, m, l);
    check_curand(
      curandSetPseudoRandomGeneratorSeed(rng_, static_cast<unsigned long long>(seed)),
      "curandSetPseudoRandomGeneratorSeed"
    );
    check_curand(
      curandGenerateNormalDouble(rng_, dY_, padded_random_elems(p, l), 0.0, 1.0),
      "curandGenerateNormalDouble(Y0_implicit)"
    );
    if (use_rr_warm_start) {
      check_cublas(
        cublasDcopy(handle_, p, dRvec_, 1, dY_, 1),
        "cublasDcopy(rr->Y0_implicit)"
      );
    }

    for (int i = 0; i < power_iters; ++i) {
      implicit_at_times_mat_deflated(
        n,
        p,
        m,
        l,
        dY_,
        dZ_,
        prev_v_cols,
        "implicit_SIMPLS_SAT_times_Y"
      );
      implicit_a_times_mat_deflated(
        n,
        p,
        m,
        l,
        dZ_,
        dY_,
        prev_v_cols,
        "implicit_SIMPLS_S_times_Z"
      );
    }

    subtract_left_projection_inplace(
      p,
      l,
      prev_v_cols,
      dY_,
      "implicit_SIMPLS_project_before_qr"
    );
    orthonormalize_qr_inplace(p, l);
    implicit_bsmall_from_deflated_basis(
      n,
      p,
      m,
      l,
      dY_,
      dBsmall_,
      prev_v_cols,
      "implicit_SIMPLS_Bsmall=QTA"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, m, &one_, dBsmall_, l, dBsmall_, l, &zero_, dGram_, l),
      "cublasDgemm(implicit_Bsmall*Bsmall^T)"
    );

    finalize_left_block_from_gram_inplace(l, k);
    copy_top_singular_values(l, k, hSvals);

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, k, l, &one_, dY_, p, dOmega_, l, &zero_, dQ_, p),
      "cublasDgemm(implicit_Ublock=Q*Uhat)"
    );
  }

  bool simpls_fast_append_component_from_block(
    int n,
    int p,
    int m,
    int a_idx,
    int col_idx,
    int prev_v_cols,
    bool reorth_v,
    bool fit,
    bool materialized_crossprod
  ) {
    ensure_buffer(dTvec_, bytes_for(n, 1), bytes_Tvec_, "cudaMalloc(dTvec)");
    ensure_buffer(dPvec_, bytes_for(p, 1), bytes_Pvec_, "cudaMalloc(dPvec)");
    ensure_buffer(dQvec_, bytes_for(m, 1), bytes_Qvec_, "cudaMalloc(dQvec)");
    ensure_buffer(dRvec_, bytes_for(p, 1), bytes_Rvec_, "cudaMalloc(dRvec)");
    ensure_buffer(dCoeff_, bytes_for(std::max(prev_v_cols, 1), 1), bytes_Coeff_, "cudaMalloc(dCoeff)");

    const double alpha = 1.0;
    const double beta = 0.0;
    const double minus_one = -1.0;

    check_cublas(
      cublasDcopy(handle_, p, dQ_ + static_cast<size_t>(col_idx) * static_cast<size_t>(p), 1, dRvec_, 1),
      "cublasDcopy(Ublock_col->r)"
    );
    x_times_vec(n, p, dRvec_, dTvec_, "cublasGemm(t=X*r)");

    double tnorm = 0.0;
    check_cublas(cublasDnrm2(handle_, n, dTvec_, 1, &tnorm), "cublasDnrm2(t)");
    if (!std::isfinite(tnorm) || tnorm <= 0.0) {
      return false;
    }

    const double inv_tnorm = 1.0 / tnorm;
    check_cublas(cublasDscal(handle_, n, &inv_tnorm, dTvec_, 1), "cublasDscal(t)");
    check_cublas(cublasDscal(handle_, p, &inv_tnorm, dRvec_, 1), "cublasDscal(r)");

    xt_times_vec(n, p, dTvec_, dPvec_, "cublasGemm(p=X^T*t)");
    yt_times_vec(n, m, dTvec_, dQvec_, "cublasGemm(q=Y^T*t)");

    if (prev_v_cols > 0) {
      check_cublas(
        cublasDgemm(
          handle_,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          prev_v_cols,
          1,
          p,
          &alpha,
          dVVmat_,
          p,
          dPvec_,
          p,
          &beta,
          dCoeff_,
          prev_v_cols
        ),
        "cublasDgemm(V^T*p)"
      );
      check_cublas(
        cublasDgemm(
          handle_,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          p,
          1,
          prev_v_cols,
          &minus_one,
          dVVmat_,
          p,
          dCoeff_,
          prev_v_cols,
          &alpha,
          dPvec_,
          p
        ),
        "cublasDgemm(p-=V*coeff)"
      );
      if (reorth_v) {
        check_cublas(
          cublasDgemm(
            handle_,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            prev_v_cols,
            1,
            p,
            &alpha,
            dVVmat_,
            p,
            dPvec_,
            p,
            &beta,
            dCoeff_,
            prev_v_cols
          ),
          "cublasDgemm(V^T*v)"
        );
        check_cublas(
          cublasDgemm(
            handle_,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            p,
            1,
            prev_v_cols,
            &minus_one,
            dVVmat_,
            p,
            dCoeff_,
            prev_v_cols,
            &alpha,
            dPvec_,
            p
          ),
          "cublasDgemm(v-=V*coeff)"
        );
      }
    }

    double vnorm = 0.0;
    check_cublas(cublasDnrm2(handle_, p, dPvec_, 1, &vnorm), "cublasDnrm2(v)");
    if (!std::isfinite(vnorm) || vnorm <= 0.0) {
      return false;
    }
    const double inv_vnorm = 1.0 / vnorm;
    check_cublas(cublasDscal(handle_, p, &inv_vnorm, dPvec_, 1), "cublasDscal(v)");

    if (materialized_crossprod) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, 1, p, &alpha, dA_, p, dPvec_, p, &beta, dZ_, m),
        "cublasDgemm(v^T*S)"
      );
      check_cublas(
        cublasDger(handle_, p, m, &minus_one, dPvec_, 1, dZ_, 1, dA_, p),
        "cublasDger(deflate)"
      );
    }

    check_cublas(
      cublasDcopy(handle_, p, dRvec_, 1, dRRmat_ + static_cast<size_t>(a_idx) * static_cast<size_t>(p), 1),
      "cublasDcopy(r->RR)"
    );
    check_cublas(
      cublasDcopy(handle_, m, dQvec_, 1, dQQmat_ + static_cast<size_t>(a_idx) * static_cast<size_t>(m), 1),
      "cublasDcopy(q->QQ)"
    );
    check_cublas(
      cublasDcopy(handle_, p, dPvec_, 1, dVVmat_ + static_cast<size_t>(a_idx) * static_cast<size_t>(p), 1),
      "cublasDcopy(v->VV)"
    );
    check_cublas(
      cublasDger(handle_, p, m, &alpha, dRvec_, 1, dQvec_, 1, dBcur_, p),
      "cublasDger(Bcur+=rq^T)"
    );

    if (fit) {
      check_cublas(
        cublasDger(handle_, n, m, &alpha, dTvec_, 1, dQvec_, 1, dYfit_, n),
        "cublasDger(Yfit+=tq^T)"
      );
    }

    return true;
  }

  void simpls_fast_copy_rr(double* hRR, int p, int max_ncomp) {
    check_cuda(cudaMemcpy(hRR, dRRmat_, bytes_for(p, max_ncomp), cudaMemcpyDeviceToHost), "cudaMemcpy(RR)");
  }

  void simpls_fast_copy_qq(double* hQQ, int m, int max_ncomp) {
    check_cuda(cudaMemcpy(hQQ, dQQmat_, bytes_for(m, max_ncomp), cudaMemcpyDeviceToHost), "cudaMemcpy(QQ)");
  }

  void simpls_fast_copy_bcur(double* hB, int p, int m) {
    check_cuda(cudaMemcpy(hB, dBcur_, bytes_for(p, m), cudaMemcpyDeviceToHost), "cudaMemcpy(Bcur)");
  }

  void simpls_fast_copy_yfit(double* hYfit, int n, int m) {
    check_cuda(cudaMemcpy(hYfit, dYfit_, bytes_for(n, m), cudaMemcpyDeviceToHost), "cudaMemcpy(Yfit)");
  }

  void simpls_fast_component_stats(
    const double* hR,
    int n,
    int p,
    int m,
    double* hT,
    double* hP,
    double* hQ,
    double* hTnorm
  ) {
    if (dX_ == nullptr || dYtrain_ == nullptr) {
      throw std::runtime_error("training matrices are not initialized for GPU workspace");
    }
    ensure_buffer(dTvec_, bytes_for(n, 1), bytes_Tvec_, "cudaMalloc(dTvec)");
    ensure_buffer(dPvec_, bytes_for(p, 1), bytes_Pvec_, "cudaMalloc(dPvec)");
    ensure_buffer(dQvec_, bytes_for(m, 1), bytes_Qvec_, "cudaMalloc(dQvec)");
    ensure_buffer(dRvec_, bytes_for(p, 1), bytes_Rvec_, "cudaMalloc(dRvec)");

    check_cuda(cudaMemcpy(dRvec_, hR, bytes_for(p, 1), cudaMemcpyHostToDevice), "cudaMemcpy(rr)");

    const double alpha = 1.0;
    const double beta = 0.0;
    x_times_vec(n, p, dRvec_, dTvec_, "cublasGemm(t=X*r)");

    double tnorm = 0.0;
    check_cublas(cublasDnrm2(handle_, n, dTvec_, 1, &tnorm), "cublasDnrm2(t)");
    if (!std::isfinite(tnorm) || tnorm <= 0.0) {
      throw std::runtime_error("invalid tnorm in cuda_simpls_fast_component_stats");
    }

    const double inv_tnorm = 1.0 / tnorm;
    check_cublas(cublasDscal(handle_, n, &inv_tnorm, dTvec_, 1), "cublasDscal(t)");
    check_cublas(cublasDscal(handle_, p, &inv_tnorm, dRvec_, 1), "cublasDscal(r)");

    xt_times_vec(n, p, dTvec_, dPvec_, "cublasGemm(p=X^T*t)");
    yt_times_vec(n, m, dTvec_, dQvec_, "cublasGemm(q=Y^T*t)");

    check_cuda(cudaMemcpy(hT, dTvec_, bytes_for(n, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(t)");
    check_cuda(cudaMemcpy(hP, dPvec_, bytes_for(p, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(p)");
    check_cuda(cudaMemcpy(hQ, dQvec_, bytes_for(m, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(q)");
    *hTnorm = tnorm;
  }

  void simpls_fast_rank1_fit_update(
    const double* hT,
    int n,
    const double* hQ,
    int m,
    double* hDelta
  ) {
    ensure_buffer(dYfit_, bytes_for(n, m), bytes_Yfit_, "cudaMalloc(dYfit)");
    ensure_buffer(dTvec_, bytes_for(n, 1), bytes_Tvec_, "cudaMalloc(dTvec)");
    ensure_buffer(dQvec_, bytes_for(m, 1), bytes_Qvec_, "cudaMalloc(dQvec)");
    check_cuda(cudaMemcpy(dTvec_, hT, bytes_for(n, 1), cudaMemcpyHostToDevice), "cudaMemcpy(fit_t)");
    check_cuda(cudaMemcpy(dQvec_, hQ, bytes_for(m, 1), cudaMemcpyHostToDevice), "cudaMemcpy(fit_q)");
    const double alpha = 1.0;
    check_cublas(
      cublasDger(handle_, n, m, &alpha, dTvec_, 1, dQvec_, 1, dYfit_, n),
      "cublasDger(Yfit+=tq^T)"
    );
    check_cuda(cudaMemcpy(hDelta, dYfit_, bytes_for(n, m), cudaMemcpyDeviceToHost), "cudaMemcpy(Yfit_cur)");
  }

  void solve_spd_system_inplace(int k, int nrhs) {
    int lwork_potrf = 0;
    check_cusolver(
      cusolverDnDpotrf_bufferSize(
        solver_,
        CUBLAS_FILL_MODE_LOWER,
        k,
        dGram_,
        k,
        &lwork_potrf
      ),
      "cusolverDnDpotrf_bufferSize"
    );
    ensure_buffer(
      dWorkEig_,
      sizeof(double) * static_cast<size_t>(lwork_potrf),
      bytes_WorkEig_,
      "cudaMalloc(dWorkEig)"
    );
    check_cusolver(
      cusolverDnDpotrf(
        solver_,
        CUBLAS_FILL_MODE_LOWER,
        k,
        dGram_,
        k,
        dWorkEig_,
        lwork_potrf,
        dInfo_
      ),
      "cusolverDnDpotrf"
    );
    check_cuda(cudaMemcpy(&hInfo_, dInfo_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(info_potrf)");
    if (hInfo_ != 0) {
      throw std::runtime_error("cusolverDnDpotrf returned non-zero info");
    }

    check_cublas(
      cublasDtrsm(
        handle_,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        k,
        nrhs,
        &one_,
        dGram_,
        k,
        dBsmall_,
        k
      ),
      "cublasDtrsm(lower)"
    );
    check_cublas(
      cublasDtrsm(
        handle_,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        CUBLAS_DIAG_NON_UNIT,
        k,
        nrhs,
        &one_,
        dGram_,
        k,
        dBsmall_,
        k
      ),
      "cublasDtrsm(lower^T)"
    );
  }

  PLSSVDGPUResult plssvd_fit(
    const arma::mat& Xtrain,
    const arma::mat& Ytrain,
    const arma::ivec& ncomp,
    bool fit,
    const SVDOptions& opt
  ) {
    const int n = static_cast<int>(Xtrain.n_rows);
    const int p = static_cast<int>(Xtrain.n_cols);
    const int m = static_cast<int>(Ytrain.n_cols);
    const int max_rank = std::min(p, m);
    const int target = std::min(max_rank, std::max(static_cast<int>(ncomp.max()), 1));
    if (target < 1) {
      throw std::runtime_error("cuda_plssvd_fit target rank is < 1");
    }
    const int l = std::min(max_rank, target + std::max(opt.oversample, 0));

    ensure_capacity(p, m, l);
    set_pls_training_matrices(Xtrain.memptr(), n, p, Ytrain.memptr(), m, fit);
    release_pls_ytrain_buffer();
    ensure_buffer(dRRmat_, bytes_for(n, target), bytes_RRmat_, "cudaMalloc(dRRmat)");
    ensure_buffer(dQQmat_, bytes_for(target, target), bytes_QQmat_, "cudaMalloc(dQQmat)");
    ensure_buffer(dBcur_, bytes_for(p, m), bytes_Bcur_, "cudaMalloc(dBcur)");
    if (fit) {
      ensure_buffer(dYfit_, bytes_for(n, m), bytes_Yfit_, "cudaMalloc(dYfit)");
    }

    check_curand(
      curandSetPseudoRandomGeneratorSeed(rng_, static_cast<unsigned long long>(opt.seed)),
      "curandSetPseudoRandomGeneratorSeed"
    );
    check_curand(
      curandGenerateNormalDouble(rng_, dOmega_, padded_random_elems(m, l), 0.0, 1.0),
      "curandGenerateNormalDouble(Omega)"
    );

    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, l, m, &alpha, dA_, p, dOmega_, m, &beta, dY_, p),
      "cublasDgemm(S*Omega)"
    );
    for (int i = 0; i < std::max(opt.power_iters, 0); ++i) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, l, p, &alpha, dA_, p, dY_, p, &beta, dZ_, m),
        "cublasDgemm(S^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, l, m, &alpha, dA_, p, dZ_, m, &beta, dY_, p),
        "cublasDgemm(S*Z)"
      );
    }

    orthonormalize_qr_inplace(p, l);
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, m, p, &alpha, dY_, p, dA_, p, &beta, dBsmall_, l),
      "cublasDgemm(Bsmall=Q^T*S)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, m, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );
    finalize_left_block_from_gram_inplace(l, target);
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, target, l, &alpha, dY_, p, dOmega_, l, &beta, dQ_, p),
      "cublasDgemm(U=Q*Uhat)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, target, m, l, &alpha, dOmega_, l, dBsmall_, l, &beta, dZ_, target),
      "cublasDgemm(RHS=Uhat^T*Bsmall)"
    );

    ensure_buffer(dRRmat_, bytes_for(n, target), bytes_RRmat_, "cudaMalloc(dRRmat)");
    x_times_mat(n, p, target, dQ_, dRRmat_, "cublasGemm(T=X*U)");
    ensure_buffer(dQQmat_, bytes_for(target, target), bytes_QQmat_, "cudaMalloc(dQQmat)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, target, target, n, &alpha, dRRmat_, n, dRRmat_, n, &beta, dQQmat_, target),
      "cublasDgemm(G=T^T*T)"
    );

    PLSSVDGPUResult out;
    out.R.set_size(static_cast<arma::uword>(p), static_cast<arma::uword>(target));
    out.Ttrain.set_size(static_cast<arma::uword>(n), static_cast<arma::uword>(target));
    check_cuda(cudaMemcpy(out.R.memptr(), dQ_, bytes_for(p, target), cudaMemcpyDeviceToHost), "cudaMemcpy(R_plssvd)");
    check_cuda(cudaMemcpy(out.Ttrain.memptr(), dRRmat_, bytes_for(n, target), cudaMemcpyDeviceToHost), "cudaMemcpy(T_plssvd)");

    arma::vec s_host(target, arma::fill::zeros);
    copy_top_singular_values(l, target, s_host.memptr());
    arma::mat rhs_host(static_cast<arma::uword>(target), static_cast<arma::uword>(m), arma::fill::zeros);
    check_cuda(cudaMemcpy(rhs_host.memptr(), dZ_, bytes_for(target, m), cudaMemcpyDeviceToHost), "cudaMemcpy(RHS_plssvd)");
    out.Q.set_size(static_cast<arma::uword>(m), static_cast<arma::uword>(target));
    out.Q.zeros();
    for (int j = 0; j < target; ++j) {
      const double sj = s_host(static_cast<arma::uword>(j));
      if (std::isfinite(sj) && sj > 0.0) {
        out.Q.col(static_cast<arma::uword>(j)) = (rhs_host.row(static_cast<arma::uword>(j)) / sj).t();
      }
    }

    out.B.set_size(static_cast<arma::uword>(p), static_cast<arma::uword>(m), ncomp.n_elem);
    out.B.zeros();
    out.C_latent.set_size(
      static_cast<arma::uword>(target),
      static_cast<arma::uword>(target),
      ncomp.n_elem
    );
    out.C_latent.zeros();
    out.W_latent.set_size(
      static_cast<arma::uword>(target),
      static_cast<arma::uword>(m),
      ncomp.n_elem
    );
    out.W_latent.zeros();
    out.R2Y.set_size(ncomp.n_elem);
    out.R2Y.zeros();
    if (fit) {
      out.Yfit.set_size(static_cast<arma::uword>(n), static_cast<arma::uword>(m), ncomp.n_elem);
    }

    for (arma::uword a = 0; a < ncomp.n_elem; ++a) {
      const int requested_mc = static_cast<int>(ncomp(a));
      const int mc = std::min(std::max(requested_mc, 1), target);
      check_cuda(
        cudaMemcpy2D(
          dGram_,
          bytes_for(mc, 1),
          dQQmat_,
          bytes_for(target, 1),
          bytes_for(mc, 1),
          static_cast<size_t>(mc),
          cudaMemcpyDeviceToDevice
        ),
        "cudaMemcpy2D(G_a)"
      );
      check_cuda(
        cudaMemcpy2D(
          dBsmall_,
          bytes_for(mc, 1),
          dZ_,
          bytes_for(target, 1),
          bytes_for(mc, 1),
          static_cast<size_t>(m),
          cudaMemcpyDeviceToDevice
        ),
        "cudaMemcpy2D(RHS_a)"
      );
      solve_spd_system_inplace(mc, m);

      arma::mat w_host(static_cast<arma::uword>(mc), static_cast<arma::uword>(m), arma::fill::zeros);
      check_cuda(cudaMemcpy(w_host.memptr(), dBsmall_, bytes_for(mc, m), cudaMemcpyDeviceToHost), "cudaMemcpy(W_latent)");
      out.W_latent.slice(a).submat(0, 0, mc - 1, m - 1) = w_host;

      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, m, mc, &alpha, dQ_, p, dBsmall_, mc, &beta, dBcur_, p),
        "cublasDgemm(B=U*W)"
      );
      check_cuda(
        cudaMemcpy(out.B.slice(a).memptr(), dBcur_, bytes_for(p, m), cudaMemcpyDeviceToHost),
        "cudaMemcpy(B_slice)"
      );

      if (fit) {
        check_cublas(
          cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, mc, &alpha, dRRmat_, n, dBsmall_, mc, &beta, dYfit_, n),
          "cublasDgemm(Yfit=T*W)"
        );
        check_cuda(
          cudaMemcpy(out.Yfit.slice(a).memptr(), dYfit_, bytes_for(n, m), cudaMemcpyDeviceToHost),
          "cudaMemcpy(Yfit_slice)"
        );
        out.R2Y(a) = rq_value_local(Ytrain, out.Yfit.slice(a));
      }
    }

    return out;
  }

  PLSSVDGPUResult plssvd_fit_implicit_xprod(
    const arma::mat& Xtrain,
    const arma::mat& Ytrain,
    const arma::ivec& ncomp,
    bool fit,
    const SVDOptions& opt
  ) {
    const int n = static_cast<int>(Xtrain.n_rows);
    const int p = static_cast<int>(Xtrain.n_cols);
    const int m = static_cast<int>(Ytrain.n_cols);
    const int max_rank = std::min(p, m);
    const int target = std::min(max_rank, std::max(static_cast<int>(ncomp.max()), 1));
    if (target < 1) {
      throw std::runtime_error("cuda_plssvd_fit_implicit_xprod target rank is < 1");
    }
    const int l = std::min(max_rank, target + std::max(opt.oversample, 0));

    ensure_matrix_free_capacity(p, m, l);
    set_pls_training_matrices(Xtrain.memptr(), n, p, Ytrain.memptr(), m, fit, false);
    ensure_matrix_free_capacity(p, m, l);

    check_curand(
      curandSetPseudoRandomGeneratorSeed(rng_, static_cast<unsigned long long>(opt.seed)),
      "curandSetPseudoRandomGeneratorSeed"
    );
    check_curand(
      curandGenerateNormalDouble(rng_, dOmega_, padded_random_elems(m, l), 0.0, 1.0),
      "curandGenerateNormalDouble(Omega)"
    );

    const double alpha = 1.0;
    const double beta = 0.0;

    implicit_a_times_mat(n, p, m, l, dOmega_, dY_, "implicit_A_times_Omega");
    for (int i = 0; i < std::max(opt.power_iters, 0); ++i) {
      implicit_at_times_mat(n, p, m, l, dY_, dZ_, "implicit_AT_times_Y");
      implicit_a_times_mat(n, p, m, l, dZ_, dY_, "implicit_A_times_Z");
    }

    orthonormalize_qr_inplace(p, l);
    implicit_bsmall_from_basis(n, p, m, l, dY_, dBsmall_, "implicit_Bsmall=QTA");
    release_pls_ytrain_buffer();

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, m, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );
    finalize_left_block_from_gram_inplace(l, target);
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, target, l, &alpha, dY_, p, dOmega_, l, &beta, dQ_, p),
      "cublasDgemm(U=Q*Uhat)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, target, m, l, &alpha, dOmega_, l, dBsmall_, l, &beta, dZ_, target),
      "cublasDgemm(RHS=Uhat^T*Bsmall)"
    );

    ensure_buffer(dRRmat_, bytes_for(n, target), bytes_RRmat_, "cudaMalloc(dRRmat)");
    x_times_mat(n, p, target, dQ_, dRRmat_, "cublasGemm(T=X*U)");
    ensure_buffer(dQQmat_, bytes_for(target, target), bytes_QQmat_, "cudaMalloc(dQQmat)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, target, target, n, &alpha, dRRmat_, n, dRRmat_, n, &beta, dQQmat_, target),
      "cublasDgemm(G=T^T*T)"
    );

    ensure_buffer(dBcur_, bytes_for(p, m), bytes_Bcur_, "cudaMalloc(dBcur)");
    if (fit) {
      ensure_buffer(dYfit_, bytes_for(n, m), bytes_Yfit_, "cudaMalloc(dYfit)");
    }

    PLSSVDGPUResult out;
    out.R.set_size(static_cast<arma::uword>(p), static_cast<arma::uword>(target));
    out.Ttrain.set_size(static_cast<arma::uword>(n), static_cast<arma::uword>(target));
    check_cuda(cudaMemcpy(out.R.memptr(), dQ_, bytes_for(p, target), cudaMemcpyDeviceToHost), "cudaMemcpy(R_plssvd)");
    check_cuda(cudaMemcpy(out.Ttrain.memptr(), dRRmat_, bytes_for(n, target), cudaMemcpyDeviceToHost), "cudaMemcpy(T_plssvd)");

    arma::vec s_host(target, arma::fill::zeros);
    copy_top_singular_values(l, target, s_host.memptr());
    arma::mat rhs_host(static_cast<arma::uword>(target), static_cast<arma::uword>(m), arma::fill::zeros);
    check_cuda(cudaMemcpy(rhs_host.memptr(), dZ_, bytes_for(target, m), cudaMemcpyDeviceToHost), "cudaMemcpy(RHS_plssvd)");
    out.Q.set_size(static_cast<arma::uword>(m), static_cast<arma::uword>(target));
    out.Q.zeros();
    for (int j = 0; j < target; ++j) {
      const double sj = s_host(static_cast<arma::uword>(j));
      if (std::isfinite(sj) && sj > 0.0) {
        out.Q.col(static_cast<arma::uword>(j)) = (rhs_host.row(static_cast<arma::uword>(j)) / sj).t();
      }
    }

    out.B.set_size(static_cast<arma::uword>(p), static_cast<arma::uword>(m), ncomp.n_elem);
    out.B.zeros();
    out.C_latent.set_size(
      static_cast<arma::uword>(target),
      static_cast<arma::uword>(target),
      ncomp.n_elem
    );
    out.C_latent.zeros();
    out.W_latent.set_size(
      static_cast<arma::uword>(target),
      static_cast<arma::uword>(m),
      ncomp.n_elem
    );
    out.W_latent.zeros();
    out.R2Y.set_size(ncomp.n_elem);
    out.R2Y.zeros();
    if (fit) {
      out.Yfit.set_size(static_cast<arma::uword>(n), static_cast<arma::uword>(m), ncomp.n_elem);
    }

    for (arma::uword a = 0; a < ncomp.n_elem; ++a) {
      const int requested_mc = static_cast<int>(ncomp(a));
      const int mc = std::min(std::max(requested_mc, 1), target);
      check_cuda(
        cudaMemcpy2D(
          dGram_,
          bytes_for(mc, 1),
          dQQmat_,
          bytes_for(target, 1),
          bytes_for(mc, 1),
          static_cast<size_t>(mc),
          cudaMemcpyDeviceToDevice
        ),
        "cudaMemcpy2D(G_a)"
      );
      check_cuda(
        cudaMemcpy2D(
          dBsmall_,
          bytes_for(mc, 1),
          dZ_,
          bytes_for(target, 1),
          bytes_for(mc, 1),
          static_cast<size_t>(m),
          cudaMemcpyDeviceToDevice
        ),
        "cudaMemcpy2D(RHS_a)"
      );
      solve_spd_system_inplace(mc, m);

      arma::mat w_host(static_cast<arma::uword>(mc), static_cast<arma::uword>(m), arma::fill::zeros);
      check_cuda(cudaMemcpy(w_host.memptr(), dBsmall_, bytes_for(mc, m), cudaMemcpyDeviceToHost), "cudaMemcpy(W_latent)");
      out.W_latent.slice(a).submat(0, 0, mc - 1, m - 1) = w_host;

      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, m, mc, &alpha, dQ_, p, dBsmall_, mc, &beta, dBcur_, p),
        "cublasDgemm(B=U*W)"
      );
      check_cuda(
        cudaMemcpy(out.B.slice(a).memptr(), dBcur_, bytes_for(p, m), cudaMemcpyDeviceToHost),
        "cudaMemcpy(B_slice)"
      );

      if (fit) {
        check_cublas(
          cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, mc, &alpha, dRRmat_, n, dBsmall_, mc, &beta, dYfit_, n),
          "cublasDgemm(Yfit=T*W)"
        );
        check_cuda(
          cudaMemcpy(out.Yfit.slice(a).memptr(), dYfit_, bytes_for(n, m), cudaMemcpyDeviceToHost),
          "cudaMemcpy(Yfit_slice)"
        );
        out.R2Y(a) = rq_value_local(Ytrain, out.Yfit.slice(a));
      }
    }

    return out;
  }

  void orthonormalize_qr_inplace(int m, int l) {
    if (env_int_or("FASTPLS_GPU_QR", 1, 0, 1) == 0) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, l, m, &one_, dY_, m, dY_, m, &zero_, dGram_, l),
        "cublasDgemm(Y^T*Y)"
      );
      hGram_host_.set_size(static_cast<arma::uword>(l), static_cast<arma::uword>(l));
      check_cuda(cudaMemcpy(hGram_host_.memptr(), dGram_, bytes_for(l, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Gram)");
      arma::mat T = inv_sqrt_psd(hGram_host_);
      check_cuda(cudaMemcpy(dOmega_, T.memptr(), bytes_for(l, l), cudaMemcpyHostToDevice), "cudaMemcpy(T)");
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, l, &one_, dY_, m, dOmega_, l, &zero_, dQ_, m),
        "cublasDgemm(Q=Y*T)"
      );
      check_cuda(cudaMemcpy(dY_, dQ_, bytes_for(m, l), cudaMemcpyDeviceToDevice), "cudaMemcpy(Q->Y)");
      return;
    }

    int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    check_cusolver(cusolverDnDgeqrf_bufferSize(solver_, m, l, dY_, m, &lwork_geqrf), "cusolverDnDgeqrf_bufferSize");
    check_cusolver(cusolverDnDorgqr_bufferSize(solver_, m, l, l, dY_, m, dTau_, &lwork_orgqr), "cusolverDnDorgqr_bufferSize");
    const int lwork = std::max(lwork_geqrf, lwork_orgqr);
    ensure_buffer(dWorkQR_, sizeof(double) * static_cast<size_t>(lwork), bytes_WorkQR_, "cudaMalloc(dWorkQR)");

    check_cusolver(cusolverDnDgeqrf(solver_, m, l, dY_, m, dTau_, dWorkQR_, lwork_geqrf, dInfo_), "cusolverDnDgeqrf");
    check_cuda(cudaMemcpy(&hInfo_, dInfo_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(info_geqrf)");
    if (hInfo_ != 0) {
      throw std::runtime_error("cusolverDnDgeqrf returned non-zero info");
    }

    check_cusolver(cusolverDnDorgqr(solver_, m, l, l, dY_, m, dTau_, dWorkQR_, lwork_orgqr, dInfo_), "cusolverDnDorgqr");
    check_cuda(cudaMemcpy(&hInfo_, dInfo_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(info_orgqr)");
    if (hInfo_ != 0) {
      throw std::runtime_error("cusolverDnDorgqr returned non-zero info");
    }
  }

  bool use_qless_qr_mode(int l) const {
    return (env_int_or("FASTPLS_GPU_QLESS_QR", 0, 0, 1) == 1) && (l > 1);
  }

  arma::mat prepare_qless_transform(int m, int l) {
    ensure_buffer(dTransform_, bytes_for(l, l), bytes_Transform_, "cudaMalloc(dTransform)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, l, m, &one_, dY_, m, dY_, m, &zero_, dGram_, l),
      "cublasDgemm(Y^T*Y)"
    );
    hGram_host_.set_size(static_cast<arma::uword>(l), static_cast<arma::uword>(l));
    check_cuda(cudaMemcpy(hGram_host_.memptr(), dGram_, bytes_for(l, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Gram)");
    arma::mat T = inv_sqrt_psd(hGram_host_);
    check_cuda(cudaMemcpy(dTransform_, T.memptr(), bytes_for(l, l), cudaMemcpyHostToDevice), "cudaMemcpy(T)");
    return T;
  }

  void form_bsmall_qless(int m, int n, int l) {
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, n, m, &one_, dY_, m, dA_, m, &zero_, dBsmall_, l),
      "cublasDgemm(Y^T*A)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, l, n, l, &one_, dTransform_, l, dBsmall_, l, &zero_, dZ_, l),
      "cublasDgemm(T*Y^T*A)"
    );
    check_cuda(cudaMemcpy(dBsmall_, dZ_, bytes_for(l, n), cudaMemcpyDeviceToDevice), "cudaMemcpy(Bsmall_qless)");
  }

  void form_u_from_qless(int m, int l, int k) {
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, l, k, l, &one_, dTransform_, l, dOmega_, l, &zero_, dZ_, l),
      "cublasDgemm(T*Uhat)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, k, l, &one_, dY_, m, dZ_, l, &zero_, dQ_, m),
      "cublasDgemm(Y*(T*Uhat))"
    );
  }

  void finalize_left_block_from_gram_inplace(int l, int k) {
    if (env_int_or("FASTPLS_GPU_EIG", 1, 0, 1) == 0) {
      hGram_host_.set_size(static_cast<arma::uword>(l), static_cast<arma::uword>(l));
      check_cuda(cudaMemcpy(hGram_host_.memptr(), dGram_, bytes_for(l, l), cudaMemcpyDeviceToHost), "cudaMemcpy(BsmallGram)");
      arma::vec evals;
      arma::mat evecs;
      if (!arma::eig_sym(evals, evecs, hGram_host_)) {
        throw std::runtime_error("eig_sym failed in finalize_left_block_from_gram_inplace");
      }
      arma::uvec ord = arma::sort_index(evals, "descend");
      arma::mat Uhat = evecs.cols(ord.head(static_cast<arma::uword>(k)));
      check_cuda(cudaMemcpy(dOmega_, Uhat.memptr(), bytes_for(l, k), cudaMemcpyHostToDevice), "cudaMemcpy(Uhat)");
      return;
    }

    int lwork_syevd = 0;
    check_cusolver(
      cusolverDnDsyevd_bufferSize(
        solver_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        l,
        dGram_,
        l,
        dEvals_,
        &lwork_syevd
      ),
      "cusolverDnDsyevd_bufferSize"
    );
    ensure_buffer(dWorkEig_, sizeof(double) * static_cast<size_t>(lwork_syevd), bytes_WorkEig_, "cudaMalloc(dWorkEig)");
    check_cusolver(
      cusolverDnDsyevd(
        solver_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        l,
        dGram_,
        l,
        dEvals_,
        dWorkEig_,
        lwork_syevd,
        dInfo_
      ),
      "cusolverDnDsyevd"
    );
    check_cuda(cudaMemcpy(&hInfo_, dInfo_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(info_syevd)");
    if (hInfo_ != 0) {
      throw std::runtime_error("cusolverDnDsyevd returned non-zero info");
    }

    for (int j = 0; j < k; ++j) {
      const int src_col = l - 1 - j;
      check_cublas(
        cublasDcopy(handle_, l, dGram_ + static_cast<size_t>(src_col) * static_cast<size_t>(l), 1,
                    dOmega_ + static_cast<size_t>(j) * static_cast<size_t>(l), 1),
        "cublasDcopy(Uhat)"
      );
    }
  }

  void copy_top_singular_values(int l, int k, double* hSvals) {
    if (hSvals == nullptr) {
      return;
    }
    arma::vec evals_host(l);
    check_cuda(cudaMemcpy(evals_host.memptr(), dEvals_, bytes_for(l, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(evals)");
    for (int j = 0; j < k; ++j) {
      const int idx = l - 1 - j;
      hSvals[j] = std::sqrt(std::max(evals_host(static_cast<arma::uword>(idx)), 0.0));
    }
  }

  void sample_y(
    const double* hA,
    int m,
    int n,
    const double* hOmega,
    int l,
    int power_iters,
    double* hY
  ) {
    ensure_capacity(m, n, l);

    check_cuda(cudaMemcpy(dA_, hA, bytes_for(m, n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    check_cuda(cudaMemcpy(dOmega_, hOmega, bytes_for(n, l), cudaMemcpyHostToDevice), "cudaMemcpy(Omega)");

    const double alpha = 1.0;
    const double beta = 0.0;

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dOmega_, n, &beta, dY_, m),
      "cublasDgemm(A*Omega)"
    );

    for (int i = 0; i < power_iters; ++i) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dZ_, n, &beta, dY_, m),
        "cublasDgemm(A*Z)"
      );
    }

    check_cuda(cudaMemcpy(hY, dY_, bytes_for(m, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Y)");
  }

  void set_matrix(const double* hA, int m, int n) {
    ensure_capacity(m, n, 1);
    check_cuda(cudaMemcpy(dA_, hA, bytes_for(m, n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
  }

  void refresh_left_block(
    const double* hA,
    int m,
    int n,
    const double* hY0,
    int l,
    int power_iters,
    double* hY
  ) {
    ensure_capacity(m, n, l);
    hOmega_host_.set_size(static_cast<arma::uword>(n), static_cast<arma::uword>(l));

    check_cuda(cudaMemcpy(dA_, hA, bytes_for(m, n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    check_cuda(cudaMemcpy(dY_, hY0, bytes_for(m, l), cudaMemcpyHostToDevice), "cudaMemcpy(Y0)");

    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < power_iters; ++i) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cuda(cudaMemcpy(hOmega_host_.memptr(), dZ_, bytes_for(n, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Z_host)");

      arma::mat Qz;
      arma::mat Rz;
      arma::qr_econ(Qz, Rz, hOmega_host_);
      if (Qz.n_cols > static_cast<arma::uword>(l)) {
        Qz = Qz.cols(0, static_cast<arma::uword>(l - 1));
      }
      check_cuda(cudaMemcpy(dOmega_, Qz.memptr(), bytes_for(n, static_cast<int>(Qz.n_cols)), cudaMemcpyHostToDevice), "cudaMemcpy(Qz)");
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, static_cast<int>(Qz.n_cols), n, &alpha, dA_, m, dOmega_, n, &beta, dY_, m),
        "cublasDgemm(A*Qz)"
      );
    }

    check_cuda(cudaMemcpy(hY, dY_, bytes_for(m, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Y)");
  }

  void refresh_left_block_u(
    const double* hA,
    int m,
    int n,
    const double* hY0,
    int l,
    int k,
    int power_iters,
    double* hUblock,
    double* hSvals
  ) {
    ensure_capacity(m, n, l);
    check_cuda(cudaMemcpy(dA_, hA, bytes_for(m, n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    check_cuda(cudaMemcpy(dY_, hY0, bytes_for(m, l), cudaMemcpyHostToDevice), "cudaMemcpy(Y0)");

    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < power_iters; ++i) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dZ_, n, &beta, dY_, m),
        "cublasDgemm(A*Z)"
      );
    }
    const bool use_qless = use_qless_qr_mode(l);
    if (use_qless) {
      prepare_qless_transform(m, l);
      form_bsmall_qless(m, n, l);
    } else {
      orthonormalize_qr_inplace(m, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, n, m, &alpha, dY_, m, dA_, m, &beta, dBsmall_, l),
        "cublasDgemm(Bsmall=Q^T*A)"
      );
    }
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, n, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );

    finalize_left_block_from_gram_inplace(l, k);
    copy_top_singular_values(l, k, hSvals);
    if (use_qless) {
      form_u_from_qless(m, l, k);
    } else {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, k, l, &alpha, dY_, m, dOmega_, l, &beta, dQ_, m),
        "cublasDgemm(Ublock=Q*Uhat)"
      );
    }
    check_cuda(cudaMemcpy(hUblock, dQ_, bytes_for(m, k), cudaMemcpyDeviceToHost), "cudaMemcpy(Ublock)");
  }

  void refresh_left_block_u_resident(
    int m,
    int n,
    const double* hY0,
    int l,
    int k,
    unsigned int seed,
    int power_iters,
    double* hUblock,
    double* hSvals
  ) {
    ensure_capacity(m, n, l);
    if (hY0 != nullptr) {
      check_cuda(cudaMemcpy(dY_, hY0, bytes_for(m, l), cudaMemcpyHostToDevice), "cudaMemcpy(Y0)");
    } else {
      check_curand(curandSetPseudoRandomGeneratorSeed(rng_, static_cast<unsigned long long>(seed)), "curandSetPseudoRandomGeneratorSeed");
      check_curand(curandGenerateNormalDouble(rng_, dY_, padded_random_elems(m, l), 0.0, 1.0), "curandGenerateNormalDouble(Y0)");
    }

    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < power_iters; ++i) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dZ_, n, &beta, dY_, m),
        "cublasDgemm(A*Z)"
      );
    }
    const bool use_qless = use_qless_qr_mode(l);
    if (use_qless) {
      prepare_qless_transform(m, l);
      form_bsmall_qless(m, n, l);
    } else {
      orthonormalize_qr_inplace(m, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, n, m, &alpha, dY_, m, dA_, m, &beta, dBsmall_, l),
        "cublasDgemm(Bsmall=Q^T*A)"
      );
    }
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, n, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );

    finalize_left_block_from_gram_inplace(l, k);
    copy_top_singular_values(l, k, hSvals);
    if (use_qless) {
      form_u_from_qless(m, l, k);
    } else {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, k, l, &alpha, dY_, m, dOmega_, l, &beta, dQ_, m),
        "cublasDgemm(Ublock=Q*Uhat)"
      );
    }
    check_cuda(cudaMemcpy(hUblock, dQ_, bytes_for(m, k), cudaMemcpyDeviceToHost), "cudaMemcpy(Ublock)");
  }

  SVDResult resident_svd(
    const arma::mat& A,
    int k,
    const SVDOptions& opt
  ) {
    const int m = static_cast<int>(A.n_rows);
    const int n = static_cast<int>(A.n_cols);
    const int max_rank = std::min(m, n);
    const int target = std::min(max_rank, std::max(k, 1));
    const int l = std::min(max_rank, target + std::max(opt.oversample, 0));

    ensure_capacity(m, n, l);
    set_matrix(A.memptr(), m, n);

    check_curand(curandSetPseudoRandomGeneratorSeed(rng_, static_cast<unsigned long long>(opt.seed)), "curandSetPseudoRandomGeneratorSeed");
    check_curand(curandGenerateNormalDouble(rng_, dOmega_, padded_random_elems(n, l), 0.0, 1.0), "curandGenerateNormalDouble(Omega)");

    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dOmega_, n, &beta, dY_, m),
      "cublasDgemm(A*Omega)"
    );

    const int power_iters = std::max(opt.power_iters, 0);
    for (int i = 0; i < power_iters; ++i) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dZ_, n, &beta, dY_, m),
        "cublasDgemm(A*Z)"
      );
    }
    const bool use_qless = use_qless_qr_mode(l);
    arma::mat T_host;
    if (use_qless) {
      T_host = prepare_qless_transform(m, l);
      form_bsmall_qless(m, n, l);
    } else {
      orthonormalize_qr_inplace(m, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, n, m, &alpha, dY_, m, dA_, m, &beta, dBsmall_, l),
        "cublasDgemm(Bsmall=Q^T*A)"
      );
    }

    const int gpu_finalize_threshold = env_int_or("FASTPLS_GPU_FINALIZE_THRESHOLD", 32, 1, 4096);
    if (l <= gpu_finalize_threshold) {
      arma::mat hQ(m, l);
      arma::mat hB(l, n);
      check_cuda(cudaMemcpy(hQ.memptr(), dY_, bytes_for(m, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Q_or_Y_host)");
      if (use_qless) {
        hQ = hQ * T_host;
      }
      check_cuda(cudaMemcpy(hB.memptr(), dBsmall_, bytes_for(l, n), cudaMemcpyDeviceToHost), "cudaMemcpy(Bsmall_host)");
      return finalize_from_q_bsmall(hQ, hB, target, opt.left_only);
    }

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, n, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );

    finalize_left_block_from_gram_inplace(l, target);
    if (use_qless) {
      form_u_from_qless(m, l, target);
    } else {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, target, l, &alpha, dY_, m, dOmega_, l, &beta, dQ_, m),
        "cublasDgemm(U=Q*Uhat)"
      );
    }

    SVDResult out;
    out.U.set_size(static_cast<arma::uword>(m), static_cast<arma::uword>(target));
    check_cuda(cudaMemcpy(out.U.memptr(), dQ_, bytes_for(m, target), cudaMemcpyDeviceToHost), "cudaMemcpy(U)");

    arma::vec evals_host(l);
    check_cuda(cudaMemcpy(evals_host.memptr(), dEvals_, bytes_for(l, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(evals)");
    out.s.set_size(static_cast<arma::uword>(target));
    for (int j = 0; j < target; ++j) {
      const int idx = l - 1 - j;
      out.s(static_cast<arma::uword>(j)) = std::sqrt(std::max(evals_host(static_cast<arma::uword>(idx)), 0.0));
    }

    if (!opt.left_only) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, target, n, l, &alpha, dOmega_, l, dBsmall_, l, &beta, dZ_, target),
        "cublasDgemm(Vt=Uhat^T*Bsmall)"
      );
      out.Vt.set_size(static_cast<arma::uword>(target), static_cast<arma::uword>(n));
      check_cuda(cudaMemcpy(out.Vt.memptr(), dZ_, bytes_for(target, n), cudaMemcpyDeviceToHost), "cudaMemcpy(Vt)");
      for (int j = 0; j < target; ++j) {
        const double sj = out.s(static_cast<arma::uword>(j));
        if (std::isfinite(sj) && sj > 0.0) {
          out.Vt.row(static_cast<arma::uword>(j)) /= sj;
        }
      }
    }

    return out;
  }

  void project_left_row(const double* hV, int m, int n, double* hVS) {
    ensure_capacity(m, n, 1);
    check_cuda(cudaMemcpy(dY_, hV, bytes_for(m, 1), cudaMemcpyHostToDevice), "cudaMemcpy(v)");
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
      "cublasDgemm(A^T*v)"
    );
    check_cuda(cudaMemcpy(hVS, dZ_, bytes_for(n, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(vS)");
  }

  void deflate_left_rank1(const double* hV, const double* hVS, int m, int n) {
    ensure_capacity(m, n, 1);
    check_cuda(cudaMemcpy(dY_, hV, bytes_for(m, 1), cudaMemcpyHostToDevice), "cudaMemcpy(v)");
    check_cuda(cudaMemcpy(dZ_, hVS, bytes_for(n, 1), cudaMemcpyHostToDevice), "cudaMemcpy(vS)");
    const double alpha = -1.0;
    check_cublas(
      cublasDger(handle_, m, n, &alpha, dY_, 1, dZ_, 1, dA_, m),
      "cublasDger(deflate)"
    );
  }

 private:
  void ensure_stream() {
    if (env_int_or("FASTPLS_CUDA_WORKSPACE_STREAMS", 0, 0, 1) == 0) {
      return;
    }
    if (!stream_ready_) {
      check_cuda(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags"
      );
      stream_ready_ = true;
    }
  }

  static size_t bytes_for(int m, int n) {
    return sizeof(double) * static_cast<size_t>(m) * static_cast<size_t>(n);
  }

  static size_t bytes_for_padded_random(int m, int n) {
    size_t elems = static_cast<size_t>(m) * static_cast<size_t>(n);
    if ((elems % 2U) != 0U) {
      ++elems;
    }
    return sizeof(double) * elems;
  }

  static size_t padded_random_elems(int m, int n) {
    size_t elems = static_cast<size_t>(m) * static_cast<size_t>(n);
    if ((elems % 2U) != 0U) {
      ++elems;
    }
    return elems;
  }

  void ensure_buffer(double*& ptr, size_t required, size_t& current, const char* where) {
    if (required <= current && ptr != nullptr) {
      return;
    }
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      current = 0;
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), required), where);
    current = required;
  }

  void crossprod_training_xy(int n, int p, int m) {
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, p, m, n, &alpha, dX_, n, dYtrain_, n, &beta, dA_, p),
      "cublasDgemm(S=X^T*Y)"
    );
  }

  void x_times_vec(int n, int p, const double* dVec, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, p, &alpha, dX_, n, dVec, p, &beta, dOut, n),
      where
    );
  }

  void xt_times_vec(int n, int p, const double* dVec, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, p, 1, n, &alpha, dX_, n, dVec, n, &beta, dOut, p),
      where
    );
  }

  void yt_times_vec(int n, int m, const double* dVec, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, 1, n, &alpha, dYtrain_, n, dVec, n, &beta, dOut, m),
      where
    );
  }

  void x_times_mat(int n, int p, int k, const double* dRight, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, k, p, &alpha, dX_, n, dRight, p, &beta, dOut, n),
      where
    );
  }

  void implicit_a_times_mat(int n, int p, int m, int k, const double* dRight, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    ensure_buffer(dTransform_, bytes_for(n, k), bytes_Transform_, "cudaMalloc(dTransform)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, k, m, &alpha, dYtrain_, n,
                  dRight, m, &beta, dTransform_, n),
      where
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, p, k, n, &alpha, dX_, n,
                  dTransform_, n, &beta, dOut, p),
      where
    );
  }

  void implicit_at_times_mat(int n, int p, int m, int k, const double* dRight, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    ensure_buffer(dTransform_, bytes_for(n, k), bytes_Transform_, "cudaMalloc(dTransform)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, k, p, &alpha, dX_, n,
                  dRight, p, &beta, dTransform_, n),
      where
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, k, n, &alpha, dYtrain_, n,
                  dTransform_, n, &beta, dOut, m),
      where
    );
  }

  void implicit_bsmall_from_basis(int n, int p, int m, int l, const double* dBasis, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    ensure_buffer(dTransform_, bytes_for(n, l), bytes_Transform_, "cudaMalloc(dTransform)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, l, p, &alpha, dX_, n,
                  dBasis, p, &beta, dTransform_, n),
      where
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, m, n, &alpha, dTransform_, n,
                  dYtrain_, n, &beta, dOut, l),
      where
    );
  }

  void subtract_left_projection_inplace(
    int p,
    int k,
    int prev_v_cols,
    double* dMat,
    const char* where
  ) {
    if (prev_v_cols <= 0) {
      return;
    }
    const double alpha = 1.0;
    const double beta = 0.0;
    const double minus_one = -1.0;
    ensure_buffer(dCoeff_, bytes_for(prev_v_cols, k), bytes_Coeff_, "cudaMalloc(dCoeff)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, prev_v_cols, k, p,
                  &alpha, dVVmat_, p, dMat, p, &beta, dCoeff_, prev_v_cols),
      where
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, k, prev_v_cols,
                  &minus_one, dVVmat_, p, dCoeff_, prev_v_cols, &alpha, dMat, p),
      where
    );
  }

  void implicit_a_times_mat_deflated(
    int n,
    int p,
    int m,
    int k,
    const double* dRight,
    double* dOut,
    int prev_v_cols,
    const char* where
  ) {
    implicit_a_times_mat(n, p, m, k, dRight, dOut, where);
    subtract_left_projection_inplace(p, k, prev_v_cols, dOut, where);
  }

  void implicit_at_times_mat_deflated(
    int n,
    int p,
    int m,
    int k,
    const double* dRight,
    double* dOut,
    int prev_v_cols,
    const char* where
  ) {
    if (prev_v_cols <= 0) {
      implicit_at_times_mat(n, p, m, k, dRight, dOut, where);
      return;
    }
    ensure_buffer(dQ_, bytes_for(p, k), bytes_Q_, "cudaMalloc(dQ)");
    check_cuda(cudaMemcpy(dQ_, dRight, bytes_for(p, k), cudaMemcpyDeviceToDevice), "cudaMemcpy(deflated_basis)");
    subtract_left_projection_inplace(p, k, prev_v_cols, dQ_, where);
    implicit_at_times_mat(n, p, m, k, dQ_, dOut, where);
  }

  void implicit_bsmall_from_deflated_basis(
    int n,
    int p,
    int m,
    int l,
    const double* dBasis,
    double* dOut,
    int prev_v_cols,
    const char* where
  ) {
    if (prev_v_cols <= 0) {
      implicit_bsmall_from_basis(n, p, m, l, dBasis, dOut, where);
      return;
    }
    ensure_buffer(dQ_, bytes_for(p, l), bytes_Q_, "cudaMalloc(dQ)");
    check_cuda(cudaMemcpy(dQ_, dBasis, bytes_for(p, l), cudaMemcpyDeviceToDevice), "cudaMemcpy(deflated_bsmall_basis)");
    subtract_left_projection_inplace(p, l, prev_v_cols, dQ_, where);
    implicit_bsmall_from_basis(n, p, m, l, dQ_, dOut, where);
  }

  void ensure_int_buffer(int*& ptr, size_t required, size_t& current, const char* where) {
    if (required <= current && ptr != nullptr) {
      return;
    }
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      current = 0;
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), required), where);
    current = required;
  }

  void release() {
    if (stream_ready_) {
      cudaStreamSynchronize(stream_);
    }
    if (dX_ != nullptr) cudaFree(dX_);
    if (dYtrain_ != nullptr) cudaFree(dYtrain_);
    if (dA_ != nullptr) cudaFree(dA_);
    if (dOmega_ != nullptr) cudaFree(dOmega_);
    if (dY_ != nullptr) cudaFree(dY_);
    if (dZ_ != nullptr) cudaFree(dZ_);
    if (dQ_ != nullptr) cudaFree(dQ_);
    if (dTransform_ != nullptr) cudaFree(dTransform_);
    if (dRRmat_ != nullptr) cudaFree(dRRmat_);
    if (dQQmat_ != nullptr) cudaFree(dQQmat_);
    if (dVVmat_ != nullptr) cudaFree(dVVmat_);
    if (dBcur_ != nullptr) cudaFree(dBcur_);
    if (dCoeff_ != nullptr) cudaFree(dCoeff_);
    if (dTvec_ != nullptr) cudaFree(dTvec_);
    if (dPvec_ != nullptr) cudaFree(dPvec_);
    if (dQvec_ != nullptr) cudaFree(dQvec_);
    if (dRvec_ != nullptr) cudaFree(dRvec_);
    if (dYfit_ != nullptr) cudaFree(dYfit_);
    if (dBsmall_ != nullptr) cudaFree(dBsmall_);
    if (dGram_ != nullptr) cudaFree(dGram_);
    if (dTau_ != nullptr) cudaFree(dTau_);
    if (dEvals_ != nullptr) cudaFree(dEvals_);
    if (dWorkQR_ != nullptr) cudaFree(dWorkQR_);
    if (dWorkEig_ != nullptr) cudaFree(dWorkEig_);
    if (dInfo_ != nullptr) cudaFree(dInfo_);
    dX_ = nullptr;
    dYtrain_ = nullptr;
    dA_ = nullptr;
    dOmega_ = nullptr;
    dY_ = nullptr;
    dZ_ = nullptr;
    dQ_ = nullptr;
    dTransform_ = nullptr;
    dRRmat_ = nullptr;
    dQQmat_ = nullptr;
    dVVmat_ = nullptr;
    dBcur_ = nullptr;
    dCoeff_ = nullptr;
    dTvec_ = nullptr;
    dPvec_ = nullptr;
    dQvec_ = nullptr;
    dRvec_ = nullptr;
    dYfit_ = nullptr;
    dBsmall_ = nullptr;
    dGram_ = nullptr;
    dTau_ = nullptr;
    dEvals_ = nullptr;
    dWorkQR_ = nullptr;
    dWorkEig_ = nullptr;
    dInfo_ = nullptr;
    bytes_X_ = bytes_Ytrain_ = 0;
    bytes_A_ = bytes_Omega_ = bytes_Y_ = bytes_Z_ = 0;
    bytes_Q_ = bytes_Bsmall_ = bytes_Gram_ = 0;
    bytes_Transform_ = 0;
    bytes_RRmat_ = bytes_QQmat_ = bytes_VVmat_ = bytes_Bcur_ = bytes_Coeff_ = 0;
    bytes_Tvec_ = bytes_Pvec_ = bytes_Qvec_ = bytes_Rvec_ = bytes_Yfit_ = 0;
    bytes_Tau_ = bytes_Evals_ = bytes_WorkQR_ = bytes_WorkEig_ = bytes_Info_ = 0;
    if (handle_ready_) {
      cublasDestroy(handle_);
      handle_ready_ = false;
    }
    if (solver_ready_) {
      cusolverDnDestroy(solver_);
      solver_ready_ = false;
    }
    if (rng_ready_) {
      curandDestroyGenerator(rng_);
      rng_ready_ = false;
    }
    if (stream_ready_) {
      cudaStreamDestroy(stream_);
      stream_ready_ = false;
      stream_ = nullptr;
    }
  }

  double* dX_ = nullptr;
  double* dYtrain_ = nullptr;
  double* dA_ = nullptr;
  double* dOmega_ = nullptr;
  double* dY_ = nullptr;
  double* dZ_ = nullptr;
  double* dQ_ = nullptr;
  double* dTransform_ = nullptr;
  double* dRRmat_ = nullptr;
  double* dQQmat_ = nullptr;
  double* dVVmat_ = nullptr;
  double* dBcur_ = nullptr;
  double* dCoeff_ = nullptr;
  double* dTvec_ = nullptr;
  double* dPvec_ = nullptr;
  double* dQvec_ = nullptr;
  double* dRvec_ = nullptr;
  double* dYfit_ = nullptr;
  double* dBsmall_ = nullptr;
  double* dGram_ = nullptr;
  double* dTau_ = nullptr;
  double* dEvals_ = nullptr;
  double* dWorkQR_ = nullptr;
  double* dWorkEig_ = nullptr;
  int* dInfo_ = nullptr;
  size_t bytes_X_ = 0;
  size_t bytes_Ytrain_ = 0;
  size_t bytes_A_ = 0;
  size_t bytes_Omega_ = 0;
  size_t bytes_Y_ = 0;
  size_t bytes_Z_ = 0;
  size_t bytes_Q_ = 0;
  size_t bytes_Transform_ = 0;
  size_t bytes_RRmat_ = 0;
  size_t bytes_QQmat_ = 0;
  size_t bytes_VVmat_ = 0;
  size_t bytes_Bcur_ = 0;
  size_t bytes_Coeff_ = 0;
  size_t bytes_Tvec_ = 0;
  size_t bytes_Pvec_ = 0;
  size_t bytes_Qvec_ = 0;
  size_t bytes_Rvec_ = 0;
  size_t bytes_Yfit_ = 0;
  size_t bytes_Bsmall_ = 0;
  size_t bytes_Gram_ = 0;
  size_t bytes_Tau_ = 0;
  size_t bytes_Evals_ = 0;
  size_t bytes_WorkQR_ = 0;
  size_t bytes_WorkEig_ = 0;
  size_t bytes_Info_ = 0;
  cublasHandle_t handle_ = nullptr;
  bool handle_ready_ = false;
  cusolverDnHandle_t solver_ = nullptr;
  bool solver_ready_ = false;
  curandGenerator_t rng_ = nullptr;
  bool rng_ready_ = false;
  cudaStream_t stream_ = nullptr;
  bool stream_ready_ = false;
  int hInfo_ = 0;
  const double one_ = 1.0;
  const double zero_ = 0.0;
  arma::mat hOmega_host_;
  arma::mat hGram_host_;
};

thread_local CudaRSVDWorkspace g_workspace;

bool runtime_available() {
  int n_devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&n_devices);
  return status == cudaSuccess && n_devices > 0;
}

}  // namespace kodama_fastpls_cuda

extern "C" bool kodama_fastpls_simpls_fit_cuda(
  const double* x_colmajor,
  int n,
  int p,
  const double* y_colmajor,
  int m,
  int max_components,
  int seed,
  double* rr_colmajor,
  double* qq_colmajor
) {
  using namespace kodama_fastpls_cuda;
  if (!runtime_available()) return false;
  if (n < 1 || p < 1 || m < 1 || max_components < 1) return false;

  const int max_ncomp = std::min(max_components, std::min(p, std::max(1, n - 1)));
  const int refresh_block = env_int_or("FASTPLS_FAST_BLOCK", 1, 1, 16);
  const int inc_power_iters = env_int_or("FASTPLS_FAST_INC_ITERS", 2, 1, 6);
  const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  const bool use_implicit_xprod = env_int_or("FASTPLS_GPU_SIMPLS_XPROD", 0, 0, 1) == 1;

  try {
    g_workspace.set_pls_training_matrices(x_colmajor, n, p, y_colmajor, m, false, !use_implicit_xprod);
    g_workspace.simpls_fast_begin_device_loop(n, p, m, max_ncomp, false);

    bool has_rr_prev = false;
    int a = 0;
    while (a < max_ncomp) {
      const int remaining = max_ncomp - a;
      const int k_block = std::min(refresh_block, remaining);
      std::vector<double> shat(static_cast<std::size_t>(k_block), 0.0);
      if (use_implicit_xprod) {
        g_workspace.simpls_fast_refresh_block_implicit_resident(
          n, p, m, k_block, k_block, a, has_rr_prev, static_cast<unsigned int>(seed + a), inc_power_iters, shat.data());
      } else {
        g_workspace.simpls_fast_refresh_block_resident(
          p, m, k_block, k_block, has_rr_prev, static_cast<unsigned int>(seed + a), inc_power_iters, shat.data());
      }

      bool stop_now = false;
      for (int j = 0; j < k_block && a < max_ncomp;) {
        bool appended = g_workspace.simpls_fast_append_component_from_block(
          n, p, m, a, j, a, reorth_v == 1, false, !use_implicit_xprod);
        if (!appended) {
          for (int retry = 0; retry < 8 && !appended; ++retry) {
            const int retry_l = std::min(std::min(p, m), std::max(2, std::min(32, k_block * (retry + 2))));
            double retry_shat = 0.0;
            const unsigned int retry_seed = static_cast<unsigned int>(seed + a + 7919 * (retry + 1));
            const int retry_power_iters = std::min(inc_power_iters + retry + 1, 8);
            if (use_implicit_xprod) {
              g_workspace.simpls_fast_refresh_block_implicit_resident(
                n, p, m, retry_l, 1, a, false, retry_seed, retry_power_iters, &retry_shat);
            } else {
              g_workspace.simpls_fast_refresh_block_resident(
                p, m, retry_l, 1, false, retry_seed, retry_power_iters, &retry_shat);
            }
            appended = g_workspace.simpls_fast_append_component_from_block(
              n, p, m, a, 0, a, reorth_v == 1, false, !use_implicit_xprod);
          }
        }
        if (!appended) {
          stop_now = true;
          break;
        }
        has_rr_prev = true;
        ++a;
        ++j;
      }
      if (stop_now) break;
    }

    g_workspace.simpls_fast_copy_rr(rr_colmajor, p, max_ncomp);
    g_workspace.simpls_fast_copy_qq(qq_colmajor, m, max_ncomp);
    return true;
  } catch (...) {
    return false;
  }
}
