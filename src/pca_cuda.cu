// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "pca_cuda_backend.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace kodama::detail {
namespace {

void check_cuda(const cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(const cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
      std::string(operation) + ": cuBLAS status " + std::to_string(static_cast<int>(status))
    );
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::size_t bytes) {
    check_cuda(cudaMalloc(&data_, bytes), "cudaMalloc(PCA)");
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
  }

  float* get() const { return static_cast<float*>(data_); }

 private:
  void* data_ = nullptr;
};

class CublasHandle {
 public:
  CublasHandle() {
    check_cublas(cublasCreate(&handle_), "cublasCreate(PCA)");
  }

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

  ~CublasHandle() {
    if (handle_ != nullptr) cublasDestroy(handle_);
  }

  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

std::vector<float> logical_column_major(
  const std::vector<float>& matrix,
  const int rows,
  const int cols,
  const bool transpose
) {
  const int output_rows = transpose ? cols : rows;
  const int output_cols = transpose ? rows : cols;
  std::vector<float> out(static_cast<std::size_t>(output_rows) * output_cols);
  for (int col = 0; col < output_cols; ++col) {
    for (int row = 0; row < output_rows; ++row) {
      out[static_cast<std::size_t>(col) * output_rows + row] = transpose ?
        matrix[static_cast<std::size_t>(col) * cols + row] :
        matrix[static_cast<std::size_t>(row) * cols + col];
    }
  }
  return out;
}

}  // namespace

std::vector<float> cuda_pca_matrix_multiply(
  const std::vector<float>& left,
  const int left_rows,
  const int left_cols,
  const std::vector<float>& right,
  const int right_rows,
  const int right_cols,
  const bool transpose_left,
  const bool transpose_right,
  const int gpu_device
) {
  const int output_rows = transpose_left ? left_cols : left_rows;
  const int inner_left = transpose_left ? left_rows : left_cols;
  const int inner_right = transpose_right ? right_cols : right_rows;
  const int output_cols = transpose_right ? right_rows : right_cols;
  if (left_rows < 1 || left_cols < 1 || right_rows < 1 || right_cols < 1 ||
      left.size() != static_cast<std::size_t>(left_rows) * left_cols ||
      right.size() != static_cast<std::size_t>(right_rows) * right_cols ||
      inner_left != inner_right) {
    throw std::invalid_argument("Non-conformable CUDA PCA matrix multiplication.");
  }
  check_cuda(cudaSetDevice(gpu_device), "cudaSetDevice(PCA)");

  const std::vector<float> left_column_major = logical_column_major(
    left, left_rows, left_cols, transpose_left
  );
  const std::vector<float> right_column_major = logical_column_major(
    right, right_rows, right_cols, transpose_right
  );
  std::vector<float> output_column_major(
    static_cast<std::size_t>(output_rows) * output_cols
  );
  const std::size_t left_bytes = left_column_major.size() * sizeof(float);
  const std::size_t right_bytes = right_column_major.size() * sizeof(float);
  const std::size_t output_bytes = output_column_major.size() * sizeof(float);
  DeviceBuffer device_left(left_bytes);
  DeviceBuffer device_right(right_bytes);
  DeviceBuffer device_output(output_bytes);
  CublasHandle handle;
  check_cuda(
    cudaMemcpy(device_left.get(), left_column_major.data(), left_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(PCA left H2D)"
  );
  check_cuda(
    cudaMemcpy(device_right.get(), right_column_major.data(), right_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(PCA right H2D)"
  );
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(
    cublasSgemm(
      handle.get(),
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      output_rows,
      output_cols,
      inner_left,
      &alpha,
      device_left.get(),
      output_rows,
      device_right.get(),
      inner_left,
      &beta,
      device_output.get(),
      output_rows
    ),
    "cublasSgemm(PCA)"
  );
  check_cuda(
    cudaMemcpy(output_column_major.data(), device_output.get(), output_bytes, cudaMemcpyDeviceToHost),
    "cudaMemcpy(PCA output D2H)"
  );

  std::vector<float> output(static_cast<std::size_t>(output_rows) * output_cols);
  for (int row = 0; row < output_rows; ++row) {
    for (int col = 0; col < output_cols; ++col) {
      output[static_cast<std::size_t>(row) * output_cols + col] =
        output_column_major[static_cast<std::size_t>(col) * output_rows + row];
    }
  }
  return output;
}

}  // namespace kodama::detail
