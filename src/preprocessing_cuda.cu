// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "preprocessing_backend.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace kodama::detail {
namespace {

using Clock = std::chrono::steady_clock;

void cuda_check(const cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::size_t size) : size_(size) {
    if (size_ != 0) cuda_check(cudaMalloc(&data_, size_ * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { if (data_ != nullptr) cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return size_; }
 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

__device__ std::uint32_t ordered_key(const float value) {
  const std::uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000U) != 0U ? ~bits : (bits ^ 0x80000000U);
}

__device__ float key_value(const std::uint32_t key) {
  const std::uint32_t bits = (key & 0x80000000U) != 0U ?
    (key ^ 0x80000000U) : ~key;
  return __uint_as_float(bits);
}

template <class Getter>
__device__ float select_value(Getter getter, const int count, int rank, const bool remove_nan) {
  if (!remove_nan) {
    for (int i = 0; i < count; ++i) if (isnan(getter(i))) return nanf("");
  }
  std::uint32_t prefix = 0U;
  std::uint32_t mask = 0U;
  for (int bit = 31; bit >= 0; --bit) {
    const std::uint32_t bit_mask = 1U << bit;
    int zeros = 0;
    for (int i = 0; i < count; ++i) {
      const float value = getter(i);
      if (remove_nan && isnan(value)) continue;
      const std::uint32_t key = ordered_key(value);
      if ((key & mask) == prefix && (key & bit_mask) == 0U) ++zeros;
    }
    if (rank >= zeros) {
      prefix |= bit_mask;
      rank -= zeros;
    }
    mask |= bit_mask;
  }
  return key_value(prefix);
}

template <class Getter>
__device__ float device_median(Getter getter, const int count, const bool remove_nan) {
  int valid = count;
  if (remove_nan) {
    valid = 0;
    for (int i = 0; i < count; ++i) if (!isnan(getter(i))) ++valid;
  }
  if (valid == 0) return nanf("");
  const float upper = select_value(getter, count, valid / 2, remove_nan);
  if ((valid & 1) != 0) return upper;
  const float lower = select_value(getter, count, valid / 2 - 1, remove_nan);
  return (lower + upper) * 0.5f;
}

__global__ void pqn_row_sum_divide(float* matrix, float* sums, const int rows, const int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int column = 0; column < cols; ++column) {
    const float value = matrix[row * cols + column];
    if (!isnan(value)) sum += fabsf(value);
  }
  sums[row] = sum;
  for (int column = 0; column < cols; ++column) matrix[row * cols + column] /= sum;
}

__global__ void column_medians(
  const float* matrix, float* medians, const int rows, const int cols
) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= cols) return;
  const auto getter = [=] __device__ (const int row) { return matrix[row * cols + column]; };
  medians[column] = device_median(getter, rows, true);
}

__global__ void pqn_finish(
  float* matrix, const float* sums, const float* reference, float* coefficients,
  const int rows, const int cols
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const auto getter = [=] __device__ (const int column) {
    return matrix[row * cols + column] / reference[column];
  };
  const float coefficient = device_median(getter, cols, true);
  coefficients[row] = coefficient * sums[row];
  for (int column = 0; column < cols; ++column) matrix[row * cols + column] /= coefficient;
}

__global__ void normalize_rows(
  float* matrix, float* coefficients, const int rows, const int cols,
  const int method, const bool test_matrix
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float coefficient = 0.0f;
  if (method == static_cast<int>(NormalizationMethod::Median)) {
    const auto getter = [=] __device__ (const int column) { return matrix[row * cols + column]; };
    coefficient = device_median(getter, cols, test_matrix);
  } else {
    for (int column = 0; column < cols; ++column) {
      const float value = matrix[row * cols + column];
      if (test_matrix && isnan(value)) continue;
      coefficient += method == static_cast<int>(NormalizationMethod::Sqrt) ?
        value * value : (test_matrix ? fabsf(value) : value);
    }
    if (method == static_cast<int>(NormalizationMethod::Sqrt)) coefficient = sqrtf(coefficient);
  }
  coefficients[row] = coefficient;
  for (int column = 0; column < cols; ++column) matrix[row * cols + column] /= coefficient;
}

__global__ void scaling_statistics(
  const float* matrix, float* centers, float* scales,
  const int rows, const int cols, const int method
) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= cols) return;
  float sum = 0.0f;
  for (int row = 0; row < rows; ++row) sum += matrix[row * cols + column];
  const float center = sum / static_cast<float>(rows);
  centers[column] = center;
  float scale = 1.0f;
  if (method == static_cast<int>(ScalingMethod::Autoscaling) ||
      method == static_cast<int>(ScalingMethod::ParetoScaling)) {
    float squares = 0.0f;
    for (int row = 0; row < rows; ++row) {
      const float delta = matrix[row * cols + column] - center;
      squares += delta * delta;
    }
    const float sd = sqrtf(squares / static_cast<float>(rows - 1));
    scale = method == static_cast<int>(ScalingMethod::ParetoScaling) ? sqrtf(sd) : sd;
  } else if (method == static_cast<int>(ScalingMethod::RangeScaling)) {
    float minimum = matrix[column];
    float maximum = matrix[column];
    for (int row = 1; row < rows; ++row) {
      const float value = matrix[row * cols + column];
      if (isnan(value) || isnan(minimum) || isnan(maximum)) {
        minimum = nanf("");
        maximum = minimum;
        break;
      }
      minimum = fminf(minimum, value);
      maximum = fmaxf(maximum, value);
    }
    scale = maximum - minimum;
  }
  scales[column] = scale;
}

__global__ void apply_scaling(
  float* matrix, const float* centers, const float* scales,
  const int items, const int cols
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= items) return;
  const int column = index % cols;
  matrix[index] = (matrix[index] - centers[column]) / scales[column];
}

void upload(DeviceBuffer<float>& device, const std::vector<float>& host) {
  if (!host.empty()) cuda_check(cudaMemcpy(device.get(), host.data(), host.size() * sizeof(float),
    cudaMemcpyHostToDevice), "cudaMemcpy host to device");
}

void download(std::vector<float>& host, const DeviceBuffer<float>& device) {
  if (!host.empty()) cuda_check(cudaMemcpy(host.data(), device.get(), host.size() * sizeof(float),
    cudaMemcpyDeviceToHost), "cudaMemcpy device to host");
}

dim3 grid(const int count, const int block = 128) {
  return dim3(static_cast<unsigned>((count + block - 1) / block));
}

}  // namespace

NormalizationResult preprocessing_normalization_cuda(
  const std::vector<float>& train, const int train_rows,
  const std::vector<float>& test, const int test_rows, const int variables,
  const NormalizationOptions& options
) {
  const auto started = Clock::now();
  cuda_check(cudaSetDevice(options.gpu_device), "cudaSetDevice");
  DeviceBuffer<float> d_train(train.size()), d_test(test.size());
  DeviceBuffer<float> d_train_coeff(static_cast<std::size_t>(train_rows));
  DeviceBuffer<float> d_test_coeff(static_cast<std::size_t>(test_rows));
  upload(d_train, train); upload(d_test, test);

  NormalizationResult result;
  result.train = train; result.test = test;
  result.train_coefficients.assign(static_cast<std::size_t>(train_rows), 1.0f);
  result.test_coefficients.assign(static_cast<std::size_t>(test_rows), 1.0f);
  result.train_rows = train_rows; result.test_rows = test_rows; result.variables = variables;
  result.method = options.method; result.backend = Backend::CUDA;
  cuda_check(cudaMemcpy(d_train_coeff.get(), result.train_coefficients.data(),
    result.train_coefficients.size() * sizeof(float), cudaMemcpyHostToDevice), "initialize coefficients");
  if (test_rows > 0) cuda_check(cudaMemcpy(d_test_coeff.get(), result.test_coefficients.data(),
    result.test_coefficients.size() * sizeof(float), cudaMemcpyHostToDevice), "initialize test coefficients");

  constexpr int block = 128;
  if (options.method == NormalizationMethod::PQN) {
    DeviceBuffer<float> d_train_sums(static_cast<std::size_t>(train_rows));
    DeviceBuffer<float> d_test_sums(static_cast<std::size_t>(test_rows));
    DeviceBuffer<float> d_reference(static_cast<std::size_t>(variables));
    pqn_row_sum_divide<<<grid(train_rows), block>>>(d_train.get(), d_train_sums.get(), train_rows, variables);
    if (options.reference.empty()) {
      column_medians<<<grid(variables), block>>>(d_train.get(), d_reference.get(), train_rows, variables);
      result.reference.resize(static_cast<std::size_t>(variables));
    } else {
      if (options.reference.size() != static_cast<std::size_t>(variables)) {
        throw std::invalid_argument("The PQN reference must contain one value per variable.");
      }
      result.reference = options.reference;
      cuda_check(cudaMemcpy(d_reference.get(), result.reference.data(), result.reference.size() * sizeof(float),
        cudaMemcpyHostToDevice), "upload PQN reference");
    }
    pqn_finish<<<grid(train_rows), block>>>(d_train.get(), d_train_sums.get(), d_reference.get(),
      d_train_coeff.get(), train_rows, variables);
    if (test_rows > 0) {
      pqn_row_sum_divide<<<grid(test_rows), block>>>(d_test.get(), d_test_sums.get(), test_rows, variables);
      pqn_finish<<<grid(test_rows), block>>>(d_test.get(), d_test_sums.get(), d_reference.get(),
        d_test_coeff.get(), test_rows, variables);
    }
    if (options.reference.empty()) download(result.reference, d_reference);
  } else if (options.method != NormalizationMethod::None) {
    normalize_rows<<<grid(train_rows), block>>>(d_train.get(), d_train_coeff.get(), train_rows,
      variables, static_cast<int>(options.method), false);
    if (test_rows > 0) normalize_rows<<<grid(test_rows), block>>>(d_test.get(), d_test_coeff.get(),
      test_rows, variables, static_cast<int>(options.method), true);
  }
  cuda_check(cudaGetLastError(), "preprocessing normalization kernel");
  cuda_check(cudaDeviceSynchronize(), "preprocessing normalization synchronize");
  download(result.train, d_train); download(result.test, d_test);
  download(result.train_coefficients, d_train_coeff);
  download(result.test_coefficients, d_test_coeff);
  result.runtime_seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

ScalingResult preprocessing_scaling_cuda(
  const std::vector<float>& train, const int train_rows,
  const std::vector<float>& test, const int test_rows, const int variables,
  const ScalingOptions& options
) {
  const auto started = Clock::now();
  cuda_check(cudaSetDevice(options.gpu_device), "cudaSetDevice");
  DeviceBuffer<float> d_train(train.size()), d_test(test.size());
  DeviceBuffer<float> d_center(static_cast<std::size_t>(variables));
  DeviceBuffer<float> d_scale(static_cast<std::size_t>(variables));
  upload(d_train, train); upload(d_test, test);
  ScalingResult result;
  result.train = train; result.test = test;
  result.center.assign(static_cast<std::size_t>(variables), 0.0f);
  result.scale.assign(static_cast<std::size_t>(variables), 1.0f);
  result.train_rows = train_rows; result.test_rows = test_rows; result.variables = variables;
  result.method = options.method; result.backend = Backend::CUDA;
  cuda_check(cudaMemcpy(d_center.get(), result.center.data(), result.center.size() * sizeof(float),
    cudaMemcpyHostToDevice), "initialize centers");
  cuda_check(cudaMemcpy(d_scale.get(), result.scale.data(), result.scale.size() * sizeof(float),
    cudaMemcpyHostToDevice), "initialize scales");
  constexpr int block = 128;
  if (options.method != ScalingMethod::None) {
    scaling_statistics<<<grid(variables), block>>>(d_train.get(), d_center.get(), d_scale.get(),
      train_rows, variables, static_cast<int>(options.method));
    apply_scaling<<<grid(static_cast<int>(train.size())), block>>>(d_train.get(), d_center.get(),
      d_scale.get(), static_cast<int>(train.size()), variables);
    if (test_rows > 0) apply_scaling<<<grid(static_cast<int>(test.size())), block>>>(d_test.get(),
      d_center.get(), d_scale.get(), static_cast<int>(test.size()), variables);
  }
  cuda_check(cudaGetLastError(), "preprocessing scaling kernel");
  cuda_check(cudaDeviceSynchronize(), "preprocessing scaling synchronize");
  download(result.train, d_train); download(result.test, d_test);
  download(result.center, d_center); download(result.scale, d_scale);
  result.runtime_seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

}  // namespace kodama::detail
