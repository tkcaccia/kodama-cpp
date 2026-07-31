// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "preprocessing_backend.hpp"

#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace kodama::detail {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* kPreprocessingMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

uint ordered_key(float value) {
  uint bits = as_type<uint>(value);
  return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

float key_value(uint key) {
  uint bits = (key & 0x80000000u) != 0u ? (key ^ 0x80000000u) : ~key;
  return as_type<float>(bits);
}

float segment_value(
  device const float* matrix, device const float* reference,
  uint segment, uint item, uint rows, uint cols, uint mode
) {
  if (mode == 0u) return matrix[segment * cols + item];
  if (mode == 1u) return matrix[item * cols + segment];
  return matrix[segment * cols + item] / reference[item];
}

float select_segment(
  device const float* matrix, device const float* reference,
  uint segment, uint count, uint rows, uint cols, uint mode,
  uint rank, bool remove_nan
) {
  if (!remove_nan) {
    for (uint i = 0; i < count; ++i) {
      if (isnan(segment_value(matrix, reference, segment, i, rows, cols, mode))) return NAN;
    }
  }
  uint prefix = 0u;
  uint mask = 0u;
  for (int bit = 31; bit >= 0; --bit) {
    uint bit_mask = 1u << uint(bit);
    uint zeros = 0u;
    for (uint i = 0; i < count; ++i) {
      float value = segment_value(matrix, reference, segment, i, rows, cols, mode);
      if (remove_nan && isnan(value)) continue;
      uint key = ordered_key(value);
      if ((key & mask) == prefix && (key & bit_mask) == 0u) ++zeros;
    }
    if (rank >= zeros) {
      prefix |= bit_mask;
      rank -= zeros;
    }
    mask |= bit_mask;
  }
  return key_value(prefix);
}

float segment_median(
  device const float* matrix, device const float* reference,
  uint segment, uint count, uint rows, uint cols, uint mode, bool remove_nan
) {
  uint valid = count;
  if (remove_nan) {
    valid = 0u;
    for (uint i = 0; i < count; ++i) {
      if (!isnan(segment_value(matrix, reference, segment, i, rows, cols, mode))) ++valid;
    }
  }
  if (valid == 0u) return NAN;
  float upper = select_segment(matrix, reference, segment, count, rows, cols, mode,
                               valid / 2u, remove_nan);
  if ((valid & 1u) != 0u) return upper;
  float lower = select_segment(matrix, reference, segment, count, rows, cols, mode,
                               valid / 2u - 1u, remove_nan);
  return (lower + upper) * 0.5f;
}

kernel void pqn_row_sum_divide(
  device float* matrix [[buffer(0)]], device float* sums [[buffer(1)]],
  constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= rows) return;
  float sum = 0.0f;
  for (uint column = 0; column < cols; ++column) {
    float value = matrix[row * cols + column];
    if (!isnan(value)) sum += abs(value);
  }
  sums[row] = sum;
  for (uint column = 0; column < cols; ++column) matrix[row * cols + column] /= sum;
}

kernel void preprocessing_column_medians(
  device const float* matrix [[buffer(0)]], device float* medians [[buffer(1)]],
  constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
  uint column [[thread_position_in_grid]]
) {
  if (column >= cols) return;
  medians[column] = segment_median(matrix, medians, column, rows, rows, cols, 1u, true);
}

kernel void pqn_finish(
  device float* matrix [[buffer(0)]], device const float* sums [[buffer(1)]],
  device const float* reference [[buffer(2)]], device float* coefficients [[buffer(3)]],
  constant uint& rows [[buffer(4)]], constant uint& cols [[buffer(5)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= rows) return;
  float coefficient = segment_median(matrix, reference, row, cols, rows, cols, 2u, true);
  coefficients[row] = coefficient * sums[row];
  for (uint column = 0; column < cols; ++column) matrix[row * cols + column] /= coefficient;
}

kernel void preprocessing_normalize_rows(
  device float* matrix [[buffer(0)]], device float* coefficients [[buffer(1)]],
  constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
  constant uint& method [[buffer(4)]], constant uint& test_matrix [[buffer(5)]],
  uint row [[thread_position_in_grid]]
) {
  if (row >= rows) return;
  float coefficient = 0.0f;
  if (method == 2u) {
    coefficient = segment_median(matrix, coefficients, row, cols, rows, cols, 0u,
                                 test_matrix != 0u);
  } else {
    for (uint column = 0; column < cols; ++column) {
      float value = matrix[row * cols + column];
      if (test_matrix != 0u && isnan(value)) continue;
      coefficient += method == 3u ? value * value :
        (test_matrix != 0u ? abs(value) : value);
    }
    if (method == 3u) coefficient = sqrt(coefficient);
  }
  coefficients[row] = coefficient;
  for (uint column = 0; column < cols; ++column) matrix[row * cols + column] /= coefficient;
}

kernel void preprocessing_scaling_statistics(
  device const float* matrix [[buffer(0)]], device float* centers [[buffer(1)]],
  device float* scales [[buffer(2)]], constant uint& rows [[buffer(3)]],
  constant uint& cols [[buffer(4)]], constant uint& method [[buffer(5)]],
  uint column [[thread_position_in_grid]]
) {
  if (column >= cols) return;
  float sum = 0.0f;
  for (uint row = 0; row < rows; ++row) sum += matrix[row * cols + column];
  float center = sum / float(rows);
  centers[column] = center;
  float scale = 1.0f;
  if (method == 2u || method == 4u) {
    float squares = 0.0f;
    for (uint row = 0; row < rows; ++row) {
      float delta = matrix[row * cols + column] - center;
      squares += delta * delta;
    }
    float sd = sqrt(squares / float(rows - 1u));
    scale = method == 4u ? sqrt(sd) : sd;
  } else if (method == 3u) {
    float minimum = matrix[column];
    float maximum = matrix[column];
    for (uint row = 1; row < rows; ++row) {
      float value = matrix[row * cols + column];
      if (isnan(value) || isnan(minimum) || isnan(maximum)) {
        minimum = NAN;
        maximum = NAN;
        break;
      }
      minimum = min(minimum, value);
      maximum = max(maximum, value);
    }
    scale = maximum - minimum;
  }
  scales[column] = scale;
}

kernel void preprocessing_apply_scaling(
  device float* matrix [[buffer(0)]], device const float* centers [[buffer(1)]],
  device const float* scales [[buffer(2)]], constant uint& items [[buffer(3)]],
  constant uint& cols [[buffer(4)]], uint index [[thread_position_in_grid]]
) {
  if (index >= items) return;
  uint column = index % cols;
  matrix[index] = (matrix[index] - centers[column]) / scales[column];
}
)METAL";

struct MetalPipelines {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pqn_sum = nil;
  id<MTLComputePipelineState> column_median = nil;
  id<MTLComputePipelineState> pqn_finish = nil;
  id<MTLComputePipelineState> normalize = nil;
  id<MTLComputePipelineState> statistics = nil;
  id<MTLComputePipelineState> apply = nil;
};

MetalPipelines& pipelines() {
  static MetalPipelines value;
  static std::once_flag once;
  static std::exception_ptr error;
  std::call_once(once, [&]() {
    try {
      value.device = MTLCreateSystemDefaultDevice();
      if (value.device == nil) throw std::runtime_error("No Apple Metal device is available.");
      value.queue = [value.device newCommandQueue];
      MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
      options.fastMathEnabled = NO;
      NSError* native_error = nil;
      id<MTLLibrary> library = [value.device
        newLibraryWithSource:[NSString stringWithUTF8String:kPreprocessingMetalSource]
        options:options error:&native_error];
      if (library == nil) {
        throw std::runtime_error(std::string("Failed to compile preprocessing Metal kernels: ") +
          [[native_error localizedDescription] UTF8String]);
      }
      auto make = [&](NSString* name) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (function == nil) throw std::runtime_error("A preprocessing Metal function is missing.");
        id<MTLComputePipelineState> state =
          [value.device newComputePipelineStateWithFunction:function error:&native_error];
        if (state == nil) throw std::runtime_error([[native_error localizedDescription] UTF8String]);
        return state;
      };
      value.pqn_sum = make(@"pqn_row_sum_divide");
      value.column_median = make(@"preprocessing_column_medians");
      value.pqn_finish = make(@"pqn_finish");
      value.normalize = make(@"preprocessing_normalize_rows");
      value.statistics = make(@"preprocessing_scaling_statistics");
      value.apply = make(@"preprocessing_apply_scaling");
    } catch (...) { error = std::current_exception(); }
  });
  if (error) std::rethrow_exception(error);
  return value;
}

id<MTLBuffer> buffer(MetalPipelines& state, const std::size_t bytes, const void* source = nullptr) {
  const std::size_t allocated = std::max<std::size_t>(bytes, sizeof(float));
  id<MTLBuffer> out = [state.device newBufferWithLength:allocated options:MTLResourceStorageModeShared];
  if (out == nil) throw std::runtime_error("Failed to allocate a preprocessing Metal buffer.");
  if (source != nullptr && bytes != 0) std::memcpy([out contents], source, bytes);
  return out;
}

void encode(
  id<MTLCommandBuffer> command, id<MTLComputePipelineState> pipeline,
  const std::vector<id<MTLBuffer>>& buffers, const std::vector<std::uint32_t>& constants,
  const std::size_t count
) {
  if (count == 0) return;
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  NSUInteger index = 0;
  for (id<MTLBuffer> item : buffers) [encoder setBuffer:item offset:0 atIndex:index++];
  for (const std::uint32_t value : constants) {
    [encoder setBytes:&value length:sizeof(value) atIndex:index++];
  }
  const NSUInteger width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 128);
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder endEncoding];
}

void finish(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) {
    throw std::runtime_error(std::string("Preprocessing Metal command failed: ") +
      [[command.error localizedDescription] UTF8String]);
  }
}

}  // namespace

NormalizationResult preprocessing_normalization_metal(
  const std::vector<float>& train, const int train_rows,
  const std::vector<float>& test, const int test_rows, const int variables,
  const NormalizationOptions& options
) {
  if (options.gpu_device != 0) throw std::invalid_argument("The Metal backend currently exposes device 0.");
  const auto started = Clock::now();
  MetalPipelines& state = pipelines();
  id<MTLBuffer> d_train = buffer(state, train.size() * sizeof(float), train.data());
  id<MTLBuffer> d_test = buffer(state, test.size() * sizeof(float), test.data());
  std::vector<float> ones_train(static_cast<std::size_t>(train_rows), 1.0f);
  std::vector<float> ones_test(static_cast<std::size_t>(test_rows), 1.0f);
  id<MTLBuffer> d_train_coeff = buffer(state, ones_train.size() * sizeof(float), ones_train.data());
  id<MTLBuffer> d_test_coeff = buffer(state, ones_test.size() * sizeof(float), ones_test.data());
  id<MTLCommandBuffer> command = [state.queue commandBuffer];

  NormalizationResult result;
  result.train = train; result.test = test;
  result.train_coefficients = ones_train; result.test_coefficients = ones_test;
  result.train_rows = train_rows; result.test_rows = test_rows; result.variables = variables;
  result.method = options.method; result.backend = Backend::Metal;
  const std::uint32_t tr = train_rows, te = test_rows, cols = variables;
  if (options.method == NormalizationMethod::PQN) {
    id<MTLBuffer> train_sums = buffer(state, static_cast<std::size_t>(train_rows) * sizeof(float));
    id<MTLBuffer> test_sums = buffer(state, static_cast<std::size_t>(test_rows) * sizeof(float));
    if (!options.reference.empty() && options.reference.size() != static_cast<std::size_t>(variables)) {
      throw std::invalid_argument("The PQN reference must contain one value per variable.");
    }
    result.reference = options.reference.empty() ?
      std::vector<float>(static_cast<std::size_t>(variables)) : options.reference;
    id<MTLBuffer> reference = buffer(state, result.reference.size() * sizeof(float), result.reference.data());
    encode(command, state.pqn_sum, {d_train, train_sums}, {tr, cols}, train_rows);
    if (options.reference.empty()) {
      encode(command, state.column_median, {d_train, reference}, {tr, cols}, variables);
    }
    encode(command, state.pqn_finish, {d_train, train_sums, reference, d_train_coeff},
      {tr, cols}, train_rows);
    if (test_rows > 0) {
      encode(command, state.pqn_sum, {d_test, test_sums}, {te, cols}, test_rows);
      encode(command, state.pqn_finish, {d_test, test_sums, reference, d_test_coeff},
        {te, cols}, test_rows);
    }
    finish(command);
    std::memcpy(result.reference.data(), [reference contents], result.reference.size() * sizeof(float));
  } else {
    if (options.method != NormalizationMethod::None) {
      const std::uint32_t method = static_cast<std::uint32_t>(options.method);
      encode(command, state.normalize, {d_train, d_train_coeff}, {tr, cols, method, 0u}, train_rows);
      if (test_rows > 0) encode(command, state.normalize, {d_test, d_test_coeff},
        {te, cols, method, 1u}, test_rows);
    }
    finish(command);
  }
  std::memcpy(result.train.data(), [d_train contents], result.train.size() * sizeof(float));
  if (!result.test.empty()) std::memcpy(result.test.data(), [d_test contents], result.test.size() * sizeof(float));
  std::memcpy(result.train_coefficients.data(), [d_train_coeff contents], result.train_coefficients.size() * sizeof(float));
  if (!result.test_coefficients.empty()) std::memcpy(result.test_coefficients.data(), [d_test_coeff contents], result.test_coefficients.size() * sizeof(float));
  result.runtime_seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

ScalingResult preprocessing_scaling_metal(
  const std::vector<float>& train, const int train_rows,
  const std::vector<float>& test, const int test_rows, const int variables,
  const ScalingOptions& options
) {
  if (options.gpu_device != 0) throw std::invalid_argument("The Metal backend currently exposes device 0.");
  const auto started = Clock::now();
  MetalPipelines& state = pipelines();
  id<MTLBuffer> d_train = buffer(state, train.size() * sizeof(float), train.data());
  id<MTLBuffer> d_test = buffer(state, test.size() * sizeof(float), test.data());
  std::vector<float> centers(static_cast<std::size_t>(variables), 0.0f);
  std::vector<float> scales(static_cast<std::size_t>(variables), 1.0f);
  id<MTLBuffer> d_center = buffer(state, centers.size() * sizeof(float), centers.data());
  id<MTLBuffer> d_scale = buffer(state, scales.size() * sizeof(float), scales.data());
  id<MTLCommandBuffer> command = [state.queue commandBuffer];
  const std::uint32_t tr = train_rows, te = test_rows, cols = variables;
  if (options.method != ScalingMethod::None) {
    const std::uint32_t method = static_cast<std::uint32_t>(options.method);
    encode(command, state.statistics, {d_train, d_center, d_scale}, {tr, cols, method}, variables);
    encode(command, state.apply, {d_train, d_center, d_scale},
      {static_cast<std::uint32_t>(train.size()), cols}, train.size());
    if (test_rows > 0) encode(command, state.apply, {d_test, d_center, d_scale},
      {static_cast<std::uint32_t>(test.size()), cols}, test.size());
  }
  finish(command);
  ScalingResult result;
  result.train = train; result.test = test; result.center = centers; result.scale = scales;
  result.train_rows = train_rows; result.test_rows = test_rows; result.variables = variables;
  result.method = options.method; result.backend = Backend::Metal;
  std::memcpy(result.train.data(), [d_train contents], result.train.size() * sizeof(float));
  if (!result.test.empty()) std::memcpy(result.test.data(), [d_test contents], result.test.size() * sizeof(float));
  std::memcpy(result.center.data(), [d_center contents], result.center.size() * sizeof(float));
  std::memcpy(result.scale.data(), [d_scale contents], result.scale.size() * sizeof(float));
  result.runtime_seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

}  // namespace kodama::detail
