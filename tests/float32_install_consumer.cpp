// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <iostream>
#include <vector>

#include "kodama/kodama.hpp"

int main() {
  const std::size_t n = 6;
  const std::size_t p = 2;
  std::vector<float> x = {
    1.0f, 0.0f,
    1.1f, 0.1f,
    0.9f, -0.1f,
    -1.0f, 0.0f,
    -1.1f, -0.1f,
    -0.9f, 0.1f
  };
  std::vector<int> y = {1, 1, 1, 2, 2, 2};
  std::vector<int> constrain;

  kodama::MatrixView view{x.data(), n, p};
  if (view.value_type != kodama::MatrixValueType::Float32) return 2;

  kodama::KNNOptions options;
  options.cv.folds = 3;
  options.cv.seed = 3;
  options.k = 1;
  options.n_threads = 1;

  const kodama::KNNCVResult result = kodama::KNNCV_CPU(view, y, constrain, options);
  std::cout << result.global_accuracy << "\n";
  return result.global_accuracy >= 0.99 ? 0 : 1;
}
