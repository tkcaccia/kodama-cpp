// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "kodama/kodama.hpp"

int main() {
  const int n_per_class = 50;
  const int classes = 3;
  const int p = 6;
  const int n = n_per_class * classes;
  std::vector<float> x(static_cast<std::size_t>(n * p));
  std::vector<int> y(static_cast<std::size_t>(n));
  std::vector<int> constrain(static_cast<std::size_t>(n));

  std::mt19937_64 rng(44);
  std::normal_distribution<double> noise(0.0, 0.25);
  for (int c = 0; c < classes; ++c) {
    for (int i = 0; i < n_per_class; ++i) {
      const int row = c * n_per_class + i;
      y[static_cast<std::size_t>(row)] = c + 1;
      constrain[static_cast<std::size_t>(row)] = row / 2;  // keep paired replicates in the same CV fold
      for (int j = 0; j < p; ++j) {
        const double signal = (j == c || j == c + 3) ? 2.5 : -0.5;
        x[static_cast<std::size_t>(row * p + j)] = static_cast<float>(signal + noise(rng));
      }
    }
  }

  kodama::MatrixView view{x.data(), static_cast<std::size_t>(n), static_cast<std::size_t>(p)};

  kodama::KNNOptions knn;
  knn.cv.folds = 5;
  knn.cv.seed = 7;
  knn.k = 10;
  knn.metric = kodama::DistanceMetric::Cosine;
  kodama::KNNCVResult kres = kodama::KNNCV(view, y, constrain, knn);

  kodama::PLSOptions pls;
  pls.cv.folds = 5;
  pls.cv.seed = 7;
  pls.max_components = 4;
  kodama::PLSCVResult pres = kodama::PLSDACV(view, y, constrain, pls);

  std::cout << "KNNCV accuracy: " << kres.global_accuracy
            << " runtime: " << kres.runtime_seconds << " sec\n";
  std::cout << "PLSDACV accuracy: " << pres.global_accuracy
            << " selected components: " << pres.selected_components
            << " runtime: " << pres.runtime_seconds << " sec\n";
  return (kres.global_accuracy > 0.8 && pres.global_accuracy > 0.8) ? 0 : 1;
}
