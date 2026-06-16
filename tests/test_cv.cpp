#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <set>
#include <vector>

#include "kodama/kodama.hpp"

namespace {

struct ToyData {
  std::vector<double> x;
  std::vector<int> y;
  std::vector<int> constrain;
  int n = 0;
  int p = 0;
};

ToyData make_toy_data() {
  ToyData d;
  const int classes = 3;
  const int n_per_class = 50;
  d.p = 6;
  d.n = classes * n_per_class;
  d.x.assign(static_cast<std::size_t>(d.n * d.p), 0.0);
  d.y.assign(static_cast<std::size_t>(d.n), 0);
  d.constrain.assign(static_cast<std::size_t>(d.n), 0);

  std::mt19937_64 rng(44);
  std::normal_distribution<double> noise(0.0, 0.25);
  for (int c = 0; c < classes; ++c) {
    for (int i = 0; i < n_per_class; ++i) {
      const int row = c * n_per_class + i;
      d.y[static_cast<std::size_t>(row)] = 10 + c;
      d.constrain[static_cast<std::size_t>(row)] = row / 2;
      for (int j = 0; j < d.p; ++j) {
        const double signal = (j == c || j == c + 3) ? 2.5 : -0.5;
        d.x[static_cast<std::size_t>(row * d.p + j)] = signal + noise(rng);
      }
    }
  }
  return d;
}

void check_constrained_folds(const std::vector<int>& constrain, const std::vector<int>& folds) {
  for (std::size_t i = 0; i < constrain.size(); ++i) {
    for (std::size_t j = i + 1; j < constrain.size(); ++j) {
      if (constrain[i] == constrain[j] && folds[i] != folds[j]) {
        throw std::runtime_error("Constraint group was split across folds.");
      }
    }
  }
}

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

}  // namespace

int main() {
  ToyData d = make_toy_data();
  kodama::MatrixView view{d.x.data(), static_cast<std::size_t>(d.n), static_cast<std::size_t>(d.p)};

  kodama::KNNOptions knn;
  knn.cv.folds = 5;
  knn.cv.seed = 1;
  knn.k = 7;
  knn.metric = kodama::DistanceMetric::Cosine;
  kodama::KNNCVResult kres = kodama::KNNCV(view, d.y, d.constrain, knn);
  require(kres.predicted_labels.size() == d.y.size(), "KNNCV prediction size mismatch.");
  require(kres.fold_assignments.size() == d.y.size(), "KNNCV fold size mismatch.");
  check_constrained_folds(d.constrain, kres.fold_assignments);
  require(kres.global_accuracy > 0.95, "KNNCV accuracy unexpectedly low.");
  require(kres.confusion.n_labels == 3, "KNNCV confusion matrix label count mismatch.");

  kodama::PLSOptions pls;
  pls.cv.folds = 5;
  pls.cv.seed = 1;
  pls.max_components = 4;
  pls.mode = kodama::PLSMode::PLS_DA;
  kodama::PLSCVResult pres = kodama::PLSCV(view, d.y, d.constrain, pls);
  require(pres.predicted_labels.size() == d.y.size(), "PLSCV prediction size mismatch.");
  require(pres.accuracy_by_components.size() == 4, "PLSCV component accuracy size mismatch.");
  require(pres.selected_components >= 1, "PLSCV selected too few components.");
  require(pres.selected_components <= 4, "PLSCV selected too many components.");
  require(pres.global_accuracy > 0.90, "PLS-DA accuracy unexpectedly low.");

  pls.mode = kodama::PLSMode::PLS_LDA;
  kodama::PLSCVResult lres = kodama::PLSCV(view, d.y, d.constrain, pls);
  require(lres.global_accuracy > 0.90, "PLS-LDA accuracy unexpectedly low.");

  std::cout << "All kodama-cpp CV tests passed.\n";
  return 0;
}
