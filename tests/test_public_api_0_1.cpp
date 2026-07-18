// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"

#include <string_view>
#include <vector>

int main() {
  static_assert(KODAMA_VERSION_MAJOR == 0);
  static_assert(KODAMA_VERSION_MINOR == 1);
  static_assert(KODAMA_VERSION_PATCH == 0);
  static_assert(std::string_view(kodama::version_string) == "0.1.0");

  using Labels = const std::vector<int>&;
  using KNNFn = kodama::KNNCVResult (*)(
      kodama::MatrixView, Labels, Labels, const kodama::KNNOptions&);
  using PLSFn = kodama::PLSCVResult (*)(
      kodama::MatrixView, Labels, Labels, const kodama::PLSOptions&);
  using CoreFn = kodama::CoreResult (*)(
      kodama::MatrixView, Labels, Labels, Labels, const kodama::CoreOptions&);
  using MatrixFn = kodama::KODAMAMatrixResult (*)(
      kodama::MatrixView, Labels, Labels, Labels, const kodama::KODAMAMatrixOptions&);
  using GraphMatrixFn = kodama::KODAMAMatrixResult (*)(
      const kodama::NeighborGraph&, int, Labels, Labels, Labels,
      const kodama::KODAMAMatrixOptions&);
  using DataGraphMatrixFn = kodama::KODAMAMatrixResult (*)(
      kodama::MatrixView, const kodama::NeighborGraph&, Labels, Labels, Labels,
      const kodama::KODAMAMatrixOptions&);
  using GraphFn = kodama::NeighborGraph (*)(
      kodama::MatrixView, const kodama::GraphClusterOptions&);
  using ClusterFn = kodama::GraphClusterResult (*)(
      const kodama::NeighborGraph&, int, const kodama::GraphClusterOptions&);
  using UMAPFn = kodama::EmbeddingResult (*)(
      const kodama::NeighborGraph&, const kodama::UMAPOptions&);
  using TSNEFn = kodama::EmbeddingResult (*)(
      const kodama::NeighborGraph&, const kodama::OpenTSNEOptions&);
  using PCAFn = kodama::PCAResult (*)(
      kodama::MatrixView, const kodama::PCAOptions&);

  const KNNFn knn[] = {
      &kodama::KNNCV, &kodama::KNNCV_CPU, &kodama::KNNCV_CUDA,
      &kodama::KNNCV_METAL};
  const PLSFn pls[] = {
      &kodama::PLSLDACV, &kodama::PLSLDACV_CPU, &kodama::PLSLDACV_CUDA,
      &kodama::PLSLDACV_METAL};
  const CoreFn core[] = {
      &kodama::CoreKNN, &kodama::CoreKNN_CPU, &kodama::CoreKNN_CUDA,
      &kodama::CoreKNN_METAL, &kodama::CorePLSLDA, &kodama::CorePLSLDA_CPU,
      &kodama::CorePLSLDA_CUDA, &kodama::CorePLSLDA_METAL};
  const MatrixFn matrix[] = {
      &kodama::KODAMAMatrix, &kodama::KODAMAMatrix_CPU,
      &kodama::KODAMAMatrix_CUDA, &kodama::KODAMAMatrix_METAL};
  const GraphMatrixFn matrix_graph[] = {
      &kodama::KODAMAMatrixFromGraph, &kodama::KODAMAMatrixFromGraph_CPU,
      &kodama::KODAMAMatrixFromGraph_CUDA,
      &kodama::KODAMAMatrixFromGraph_METAL};
  const DataGraphMatrixFn matrix_data_graph[] = {
      &kodama::KODAMAMatrixFromGraphData,
      &kodama::KODAMAMatrixFromGraphData_CPU,
      &kodama::KODAMAMatrixFromGraphData_CUDA,
      &kodama::KODAMAMatrixFromGraphData_METAL};
  const GraphFn graph[] = {
      &kodama::KODAMAKNNGraph, &kodama::KODAMAKNNGraph_CPU,
      &kodama::KODAMAKNNGraph_CUDA, &kodama::KODAMAKNNGraph_METAL};
  const ClusterFn cluster[] = {
      &kodama::KODAMAGraphCluster, &kodama::KODAMAGraphCluster_CPU};
  const UMAPFn umap[] = {&kodama::KODAMAUMAP_CPU, &kodama::KODAMAUMAP_CUDA};
  const TSNEFn tsne[] = {
      &kodama::KODAMAOpenTSNE_CPU, &kodama::KODAMAOpenTSNE_CUDA};
  const PCAFn pca[] = {
      &kodama::PCA, &kodama::PCA_CPU, &kodama::PCA_CUDA, &kodama::PCA_METAL};

  return knn[0] && pls[0] && core[0] && matrix[0] && matrix_graph[0] &&
                 matrix_data_graph[0] && graph[0] && cluster[0] && umap[0] &&
                 tsne[0] && pca[0]
             ? 0
             : 1;
}
