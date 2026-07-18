# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18)

project(kodama-cpp
  VERSION 0.1.0
  DESCRIPTION "Standalone float32 KODAMA kernels for CPU, CUDA, and Apple Metal"
  LANGUAGES CXX
)

option(KODAMA_BUILD_TESTS "Build kodama-cpp tests" ON)
option(KODAMA_BUILD_EXAMPLES "Build kodama-cpp examples" ON)
option(KODAMA_ENABLE_CUDA "Build package-owned CUDA backends" OFF)
if(APPLE)
  option(KODAMA_ENABLE_METAL "Build native Apple Metal backends" ON)
  option(KODAMA_ENABLE_OPENMP "Build optional OpenMP parallel CPU loops" OFF)
else()
  option(KODAMA_ENABLE_METAL "Build native Apple Metal backends" OFF)
  option(KODAMA_ENABLE_OPENMP "Build optional OpenMP parallel CPU loops" ON)
endif()

add_library(kodama_cpp
  src/common.cpp
  src/core.cpp
  src/graph_cluster.cpp
  src/kodama_matrix.cpp
  src/knncv.cpp
  src/native_knn.cpp
  src/pca.cpp
  src/plscv.cpp
  src/visualization.cpp
)

add_library(kodama::kodama_cpp ALIAS kodama_cpp)
set_target_properties(kodama_cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(kodama_cpp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(kodama_cpp PUBLIC cxx_std_17)

if(KODAMA_ENABLE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "KODAMA_ENABLE_METAL=ON is supported only on Apple platforms.")
  endif()
  enable_language(OBJCXX)
  target_sources(kodama_cpp PRIVATE src/metal_backend.mm)
  set_source_files_properties(src/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  target_compile_definitions(kodama_cpp PRIVATE KODAMA_ENABLE_METAL=1)
  target_link_libraries(kodama_cpp PRIVATE
    "-framework Metal"
    "-framework MetalPerformanceShaders"
    "-framework Foundation"
  )
else()
  target_sources(kodama_cpp PRIVATE src/metal_backend_stub.cpp)
endif()

set(KODAMA_CONFIG_NEEDS_OPENMP OFF)
if(KODAMA_ENABLE_OPENMP)
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(kodama_cpp PRIVATE OpenMP::OpenMP_CXX)
    set(KODAMA_CONFIG_NEEDS_OPENMP ON)
  else()
    message(WARNING "KODAMA_ENABLE_OPENMP=ON, but OpenMP CXX was not found; building without OpenMP.")
  endif()
endif()

if(KODAMA_ENABLE_CUDA)
  enable_language(CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  find_path(CUDA_TOOLKIT_INCLUDE_DIR
    NAMES cublas_v2.h cuda_runtime_api.h
    HINTS
      $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
      /usr/local/cuda/include
      /usr/local/cuda-13.0/targets/x86_64-linux/include
      /usr/include
  )
  find_library(CUDA_CUBLAS_LIBRARY
    NAMES cublas
    HINTS
      $ENV{CONDA_PREFIX}/targets/x86_64-linux/lib
      /usr/local/cuda/lib64
      /usr/local/cuda-13.0/targets/x86_64-linux/lib
      /usr/lib/x86_64-linux-gnu
  )
  find_library(CUDA_RUNTIME_LIBRARY
    NAMES cudart
    HINTS
      $ENV{CONDA_PREFIX}/targets/x86_64-linux/lib
      /usr/local/cuda/lib64
      /usr/local/cuda-13.0/targets/x86_64-linux/lib
      /usr/lib/x86_64-linux-gnu
  )
  find_library(CUDA_CUSOLVER_LIBRARY
    NAMES cusolver
    HINTS
      $ENV{CONDA_PREFIX}/targets/x86_64-linux/lib
      /usr/local/cuda/lib64
      /usr/local/cuda-13.0/targets/x86_64-linux/lib
      /usr/lib/x86_64-linux-gnu
  )
  find_library(CUDA_CURAND_LIBRARY
    NAMES curand
    HINTS
      $ENV{CONDA_PREFIX}/targets/x86_64-linux/lib
      /usr/local/cuda/lib64
      /usr/local/cuda-13.0/targets/x86_64-linux/lib
      /usr/lib/x86_64-linux-gnu
  )
  find_library(CUDA_CUFFT_LIBRARY
    NAMES cufft
    HINTS
      $ENV{CONDA_PREFIX}/targets/x86_64-linux/lib
      /usr/local/cuda/lib64
      /usr/local/cuda-13.0/targets/x86_64-linux/lib
      /usr/lib/x86_64-linux-gnu
  )
  if(NOT CUDA_TOOLKIT_INCLUDE_DIR OR NOT CUDA_CUBLAS_LIBRARY OR NOT CUDA_RUNTIME_LIBRARY OR NOT CUDA_CUSOLVER_LIBRARY OR NOT CUDA_CURAND_LIBRARY OR NOT CUDA_CUFFT_LIBRARY)
    message(FATAL_ERROR "KODAMA_ENABLE_CUDA=ON requires CUDA headers, libcublas, libcusolver, libcurand, libcufft, and libcudart.")
  endif()
  target_compile_definitions(kodama_cpp PRIVATE KODAMA_ENABLE_CUDA=1)
  target_include_directories(kodama_cpp PRIVATE ${CUDA_TOOLKIT_INCLUDE_DIR})
  target_link_libraries(kodama_cpp PRIVATE ${CUDA_CUBLAS_LIBRARY} ${CUDA_CUSOLVER_LIBRARY} ${CUDA_CURAND_LIBRARY} ${CUDA_CUFFT_LIBRARY} ${CUDA_RUNTIME_LIBRARY})
  target_sources(kodama_cpp PRIVATE src/pls_lda_cuda.cu src/kodama_matrix_cuda.cu src/embedding_cuda_kernels.cu src/cuda_simpls_float.cpp src/native_cuda_backend.cu src/pca_cuda.cu)
  set_property(TARGET kodama_cpp PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
  set_source_files_properties(src/pls_lda_cuda.cu src/kodama_matrix_cuda.cu src/embedding_cuda_kernels.cu src/native_cuda_backend.cu src/pca_cuda.cu PROPERTIES LANGUAGE CUDA)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS kodama_cpp
  EXPORT kodama-cppTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT kodama-cppTargets
  NAMESPACE kodama::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kodama-cpp
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/kodama-cppConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/kodama-cppConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/kodama-cppConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kodama-cpp
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/kodama-cppConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/kodama-cppConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kodama-cpp
)
install(FILES
  LICENSE
  PROVENANCE.md
  THIRD_PARTY_NOTICES.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/kodama-cpp
)
install(DIRECTORY licenses/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/kodama-cpp/third-party
)
install(FILES
  API_STABILITY.md
  CHANGELOG.md
  CITATION.cff
  CONTRIBUTING.md
  README.md
  RELEASE_CHECKLIST.md
  docs/previous-version-delta.md
  docs/release-validation.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/kodama-cpp
)

if(KODAMA_BUILD_EXAMPLES)
  add_executable(kodama_cv_example examples/run_cv.cpp)
  target_link_libraries(kodama_cv_example PRIVATE kodama_cpp)
  add_executable(kodama_mnist70_sanity benchmarks/run_mnist70_sanity.cpp)
  target_link_libraries(kodama_mnist70_sanity PRIVATE kodama_cpp)
  add_executable(kodama_cv_sanity_bin benchmarks/run_cv_sanity_bin.cpp)
  target_link_libraries(kodama_cv_sanity_bin PRIVATE kodama_cpp)
  add_executable(kodama_core_sanity_bin benchmarks/run_core_sanity_bin.cpp)
  target_link_libraries(kodama_core_sanity_bin PRIVATE kodama_cpp)
  add_executable(kodama_native_backend_benchmark benchmarks/run_native_backend_benchmark.cpp)
  target_link_libraries(kodama_native_backend_benchmark PRIVATE kodama_cpp)
  add_executable(kodama_backend_benchmark_bin benchmarks/run_kodama_backend_bin.cpp)
  target_link_libraries(kodama_backend_benchmark_bin PRIVATE kodama_cpp)
  if(KODAMA_ENABLE_CUDA)
    add_executable(kodama_cuda_knn_benchmark benchmarks/run_cuda_knn_benchmark.cpp)
    target_link_libraries(kodama_cuda_knn_benchmark PRIVATE kodama_cpp)
  endif()
  if(KODAMA_ENABLE_CUDA)
    target_compile_definitions(kodama_mnist70_sanity PRIVATE KODAMA_ENABLE_CUDA=1)
    target_compile_definitions(kodama_cv_sanity_bin PRIVATE KODAMA_ENABLE_CUDA=1)
    target_compile_definitions(kodama_core_sanity_bin PRIVATE KODAMA_ENABLE_CUDA=1)
  endif()
endif()

if(KODAMA_BUILD_TESTS)
  enable_testing()
  if(UNIX)
    add_test(
      NAME kodama_license_audit
      COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_license_headers.sh
    )
  endif()
  add_executable(kodama_cpp_tests tests/test_cv.cpp)
  target_link_libraries(kodama_cpp_tests PRIVATE kodama_cpp)
  add_executable(kodama_public_api_0_1_tests tests/test_public_api_0_1.cpp)
  target_link_libraries(kodama_public_api_0_1_tests PRIVATE kodama_cpp)
  if(KODAMA_ENABLE_METAL)
    target_compile_definitions(kodama_cpp_tests PRIVATE KODAMA_ENABLE_METAL=1)
    add_executable(kodama_metal_tests tests/test_metal.cpp)
    target_link_libraries(kodama_metal_tests PRIVATE kodama_cpp)
    add_test(NAME kodama_metal_tests COMMAND kodama_metal_tests)
    set_tests_properties(kodama_metal_tests PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  if(KODAMA_ENABLE_CUDA)
    target_compile_definitions(kodama_cpp_tests PRIVATE KODAMA_ENABLE_CUDA=1)
  endif()
  add_test(NAME kodama_cpp_tests COMMAND kodama_cpp_tests)
  add_test(NAME kodama_public_api_0_1_tests COMMAND kodama_public_api_0_1_tests)
  if(KODAMA_BUILD_EXAMPLES)
    if(KODAMA_ENABLE_CUDA)
      set(KODAMA_FLOAT32_SMOKE_CUDA ON)
    else()
      set(KODAMA_FLOAT32_SMOKE_CUDA OFF)
    endif()
    add_test(
      NAME kodama_float32_binary_smoke
      COMMAND
        ${CMAKE_COMMAND} -E env
          KODAMA_SMOKE_SKIP_BUILD=1
          KODAMA_BUILD_DIR=$<TARGET_FILE_DIR:kodama_cv_sanity_bin>
          KODAMA_ENABLE_CUDA=${KODAMA_FLOAT32_SMOKE_CUDA}
          bash ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/run_float32_smoke.sh
    )
  endif()
endif()
