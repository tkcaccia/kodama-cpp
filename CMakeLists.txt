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
option(KODAMA_ENABLE_WARNINGS "Enable compiler warnings for kodama-cpp" ON)
set(KODAMA_SANITIZER "none" CACHE STRING "Runtime sanitizer: none, address, undefined, or thread")
set_property(CACHE KODAMA_SANITIZER PROPERTY STRINGS none address undefined thread)
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
  src/passing_message.cpp
  src/preprocessing.cpp
  src/plscv.cpp
  src/resident_ivf.cpp
  src/spatial_features.cpp
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

if(KODAMA_ENABLE_WARNINGS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(kodama_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-Wpedantic>
  )
endif()

if(NOT KODAMA_SANITIZER STREQUAL "none")
  if(KODAMA_ENABLE_CUDA)
    message(FATAL_ERROR "KODAMA_SANITIZER is intended for CPU/Metal validation builds; disable CUDA.")
  endif()
  if(NOT KODAMA_SANITIZER MATCHES "^(address|undefined|thread)$")
    message(FATAL_ERROR "KODAMA_SANITIZER must be none, address, undefined, or thread.")
  endif()
  # Inline standard-library operations may execute in consumers of this static
  # library. Instrument both the library and linked test/example targets so
  # libc++ container annotations remain consistent across translation units.
  target_compile_options(kodama_cpp PUBLIC -fsanitize=${KODAMA_SANITIZER} -fno-omit-frame-pointer)
  target_link_options(kodama_cpp PUBLIC -fsanitize=${KODAMA_SANITIZER})
endif()

if(KODAMA_ENABLE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "KODAMA_ENABLE_METAL=ON is supported only on Apple platforms.")
  endif()
  enable_language(OBJCXX)
  target_sources(kodama_cpp PRIVATE src/metal_backend.mm src/preprocessing_metal.mm)
  set_source_files_properties(src/metal_backend.mm src/preprocessing_metal.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
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
  message(STATUS "kodama-cpp CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES} (set CMAKE_CUDA_ARCHITECTURES explicitly for release binaries)")
  set(_KODAMA_CONDA_CUDA_LIB_DIRS)
  if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    list(APPEND _KODAMA_CONDA_CUDA_LIB_DIRS
      "$ENV{CONDA_PREFIX}/lib"
      "$ENV{CONDA_PREFIX}/targets/x86_64-linux/lib"
    )
    # Do not bake the nvcc toolkit's implicit /usr/local/cuda RPATH into build
    # executables. The documented LD_LIBRARY_PATH then selects one coherent
    # conda CUDA runtime instead of mixing toolkit and environment libraries.
    set(CMAKE_SKIP_BUILD_RPATH TRUE)
  endif()
  macro(kodama_find_cuda_library output name)
    unset(${output} CACHE)
    if(_KODAMA_CONDA_CUDA_LIB_DIRS)
      find_library(${output} NAMES ${name} PATHS ${_KODAMA_CONDA_CUDA_LIB_DIRS} NO_DEFAULT_PATH)
    endif()
    if(NOT ${output})
      find_library(${output}
        NAMES ${name}
        HINTS
          /usr/local/cuda/lib64
          /usr/local/cuda-13.0/targets/x86_64-linux/lib
          /usr/lib/x86_64-linux-gnu
      )
    endif()
  endmacro()
  unset(CUDA_TOOLKIT_INCLUDE_DIR CACHE)
  find_path(CUDA_TOOLKIT_INCLUDE_DIR
    NAMES cublas_v2.h cuda_runtime_api.h
    HINTS
      $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
      /usr/local/cuda/include
      /usr/local/cuda-13.0/targets/x86_64-linux/include
      /usr/include
  )
  kodama_find_cuda_library(CUDA_CUBLAS_LIBRARY cublas)
  kodama_find_cuda_library(CUDA_CUBLASLT_LIBRARY cublasLt)
  kodama_find_cuda_library(CUDA_RUNTIME_LIBRARY cudart)
  kodama_find_cuda_library(CUDA_CUSOLVER_LIBRARY cusolver)
  kodama_find_cuda_library(CUDA_CURAND_LIBRARY curand)
  kodama_find_cuda_library(CUDA_CUFFT_LIBRARY cufft)
  if(NOT CUDA_TOOLKIT_INCLUDE_DIR OR NOT CUDA_CUBLAS_LIBRARY OR NOT CUDA_CUBLASLT_LIBRARY OR NOT CUDA_RUNTIME_LIBRARY OR NOT CUDA_CUSOLVER_LIBRARY OR NOT CUDA_CURAND_LIBRARY OR NOT CUDA_CUFFT_LIBRARY)
    message(FATAL_ERROR "KODAMA_ENABLE_CUDA=ON requires CUDA headers, libcublas, libcublasLt, libcusolver, libcurand, libcufft, and libcudart.")
  endif()
  target_compile_definitions(kodama_cpp PRIVATE KODAMA_ENABLE_CUDA=1)
  target_include_directories(kodama_cpp PRIVATE ${CUDA_TOOLKIT_INCLUDE_DIR})
  if(_KODAMA_CONDA_CUDA_LIB_DIRS)
    target_link_directories(kodama_cpp BEFORE PRIVATE ${_KODAMA_CONDA_CUDA_LIB_DIRS})
  endif()
  target_link_libraries(kodama_cpp PRIVATE ${CUDA_CUBLAS_LIBRARY} ${CUDA_CUBLASLT_LIBRARY} ${CUDA_CUSOLVER_LIBRARY} ${CUDA_CURAND_LIBRARY} ${CUDA_CUFFT_LIBRARY} ${CUDA_RUNTIME_LIBRARY})
  target_sources(kodama_cpp PRIVATE src/pls_lda_cuda.cu src/kodama_matrix_cuda.cu src/embedding_cuda_kernels.cu src/cuda_simpls_float.cpp src/native_cuda_backend.cu src/pca_cuda.cu src/preprocessing_cuda.cu)
  target_compile_options(kodama_cpp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  set_source_files_properties(src/pls_lda_cuda.cu src/kodama_matrix_cuda.cu src/embedding_cuda_kernels.cu src/native_cuda_backend.cu src/pca_cuda.cu src/preprocessing_cuda.cu PROPERTIES LANGUAGE CUDA)
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
  docs/backend-validation.md
  docs/accelerator-entrypoint-validation.md
  docs/coverage.md
  docs/getting-started.md
  docs/mainpage.md
  docs/r-api.md
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
  add_executable(kodama_graph_storage_lifecycle benchmarks/run_graph_storage_lifecycle.cpp)
  target_link_libraries(kodama_graph_storage_lifecycle PRIVATE kodama_cpp)
  add_executable(kodama_resident_ivf_benchmark benchmarks/run_resident_ivf_benchmark.cpp)
  target_link_libraries(kodama_resident_ivf_benchmark PRIVATE kodama_cpp)

  add_executable(kodama_graph_backend_benchmark benchmarks/run_kodama_graph_backend_benchmark.cpp)
  target_link_libraries(kodama_graph_backend_benchmark PRIVATE kodama_cpp)
  add_executable(kodama_backend_benchmark_bin benchmarks/run_kodama_backend_bin.cpp)
  target_link_libraries(kodama_backend_benchmark_bin PRIVATE kodama_cpp)
  add_executable(kodama_dissimilarity_benchmark benchmarks/run_dissimilarity_benchmark.cpp)
  target_link_libraries(kodama_dissimilarity_benchmark PRIVATE kodama_cpp)
  if(KODAMA_ENABLE_CUDA)
    add_executable(kodama_cuda_knn_benchmark benchmarks/run_cuda_knn_benchmark.cpp)
    target_link_libraries(kodama_cuda_knn_benchmark PRIVATE kodama_cpp)
  endif()
  if(KODAMA_ENABLE_CUDA)
    target_compile_definitions(kodama_mnist70_sanity PRIVATE KODAMA_ENABLE_CUDA=1)
    target_compile_definitions(kodama_cv_sanity_bin PRIVATE KODAMA_ENABLE_CUDA=1)
    target_compile_definitions(kodama_core_sanity_bin PRIVATE KODAMA_ENABLE_CUDA=1)
    target_compile_definitions(kodama_backend_benchmark_bin PRIVATE KODAMA_ENABLE_CUDA=1)
  endif()
endif()

if(KODAMA_BUILD_TESTS)
  enable_testing()
  if(UNIX)
    add_test(
      NAME kodama_license_audit
      COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_license_headers.sh
    )
    add_test(
      NAME kodama_standalone_namespace_audit
      COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_standalone_namespace.sh
        ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_FILE:kodama_cpp>
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
