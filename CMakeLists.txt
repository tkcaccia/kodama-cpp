cmake_minimum_required(VERSION 3.18)

project(kodama-cpp
  VERSION 0.1.0
  DESCRIPTION "Standalone C++ kernels for high-performance KODAMA-related algorithms"
  LANGUAGES CXX
)

option(KODAMA_BUILD_TESTS "Build kodama-cpp tests" ON)
option(KODAMA_BUILD_EXAMPLES "Build kodama-cpp examples" ON)
option(KODAMA_ENABLE_CUDA "Build CUDA FAISS/cuVS backends" OFF)

add_library(kodama_cpp
  src/common.cpp
  src/core.cpp
  src/graph_cluster.cpp
  src/kodama_matrix.cpp
  src/knncv.cpp
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

find_path(FAISS_INCLUDE_DIR
  NAMES faiss/IndexIVFFlat.h
  HINTS /opt/homebrew/include /usr/local/include $ENV{CONDA_PREFIX}/include
)
find_library(FAISS_LIBRARY
  NAMES faiss
  HINTS /opt/homebrew/lib /usr/local/lib $ENV{CONDA_PREFIX}/lib
)
if(NOT FAISS_INCLUDE_DIR OR NOT FAISS_LIBRARY)
  message(FATAL_ERROR "FAISS is mandatory for kodama-cpp. Install FAISS or set FAISS_INCLUDE_DIR and FAISS_LIBRARY.")
endif()
target_include_directories(kodama_cpp PRIVATE ${FAISS_INCLUDE_DIR})
target_link_libraries(kodama_cpp PRIVATE ${FAISS_LIBRARY})
target_compile_definitions(kodama_cpp PRIVATE KODAMA_ENABLE_FAISS=1)

find_path(OPENMP_OMP_INCLUDE_DIR
  NAMES omp.h
  HINTS
    /opt/homebrew/opt/libomp/include
    /usr/local/opt/libomp/include
    $ENV{CONDA_PREFIX}/include
)
find_library(OPENMP_OMP_LIBRARY
  NAMES omp gomp iomp5
  HINTS
    /opt/homebrew/opt/libomp/lib
    /usr/local/opt/libomp/lib
    $ENV{CONDA_PREFIX}/lib
)
if(OPENMP_OMP_INCLUDE_DIR)
  target_include_directories(kodama_cpp PRIVATE ${OPENMP_OMP_INCLUDE_DIR})
endif()
if(OPENMP_OMP_LIBRARY)
  target_link_libraries(kodama_cpp PRIVATE ${OPENMP_OMP_LIBRARY})
else()
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(kodama_cpp PRIVATE OpenMP::OpenMP_CXX)
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
  find_path(ARMADILLO_INCLUDE_DIR
    NAMES armadillo
    HINTS /usr/include /usr/local/include $ENV{CONDA_PREFIX}/include
  )
  find_library(MKL_RT_LIBRARY
    NAMES mkl_rt
    HINTS $ENV{CONDA_PREFIX}/lib
  )
  if(NOT CUDA_TOOLKIT_INCLUDE_DIR OR NOT CUDA_CUBLAS_LIBRARY OR NOT CUDA_RUNTIME_LIBRARY OR NOT CUDA_CUSOLVER_LIBRARY OR NOT CUDA_CURAND_LIBRARY OR NOT CUDA_CUFFT_LIBRARY)
    message(FATAL_ERROR "KODAMA_ENABLE_CUDA=ON requires CUDA headers, libcublas, libcusolver, libcurand, libcufft, and libcudart.")
  endif()
  if(NOT ARMADILLO_INCLUDE_DIR)
    message(FATAL_ERROR "KODAMA_ENABLE_CUDA=ON requires Armadillo headers for the fastPLS CUDA SIMPLS backend.")
  endif()
  find_library(FAISS_GPU_LIBRARY
    NAMES faiss_gpu
    HINTS /usr/local/lib $ENV{CONDA_PREFIX}/lib
  )
  if(NOT FAISS_GPU_LIBRARY AND EXISTS "${FAISS_INCLUDE_DIR}/faiss/gpu/GpuIndexIVFFlat.h")
    set(FAISS_GPU_LIBRARY ${FAISS_LIBRARY})
  endif()
  find_path(CUVS_INCLUDE_DIR
    NAMES cuvs/neighbors/ivf_flat.hpp cuvs/neighbors/cagra.hpp
    HINTS /usr/local/include $ENV{CONDA_PREFIX}/include
  )
  if(NOT FAISS_GPU_LIBRARY)
    message(FATAL_ERROR "KODAMA_ENABLE_CUDA=ON requires FAISS GPU library (libfaiss_gpu).")
  endif()
  if(NOT CUVS_INCLUDE_DIR)
    message(FATAL_ERROR "KODAMA_ENABLE_CUDA=ON requires cuVS headers.")
  endif()
  find_library(CUVS_LIBRARY
    NAMES cuvs
    HINTS /usr/local/lib $ENV{CONDA_PREFIX}/lib
  )
  if(NOT CUVS_LIBRARY)
    message(FATAL_ERROR "KODAMA_ENABLE_CUDA=ON requires libcuvs.")
  endif()
  find_path(CUGRAPH_INCLUDE_DIR
    NAMES cugraph_c/community_algorithms.h
    HINTS /usr/local/include $ENV{CONDA_PREFIX}/include
  )
  find_library(CUGRAPH_LIBRARY
    NAMES cugraph_c
    HINTS /usr/local/lib $ENV{CONDA_PREFIX}/lib
  )
  if(CUGRAPH_INCLUDE_DIR AND CUGRAPH_LIBRARY)
    target_compile_definitions(kodama_cpp PRIVATE KODAMA_ENABLE_CUGRAPH=1)
    target_include_directories(kodama_cpp PRIVATE ${CUGRAPH_INCLUDE_DIR})
    target_link_libraries(kodama_cpp PRIVATE ${CUGRAPH_LIBRARY})
    message(STATUS "RAPIDS cuGraph found: enabling CUDA Louvain/Leiden graph clustering")
  else()
    message(STATUS "RAPIDS cuGraph not found: CUDA graph clustering will report unavailable")
  endif()
  target_compile_definitions(kodama_cpp PRIVATE KODAMA_ENABLE_CUDA=1 KODAMA_ENABLE_CUVS=1)
  target_include_directories(kodama_cpp PRIVATE ${CUDA_TOOLKIT_INCLUDE_DIR} ${CUVS_INCLUDE_DIR} ${ARMADILLO_INCLUDE_DIR})
  target_compile_options(kodama_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-I${ARMADILLO_INCLUDE_DIR}>)
  target_link_libraries(kodama_cpp PRIVATE ${FAISS_GPU_LIBRARY} ${CUVS_LIBRARY} ${CUDA_CUBLAS_LIBRARY} ${CUDA_CUSOLVER_LIBRARY} ${CUDA_CURAND_LIBRARY} ${CUDA_CUFFT_LIBRARY} ${CUDA_RUNTIME_LIBRARY})
  if(MKL_RT_LIBRARY)
    target_link_libraries(kodama_cpp PRIVATE ${MKL_RT_LIBRARY})
  endif()
  find_library(CONDA_STDCXX_LIBRARY
    NAMES stdc++
    HINTS $ENV{CONDA_PREFIX}/lib
    NO_DEFAULT_PATH
  )
  if(CONDA_STDCXX_LIBRARY)
    target_link_libraries(kodama_cpp PRIVATE ${CONDA_STDCXX_LIBRARY})
  endif()
  target_sources(kodama_cpp PRIVATE src/pls_cknn_cuda.cu src/pls_lda_cuda.cu src/kodama_matrix_cuda.cu src/embedding_cuda_kernels.cu src/fastpls_cuda_simpls.cpp)
  set_property(TARGET kodama_cpp PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
  set_source_files_properties(src/pls_cknn_cuda.cu src/pls_lda_cuda.cu src/kodama_matrix_cuda.cu src/embedding_cuda_kernels.cu PROPERTIES LANGUAGE CUDA)
endif()

include(GNUInstallDirs)
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

if(KODAMA_BUILD_EXAMPLES)
  add_executable(kodama_cv_example examples/run_cv.cpp)
  target_link_libraries(kodama_cv_example PRIVATE kodama_cpp)
  add_executable(kodama_mnist70_sanity benchmarks/run_mnist70_sanity.cpp)
  target_link_libraries(kodama_mnist70_sanity PRIVATE kodama_cpp)
  add_executable(kodama_cv_sanity_bin benchmarks/run_cv_sanity_bin.cpp)
  target_link_libraries(kodama_cv_sanity_bin PRIVATE kodama_cpp)
  add_executable(kodama_core_sanity_bin benchmarks/run_core_sanity_bin.cpp)
  target_link_libraries(kodama_core_sanity_bin PRIVATE kodama_cpp)
  target_include_directories(kodama_core_sanity_bin PRIVATE ${FAISS_INCLUDE_DIR})
  if(KODAMA_ENABLE_CUDA)
    target_compile_definitions(kodama_mnist70_sanity PRIVATE KODAMA_ENABLE_CUDA=1)
    target_compile_definitions(kodama_cv_sanity_bin PRIVATE KODAMA_ENABLE_CUDA=1)
    target_compile_definitions(kodama_core_sanity_bin PRIVATE KODAMA_ENABLE_CUDA=1)
  endif()
endif()

if(KODAMA_BUILD_TESTS)
  enable_testing()
  add_executable(kodama_cpp_tests tests/test_cv.cpp)
  target_link_libraries(kodama_cpp_tests PRIVATE kodama_cpp)
  if(KODAMA_ENABLE_CUDA)
    target_compile_definitions(kodama_cpp_tests PRIVATE KODAMA_ENABLE_CUDA=1)
  endif()
  add_test(NAME kodama_cpp_tests COMMAND kodama_cpp_tests)
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
