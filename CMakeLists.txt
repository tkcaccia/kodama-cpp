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
  src/knncv.cpp
  src/plscv.cpp
)

add_library(kodama::kodama_cpp ALIAS kodama_cpp)

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

if(KODAMA_ENABLE_CUDA)
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
  if(NOT CUDA_TOOLKIT_INCLUDE_DIR OR NOT CUDA_CUBLAS_LIBRARY OR NOT CUDA_RUNTIME_LIBRARY)
    message(FATAL_ERROR "KODAMA_ENABLE_CUDA=ON requires CUDA headers, libcublas, and libcudart.")
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
  target_compile_definitions(kodama_cpp PRIVATE KODAMA_ENABLE_CUDA=1 KODAMA_ENABLE_CUVS=1)
  target_include_directories(kodama_cpp PRIVATE ${CUDA_TOOLKIT_INCLUDE_DIR} ${CUVS_INCLUDE_DIR})
  target_link_libraries(kodama_cpp PRIVATE ${FAISS_GPU_LIBRARY} ${CUVS_LIBRARY} ${CUDA_CUBLAS_LIBRARY} ${CUDA_RUNTIME_LIBRARY})
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
endif()

if(KODAMA_BUILD_TESTS)
  enable_testing()
  add_executable(kodama_cpp_tests tests/test_cv.cpp)
  target_link_libraries(kodama_cpp_tests PRIVATE kodama_cpp)
  if(KODAMA_ENABLE_CUDA)
    target_compile_definitions(kodama_cpp_tests PRIVATE KODAMA_ENABLE_CUDA=1)
  endif()
  add_test(NAME kodama_cpp_tests COMMAND kodama_cpp_tests)
endif()
