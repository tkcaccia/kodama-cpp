# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18)

project(kodama-python LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(KODAMA_CPP_ROOT "" CACHE PATH "Path to a kodama-cpp source checkout")
set(KODAMA_CPP_BUILD_DIR "" CACHE PATH "Path to a built kodama-cpp build directory")
set(KODAMA_CPP_EXTRA_LIBS "" CACHE STRING "Extra libraries needed when linking a static kodama-cpp")
set(KODAMA_CPP_RESOLVED_EXTRA_LIBS "")
set(KODAMA_CPP_WITH_CUDA OFF)

if(KODAMA_CPP_ROOT)
  if(NOT KODAMA_CPP_BUILD_DIR)
    set(KODAMA_CPP_BUILD_DIR "${KODAMA_CPP_ROOT}/build")
  endif()
  if(NOT EXISTS "${KODAMA_CPP_ROOT}/include/kodama/kodama.hpp")
    message(FATAL_ERROR "KODAMA_CPP_ROOT does not contain include/kodama/kodama.hpp")
  endif()
  if(EXISTS "${KODAMA_CPP_BUILD_DIR}/libkodama_cpp.a")
    set(KODAMA_CPP_LIBRARY "${KODAMA_CPP_BUILD_DIR}/libkodama_cpp.a")
  elseif(EXISTS "${KODAMA_CPP_BUILD_DIR}/libkodama_cpp${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(KODAMA_CPP_LIBRARY "${KODAMA_CPP_BUILD_DIR}/libkodama_cpp${CMAKE_SHARED_LIBRARY_SUFFIX}")
  else()
    message(FATAL_ERROR "Cannot find libkodama_cpp in KODAMA_CPP_BUILD_DIR=${KODAMA_CPP_BUILD_DIR}")
  endif()
  add_library(kodama::kodama_cpp UNKNOWN IMPORTED)
  set_target_properties(kodama::kodama_cpp PROPERTIES
    IMPORTED_LOCATION "${KODAMA_CPP_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${KODAMA_CPP_ROOT}/include"
  )
  if(EXISTS "${KODAMA_CPP_BUILD_DIR}/CMakeCache.txt")
    file(STRINGS "${KODAMA_CPP_BUILD_DIR}/CMakeCache.txt"
      KODAMA_CPP_CUDA_CACHE_LINE
      REGEX "^KODAMA_ENABLE_CUDA:BOOL=ON$")
    if(KODAMA_CPP_CUDA_CACHE_LINE)
      set(KODAMA_CPP_WITH_CUDA ON)
    endif()
  endif()
else()
  find_package(kodama-cpp CONFIG REQUIRED)
endif()

if(KODAMA_CPP_WITH_CUDA)
  if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
    set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}")
  endif()
  find_package(CUDAToolkit REQUIRED)
  list(APPEND KODAMA_CPP_RESOLVED_EXTRA_LIBS
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cusolver
    CUDA::cusparse
    CUDA::curand
    CUDA::cufft
  )
endif()

if(APPLE)
  set(KODAMA_PY_OPENMP_NAMES omp gomp iomp5)
else()
  set(KODAMA_PY_OPENMP_NAMES gomp omp iomp5)
endif()
find_library(KODAMA_PY_OPENMP_LIBRARY
  NAMES ${KODAMA_PY_OPENMP_NAMES}
  PATHS
    "$ENV{CONDA_PREFIX}/lib"
    /opt/homebrew/opt/libomp/lib
    /usr/local/lib
    /usr/lib
  NO_DEFAULT_PATH
)
if(KODAMA_PY_OPENMP_LIBRARY)
  list(APPEND KODAMA_CPP_RESOLVED_EXTRA_LIBS "${KODAMA_PY_OPENMP_LIBRARY}")
endif()

pybind11_add_module(_core MODULE src/kodama_cpp_python/bindings.cpp)
target_compile_features(_core PRIVATE cxx_std_17)
target_link_libraries(_core PRIVATE kodama::kodama_cpp ${KODAMA_CPP_RESOLVED_EXTRA_LIBS} ${KODAMA_CPP_EXTRA_LIBS})
if(APPLE)
  find_library(KODAMA_PY_METAL_FRAMEWORK Metal REQUIRED)
  find_library(KODAMA_PY_MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
  find_library(KODAMA_PY_FOUNDATION_FRAMEWORK Foundation REQUIRED)
  target_link_libraries(_core PRIVATE
    ${KODAMA_PY_METAL_FRAMEWORK}
    ${KODAMA_PY_MPS_FRAMEWORK}
    ${KODAMA_PY_FOUNDATION_FRAMEWORK}
  )
  target_link_options(_core PRIVATE
    "-Wl,-rpath,/opt/homebrew/opt/libomp/lib"
    "-Wl,-rpath,/opt/homebrew/opt/llvm/lib"
  )
endif()

install(TARGETS _core DESTINATION kodama)
