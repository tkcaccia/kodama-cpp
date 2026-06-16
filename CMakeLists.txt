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
  find_library(FAISS_GPU_LIBRARY
    NAMES faiss_gpu
    HINTS /usr/local/lib $ENV{CONDA_PREFIX}/lib
  )
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
  target_compile_definitions(kodama_cpp PRIVATE KODAMA_ENABLE_CUDA=1 KODAMA_ENABLE_CUVS=1)
  target_include_directories(kodama_cpp PRIVATE ${CUVS_INCLUDE_DIR})
  target_link_libraries(kodama_cpp PRIVATE ${FAISS_GPU_LIBRARY})
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
  add_test(NAME kodama_cpp_tests COMMAND kodama_cpp_tests)
endif()
