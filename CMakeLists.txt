cmake_minimum_required(VERSION 3.18)

project(kodama-cpp
  VERSION 0.1.0
  DESCRIPTION "Standalone C++ kernels for high-performance KODAMA-related algorithms"
  LANGUAGES CXX
)

option(KODAMA_BUILD_TESTS "Build kodama-cpp tests" ON)
option(KODAMA_BUILD_EXAMPLES "Build kodama-cpp examples" ON)
option(KODAMA_ENABLE_FAISS "Enable optional FAISS backend when available" OFF)
option(KODAMA_ENABLE_CUVS "Enable optional cuVS backend when available" OFF)

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

if(KODAMA_ENABLE_FAISS)
  find_path(FAISS_INCLUDE_DIR faiss/IndexFlat.h)
  find_library(FAISS_LIBRARY faiss)
  if(FAISS_INCLUDE_DIR AND FAISS_LIBRARY)
    target_compile_definitions(kodama_cpp PRIVATE KODAMA_ENABLE_FAISS=1)
    target_include_directories(kodama_cpp PRIVATE ${FAISS_INCLUDE_DIR})
    target_link_libraries(kodama_cpp PRIVATE ${FAISS_LIBRARY})
  else()
    message(WARNING "KODAMA_ENABLE_FAISS=ON but FAISS was not found. Building CPU-only fallback.")
  endif()
endif()

if(KODAMA_ENABLE_CUVS)
  target_compile_definitions(kodama_cpp PRIVATE KODAMA_ENABLE_CUVS=1)
  message(STATUS "cuVS backend requested. First phase exposes the build flag; native kernels are implemented in a later CUDA pass.")
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

