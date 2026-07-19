#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${KODAMA_COVERAGE_BUILD_DIR:-"$ROOT_DIR/build-coverage"}
OUTPUT_DIR=${KODAMA_COVERAGE_OUTPUT_DIR:-"$ROOT_DIR/coverage"}
CLANGXX=${KODAMA_COVERAGE_CXX:-$(command -v clang++)}
LLVM_PROFDATA=${KODAMA_LLVM_PROFDATA:-$(command -v llvm-profdata)}
LLVM_COV=${KODAMA_LLVM_COV:-$(command -v llvm-cov)}

mkdir -p "$BUILD_DIR" "$OUTPUT_DIR" "$BUILD_DIR/profiles"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER="$CLANGXX" \
  -DCMAKE_CXX_FLAGS="-O0 -g -fprofile-instr-generate -fcoverage-mapping" \
  -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fprofile-instr-generate" \
  -DKODAMA_ENABLE_CUDA=OFF \
  -DKODAMA_ENABLE_METAL=OFF \
  -DKODAMA_ENABLE_OPENMP=OFF \
  -DKODAMA_BUILD_EXAMPLES=OFF \
  -DKODAMA_BUILD_TESTS=ON

cmake --build "$BUILD_DIR" --parallel

LLVM_PROFILE_FILE="$BUILD_DIR/profiles/cv-%p.profraw" "$BUILD_DIR/kodama_cpp_tests"
LLVM_PROFILE_FILE="$BUILD_DIR/profiles/api-%p.profraw" "$BUILD_DIR/kodama_public_api_0_1_tests"

"$LLVM_PROFDATA" merge -sparse "$BUILD_DIR"/profiles/*.profraw -o "$OUTPUT_DIR/kodama.profdata"

COMMON_ARGS=(
  "$BUILD_DIR/kodama_cpp_tests"
  "-object=$BUILD_DIR/kodama_public_api_0_1_tests"
  "-instr-profile=$OUTPUT_DIR/kodama.profdata"
  "-ignore-filename-regex=.*/(tests|benchmarks|examples|build[^/]*)/.*"
)

"$LLVM_COV" report "${COMMON_ARGS[@]}" > "$OUTPUT_DIR/coverage-summary.txt"
"$LLVM_COV" export "${COMMON_ARGS[@]}" -format=lcov > "$OUTPUT_DIR/coverage.lcov"
"$LLVM_COV" show "${COMMON_ARGS[@]}" \
  -format=html \
  -output-dir="$OUTPUT_DIR/html" \
  -show-line-counts-or-regions \
  -show-instantiations=false

cat "$OUTPUT_DIR/coverage-summary.txt"
