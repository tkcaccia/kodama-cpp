#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

failures=0
checked=0
truncated_name='Stefano Cacc''ia'

is_source_file() {
  local file="$1"
  case "$file" in
    CMakeLists.txt|*/CMakeLists.txt|*.c|*.cc|*.cpp|*.cxx|*.cu|*.cuh|*.h|*.hh|*.hpp|*.hxx|*.m|*.mm|*.py|*.R|*.r|*.sh|*.cmake|*.cmake.in)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

report_missing() {
  printf 'license-audit: %s: missing %s\n' "$1" "$2" >&2
  failures=$((failures + 1))
}

require_literal() {
  local file="$1"
  local literal="$2"
  if ! grep -Fq "$literal" "$file"; then
    report_missing "$file" "$literal"
  fi
}

source_files() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git ls-files -z
    local release_candidate_files=(
      benchmarks/run_jmlr_ensemble_convergence.R
      benchmarks/run_jmlr_m_tcycle_sensitivity.R
      benchmarks/run_jmlr_release_validation.R
      benchmarks/run_jmlr_release_validation.sh
      include/kodama/version.hpp
      manuscript/build_jmlr_submission.py
      tests/test_public_api_0_1.cpp
      tools/check_license_headers.sh
      tools/make_release_archive.sh
      tools/run_classic_vs_kodama_visualization_benchmark.R
    )
    local file
    for file in "${release_candidate_files[@]}"; do
      if [[ -f "$file" ]] && ! git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
        printf '%s\0' "$file"
      fi
    done
    return
  fi
  find . \
    -path './build*' -prune -o \
    -path './.git' -prune -o \
    -type f -print0
}

while IFS= read -r -d '' file; do
  file="${file#./}"
  if ! is_source_file "$file"; then
    continue
  fi
  checked=$((checked + 1))
  require_literal "$file" "SPDX-FileCopyrightText:"
  require_literal "$file" "SPDX-License-Identifier:"
  if grep -Eq "${truncated_name}([^t]|$)" "$file"; then
    printf 'license-audit: %s: contains truncated author name %s\n' "$file" "$truncated_name" >&2
    failures=$((failures + 1))
  fi
done < <(source_files)

require_literal "src/native_knn.cpp" "SPDX-FileCopyrightText: Meta Platforms, Inc. and affiliates"
require_literal "src/native_knn.cpp" "SPDX-FileCopyrightText: 2026 Stefano Cacciatore"
require_literal "src/native_knn.cpp" "SPDX-License-Identifier: MIT"

require_literal "src/metal_backend.mm" "SPDX-FileCopyrightText: Meta Platforms, Inc. and affiliates"
require_literal "src/metal_backend.mm" "SPDX-FileCopyrightText: 2024 Sydney Bach, The Solace Project"
require_literal "src/metal_backend.mm" "SPDX-FileCopyrightText: 2026 Stefano Cacciatore"
require_literal "src/metal_backend.mm" "SPDX-License-Identifier: MIT AND Apache-2.0"
require_literal "src/metal_backend.mm" "modified standalone"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

check_digest() {
  local file="$1"
  local expected="$2"
  local actual
  actual="$(sha256_file "$file")"
  if [[ "$actual" != "$expected" ]]; then
    printf 'license-audit: %s: SHA-256 %s, expected %s\n' "$file" "$actual" "$expected" >&2
    failures=$((failures + 1))
  fi
}

check_digest "LICENSE" "b7781dc305363dd9674c0aad2c3f5bdc8b4b07044805f66f4a8064d9be2fcf88"
check_digest "licenses/FAISS-LICENSE" "52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85"
check_digest "licenses/CUVS-LICENSE" "756005f963846334943e8bfc08ef98cd254257d8467ac7a7ffd42a1be262f442"
check_digest "licenses/FAISS-MLX-LICENSE" "b8d7376c7b21f8f4895af89772e9de0558bb99515820ceb21ba4be4e96efffcc"

if ((failures > 0)); then
  printf 'License audit failed with %d finding(s).\n' "$failures" >&2
  exit 1
fi

printf 'License audit passed for %d tracked source files.\n' "$checked"
