#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

set -euo pipefail

root=${1:?repository root is required}
library=${2:-}

legacy_pattern='fastembedr_cuda_|FASTEMBEDR_CUDA_GRAPH_CAPTURE|FASTEMBEDR_TSNE_FFT_GRID'

if find "$root/src" "$root/include" "$root/tests" -type f \
    \( -name '*.cpp' -o -name '*.cu' -o -name '*.hpp' -o -name '*.h' -o -name '*.mm' \) \
    -exec grep -n -E "$legacy_pattern" {} +; then
  echo "Legacy FastEmbedR CUDA identifiers remain in standalone KODAMA sources." >&2
  exit 1
fi

if [[ -n "$library" && -f "$library" ]] && command -v nm >/dev/null 2>&1; then
  if nm -g "$library" 2>/dev/null | grep -E 'fastembedr_cuda_'; then
    echo "Legacy FastEmbedR CUDA symbols remain in the linked KODAMA library." >&2
    exit 1
  fi
fi

echo "Standalone namespace audit passed."
