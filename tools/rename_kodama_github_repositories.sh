#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

set -euo pipefail

owner="${KODAMA_GITHUB_OWNER:-tkcaccia}"
workspace="${KODAMA_CPP_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
r_checkout="$workspace/split-repos/KODAMA"

command -v gh >/dev/null 2>&1 || {
  echo "GitHub CLI (gh) is required." >&2
  exit 1
}
gh auth status -h github.com >/dev/null

repo_name() {
  gh repo view "$owner/$1" --json name --jq .name 2>/dev/null || true
}

legacy_source=$(repo_name KODAMA)
wrapper_source=$(repo_name kodama-r)
legacy_target=$(repo_name KODAMAlegacy)

if [[ "$legacy_source" != "KODAMA" ]]; then
  if [[ "$legacy_target" == "KODAMAlegacy" && "$wrapper_source" != "kodama-r" ]]; then
    echo "Repository rename already completed."
  else
    echo "Expected $owner/KODAMA and $owner/kodama-r before migration." >&2
    exit 1
  fi
else
  [[ "$wrapper_source" == "kodama-r" ]] || {
    echo "Expected $owner/kodama-r before migration." >&2
    exit 1
  }
  [[ -z "$legacy_target" ]] || {
    echo "$owner/KODAMAlegacy already exists; refusing an ambiguous rename." >&2
    exit 1
  }

  gh repo rename KODAMAlegacy --repo "$owner/KODAMA" --yes
  gh repo rename KODAMA --repo "$owner/kodama-r" --yes
fi

[[ "$(repo_name KODAMAlegacy)" == "KODAMAlegacy" ]] || {
  echo "Could not verify $owner/KODAMAlegacy." >&2
  exit 1
}
[[ "$(repo_name KODAMA)" == "KODAMA" ]] || {
  echo "Could not verify $owner/KODAMA." >&2
  exit 1
}

if [[ -d "$r_checkout/.git" ]]; then
  git -C "$r_checkout" remote set-url origin "https://github.com/$owner/KODAMA.git"
fi

echo "Repository migration complete:"
echo "  https://github.com/$owner/KODAMAlegacy"
echo "  https://github.com/$owner/KODAMA"
