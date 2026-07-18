#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Stefano Cacciatore
# SPDX-License-Identifier: MIT

set -euo pipefail

tag="${1:-v0.1.0}"
out_dir="${2:-dist}"
version="${tag#v}"
archive="$out_dir/kodama-cpp-$version.tar.gz"
checksum="$archive.sha256"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to archive a dirty worktree." >&2
  exit 1
fi
if ! git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
  echo "Missing release tag: $tag" >&2
  exit 1
fi
if [[ "$(git rev-parse HEAD)" != "$(git rev-list -n 1 "$tag")" ]]; then
  echo "HEAD is not the commit identified by $tag." >&2
  exit 1
fi

bash tools/check_license_headers.sh
mkdir -p "$out_dir"
git archive --format=tar --prefix="kodama-cpp-$version/" "$tag" | gzip -n > "$archive"

if tar -tzf "$archive" | grep -Eq '(^|/)(\.git|\.DS_Store|\._[^/]*)($|/)'; then
  echo "Release archive contains a forbidden metadata file." >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$archive" > "$checksum"
else
  shasum -a 256 "$archive" > "$checksum"
fi

printf 'Created %s\n' "$archive"
printf 'Commit %s\n' "$(git rev-parse HEAD)"
cat "$checksum"
