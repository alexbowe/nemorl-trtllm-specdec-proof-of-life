#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
repo="${NEMORL_REPO:-$repo_root/external/RL}"
patch_path="$repo_root/patches/nemorl-torch-2.9-alias-patch.patch"

if [ ! -d "$repo/.git" ] && [ ! -f "$repo/.git" ]; then
  echo "Missing Nemo-RL submodule. Run scripts/bootstrap_submodules.sh first." >&2
  exit 1
fi

cd "$repo"

if git apply --reverse --check "$patch_path" >/dev/null 2>&1; then
  echo "Nemo-RL patch already applied."
  exit 0
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Nemo-RL submodule has local changes; refusing to apply patch over them." >&2
  git status --short
  exit 1
fi

git apply --check "$patch_path"
git apply "$patch_path"
echo "Nemo-RL patch applied."
