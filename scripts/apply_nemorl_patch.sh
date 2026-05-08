#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
repo="${NEMORL_REPO:-$repo_root/external/RL}"
patches=(
  "$repo_root/patches/nemorl-torch-2.9-alias-patch.patch"
  "$repo_root/patches/nemorl-trtllm-kvcache.patch"
  "$repo_root/patches/nemorl-trtllm-clean-shutdown.patch"
  "$repo_root/patches/nemorl-trtllm-generation-clean-shutdown.patch"
)

if [ ! -d "$repo/.git" ] && [ ! -f "$repo/.git" ]; then
  echo "Missing Nemo-RL submodule. Run scripts/bootstrap_submodules.sh first." >&2
  exit 1
fi

cd "$repo"

for patch_path in "${patches[@]}"; do
  patch_name="$(basename "$patch_path")"
  if git apply --reverse --check "$patch_path" >/dev/null 2>&1; then
    echo "Nemo-RL patch already applied: $patch_name"
    continue
  fi

  if ! git apply --check "$patch_path"; then
    echo "Nemo-RL patch failed: $patch_name" >&2
    git status --short >&2
    exit 1
  fi

  git apply "$patch_path"
  echo "Nemo-RL patch applied: $patch_name"
done
