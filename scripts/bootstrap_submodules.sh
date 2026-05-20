#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"

cd "$repo_root"
git -c credential.helper= -c submodule.recurse=false submodule sync \
  external/TensorRT-LLM \
  external/RL
git -c credential.helper= -c submodule.recurse=false submodule update --init \
  external/TensorRT-LLM \
  external/RL

printf 'TensorRT-LLM: %s\n' "$(git -C "$repo_root/external/TensorRT-LLM" rev-parse HEAD)"
printf 'RL: %s\n' "$(git -C "$repo_root/external/RL" rev-parse HEAD)"
printf '\nSubmodules pinned.\n'
