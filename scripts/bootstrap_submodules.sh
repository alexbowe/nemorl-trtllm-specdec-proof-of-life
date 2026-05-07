#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

cd "$repo_root"
git submodule update --init --recursive

printf 'TensorRT-LLM: %s\n' "$(git -C "$repo_root/external/TensorRT-LLM" rev-parse HEAD)"
printf 'RL: %s\n' "$(git -C "$repo_root/external/RL" rev-parse HEAD)"
printf '\nSubmodules pinned. Run scripts/apply_trtllm_patch.sh next.\n'
