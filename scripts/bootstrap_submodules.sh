#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

trt_sha="c31be54bb2c34d52cc710358bae31fcf8a43d5ae"
rl_sha="d69c8f638e390b407b89bc561355cfb4b196e131"

cd "$repo_root"
git submodule update --init --recursive

cd "$repo_root/external/TensorRT-LLM"
git fetch origin rick/specdec-driver535-fixes
git checkout "$trt_sha"

cd "$repo_root/external/RL"
git fetch origin rick/trtllm-specdec
git checkout "$rl_sha"

printf 'TensorRT-LLM: %s\n' "$(git -C "$repo_root/external/TensorRT-LLM" rev-parse HEAD)"
printf 'RL: %s\n' "$(git -C "$repo_root/external/RL" rev-parse HEAD)"
printf '\nSubmodules pinned. Run scripts/apply_trtllm_patch.sh next.\n'
