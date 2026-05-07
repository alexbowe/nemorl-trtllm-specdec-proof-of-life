#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
trt_repo="$repo_root/external/TensorRT-LLM"
patch_path="$repo_root/patches/trtllm-rick-mamba-multitoken-decode.patch"
base_sha="c31be54bb2c34d52cc710358bae31fcf8a43d5ae"
branch="${TRTLLM_PATCH_BRANCH:-abowe/rick-specdec-multitoken-fix}"
commit_patch="${COMMIT_PATCH:-0}"

if [ ! -d "$trt_repo/.git" ] && [ ! -f "$trt_repo/.git" ]; then
  echo "Missing TRTLLM submodule. Run scripts/bootstrap_submodules.sh first." >&2
  exit 1
fi

cd "$trt_repo"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "TRTLLM submodule has local changes; refusing to apply patch over them." >&2
  git status --short
  exit 1
fi

git checkout -B "$branch" "$base_sha"

if git apply --reverse --check "$patch_path" >/dev/null 2>&1; then
  echo "Patch already applied."
else
  git apply --check "$patch_path"
  git apply "$patch_path"
  echo "Patch applied."
fi

if [ "$commit_patch" = "1" ]; then
  git add tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py
  git commit -m "fix: support multi-token Mamba specdec decode" || true
  git rev-parse HEAD
else
  echo
  echo "Left TRTLLM patched but uncommitted."
  echo "To make a fork-backed submodule SHA, rerun with COMMIT_PATCH=1, push this branch to your fork, then commit the parent submodule pointer."
fi
