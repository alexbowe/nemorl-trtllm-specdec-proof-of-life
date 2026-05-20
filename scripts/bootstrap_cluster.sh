#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"

repo_url="${NEMORL_TRTLLM_REPO_URL:-https://github.com/alexbowe/nemorl-trtllm-specdec-proof-of-life.git}"
ref="${NEMORL_TRTLLM_REF:-main}"
profile="${CLUSTER_PROFILE:-$(detect_cluster_profile)}"
dev_root="${DEV_ROOT:-$(default_dev_root "$profile")}"
install_dir="${NEMORL_TRTLLM_INSTALL_DIR:-$dev_root/nemorl-trtllm-specdec-proof-of-life}"
export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"

require_command git

mkdir -p "$dev_root"

if [ -d "$install_dir/.git" ]; then
  public_git -C "$install_dir" fetch origin "$ref"
  public_git -C "$install_dir" restore \
    --source="origin/$ref" \
    --worktree \
    --staged \
    -- .gitmodules README.md data patches requirements scripts
  public_git -C "$install_dir" checkout -B "$ref" "origin/$ref"
  public_git -C "$install_dir" -c submodule.recurse=false submodule sync \
    external/TensorRT-LLM \
    external/RL
  public_git -C "$install_dir" -c submodule.recurse=false submodule update --init \
    external/TensorRT-LLM \
    external/RL
else
  mkdir -p "$(dirname "$install_dir")"
  public_git clone --branch "$ref" "$repo_url" "$install_dir"
  public_git -C "$install_dir" -c submodule.recurse=false submodule sync \
    external/TensorRT-LLM \
    external/RL
  public_git -C "$install_dir" -c submodule.recurse=false submodule update --init \
    external/TensorRT-LLM \
    external/RL
fi

DEV_ROOT="$dev_root" CLUSTER_PROFILE="$profile" "$install_dir/scripts/srun_smoke.sh"
