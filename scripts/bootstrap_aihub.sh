#!/usr/bin/env bash
set -euo pipefail

repo_url="${NEMORL_TRTLLM_REPO_URL:-https://github.com/alexbowe/nemorl-trtllm-specdec-proof-of-life.git}"
ref="${NEMORL_TRTLLM_REF:-main}"
profile="${CLUSTER_PROFILE:-aihub}"

if [ -z "${DEV_ROOT:-}" ]; then
  user="${USER:-$(id -un)}"
  DEV_ROOT="$HOME/dev"
  for path in /lustre/fsw/portfolios/*/users/"$user"; do
    if [ -e "$path" ]; then
      DEV_ROOT="$path/dev"
      break
    fi
  done
fi
dev_root="$DEV_ROOT"
install_dir="${NEMORL_TRTLLM_INSTALL_DIR:-$dev_root/nemorl-trtllm-specdec-proof-of-life}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but was not found." >&2
  exit 1
fi

git_public() {
  git -c credential.helper= "$@"
}

mkdir -p "$dev_root"

if [ -d "$install_dir/.git" ]; then
  git_public -C "$install_dir" fetch origin "$ref"
  git_public -C "$install_dir" restore \
    --source="origin/$ref" \
    --worktree \
    --staged \
    -- .gitmodules README.md data patches requirements scripts
  git_public -C "$install_dir" checkout -B "$ref" "origin/$ref"
  git_public -C "$install_dir" submodule sync --recursive
  git_public -C "$install_dir" submodule update --init --recursive
else
  mkdir -p "$(dirname "$install_dir")"
  git_public clone --recurse-submodules --branch "$ref" "$repo_url" "$install_dir"
fi

DEV_ROOT="$dev_root" CLUSTER_PROFILE="$profile" "$install_dir/scripts/srun_smoke.sh"
