#!/usr/bin/env bash
set -euo pipefail

repo_url="${NEMORL_TRTLLM_REPO_URL:-https://github.com/alexbowe/nemorl-trtllm-specdec-proof-of-life.git}"
ref="${NEMORL_TRTLLM_REF:-main}"
profile="${CLUSTER_PROFILE:-aihub}"
export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"

if [ -z "${DEV_ROOT:-}" ]; then
  user="${USER:-$(id -un)}"
  for path in /lustre/fsw/portfolios/*/users/"$user"; do
    if [ -d "$path" ] && [ -w "$path" ]; then
      DEV_ROOT="$path/dev"
      break
    fi
  done
  if [ -z "${DEV_ROOT:-}" ]; then
    cat >&2 <<EOF
No writable AIHub Lustre user directory was found.
Set DEV_ROOT to a writable large-storage path, for example:
  DEV_ROOT=/lustre/fsw/portfolios/<portfolio>/users/$user/dev

Home is intentionally not used by default because it is usually only 10G.
EOF
    exit 1
  fi
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

check_standalone_checkout() {
  local missing=0
  local required_paths=(
    ".gitmodules"
    "data/tiny_math_grpo.jsonl"
    "requirements/runtime.txt"
    "requirements/torch-build.txt"
    "patches/trtllm-mamba-multitoken-decode.patch"
    "patches/nemorl-torch-2.9-alias-patch.patch"
    "patches/nemorl-trtllm-kvcache.patch"
    "patches/nemorl-trtllm-clean-shutdown.patch"
    "patches/nemorl-trtllm-generation-clean-shutdown.patch"
    "scripts/common.sh"
    "scripts/bootstrap_submodules.sh"
    "scripts/apply_trtllm_patch.sh"
    "scripts/apply_nemorl_patch.sh"
    "scripts/provision_runtime.sh"
    "scripts/prepare_trtllm_libs.sh"
    "scripts/preflight.sh"
    "scripts/run_tiny_grpo.sh"
    "scripts/smoke.sh"
    "scripts/srun_smoke.sh"
  )

  for path in "${required_paths[@]}"; do
    if [ ! -e "$install_dir/$path" ]; then
      echo "Missing required repo file: $install_dir/$path" >&2
      missing=1
    fi
  done

  if [ "$missing" -ne 0 ]; then
    cat >&2 <<EOF
The checkout is incomplete. Re-run bootstrap with a new NEMORL_TRTLLM_INSTALL_DIR
or inspect the checkout at:
  $install_dir
EOF
    exit 1
  fi
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

check_standalone_checkout

DEV_ROOT="$dev_root" CLUSTER_PROFILE="$profile" "$install_dir/scripts/srun_smoke.sh"
