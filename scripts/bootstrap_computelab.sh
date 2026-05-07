#!/usr/bin/env bash
set -euo pipefail

repo_url="${NEMORL_TRTLLM_REPO_URL:-https://github.com/alexbowe/nemorl-trtllm-specdec-proof-of-life.git}"
install_dir="${NEMORL_TRTLLM_INSTALL_DIR:-${DEV_ROOT:-$HOME/dev}/nemorl-trtllm-specdec-proof-of-life}"
ref="${NEMORL_TRTLLM_REF:-main}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but was not found." >&2
  exit 1
fi

if [ -d "$install_dir/.git" ]; then
  if [ -n "$(git -C "$install_dir" status --porcelain)" ]; then
    echo "Existing checkout is not clean: $install_dir" >&2
    echo "Use NEMORL_TRTLLM_INSTALL_DIR=/new/path for a fresh checkout, or clean that checkout manually." >&2
    exit 1
  fi
  git -C "$install_dir" fetch origin "$ref"
  git -C "$install_dir" checkout "$ref"
  git -C "$install_dir" pull --ff-only origin "$ref"
else
  mkdir -p "$(dirname "$install_dir")"
  git clone --recurse-submodules --branch "$ref" "$repo_url" "$install_dir"
fi

"$install_dir/scripts/computelab_srun_smoke.sh"
