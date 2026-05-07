#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/bootstrap_submodules.sh"
"$script_dir/apply_trtllm_patch.sh"
"$script_dir/apply_nemorl_patch.sh"
"$script_dir/prepare_trtllm_libs.sh"

if command -v ray >/dev/null 2>&1; then
  ray stop --force >/dev/null 2>&1 || true
fi

"$script_dir/preflight_computelab.sh"
"$script_dir/run_tiny_grpo.sh"
