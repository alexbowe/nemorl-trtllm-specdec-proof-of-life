#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/bootstrap_submodules.sh"
"$script_dir/apply_trtllm_patch.sh"
"$script_dir/apply_nemorl_patch.sh"

if [ "${NEMORL_TRTLLM_PROVISION_RUNTIME:-1}" = "1" ]; then
  "$script_dir/provision_runtime.sh"
fi

"$script_dir/prepare_trtllm_libs.sh"

if command -v ray >/dev/null 2>&1; then
  ray stop --force >/dev/null 2>&1 || true
fi

"$script_dir/preflight.sh"

if [ "${NEMORL_TRTLLM_SMOKE_MODE:-run}" = "preflight" ]; then
  exit 0
fi

"$script_dir/run_tiny_grpo.sh"
