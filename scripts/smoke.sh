#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"

profile="${CLUSTER_PROFILE:-$(detect_cluster_profile)}"
dev_root="${DEV_ROOT:-$(default_dev_root "$profile")}"
venv="${NEMORL_TRTLLM_VENV:-$(default_venv "$dev_root")}"

"$script_dir/bootstrap_submodules.sh"
if [ "${NEMORL_TRTLLM_APPLY_TRTLLM_PATCHES:-0}" = "1" ]; then
  "$script_dir/apply_trtllm_patch.sh"
fi
if [ "${NEMORL_TRTLLM_APPLY_NEMORL_PATCHES:-0}" = "1" ]; then
  "$script_dir/apply_nemorl_patch.sh"
fi

if [ "${NEMORL_TRTLLM_PROVISION_RUNTIME:-1}" = "1" ]; then
  "$script_dir/provision_runtime.sh"
fi

"$script_dir/prepare_trtllm_libs.sh"

if [ -x "$venv/bin/ray" ]; then
  "$venv/bin/ray" stop --force >/dev/null 2>&1 || true
elif command -v ray >/dev/null 2>&1; then
  ray stop --force >/dev/null 2>&1 || true
fi

"$script_dir/preflight.sh"

smoke_mode="${NEMORL_TRTLLM_SMOKE_MODE:-run}"
case "$smoke_mode" in
  preflight)
    exit 0
    ;;
  run)
    "$script_dir/run_tiny_grpo.sh"
    ;;
  *)
    "$script_dir/run_tiny_grpo.sh" "$smoke_mode"
    ;;
esac
