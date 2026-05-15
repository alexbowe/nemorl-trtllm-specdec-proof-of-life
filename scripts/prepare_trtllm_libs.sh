#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"

profile="${CLUSTER_PROFILE:-$(detect_cluster_profile)}"
dev_root="${DEV_ROOT:-$(default_dev_root "$profile")}"
trt_repo="${TRTLLM_REPO:-$repo_root/external/TensorRT-LLM}"
venv="${NEMORL_TRTLLM_VENV:-$(default_venv "$dev_root")}"
target_libs="$trt_repo/tensorrt_llm/libs"
required_lib="libnvinfer_plugin_tensorrt_llm.so"

if [ -e "$target_libs/$required_lib" ] && [ ! -L "$target_libs/$required_lib" ]; then
  echo "TRTLLM plugin libs already present: $target_libs"
  exit 0
fi

candidate_dirs=()
if [ -n "${TRTLLM_LIBS_SOURCE:-}" ]; then
  candidate_dirs+=("$TRTLLM_LIBS_SOURCE")
fi
candidate_dirs+=(
  "$dev_root/rick-tensorrt-llm-specdec-driver535-fixes/tensorrt_llm/libs"
  "$dev_root/TensorRT-LLM/tensorrt_llm/libs"
  "$venv/lib/python3.12/site-packages/tensorrt_llm/libs"
  "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs"
  "/usr/local/lib/python3.12/site-packages/tensorrt_llm/libs"
)

source_libs=""
for dir in "${candidate_dirs[@]}"; do
  if [ -f "$dir/$required_lib" ]; then
    source_libs="$dir"
    break
  fi
done

if [ -z "$source_libs" ]; then
  cat >&2 <<EOF
Missing TRTLLM plugin libs for source checkout:
  $target_libs/$required_lib

Set TRTLLM_LIBS_SOURCE=/path/to/tensorrt_llm/libs, or build TRTLLM first.
EOF
  exit 1
fi

mkdir -p "$target_libs"

link_artifact() {
  local source="$1"
  local target="$2"

  if [ -L "$target" ]; then
    if [ "$(readlink "$target")" != "$source" ]; then
      ln -sfn "$source" "$target"
    fi
  elif [ -e "$target" ]; then
    return
  else
    ln -s "$source" "$target"
  fi
}

for lib in "$source_libs"/*.so; do
  link_artifact "$lib" "$target_libs/$(basename "$lib")"
done

source_pkg="$(cd -- "$source_libs/.." && pwd)"
target_pkg="$(cd -- "$target_libs/.." && pwd)"
for artifact in "$source_pkg"/*.so "$source_pkg"/*.pyi "$source_pkg"/bindings "$source_pkg"/deep_gemm; do
  if [ -e "$artifact" ]; then
    link_artifact "$artifact" "$target_pkg/$(basename "$artifact")"
  fi
done

if [ ! -e "$target_libs/$required_lib" ]; then
  echo "Failed to link required TRTLLM plugin lib: $target_libs/$required_lib" >&2
  exit 1
fi

echo "Linked TRTLLM plugin libs from: $source_libs"
