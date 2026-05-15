#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"

repo_root="$(repo_root_for_script "${BASH_SOURCE[0]}")"
profile="${CLUSTER_PROFILE:-$(detect_cluster_profile)}"
dev_root="${DEV_ROOT:-$(default_dev_root "$profile")}"
repo="${NEMORL_REPO:-$repo_root/external/RL}"
trt_repo="${TRTLLM_REPO:-$repo_root/external/TensorRT-LLM}"
venv="${NEMORL_TRTLLM_VENV:-$(default_venv "$dev_root")}"
run_root="${RUN_ROOT:-$dev_root/nemorl-trtllm-smoke}"
requirements_file="${NEMORL_TRTLLM_REQUIREMENTS:-$repo_root/requirements/runtime.txt}"
no_deps_requirements_file="${NEMORL_TRTLLM_NO_DEPS_REQUIREMENTS:-$repo_root/requirements/no-deps.txt}"
torch_build_requirements_file="${NEMORL_TRTLLM_TORCH_BUILD_REQUIREMENTS:-$repo_root/requirements/torch-build.txt}"

require_command python

if [ ! -d "$repo" ]; then
  echo "Missing Nemo-RL checkout: $repo" >&2
  exit 1
fi

if [ ! -d "$trt_repo" ]; then
  echo "Missing TensorRT-LLM checkout: $trt_repo" >&2
  exit 1
fi

python - <<'PY'
import sys

if sys.version_info < (3, 12):
    raise SystemExit(f"Python >=3.12 is required, got {sys.version.split()[0]}")
print(f"bootstrap_python={sys.version.split()[0]}")
PY

mkdir -p "$(dirname "$venv")" "$run_root/pip-cache" "$run_root/tmp"

if [ ! -x "$venv/bin/python" ]; then
  echo "Creating venv: $venv"
  python -m venv --system-site-packages "$venv"
fi

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$run_root/pip-cache}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.nvidia.com}"
export TMPDIR="${TMPDIR:-$run_root/tmp}"
constraints="$run_root/runtime-constraints.txt"
"$venv/bin/python" - "$constraints" <<'PY'
import importlib
import sys

path = sys.argv[1]
packages = ["torch", "torchvision", "torchaudio"]
with open(path, "w", encoding="utf-8") as f:
    for package in packages:
        try:
            mod = importlib.import_module(package)
        except Exception:
            continue
        version = getattr(mod, "__version__", "").split("+", 1)[0]
        if version:
            f.write(f"{package}=={version}\n")
PY

detect_cuda_archs="${NEMORL_TRTLLM_DETECT_CUDA_ARCH_LIST:-1}"
if [ "$detect_cuda_archs" = "1" ]; then
  detected_archs="$("$venv/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(0)

archs = {
    f"{major}.{minor}"
    for index in range(torch.cuda.device_count())
    for major, minor in [torch.cuda.get_device_capability(index)]
}
print(";".join(sorted(archs)))
PY
)"
  if [ -n "$detected_archs" ]; then
    export TORCH_CUDA_ARCH_LIST="$detected_archs"
  fi
fi
printf 'torch_cuda_arch_list=%s\n' "${TORCH_CUDA_ARCH_LIST:-<unset>}"

install_torch_build_deps="${NEMORL_TRTLLM_INSTALL_TORCH_BUILD_DEPS:-}"
if [ -z "$install_torch_build_deps" ]; then
  if [ "${NEMORL_TRTLLM_SMOKE_MODE:-run}" = "preflight" ]; then
    install_torch_build_deps=0
  else
    install_torch_build_deps=1
  fi
fi

"$venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$venv/bin/python" -m pip install --constraint "$constraints" -r "$requirements_file"
"$venv/bin/python" -m pip install --no-deps --constraint "$constraints" -r "$no_deps_requirements_file"
if [ "$install_torch_build_deps" = "1" ]; then
  "$venv/bin/python" -m pip install --no-build-isolation --no-cache-dir --constraint "$constraints" -r "$torch_build_requirements_file"
else
  echo "Skipping torch build dependencies for preflight"
fi

# Install local source trees without letting pip replace the container's torch stack.
"$venv/bin/python" -m pip install --no-deps -e "$repo"

automodel_repo="$repo/3rdparty/Automodel-workspace/Automodel"
if [ -f "$automodel_repo/pyproject.toml" ] || [ -f "$automodel_repo/setup.py" ]; then
  "$venv/bin/python" -m pip install --no-deps -e "$automodel_repo"
fi

echo "Runtime ready: $venv"
