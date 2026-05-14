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
packages = ["torch", "torchvision", "torchaudio", "triton"]
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

"$venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$venv/bin/python" -m pip install --constraint "$constraints" -r "$requirements_file"
"$venv/bin/python" -m pip install --no-deps --constraint "$constraints" -r "$no_deps_requirements_file"
"$venv/bin/python" -m pip install --no-build-isolation --no-cache-dir --constraint "$constraints" -r "$torch_build_requirements_file"

# Install local source trees without letting pip replace the container's torch stack.
"$venv/bin/python" -m pip install --no-deps -e "$repo"

automodel_repo="$repo/3rdparty/Automodel-workspace/Automodel"
if [ -f "$automodel_repo/pyproject.toml" ] || [ -f "$automodel_repo/setup.py" ]; then
  "$venv/bin/python" -m pip install --no-deps -e "$automodel_repo"
fi

echo "Runtime ready: $venv"
