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

mkdir -p "$run_root" "$run_root/hf-cache" "$run_root/ray"

venv_site="$venv/lib/python3.12/site-packages"
venv_libs="$venv_site/torch/lib"
for libdir in "$venv_site"/nvidia/*/lib; do
  if [ -d "$libdir" ]; then
    venv_libs="$venv_libs:$libdir"
  fi
done

export PYTHONFAULTHANDLER=1
export TLLM_DISABLE_MPI=1
unset MPI4PY_RC_INITIALIZE
export LD_LIBRARY_PATH="$venv_libs:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//:\/usr\/local\/lib\/python3.12\/dist-packages\/torch\/lib/}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//:\/usr\/local\/lib\/python3.12\/dist-packages\/torch_tensorrt\/lib/}"
export PYTHONPATH="$repo:$trt_repo:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$run_root/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

while IFS='=' read -r name _; do
  case "$name" in
    SLURM_*) unset "$name" ;;
  esac
done < <(env)

cd "$repo"

check() {
  local name="$1"
  shift
  printf '\n== %s ==\n' "$name"
  "$venv/bin/python" -u "$@"
}

check "python and cuda" - <<'PY'
import os
import platform
import torch

print("arch", platform.machine(), flush=True)
print("python", platform.python_version(), flush=True)
print("torch", torch.__version__, flush=True)
print("cuda_available", torch.cuda.is_available(), flush=True)
print("cuda_version", torch.version.cuda, flush=True)
print("device_count", torch.cuda.device_count(), flush=True)
print("ld_library_path_has_cuda_runtime", "cuda_runtime" in os.environ.get("LD_LIBRARY_PATH", ""), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY

check "native libraries" - <<'PY'
import ctypes
import ctypes.util

required = {
    "cudart": ["libcudart.so.13", "libcudart.so.12", "libcudart.so"],
    "nccl": ["libnccl.so.2", "libnccl.so"],
}
for name, fallbacks in required.items():
    candidates = [ctypes.util.find_library(name), *fallbacks]
    for lib in filter(None, candidates):
        try:
            ctypes.CDLL(lib)
            print(f"{name}: OK {lib}", flush=True)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"Could not load native library: {name}")
print("nccl_find", ctypes.util.find_library("nccl"), flush=True)
PY

check "core imports" - <<'PY'
import importlib

for name in ["nemo_rl", "tensorrt_llm", "ray", "transformers"]:
    mod = importlib.import_module(name)
    print(f"{name}: OK {getattr(mod, '__file__', '')}", flush=True)
PY

check "vllm communicator imports" - <<'PY'
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

print("PyNcclCommunicator", PyNcclCommunicator, flush=True)
print("StatelessProcessGroup", StatelessProcessGroup, flush=True)
PY

check "nemo rl config load" - <<'PY'
from nemo_rl.utils.config import load_config
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
cfg = load_config("configs/grpo_qwen3_1.7b_specdec.yaml")
print("backend", cfg.policy.generation.backend, flush=True)
print("specdec", cfg.policy.generation.trtllm_cfg.speculative_decoding, flush=True)
PY

printf '\npreflight: OK\n'
