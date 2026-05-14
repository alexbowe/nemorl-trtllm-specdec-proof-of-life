#!/usr/bin/env bash

repo_root_for_script() {
  local script_path="$1"
  local script_dir
  script_dir="$(cd -- "$(dirname -- "$script_path")" && pwd)"
  cd -- "$script_dir/.." && pwd
}

public_git() {
  git -c credential.helper= "$@"
}

detect_cluster_profile() {
  local host
  host="$(hostname -f 2>/dev/null || hostname)"
  case "$host" in
    *cw-pdx-cs-001*|*cw-dfw-cs-001*) echo "aihub" ;;
    *computelab*) echo "computelab" ;;
    *) echo "generic" ;;
  esac
}

default_dev_root() {
  local profile="${1:-$(detect_cluster_profile)}"
  case "$profile" in
    aihub)
      local user="${USER:-$(id -un)}"
      local path
      for path in /lustre/fsw/portfolios/*/users/"$user"; do
        if [ -e "$path" ]; then
          echo "$path/dev"
          return
        fi
      done
      echo "$HOME/dev"
      ;;
    computelab)
      local user="${USER:-$(id -un)}"
      local scratch="/home/scratch.${user}_other"
      if [ -d "$scratch" ]; then
        echo "$scratch/dev"
      else
        echo "$HOME/dev"
      fi
      ;;
    *)
      echo "$HOME/dev"
      ;;
  esac
}

default_venv() {
  local dev_root="$1"
  local tag
  tag="$(python - <<'PY'
import re
import sys

py = f"py{sys.version_info.major}{sys.version_info.minor}"
try:
    import torch

    torch_version = torch.__version__
    cuda_version = torch.version.cuda or "cpu"
    tag = f"{py}-torch{torch_version}-cu{cuda_version}"
except Exception:
    tag = f"{py}-torchunknown"

print(re.sub(r"[^A-Za-z0-9._-]+", "_", tag))
PY
)"
  echo "$dev_root/venvs/trtllm-rick-$tag"
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$command_name is required but was not found." >&2
    exit 1
  fi
}
