#!/usr/bin/env bash
set -euo pipefail

trtllm_owner="${1:-alexbowe}"
rl_owner="${RL_FORK_OWNER:-}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

cd "$repo_root"

git config -f .gitmodules submodule.external/TensorRT-LLM.url "https://github.com/$trtllm_owner/TensorRT-LLM.git"
if [ -n "$rl_owner" ]; then
  git config -f .gitmodules submodule.external/RL.url "https://github.com/$rl_owner/RL.git"
fi
git submodule sync --recursive

printf 'TensorRT-LLM submodule URL now points at %s/TensorRT-LLM.\n' "$trtllm_owner"
if [ -n "$rl_owner" ]; then
  printf 'RL submodule URL now points at %s/RL.\n' "$rl_owner"
else
  printf 'RL submodule URL left unchanged; set RL_FORK_OWNER only if RL needs changes.\n'
fi
