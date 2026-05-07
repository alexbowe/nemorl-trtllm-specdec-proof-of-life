#!/usr/bin/env bash
set -euo pipefail

owner="${1:-alexbowe}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

cd "$repo_root"

git config -f .gitmodules submodule.external/TensorRT-LLM.url "https://github.com/$owner/TensorRT-LLM.git"
git config -f .gitmodules submodule.external/RL.url "https://github.com/$owner/RL.git"
git submodule sync --recursive

printf 'Submodule URLs now point at %s forks.\n' "$owner"
printf 'Create those forks on GitHub first if they do not exist yet.\n'
