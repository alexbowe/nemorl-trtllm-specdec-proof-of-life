#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_PROFILE="${CLUSTER_PROFILE:-computelab}" "$script_dir/srun_smoke.sh"
