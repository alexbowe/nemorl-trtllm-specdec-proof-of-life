#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

dev_root="${DEV_ROOT:-/home/scratch.abowe_other/dev}"
container_image="${COMPUTELAB_CONTAINER_IMAGE:-$dev_root/trtllm_pytorch2512_trt1014.sqsh}"
partition="${COMPUTELAB_PARTITION:-a100-sxm4-80gb@dvt/red-october@dvt/4gpu-128cpu-512gb}"
gpus_per_node="${COMPUTELAB_GPUS_PER_NODE:-2}"
cpus_per_task="${COMPUTELAB_CPUS_PER_TASK:-64}"
time_limit="${COMPUTELAB_TIME:-02:00:00}"

if [ ! -f "$container_image" ]; then
  echo "Missing container image: $container_image" >&2
  exit 1
fi

cd "$repo_root"

srun \
  --partition="$partition" \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node="$gpus_per_node" \
  --cpus-per-task="$cpus_per_task" \
  --time="$time_limit" \
  --container-image="$container_image" \
  --container-mounts="$repo_root:$repo_root,$dev_root:$dev_root" \
  --container-workdir="$repo_root" \
  bash -lc 'scripts/computelab_smoke.sh'
