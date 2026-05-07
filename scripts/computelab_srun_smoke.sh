#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

dev_root="${DEV_ROOT:-$(dirname "$repo_root")}"
container_image="${COMPUTELAB_CONTAINER_IMAGE:-$dev_root/trtllm_pytorch2512_trt1014.sqsh}"
partition="${COMPUTELAB_PARTITION:-a100-sxm4-80gb@dvt/red-october@dvt/4gpu-128cpu-512gb}"
gpus_per_node="${COMPUTELAB_GPUS_PER_NODE:-2}"
cpus_per_task="${COMPUTELAB_CPUS_PER_TASK:-64}"
time_limit="${COMPUTELAB_TIME:-02:00:00}"

if [ ! -f "$container_image" ]; then
  echo "Missing container image: $container_image" >&2
  echo "Set COMPUTELAB_CONTAINER_IMAGE=/path/to/trtllm_pytorch2512_trt1014.sqsh." >&2
  exit 1
fi

cd "$repo_root"

srun_log="$(mktemp "${TMPDIR:-/tmp}/nemorl-trtllm-srun.XXXXXX")"

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
  bash -lc 'scripts/computelab_smoke.sh' \
  2> >(tee "$srun_log" >&2) &

srun_pid=$!
job_id=""
missing_job_count=0

while kill -0 "$srun_pid" >/dev/null 2>&1; do
  if [ -z "$job_id" ]; then
    job_id="$(sed -n 's/.*job \([0-9][0-9]*\) queued.*/\1/p' "$srun_log" | tail -n 1)"
  fi
  if [ -n "$job_id" ]; then
    queue_line="$(squeue -h -j "$job_id" -o "%.18i %.9P %.8j %.8u %.2t %.10M %.10l %.6D %R" 2>/dev/null || true)"
    if [ -n "$queue_line" ]; then
      missing_job_count=0
      printf '%s\n' "$queue_line"
      sleep "${COMPUTELAB_QUEUE_POLL_SECONDS:-60}"
    else
      missing_job_count=$((missing_job_count + 1))
      job_state="$(sacct -n -j "$job_id" --format=State%24,ExitCode,Elapsed -P 2>/dev/null | head -n 1 || true)"
      if [ -n "$job_state" ]; then
        printf 'Slurm job %s is no longer in squeue: %s\n' "$job_id" "$job_state" >&2
      else
        printf 'Slurm job %s is no longer in squeue.\n' "$job_id" >&2
      fi
      if [ "$missing_job_count" -ge 2 ]; then
        printf 'srun is still alive after job %s disappeared; stopping local wrapper.\n' "$job_id" >&2
        kill "$srun_pid" >/dev/null 2>&1 || true
        wait "$srun_pid" || true
        exit 1
      fi
      sleep "${COMPUTELAB_MISSING_JOB_GRACE_SECONDS:-15}"
    fi
  else
    sleep 5
  fi
done

wait "$srun_pid"
