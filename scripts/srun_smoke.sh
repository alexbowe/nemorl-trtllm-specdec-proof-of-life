#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"

repo_root="$(repo_root_for_script "${BASH_SOURCE[0]}")"
profile="${CLUSTER_PROFILE:-$(detect_cluster_profile)}"
dev_root="${DEV_ROOT:-$(default_dev_root "$profile")}"
slurm_user="${SLURM_USER:-${USER:-$(id -un)}}"
job_name="${SLURM_JOB_NAME:-nemorl-trtllm-smoke-$$}"

case "$profile" in
  aihub)
    partition="${SLURM_PARTITION:-${AIHUB_PARTITION:-batch_short}}"
    gpus_per_node="${SLURM_GPUS_PER_NODE:-${AIHUB_GPUS_PER_NODE:-2}}"
    cpus_per_task="${SLURM_CPUS_PER_TASK:-${AIHUB_CPUS_PER_TASK:-32}}"
    time_limit="${SLURM_TIME:-${AIHUB_TIME:-02:00:00}}"
    account="${SLURM_ACCOUNT:-${AIHUB_ACCOUNT:-}}"
    exclude="${SLURM_EXCLUDE:-${AIHUB_EXCLUDE:-}}"
    if [ -z "$account" ]; then
      account="$(
        sacctmgr -nP show assoc where user="$(id -un)" format=account 2>/dev/null \
          | sed '/^$/d' \
          | sort -u \
          | head -n 1 \
          || true
      )"
    fi
    ;;
  computelab)
    partition="${SLURM_PARTITION:-${COMPUTELAB_PARTITION:-a100-sxm4-80gb@dvt/red-october@dvt/4gpu-128cpu-512gb}}"
    gpus_per_node="${SLURM_GPUS_PER_NODE:-${COMPUTELAB_GPUS_PER_NODE:-2}}"
    cpus_per_task="${SLURM_CPUS_PER_TASK:-${COMPUTELAB_CPUS_PER_TASK:-64}}"
    time_limit="${SLURM_TIME:-${COMPUTELAB_TIME:-02:00:00}}"
    account="${SLURM_ACCOUNT:-${COMPUTELAB_ACCOUNT:-}}"
    exclude="${SLURM_EXCLUDE:-${COMPUTELAB_EXCLUDE:-}}"
    ;;
  *)
    partition="${SLURM_PARTITION:-batch_short}"
    gpus_per_node="${SLURM_GPUS_PER_NODE:-2}"
    cpus_per_task="${SLURM_CPUS_PER_TASK:-32}"
    time_limit="${SLURM_TIME:-02:00:00}"
    account="${SLURM_ACCOUNT:-}"
    exclude="${SLURM_EXCLUDE:-}"
    ;;
esac

container_image="${CONTAINER_IMAGE:-${NEMORL_TRTLLM_CONTAINER_IMAGE:-${COMPUTELAB_CONTAINER_IMAGE:-}}}"
if [ -z "$container_image" ]; then
  if [ -f "$dev_root/trtllm_pytorch2512_trt1014.sqsh" ]; then
    container_image="$dev_root/trtllm_pytorch2512_trt1014.sqsh"
  elif [ "$profile" = "aihub" ]; then
    container_image="nvcr.io#nvidia/pytorch:25.10-py3"
  else
    container_image="nvcr.io#nvidia/pytorch:25.12-py3"
  fi
fi

mkdir -p "$dev_root"
cd "$repo_root"

srun_log="$(mktemp "${TMPDIR:-/tmp}/nemorl-trtllm-srun.XXXXXX")"
srun_args=(
  --job-name="$job_name"
  --partition="$partition"
  --nodes=1
  --ntasks=1
  --gpus-per-node="$gpus_per_node"
  --cpus-per-task="$cpus_per_task"
  --time="$time_limit"
  --container-image="$container_image"
  --container-mounts="$repo_root:$repo_root,$dev_root:$dev_root"
  --container-workdir="$repo_root"
)
if [ -n "$account" ]; then
  srun_args+=(--account="$account")
fi
if [ -n "$exclude" ]; then
  srun_args+=(--exclude="$exclude")
fi

printf 'profile=%s\n' "$profile"
printf 'dev_root=%s\n' "$dev_root"
printf 'container_image=%s\n' "$container_image"
printf 'partition=%s gpus=%s cpus=%s time=%s account=%s exclude=%s\n' \
  "$partition" "$gpus_per_node" "$cpus_per_task" "$time_limit" "${account:-<none>}" "${exclude:-<none>}"

srun "${srun_args[@]}" bash -lc 'scripts/smoke.sh' 2> >(tee "$srun_log" >&2) &

srun_pid=$!
job_id=""
missing_job_count=0
poll_seconds="${QUEUE_POLL_SECONDS:-${COMPUTELAB_QUEUE_POLL_SECONDS:-60}}"
missing_grace_seconds="${MISSING_JOB_GRACE_SECONDS:-${COMPUTELAB_MISSING_JOB_GRACE_SECONDS:-15}}"

while kill -0 "$srun_pid" >/dev/null 2>&1; do
  if [ -z "$job_id" ]; then
    job_id="$(sed -n 's/.*job \([0-9][0-9]*\) queued.*/\1/p' "$srun_log" | tail -n 1)"
    if [ -z "$job_id" ]; then
      job_id="$(
        squeue -h -u "$slurm_user" -n "$job_name" -o "%i" 2>/dev/null \
          | head -n 1 \
          || true
      )"
    fi
  fi
  if [ -n "$job_id" ]; then
    queue_line="$(squeue -h -j "$job_id" -o "%.18i %.12P %.18j %.8u %.2t %.10M %.10l %.6D %R" 2>/dev/null || true)"
    if [ -n "$queue_line" ]; then
      missing_job_count=0
      printf '%s\n' "$queue_line"
      sleep "$poll_seconds"
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
      sleep "$missing_grace_seconds"
    fi
  else
    sleep 5
  fi
done

wait "$srun_pid"
