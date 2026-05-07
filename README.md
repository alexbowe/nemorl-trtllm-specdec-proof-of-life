# Nemo-RL TRTLLM Specdec Proof Of Life

Reproducible smoke test for Nemo-RL GRPO with TRTLLM inference, Nemotron 3 Nano,
and draft-target speculative decoding.

This is not a benchmark. Success means the stack reaches one
generation/logprob/training step and exits after `max_num_steps=1`.

## Quick Run

From a computelab login shell:

```bash
curl -fsSL https://raw.githubusercontent.com/alexbowe/nemorl-trtllm-specdec-proof-of-life/main/scripts/bootstrap_computelab.sh | env \
  DEV_ROOT=/path/to/scratch/dev \
  bash
```

For this environment:

```bash
curl -fsSL https://raw.githubusercontent.com/alexbowe/nemorl-trtllm-specdec-proof-of-life/main/scripts/bootstrap_computelab.sh | env \
  DEV_ROOT=/home/scratch.abowe_other/dev \
  bash
```

The bootstrap script clones or updates this repo under `$DEV_ROOT`, reserves a
GPU node with `srun`, starts the TRTLLM `.sqsh` container, runs preflight checks,
and launches the tiny GRPO smoke.

If an old checkout exists and is dirty, choose a fresh install path:

```bash
curl -fsSL https://raw.githubusercontent.com/alexbowe/nemorl-trtllm-specdec-proof-of-life/main/scripts/bootstrap_computelab.sh | env \
  DEV_ROOT=/home/scratch.abowe_other/dev \
  NEMORL_TRTLLM_INSTALL_DIR=/home/scratch.abowe_other/dev/nemorl-trtllm-specdec-proof-of-life-fresh \
  bash
```

Expected defaults next to the repo:

- container: `$DEV_ROOT/trtllm_pytorch2512_trt1014.sqsh`
- venv: `$DEV_ROOT/venvs/trtllm-rick-py312`

Override paths as needed:

```bash
curl -fsSL https://raw.githubusercontent.com/alexbowe/nemorl-trtllm-specdec-proof-of-life/main/scripts/bootstrap_computelab.sh | env \
  DEV_ROOT=/path/to/scratch/dev \
  COMPUTELAB_CONTAINER_IMAGE=/path/to/trtllm_pytorch2512_trt1014.sqsh \
  NEMORL_TRTLLM_VENV=/path/to/venvs/trtllm-rick-py312 \
  bash
```

Useful scheduler overrides:

- `COMPUTELAB_PARTITION`
- `COMPUTELAB_GPUS_PER_NODE`
- `COMPUTELAB_CPUS_PER_TASK`
- `COMPUTELAB_TIME`
- `COMPUTELAB_QUEUE_POLL_SECONDS`

## Manual Run

If the repo is already cloned:

```bash
/path/to/nemorl-trtllm-specdec-proof-of-life/scripts/computelab_srun_smoke.sh
```

If already inside a suitable Pyxis/Enroot allocation:

```bash
scripts/computelab_smoke.sh
```

For step-by-step debugging:

```bash
scripts/bootstrap_submodules.sh
scripts/apply_trtllm_patch.sh
scripts/preflight_computelab.sh
scripts/run_tiny_grpo.sh
```

## What It Runs

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- Backend: TRTLLM
- Specdec: `draft_target`
- Task: tiny math GRPO with `hf_math_verify`
- Run size: one GRPO step, one prompt per step, two generations per prompt

Validated smoke result:

- `max_draft_len=4`
- final signal: `Max number of steps has been reached`
- observed reward: `Avg Reward: 0.5000`

The reward is only a smoke-test signal.

## Sources

- TensorRT-LLM submodule: `alexbowe/TensorRT-LLM`, branch `abowe/rick-specdec-multitoken-fix`, commit `2b617b2f2c8fbbdf41eb1720f473c1ae926522e5`
- Nemo-RL submodule: `ricklamers-nvidia/RL`, branch `rick/trtllm-specdec`, commit `d69c8f638e390b407b89bc561355cfb4b196e131`
- TRTLLM base branch: `ricklamers-nvidia/TensorRT-LLM`, branch `rick/specdec-driver535-fixes`, commit `c31be54bb2c34d52cc710358bae31fcf8a43d5ae`
- Review patch: `patches/trtllm-mamba-multitoken-decode.patch`

The TRTLLM patch fixes a Mamba decode path on the older specdec branch for
batches with multiple draft tokens per request. Current NVIDIA TRTLLM `main` has
a different speculative/MTP path, so the patch should not be ported blindly.

The Nemo-RL branch already passes `trtllm_cfg.gpu_memory_utilization` through to
TRTLLM as `KvCacheConfig(free_gpu_memory_fraction=...)`.

## Layout

- `external/TensorRT-LLM`: pinned TRTLLM submodule
- `external/RL`: pinned Nemo-RL submodule
- `patches/`: review copy of the TRTLLM patch
- `scripts/`: bootstrap, preflight, and smoke runners
- `data/tiny_math_grpo.jsonl`: toy arithmetic prompts
