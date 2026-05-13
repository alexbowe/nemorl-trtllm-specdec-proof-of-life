# Nemo-RL TRTLLM Specdec Proof Of Life

Reproducible smoke test for Nemo-RL GRPO with TRTLLM inference, Nemotron 3 Nano,
and draft-target speculative decoding.

This is not a benchmark. Success means the stack reaches one
generation/logprob/training step and exits after `max_num_steps=1`.

## Quick Run

From an AIHub login shell:

```bash
curl -fsSL https://raw.githubusercontent.com/alexbowe/nemorl-trtllm-specdec-proof-of-life/main/scripts/bootstrap_aihub.sh | bash
```

The AIHub bootstrap defaults to the first writable Lustre user directory:

```bash
/lustre/fsw/portfolios/*/users/${USER}/dev
```

It auto-detects your Slurm account when possible. Override it if needed:

```bash
curl -fsSL https://raw.githubusercontent.com/alexbowe/nemorl-trtllm-specdec-proof-of-life/main/scripts/bootstrap_aihub.sh | env \
  DEV_ROOT=/lustre/fsw/portfolios/coreai/users/${USER}/dev \
  AIHUB_ACCOUNT=coreai_lpu_software \
  bash
```

From a computelab login shell:

```bash
curl -fsSL https://raw.githubusercontent.com/alexbowe/nemorl-trtllm-specdec-proof-of-life/main/scripts/bootstrap_computelab.sh | env \
  DEV_ROOT=/home/scratch.${USER}_other/dev \
  bash
```

`DEV_ROOT` is the directory where the repo, caches, venv, and run outputs live.
On computelab, a common pattern is:

```bash
/home/scratch.${USER}_other/dev
```

The bootstrap script clones or updates this repo under `$DEV_ROOT`, reserves a
GPU node with `srun`, starts a Pyxis/Enroot container, provisions the Python
runtime if needed, runs preflight checks, and launches the tiny GRPO smoke.

The script is standalone in the sense that it does not expect pre-existing
computelab or AIHub helper scripts, a pre-existing checkout, a pre-existing
venv, or a pre-built local TRTLLM source tree. It verifies that the cloned repo
contains the required scripts, patches, requirements, and tiny dataset before
submitting the Slurm job.

It still needs normal cluster infrastructure: `git`, Slurm with Pyxis/Enroot,
network access to GitHub/Python package indexes/container registry, a writable
large-storage `DEV_ROOT`, and access to the model on Hugging Face if your
environment requires authentication.

If an old checkout exists, the bootstrap updates the top-level smoke scripts,
patches, and docs from `origin/main`, then re-syncs submodules. Existing
submodule downloads are reused.

The first run can be slow because it imports the base container and creates the
Python venv. Later runs reuse both.

Use `NEMORL_TRTLLM_INSTALL_DIR` only when you intentionally want a separate
checkout.

Expected defaults next to the repo:

- container: `$DEV_ROOT/trtllm_pytorch2512_trt1014.sqsh` if present, otherwise
  `nvcr.io#nvidia/pytorch:25.12-py3`
- venv: `$DEV_ROOT/venvs/trtllm-rick-py312`

Override paths as needed:

```bash
curl -fsSL https://raw.githubusercontent.com/alexbowe/nemorl-trtllm-specdec-proof-of-life/main/scripts/bootstrap_computelab.sh | env \
  DEV_ROOT=/path/to/scratch/dev \
  CONTAINER_IMAGE=/path/to/trtllm_pytorch2512_trt1014.sqsh \
  NEMORL_TRTLLM_VENV=/path/to/venvs/trtllm-rick-py312 \
  bash
```

Useful scheduler overrides:

- `SLURM_ACCOUNT`
- `SLURM_PARTITION`
- `SLURM_GPUS_PER_NODE`
- `SLURM_CPUS_PER_TASK`
- `SLURM_TIME`
- `QUEUE_POLL_SECONDS`
- `MISSING_JOB_GRACE_SECONDS`

The old `COMPUTELAB_*` override names still work for computelab.

## Manual Run

If the repo is already cloned:

```bash
/path/to/nemorl-trtllm-specdec-proof-of-life/scripts/computelab_srun_smoke.sh
```

If already inside a suitable Pyxis/Enroot allocation:

```bash
scripts/smoke.sh
```

For step-by-step debugging:

```bash
scripts/bootstrap_submodules.sh
scripts/apply_trtllm_patch.sh
scripts/apply_nemorl_patch.sh
scripts/provision_runtime.sh
scripts/prepare_trtllm_libs.sh
scripts/preflight.sh
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

## Fixes Applied

- Patch the TRTLLM Mamba decode path to handle multiple draft tokens per request
  with `draft_target`.
- Link built TRTLLM plugin `.so` files into the fresh source checkout, because a
  clean submodule checkout does not include compiled TRTLLM libraries.
- Relax Nemo-RL's PyTorch alias patch guard from `2.9.0` to `2.9.x` for the
  validated `2.9.1+cu130` venv.
- Pass `KvCacheConfig(enable_block_reuse=False, max_tokens=...,
  free_gpu_memory_fraction=...)` into TRTLLM for the Mamba cache path.
- Run TRTLLM generation one prompt at a time for this tiny smoke config.
- Clean up expected TRTLLM/Ray shutdown noise after the smoke has completed.
- Force anonymous public GitHub fetches in bootstrap to avoid stale credential
  helpers turning public fetches into `403` errors.

## Sources

- TensorRT-LLM submodule: `alexbowe/TensorRT-LLM`, branch `abowe/rick-specdec-multitoken-fix`, commit `2b617b2f2c8fbbdf41eb1720f473c1ae926522e5`
- Nemo-RL submodule: `ricklamers-nvidia/RL`, branch `rick/trtllm-specdec`, commit `d69c8f638e390b407b89bc561355cfb4b196e131`
- TRTLLM base branch: `ricklamers-nvidia/TensorRT-LLM`, branch `rick/specdec-driver535-fixes`, commit `c31be54bb2c34d52cc710358bae31fcf8a43d5ae`
- Review patches:
  - `patches/trtllm-mamba-multitoken-decode.patch`
  - `patches/nemorl-torch-2.9-alias-patch.patch`
  - `patches/nemorl-trtllm-kvcache.patch`
  - `patches/nemorl-trtllm-clean-shutdown.patch`
  - `patches/nemorl-trtllm-generation-clean-shutdown.patch`

The TRTLLM patch fixes a Mamba decode path on the older specdec branch for
batches with multiple draft tokens per request. Current NVIDIA TRTLLM `main` has
a different speculative/MTP path, so the patch should not be ported blindly.

The Nemo-RL patches are smoke-run support patches for the computelab
environment and this Mamba/specdec configuration.

## Layout

- `external/TensorRT-LLM`: pinned TRTLLM submodule
- `external/RL`: pinned Nemo-RL submodule
- `patches/`: review copy of the TRTLLM patch
- `scripts/`: bootstrap, preflight, and smoke runners
- `data/tiny_math_grpo.jsonl`: toy arithmetic prompts
