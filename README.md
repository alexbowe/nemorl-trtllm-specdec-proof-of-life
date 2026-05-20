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
  `nvcr.io#nvidia/pytorch:26.02-py3` on AIHub and
  `nvcr.io#nvidia/pytorch:25.12-py3` elsewhere
- venv: `$DEV_ROOT/venvs/trtllm-rick-<python-and-torch-version>`
- AIHub Slurm shape: 1 node, 2 GPUs, 32 CPUs, 128G RAM, 2 hours

Override paths as needed:

```bash
curl -fsSL https://raw.githubusercontent.com/alexbowe/nemorl-trtllm-specdec-proof-of-life/main/scripts/bootstrap_computelab.sh | env \
  DEV_ROOT=/path/to/scratch/dev \
  CONTAINER_IMAGE=/path/to/trtllm_pytorch2512_trt1014.sqsh \
  NEMORL_TRTLLM_VENV=/path/to/venvs/trtllm-rick-custom \
  bash
```

Useful scheduler overrides:

- `SLURM_ACCOUNT`
- `SLURM_PARTITION`
- `SLURM_EXCLUDE` or `AIHUB_EXCLUDE` if a node has local Pyxis/Enroot image
  import issues
- `SLURM_GPUS_PER_NODE`
- `SLURM_CPUS_PER_TASK`
- `SLURM_TIME`
- `SLURM_CONTAINER_REMAP_ROOT` / `AIHUB_CONTAINER_REMAP_ROOT`; AIHub defaults
  to `0` because some Pyxis nodes fail to start containers with root remapping.
- `QUEUE_POLL_SECONDS`
- `MISSING_JOB_GRACE_SECONDS`
- `SRUN_MAX_ATTEMPTS` / `AIHUB_SRUN_MAX_ATTEMPTS`; AIHub defaults to `5` and
  retries on node-local Pyxis import/start failures.

The cluster-specific names also work: `AIHUB_*` on AIHub and `COMPUTELAB_*` on
computelab.

Useful runtime overrides:

- `NEMORL_TRTLLM_SMOKE_MODE=preflight` runs setup plus imports/config checks only.
- `NEMORL_TRTLLM_SMOKE_MODE=ray-check` runs setup plus Nemo-RL Ray init only.
- `NEMORL_TRTLLM_SMOKE_MODE=collective-check` runs the tiny 2-GPU NCCL
  collective check used to validate the Nemo-RL/TRTLLM refit path.
- `NEMORL_TRTLLM_APPLY_TRTLLM_PATCHES=1` applies the older bundled TRTLLM
  Mamba patch when testing an older external checkout. The default submodule
  branch is already rebased and should not need it.
- `NEMORL_TRTLLM_APPLY_NEMORL_PATCHES=1` applies the older bundled Nemo-RL
  patches when testing an older external checkout. The default submodule branch
  already has the equivalent fixes.
- `NEMORL_TRTLLM_INSTALL_NEMORL=1` enables editable Nemo-RL install. By default
  the smoke runs from `PYTHONPATH`, which matches the rebased source checkout.
- `NEMORL_TRTLLM_INSTALL_TORCH_BUILD_DEPS=1` forces torch-extension deps during
  preflight. Normal GRPO runs install them automatically.
- `NEMORL_TRTLLM_DETECT_CUDA_ARCH_LIST=0` keeps the container's
  `TORCH_CUDA_ARCH_LIST`. By default the script detects the visible GPU archs
  before torch-extension builds.

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
scripts/provision_runtime.sh
scripts/prepare_trtllm_libs.sh
scripts/preflight.sh
scripts/run_tiny_grpo.sh
```

The patch scripts are only for older external checkouts.

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

- Rebase the TRTLLM specdec path onto upstream `v1.3.0rc14`. The older bundled
  Mamba multi-token patch is now opt-in for testing older TRTLLM checkouts.
- Lazy-load optional TRTLLM FlashInfer MoE communication code, so a basic
  TRTLLM import does not require FlashInfer's CUDA IPC path.
- Cast TRTLLM Mamba prefill SSM state updates to the cache dtype before writing
  them back into the Python Mamba cache.
- Derive TRTLLM Mamba speculative decode token count from the actual decode
  batch instead of assuming every step has `max_draft_len + 1` tokens.
- Disable TRTLLM Mamba replay state update for DraftTarget paths, because
  replay currently assumes a static max draft window while DraftTarget can
  execute a shorter runtime draft window during warmup/verification.
- Link built TRTLLM plugin `.so` files into the fresh source checkout, because a
  clean submodule checkout does not include compiled TRTLLM libraries.
- Link TRTLLM wheel package extensions such as `tensorrt_llm.bindings` into the
  source checkout when running patched source over the wheel install.
- Relax Nemo-RL's PyTorch alias patch guard from `2.9.0` to `2.9.x` for the
  validated `2.9.1+cu130` venv.
- Pass `KvCacheConfig(enable_block_reuse=False, max_tokens=...,
  free_gpu_memory_fraction=...)` into TRTLLM for the Mamba cache path.
- Run TRTLLM generation one prompt at a time for this tiny smoke config.
- Clean up expected TRTLLM/Ray shutdown noise after the smoke has completed.
- Force anonymous public GitHub fetches in bootstrap to avoid stale credential
  helpers turning public fetches into `403` errors.
- Install torch-extension packages without pip build isolation, so packages like
  `flash-attn` can see the container's torch during wheel build.
- Skip Git LFS smudge by default during clone/submodule setup; the smoke does
  not need LFS payloads and this avoids slow first-run checkouts.
- Align package pins like `transformers` and `datasets` with the versions
  required by the TRTLLM wheel.
- Install the TRTLLM wheel with `--no-deps` so optional transitive pins do not
  break this TRTLLM-only smoke.
- Put pip build temp files under the run root, and disable pip wheel caching for
  torch-extension builds to avoid cross-filesystem wheel rename failures.
- Put Ray temp files under a short local `/tmp/nemorl-ray-*` path to avoid long
  Lustre runtime paths in Ray sockets.
- Disable Nemo-RL's stale Ray auto-attach path during smoke runs, so each Slurm
  job starts a fresh local Ray instance.
- Make `decord` and Megatron imports lazy enough that this TRTLLM/DTensor smoke
  path does not need unused optional dependencies at import time.
- Run Nemo-RL from `PYTHONPATH` by default, avoiding editable-install Python
  metadata checks that are stricter than the validated container runtime.
- Use Nemo-RL's NCCL process group for TRTLLM refit, matching the policy worker
  side of the collective.
- Convert Mamba Conv1d refit weights from dense `[C, C, K]` layout into the
  TRTLLM depthwise `[C, K]` layout.
- Run AIHub Pyxis containers with `--no-container-remap-root` by default, which
  avoids node-level `pyxis: couldn't start container` failures seen after image
  import.
- Retry AIHub `srun` after node-local Pyxis failures, excluding the failed node
  before the next attempt.

## Sources

- TensorRT-LLM submodule: `alexbowe/TensorRT-LLM`, branch `abowe/trtllm-specdec-rebase-1.3.0rc14`, commit `94668a78402eac7c5ac6d3c7cd542eb7802d8b87`
- Nemo-RL submodule: `alexbowe/RL`, branch `abowe/trtllm-specdec-rebase`, commit `129ecfccad642f9fb231436a9da0ee067c61c25f`
- TRTLLM base: `NVIDIA/TensorRT-LLM`, tag `v1.3.0rc14`, commit `93cb6518b6d6dbd6095748189e626db731f44545`
- TRTLLM specdec source commit: `ricklamers-nvidia/TensorRT-LLM`, branch `rick/specdec-driver535-fixes`, commit `c31be54bb2c34d52cc710358bae31fcf8a43d5ae`
- Review patches:
  - `patches/trtllm-mamba-multitoken-decode.patch`
  - `patches/nemorl-torch-2.9-alias-patch.patch`
  - `patches/nemorl-trtllm-kvcache.patch`
  - `patches/nemorl-trtllm-clean-shutdown.patch`
  - `patches/nemorl-trtllm-generation-clean-shutdown.patch`
  - `patches/nemorl-ray-disable-auto-attach.patch`

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
