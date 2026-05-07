# Nemo-RL TRTLLM Specdec Proof Of Life

Small reproducible smoke test for running Nemo-RL GRPO with TRTLLM inference,
Nemotron 3 Nano, and draft-target speculative decoding.

This is a proof of life, not a benchmark. Success means the stack reaches one
generation/logprob/training step and exits after `max_num_steps=1`.

## Pinned Sources

- TensorRT-LLM: `ricklamers-nvidia/TensorRT-LLM`, branch `rick/specdec-driver535-fixes`, commit `c31be54bb2c34d52cc710358bae31fcf8a43d5ae`
- Nemo-RL: `ricklamers-nvidia/RL`, branch `rick/trtllm-specdec`, commit `d69c8f638e390b407b89bc561355cfb4b196e131`
- Local TRTLLM patch: `patches/trtllm-rick-mamba-multitoken-decode.patch`

The patch fixes the older Rick-branch Mamba decode path for specdec batches with
multiple draft tokens per request. Current NVIDIA TRTLLM `main` has a different
speculative/MTP path, so this patch should not be ported there blindly.

## Layout

- `external/TensorRT-LLM`: pinned TRTLLM submodule
- `external/RL`: pinned Nemo-RL submodule
- `patches/`: local patch needed for the Rick TRTLLM branch
- `scripts/preflight_computelab.sh`: fast environment/import/config preflight
- `scripts/run_tiny_grpo.sh`: one-step GRPO smoke run
- `data/tiny_math_grpo.jsonl`: toy arithmetic prompts

## Bootstrap

```bash
git clone --recurse-submodules https://github.com/alexbowe/nemorl-trtllm-specdec-proof-of-life.git
cd nemorl-trtllm-specdec-proof-of-life

scripts/bootstrap_submodules.sh
scripts/apply_trtllm_patch.sh
```

If you want the patched TRTLLM state to be mergeable back to Rick, create forks
first, then:

```bash
scripts/use_forks.sh alexbowe
COMMIT_PATCH=1 scripts/apply_trtllm_patch.sh
git -C external/TensorRT-LLM push -u origin abowe/rick-specdec-multitoken-fix
git add .gitmodules external/TensorRT-LLM
git commit -m "Pin patched TRTLLM specdec proof-of-life branch"
```

Until the parent repo and forks are created on GitHub, the local repo can still
be used directly from its checkout path.

## Computelab Run

This assumes the validated computelab venv/container shape is already present:

- container: `/home/scratch.abowe_other/dev/trtllm_pytorch2512_trt1014.sqsh`
- venv: `/home/scratch.abowe_other/dev/venvs/trtllm-rick-py312`

Inside a suitable Pyxis/Enroot allocation:

```bash
scripts/preflight_computelab.sh
scripts/run_tiny_grpo.sh
```

Useful overrides:

```bash
MODEL_NAME=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
SPECULATIVE_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
MAX_DRAFT_LEN=4 \
MAX_NEW_TOKENS=8 \
scripts/run_tiny_grpo.sh
```

## Expected Smoke Result

The validated run completed:

- model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- backend: `trtllm`
- specdec: `draft_target`
- max draft length: `4`
- task: tiny math GRPO with `hf_math_verify`
- final signal: `Max number of steps has been reached`
- reward in the successful smoke: `Avg Reward: 0.5000`

The reward is not meaningful. It only proves the end-to-end RL loop ran.
