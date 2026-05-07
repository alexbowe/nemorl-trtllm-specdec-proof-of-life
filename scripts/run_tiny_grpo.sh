#!/usr/bin/env bash
set -euo pipefail

mode="${1:-run}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

dev_root="${DEV_ROOT:-/home/scratch.abowe_other/dev}"
repo="${NEMORL_REPO:-$repo_root/external/RL}"
trt_repo="${TRTLLM_REPO:-$repo_root/external/TensorRT-LLM}"
venv="${NEMORL_TRTLLM_VENV:-$dev_root/venvs/trtllm-rick-py312}"
run_root="${RUN_ROOT:-$dev_root/nemorl-trtllm-smoke}"
model_name="${MODEL_NAME:-nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16}"
spec_model="${SPECULATIVE_MODEL:-$model_name}"
spec_decoding_method="${SPEC_DECODING_METHOD:-draft_target}"
max_draft_len="${MAX_DRAFT_LEN:-4}"
max_new_tokens="${MAX_NEW_TOKENS:-8}"
trtllm_gpu_memory_utilization="${TRTLLM_GPU_MEMORY_UTILIZATION:-0.2}"
trtllm_max_num_tokens="${TRTLLM_MAX_NUM_TOKENS:-2048}"
trtllm_max_batch_size="${TRTLLM_MAX_BATCH_SIZE:-1}"
generation_batch_size="${GENERATION_BATCH_SIZE:-1}"
num_generations_per_prompt="${NUM_GENERATIONS_PER_PROMPT:-2}"
train_global_batch_size="${TRAIN_GLOBAL_BATCH_SIZE:-2}"
train_micro_batch_size="${TRAIN_MICRO_BATCH_SIZE:-1}"
max_total_sequence_length="${MAX_TOTAL_SEQUENCE_LENGTH:-512}"
cluster_gpus_per_node="${CLUSTER_GPUS_PER_NODE:-2}"
inference_gpus_per_node="${INFERENCE_GPUS_PER_NODE:-1}"
dtensor_v2="${DTENSOR_V2:-false}"
dtensor_tensor_parallel_size="${DTENSOR_TENSOR_PARALLEL_SIZE:-1}"
dtensor_context_parallel_size="${DTENSOR_CONTEXT_PARALLEL_SIZE:-1}"
dtensor_cpu_offload="${DTENSOR_CPU_OFFLOAD:-false}"
dtensor_activation_checkpointing="${DTENSOR_ACTIVATION_CHECKPOINTING:-false}"
dtensor_sequence_parallel="${DTENSOR_SEQUENCE_PARALLEL:-false}"
stamp="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$run_root" "$run_root/logs" "$run_root/hf-cache" "$run_root/ray" \
  "$run_root/triton-cache" "$run_root/torchinductor-cache" "$run_root/xdg-cache"

venv_site="$venv/lib/python3.12/site-packages"
venv_libs="$venv_site/torch/lib"
for libdir in "$venv_site"/nvidia/*/lib; do
  if [ -d "$libdir" ]; then
    venv_libs="$venv_libs:$libdir"
  fi
done

export PYTHONFAULTHANDLER=1
export TLLM_DISABLE_MPI=1
unset MPI4PY_RC_INITIALIZE
export LD_LIBRARY_PATH="$venv_libs:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//:\/usr\/local\/lib\/python3.12\/dist-packages\/torch\/lib/}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//:\/usr\/local\/lib\/python3.12\/dist-packages\/torch_tensorrt\/lib/}"
export PYTHONPATH="$repo:$trt_repo:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$run_root/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$run_root/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$run_root/torchinductor-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$run_root/xdg-cache}"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.001}"
export NRL_REFIT_NUM_BUFFERS="${NRL_REFIT_NUM_BUFFERS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

while IFS='=' read -r name _; do
  case "$name" in
    SLURM_*) unset "$name" ;;
  esac
done < <(env)

config_path="$run_root/grpo_nemotron3nano_specdec_tiny_2gpu_${stamp}.yaml"
run_log="$run_root/run_${stamp}.log"

cd "$repo"

if [ "$mode" = "import-check" ]; then
  "$venv/bin/python" - <<'PY'
import importlib

required = [
    "torch",
    "nemo_rl",
    "tensorrt_llm",
]
optional = [
    "nemo_automodel",
    "megatron",
    "megatron.core",
    "megatron.bridge",
    "transformer_engine",
    "flash_attn",
    "vllm.distributed.device_communicators.pynccl",
    "vllm.distributed.utils",
]

failed = False
for name in required:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: OK {getattr(mod, '__file__', '')}")
    except Exception as exc:
        failed = True
        print(f"{name}: FAIL {type(exc).__name__}: {exc}")

for name in optional:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: optional OK {getattr(mod, '__file__', '')}")
    except Exception as exc:
        print(f"{name}: optional FAIL {type(exc).__name__}: {exc}")

raise SystemExit(1 if failed else 0)
PY
  exit $?
fi

if [ "$mode" = "model-check" ]; then
  "$venv/bin/python" - "$model_name" <<'PY'
import sys

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = sys.argv[1]
print(f"model={model_name}", flush=True)
cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
print(f"model_type={cfg.model_type}", flush=True)
print(f"architectures={getattr(cfg, 'architectures', None)}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f"tokenizer={type(tokenizer).__name__} vocab={len(tokenizer)}", flush=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
print(f"empty_model={type(model).__name__}", flush=True)
PY
  exit $?
fi

"$venv/bin/python" - "$config_path" "$run_root/logs" "$model_name" "$spec_model" "$spec_decoding_method" "$max_draft_len" "$max_new_tokens" "$trtllm_gpu_memory_utilization" "$trtllm_max_num_tokens" "$trtllm_max_batch_size" "$generation_batch_size" "$num_generations_per_prompt" "$train_global_batch_size" "$train_micro_batch_size" "$max_total_sequence_length" "$cluster_gpus_per_node" "$inference_gpus_per_node" "$dtensor_v2" "$dtensor_tensor_parallel_size" "$dtensor_context_parallel_size" "$dtensor_cpu_offload" "$dtensor_activation_checkpointing" "$dtensor_sequence_parallel" <<'PY'
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config

OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)

config_path, log_dir, model_name, spec_model, spec_decoding_method = sys.argv[1:6]
max_draft_len = int(sys.argv[6])
max_new_tokens = int(sys.argv[7])
trtllm_gpu_memory_utilization = float(sys.argv[8])
trtllm_max_num_tokens = int(sys.argv[9])
trtllm_max_batch_size = int(sys.argv[10])
generation_batch_size = int(sys.argv[11])
num_generations_per_prompt = int(sys.argv[12])
train_global_batch_size = int(sys.argv[13])
train_micro_batch_size = int(sys.argv[14])
max_total_sequence_length = int(sys.argv[15])
cluster_gpus_per_node = int(sys.argv[16])
inference_gpus_per_node = int(sys.argv[17])
dtensor_v2 = sys.argv[18].lower() in {"1", "true", "yes", "on"}
dtensor_tensor_parallel_size = int(sys.argv[19])
dtensor_context_parallel_size = int(sys.argv[20])
dtensor_cpu_offload = sys.argv[21].lower() in {"1", "true", "yes", "on"}
dtensor_activation_checkpointing = sys.argv[22].lower() in {"1", "true", "yes", "on"}
dtensor_sequence_parallel = sys.argv[23].lower() in {"1", "true", "yes", "on"}
cfg = load_config("configs/grpo_qwen3_1.7b_specdec.yaml")
run_root = Path(config_path).parent
tiny_data_path = run_root / "tiny_math_grpo.jsonl"

examples = [
    {"input": "What is 1 + 1? Return the final answer in \\boxed{}.", "output": "2"},
    {"input": "What is 2 + 3? Return the final answer in \\boxed{}.", "output": "5"},
    {"input": "What is 6 - 4? Return the final answer in \\boxed{}.", "output": "2"},
    {"input": "What is 3 * 3? Return the final answer in \\boxed{}.", "output": "9"},
]
with tiny_data_path.open("w", encoding="utf-8") as f:
    for row in examples:
        f.write(json.dumps(row) + "\n")

cfg.grpo.num_prompts_per_step = 1
cfg.grpo.num_generations_per_prompt = num_generations_per_prompt
cfg.grpo.max_num_epochs = 1
cfg.grpo.max_num_steps = 1
cfg.grpo.val_period = 0
cfg.grpo.val_at_start = False
cfg.grpo.max_val_samples = 0
cfg.grpo.val_batch_size = 1

cfg.checkpointing.enabled = False

cfg.policy.model_name = model_name
cfg.policy.tokenizer.name = model_name
cfg.policy.train_global_batch_size = train_global_batch_size
cfg.policy.train_micro_batch_size = train_micro_batch_size
cfg.policy.generation_batch_size = generation_batch_size
cfg.policy.logprob_batch_size = 1
cfg.policy.max_total_sequence_length = max_total_sequence_length
cfg.policy.sequence_packing.enabled = False
cfg.policy.dtensor_cfg._v2 = dtensor_v2
cfg.policy.dtensor_cfg.enabled = True
cfg.policy.dtensor_cfg.tensor_parallel_size = dtensor_tensor_parallel_size
cfg.policy.dtensor_cfg.context_parallel_size = dtensor_context_parallel_size
cfg.policy.dtensor_cfg.cpu_offload = dtensor_cpu_offload
cfg.policy.dtensor_cfg.activation_checkpointing = dtensor_activation_checkpointing
cfg.policy.dtensor_cfg.sequence_parallel = dtensor_sequence_parallel
cfg.policy.megatron_cfg.enabled = False
cfg.policy.megatron_cfg.tensor_model_parallel_size = 1
cfg.policy.megatron_cfg.context_parallel_size = 1
cfg.policy.megatron_cfg.pipeline_model_parallel_size = 1
cfg.policy.megatron_cfg.optimizer.use_distributed_optimizer = False
cfg.policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce = False
cfg.policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather = False

cfg.policy.generation.backend = "trtllm"
cfg.policy.generation.max_new_tokens = max_new_tokens
cfg.policy.generation.trtllm_cfg.tensor_parallel_size = 1
cfg.policy.generation.trtllm_cfg.gpu_memory_utilization = trtllm_gpu_memory_utilization
cfg.policy.generation.trtllm_cfg.max_model_len = max_total_sequence_length
cfg.policy.generation.trtllm_cfg.max_batch_size = trtllm_max_batch_size
cfg.policy.generation.trtllm_cfg.max_num_tokens = trtllm_max_num_tokens
if spec_decoding_method == "none":
    cfg.policy.generation.trtllm_cfg.speculative_decoding = None
else:
    cfg.policy.generation.trtllm_cfg.speculative_decoding.method = spec_decoding_method
    cfg.policy.generation.trtllm_cfg.speculative_decoding.max_draft_len = max_draft_len
    cfg.policy.generation.trtllm_cfg.speculative_decoding.speculative_model = spec_model
cfg.policy.generation.colocated.enabled = False
cfg.policy.generation.colocated.resources.gpus_per_node = inference_gpus_per_node
cfg.policy.generation.colocated.resources.num_nodes = None

cfg.data.max_input_seq_length = max_total_sequence_length
cfg.data.dataset_name = "ResponseDataset"
cfg.data.train_data_path = str(tiny_data_path)
cfg.data.val_data_path = None
cfg.data.input_key = "input"
cfg.data.output_key = "output"
cfg.data.train_split = None
cfg.data.val_split = None
cfg.data.prompt_file = None
cfg.data.shuffle = False
cfg.data.num_workers = 0
cfg.env.math.num_workers = 1

cfg.logger.log_dir = log_dir
cfg.logger.num_val_samples_to_print = 0
cfg.logger.wandb_enabled = False
cfg.logger.tensorboard_enabled = False
cfg.logger.mlflow_enabled = False
cfg.logger.swanlab_enabled = False
cfg.logger.monitor_gpus = False
cfg.logger.mongodb_enabled = False

cfg.cluster.gpus_per_node = cluster_gpus_per_node
cfg.cluster.num_nodes = 1

Path(config_path).parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(config=cfg, f=config_path)
resolved = OmegaConf.to_container(cfg, resolve=True)

print(f"wrote_config={config_path}")
print(f"tiny_data={tiny_data_path}")
print(f"model={resolved['policy']['model_name']}")
print(f"backend={resolved['policy']['generation']['backend']}")
print(f"cluster_gpus={resolved['cluster']['gpus_per_node']}")
print(f"train_gpus={resolved['cluster']['gpus_per_node'] - resolved['policy']['generation']['colocated']['resources']['gpus_per_node']}")
print(f"inference_gpus={resolved['policy']['generation']['colocated']['resources']['gpus_per_node']}")
print(f"max_steps={resolved['grpo']['max_num_steps']}")
print(f"max_seq={resolved['policy']['max_total_sequence_length']}")
print(f"max_new_tokens={resolved['policy']['generation']['max_new_tokens']}")
print(f"generation_batch_size={resolved['policy']['generation_batch_size']}")
print(f"train_global_batch_size={resolved['policy']['train_global_batch_size']}")
print(f"dtensor={resolved['policy']['dtensor_cfg']}")
print(f"specdec={resolved['policy']['generation']['trtllm_cfg']['speculative_decoding']}")
PY

if [ "$mode" = "config-only" ]; then
  exit 0
fi

echo "run_log=$run_log"
"$venv/bin/python" -u examples/run_grpo.py --config "$config_path" 2>&1 | tee "$run_log"
