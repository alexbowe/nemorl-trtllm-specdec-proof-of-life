#!/usr/bin/env bash
set -euo pipefail

mode="${1:-run}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"

profile="${CLUSTER_PROFILE:-$(detect_cluster_profile)}"
dev_root="${DEV_ROOT:-$(default_dev_root "$profile")}"
repo="${NEMORL_REPO:-$repo_root/external/RL}"
trt_repo="${TRTLLM_REPO:-$repo_root/external/TensorRT-LLM}"
venv="${NEMORL_TRTLLM_VENV:-$(default_venv "$dev_root")}"
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
ray_root="${RAY_ROOT:-/tmp/nemorl-ray-${USER:-user}-${SLURM_JOB_ID:-$$}}"
cluster_num_nodes="${CLUSTER_NUM_NODES:-1}"
cluster_gpus_per_node="${CLUSTER_GPUS_PER_NODE:-2}"
if [ "$cluster_num_nodes" -gt 1 ]; then
  inference_gpus_per_node="${INFERENCE_GPUS_PER_NODE:-$cluster_gpus_per_node}"
  inference_num_nodes="${INFERENCE_NUM_NODES:-1}"
else
  inference_gpus_per_node="${INFERENCE_GPUS_PER_NODE:-1}"
  inference_num_nodes="${INFERENCE_NUM_NODES:-}"
fi
dtensor_v2="${DTENSOR_V2:-false}"
dtensor_tensor_parallel_size="${DTENSOR_TENSOR_PARALLEL_SIZE:-1}"
dtensor_context_parallel_size="${DTENSOR_CONTEXT_PARALLEL_SIZE:-1}"
dtensor_cpu_offload="${DTENSOR_CPU_OFFLOAD:-false}"
dtensor_activation_checkpointing="${DTENSOR_ACTIVATION_CHECKPOINTING:-false}"
dtensor_sequence_parallel="${DTENSOR_SEQUENCE_PARALLEL:-false}"
require_specdec_metrics="${NEMORL_TRTLLM_REQUIRE_SPECDEC_METRICS:-1}"
stamp="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$run_root" "$run_root/logs" "$run_root/hf-cache" "$ray_root" "$ray_root/tmp" \
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
unset RAY_ADDRESS RAY_CLIENT_MODE RAY_JOB_ID RAY_NAMESPACE RAY_RUNTIME_ENV_URI
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR="${RAY_TMPDIR:-$ray_root}"
export TMPDIR="${NEMORL_TRTLLM_TMPDIR:-$ray_root/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export NEMO_RL_DISABLE_RAY_AUTO_ATTACH="${NEMO_RL_DISABLE_RAY_AUTO_ATTACH:-1}"
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

base_config="${NEMORL_BASE_CONFIG:-}"
if [ -z "$base_config" ]; then
  for candidate in \
    configs/grpo_qwen3_1.7b_specdec.yaml \
    examples/configs/recipes/llm/grpo-qwen3-1.7b-2n4g-megatron-trtllm.yaml \
    examples/configs/grpo_math_1B.yaml; do
    if [ -f "$candidate" ]; then
      base_config="$candidate"
      break
    fi
  done
fi
if [ -z "$base_config" ]; then
  echo "Could not find a Nemo-RL GRPO base config." >&2
  exit 1
fi

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

if [ "$mode" = "ray-check" ]; then
  "$venv/bin/python" - "$RAY_TMPDIR" <<'PY'
import sys

import ray
from nemo_rl.distributed.virtual_cluster import init_ray

log_dir = sys.argv[1]
print(f"ray_log_dir={log_dir}", flush=True)
init_ray(log_dir=log_dir)
print(f"ray_initialized={ray.is_initialized()}", flush=True)
print(f"ray_resources={ray.cluster_resources()}", flush=True)
ray.shutdown()
PY
  exit $?
fi

if [ "$mode" = "collective-check" ]; then
  "$venv/bin/python" - "$RAY_TMPDIR" <<'PY'
import contextlib
import socket
import sys

import ray

from nemo_rl.distributed.virtual_cluster import init_ray


def free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


@ray.remote(num_gpus=1)
class PolicyCollectiveActor:
    def init_collective(self, ip: str, port: int) -> str:
        import torch
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        group = StatelessProcessGroup(
            master_address=ip, port=port, rank=0, world_size=2,
        )
        group.init_nccl_communicator(device=torch.cuda.current_device())
        self.group = group
        return "policy-ok"


@ray.remote(num_gpus=1)
class TrtllmCollectiveActor:
    def init_collective(self, ip: str, port: int) -> str:
        import torch
        from nemo_rl.models.generation.trtllm.trtllm_backend import NcclExtension

        extension = NcclExtension.__new__(NcclExtension)
        extension.device_id = torch.cuda.current_device()
        extension.init_collective(
            rank_prefix=0, ip=ip, port=port, world_size=2, train_world_size=1,
        )
        self.extension = extension
        return "trtllm-ok"


log_dir = sys.argv[1]
init_ray(log_dir=log_dir)
resources = ray.cluster_resources()
print(f"ray_resources={resources}", flush=True)
if resources.get("GPU", 0) < 2:
    raise SystemExit("collective-check requires 2 GPUs")

ip = ray.util.get_node_ip_address()
port = free_port()
policy = PolicyCollectiveActor.remote()
trtllm = TrtllmCollectiveActor.remote()
results = ray.get(
    [policy.init_collective.remote(ip, port), trtllm.init_collective.remote(ip, port)],
    timeout=180,
)
print(f"collective_check={results}", flush=True)
ray.shutdown()
PY
  exit $?
fi

"$venv/bin/python" - "$base_config" "$config_path" "$run_root/logs" "$model_name" "$spec_model" "$spec_decoding_method" "$max_draft_len" "$max_new_tokens" "$trtllm_gpu_memory_utilization" "$trtllm_max_num_tokens" "$trtllm_max_batch_size" "$generation_batch_size" "$num_generations_per_prompt" "$train_global_batch_size" "$train_micro_batch_size" "$max_total_sequence_length" "$cluster_num_nodes" "$cluster_gpus_per_node" "$inference_gpus_per_node" "$inference_num_nodes" "$dtensor_v2" "$dtensor_tensor_parallel_size" "$dtensor_context_parallel_size" "$dtensor_cpu_offload" "$dtensor_activation_checkpointing" "$dtensor_sequence_parallel" <<'PY'
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config

OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)

base_config, config_path, log_dir, model_name, spec_model, spec_decoding_method = sys.argv[1:7]
max_draft_len = int(sys.argv[7])
max_new_tokens = int(sys.argv[8])
trtllm_gpu_memory_utilization = float(sys.argv[9])
trtllm_max_num_tokens = int(sys.argv[10])
trtllm_max_batch_size = int(sys.argv[11])
generation_batch_size = int(sys.argv[12])
num_generations_per_prompt = int(sys.argv[13])
train_global_batch_size = int(sys.argv[14])
train_micro_batch_size = int(sys.argv[15])
max_total_sequence_length = int(sys.argv[16])
cluster_num_nodes = int(sys.argv[17])
cluster_gpus_per_node = int(sys.argv[18])
inference_gpus_per_node = int(sys.argv[19])
inference_num_nodes = int(sys.argv[20]) if sys.argv[20] else None
dtensor_v2 = sys.argv[21].lower() in {"1", "true", "yes", "on"}
dtensor_tensor_parallel_size = int(sys.argv[22])
dtensor_context_parallel_size = int(sys.argv[23])
dtensor_cpu_offload = sys.argv[24].lower() in {"1", "true", "yes", "on"}
dtensor_activation_checkpointing = sys.argv[25].lower() in {"1", "true", "yes", "on"}
dtensor_sequence_parallel = sys.argv[26].lower() in {"1", "true", "yes", "on"}
disable_nemotron_h_fast_path = os.environ.get(
    "NEMORL_TRTLLM_DISABLE_NEMOTRON_H_FAST_PATH", "1"
).lower() in {"1", "true", "yes", "on"}
cfg = load_config(base_config)
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
grpo_source = Path("nemo_rl/algorithms/grpo.py").read_text(encoding="utf-8")
if "val_at_end" in grpo_source:
    cfg.grpo.val_at_end = False
if "seq_logprob_error_threshold" in grpo_source:
    cfg.grpo.seq_logprob_error_threshold = None

cfg.checkpointing.enabled = False
cfg.checkpointing.save_optimizer = False

cfg.policy.model_name = model_name
cfg.policy.tokenizer.name = model_name
cfg.policy.train_global_batch_size = train_global_batch_size
cfg.policy.train_micro_batch_size = train_micro_batch_size
cfg.policy.generation_batch_size = generation_batch_size
cfg.policy.logprob_batch_size = 1
cfg.policy.max_total_sequence_length = max_total_sequence_length
cfg.policy.sequence_packing.enabled = False
if disable_nemotron_h_fast_path:
    cfg.policy.hf_config_overrides = dict(cfg.policy.hf_config_overrides or {})
    cfg.policy.hf_config_overrides["use_mamba_kernels"] = False
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
run_grpo_source = Path("examples/run_grpo.py").read_text(encoding="utf-8")
if 'policy["draft"]' in run_grpo_source and "draft" not in cfg.policy:
    cfg.policy.draft = {
        "enabled": False,
        "model_name": None,
        "loss_weight": 0.1,
        "num_layers": None,
        "aux_layer_indices": None,
    }

cfg.policy.generation.backend = "trtllm"
cfg.policy.generation.max_new_tokens = max_new_tokens
cfg.policy.generation.trtllm_cfg.tensor_parallel_size = 1
cfg.policy.generation.trtllm_cfg.gpu_memory_utilization = trtllm_gpu_memory_utilization
cfg.policy.generation.trtllm_cfg.max_model_len = max_total_sequence_length
cfg.policy.generation.trtllm_cfg.max_batch_size = trtllm_max_batch_size
cfg.policy.generation.trtllm_cfg.max_num_tokens = trtllm_max_num_tokens
cfg.policy.generation.trtllm_cfg.async_engine = False
cfg.policy.generation.trtllm_cfg.return_perf_metrics = True
if spec_decoding_method == "none":
    cfg.policy.generation.trtllm_cfg.speculative_decoding = None
else:
    if cfg.policy.generation.trtllm_cfg.get("speculative_decoding") is None:
        cfg.policy.generation.trtllm_cfg.speculative_decoding = {}
    cfg.policy.generation.trtllm_cfg.speculative_decoding.method = spec_decoding_method
    cfg.policy.generation.trtllm_cfg.speculative_decoding.max_draft_len = max_draft_len
    cfg.policy.generation.trtllm_cfg.speculative_decoding.speculative_model = spec_model
cfg.policy.generation.colocated.enabled = False
cfg.policy.generation.colocated.resources.gpus_per_node = inference_gpus_per_node
cfg.policy.generation.colocated.resources.num_nodes = inference_num_nodes

cfg.data.max_input_seq_length = max_total_sequence_length
cfg.data.shuffle = False
cfg.data.num_workers = 0
cfg.data.use_multiple_dataloader = False
cfg.data.num_prompts_per_dataloader = 1
data_source = Path("nemo_rl/data/__init__.py").read_text(encoding="utf-8")
uses_nested_data_config = "train: ResponseDatasetConfig" in data_source
if uses_nested_data_config:
    for key in [
        "dataset_name",
        "train_data_path",
        "val_data_path",
        "input_key",
        "output_key",
        "train_split",
        "val_split",
        "prompt_file",
    ]:
        if key in cfg.data:
            del cfg.data[key]
    cfg.data.train = {
        "dataset_name": "ResponseDataset",
        "data_path": str(tiny_data_path),
        "input_key": "input",
        "output_key": "output",
        "prompt_file": None,
        "system_prompt_file": None,
        "processor": "math_hf_data_processor",
        "env_name": "math",
    }
    cfg.data.validation = None
    cfg.data.default = {
        "dataset_name": "ResponseDataset",
        "input_key": "input",
        "output_key": "output",
        "prompt_file": None,
        "system_prompt_file": None,
        "processor": "math_hf_data_processor",
        "env_name": "math",
    }
else:
    cfg.data.dataset_name = "ResponseDataset"
    cfg.data.train_data_path = str(tiny_data_path)
    cfg.data.val_data_path = None
    cfg.data.input_key = "input"
    cfg.data.output_key = "output"
    cfg.data.train_split = None
    cfg.data.val_split = None
    cfg.data.prompt_file = None
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
cfg.cluster.num_nodes = cluster_num_nodes

Path(config_path).parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(config=cfg, f=config_path)
resolved = OmegaConf.to_container(cfg, resolve=True)

print(f"wrote_config={config_path}")
print(f"base_config={base_config}")
print(f"tiny_data={tiny_data_path}")
print(f"model={resolved['policy']['model_name']}")
print(f"backend={resolved['policy']['generation']['backend']}")
print(f"cluster_gpus={resolved['cluster']['gpus_per_node']}")
print(f"cluster_nodes={resolved['cluster']['num_nodes']}")
if resolved["cluster"]["num_nodes"] > 1:
    train_nodes = resolved["cluster"]["num_nodes"] - resolved["policy"]["generation"]["colocated"]["resources"]["num_nodes"]
    train_gpus = resolved["cluster"]["gpus_per_node"]
else:
    train_nodes = 1
    train_gpus = resolved["cluster"]["gpus_per_node"] - resolved["policy"]["generation"]["colocated"]["resources"]["gpus_per_node"]
print(f"train_nodes={train_nodes}")
print(f"train_gpus={train_gpus}")
print(f"inference_gpus={resolved['policy']['generation']['colocated']['resources']['gpus_per_node']}")
print(f"inference_nodes={resolved['policy']['generation']['colocated']['resources']['num_nodes']}")
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

if [ "$require_specdec_metrics" = "1" ] && [ "$spec_decoding_method" != "none" ]; then
  if ! grep -q "TRTLLM Specdec Metrics:" "$run_log"; then
    echo "Expected TRTLLM specdec metrics were not reported in $run_log" >&2
    echo "Set NEMORL_TRTLLM_REQUIRE_SPECDEC_METRICS=0 to allow older checkouts." >&2
    exit 1
  fi
  echo
  echo "specdec_metrics_report=$run_log"
  sed -n '/TRTLLM Specdec Metrics:/,+6p' "$run_log"
fi
