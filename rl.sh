#!/usr/bin/env bash
set -euo pipefail

cd /mnt/carles/llm

# export TPU_VISIBLE_CHIPS=0,1,2,3
# export TPU_PROCESS_BOUNDS=1,1,1
# export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
# export HF_HUB_DISABLE_PROGRESS_BARS=1
# export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
# export PYTHONUNBUFFERED=1
export HF_HOME=/mnt/carles/.cache

python src/jaxformers/train/grpo.py \
  --config experiments/rl_verifiers_gsm8k.py \
  parallel.parallel_dims.dp_replicate=1 \
  parallel.parallel_dims.dp_shard=4 \
  parallel.parallel_dims.cp=1 \
  parallel.parallel_dims.tp=1 \
  vllm.tensor_parallel_size=4 \
  "$@"
