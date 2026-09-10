#!/usr/bin/env bash
set -euo pipefail

cd /mnt/carles/llm

# export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
# export PYTHONUNBUFFERED=1
export HF_HOME=/mnt/carles/.cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"


uv run python -u src/jaxformers/train/async_grpo.py \
  --config experiments/async_config.py \
  inference.mode=new \
  "$@"
