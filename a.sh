#!/usr/bin/env bash
set -euo pipefail

cd /mnt/carles/llm

export TPU_VISIBLE_CHIPS=0
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export PYTHONUNBUFFERED=1
export PYTHONPATH="/mnt/carles/verifiers/environments/gsm8k:/mnt/carles/verifiers${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/mnt/carles/.cache

uv run python -u src/jaxformers/train/grpo.py --config experiments/rl_verifiers_gsm8k.py
