#!/usr/bin/env bash
set -euo pipefail

export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jax_cache}"

./.venv/bin/python -u issues/microbatch/xx.py \
  --modes fsdp \
  --loss-impl reference \
  --optimizer adam \
  --use-lora \
  --lora-rank 512 \
  --seq-len 2048 \
  --microbatch-size 8 \
  --accum-steps 4
