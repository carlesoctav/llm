#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export HF_HOME="${HF_HOME:-/mnt/carles/.cache}"
export PYTHONPATH="${PYTHONPATH:-src}"

LOGGER_NAME="${LOGGER_NAME:-wandb}"
PROJECT_NAME="${PROJECT_NAME:-refactor_scan}"
MAX_TRAIN_STEP="${MAX_TRAIN_STEP:-30}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
INIT_LORA="${INIT_LORA:-}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-}"
REMAT_LAYER="${REMAT_LAYER:-True}"
PACKING="${PACKING:-True}"
ATTN_IMPL="${ATTN_IMPL:-}"
LOSS_IMPL="${LOSS_IMPL:-}"

args=(
  --config ./config/gemma_3_1b_tunix.py
  "logger_name=${LOGGER_NAME}"
  "project_name=${PROJECT_NAME}"
  "max_train_step=${MAX_TRAIN_STEP}"
  "packing=${PACKING}"
  "global_batch_size=${GLOBAL_BATCH_SIZE}"
  "remat_layer=${REMAT_LAYER}"
  "forward_impl=scan_layer"
)

if [[ -n "${INIT_LORA}" ]]; then
  args+=("init_lora=${INIT_LORA}")
fi

if [[ -n "${OPTIMIZER_NAME}" ]]; then
  args+=("optimizer_name=${OPTIMIZER_NAME}")
fi

if [[ -n "${ATTN_IMPL}" ]]; then
  args+=("attn_implementation=${ATTN_IMPL}")
fi

if [[ -n "${LOSS_IMPL}" ]]; then
  args+=("loss_implementation=${LOSS_IMPL}")
fi

exec .venv/bin/python src/jaxformers/train/ntp.py "${args[@]}" "$@"
