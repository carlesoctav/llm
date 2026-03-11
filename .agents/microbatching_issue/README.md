# Microbatching Experiments

## Context

These experiments were run to compare:

- `optax.microbatch(...)`
- our custom `jax.lax.scan(...)` microbatch path in `src/jaxformers/train/micro_batching_ntp.py`

against the setup used by the W&B run `new_grad_api_loop_dp_no_lora_sgd` in project `new_grad_accum_api`.

That reference run uses:

- script: `src/jaxformers/train/micro_batching_ntp.py`
- `model_name=huggingface_gemma3`
- `optimizer_name=sgd`
- `use_lora=False`
- `global_batch_size=32`
- `optimizer.grad_accum=4`
- `packing=True`
- `remat_layer=False`
- `attn_implementation=sdpa`
- `loss_implementation=reference`
- DP layout: `dp_replicate=4`, `dp_shard=1`

## Exact DP match: Optax vs custom scan

### `optax.microbatch`

Exact matched run:

- W&B run: `https://wandb.ai/carlesoctav/new_grad_accum_api_compare/runs/8ojae0ok`
- compile time: `148.35s`
- total memory: `29.1 GB`
- output: `3.7 GB`
- temp: `25.3 GB`
- argument: `3.7 GB`
- estimated TFLOPs/step: `54.77`

Notes:

- This matched the original DP setup.
- The run then hit the existing `find_grad_norm(...)` issue on this branch.

### Custom `scan` before bf16 fix

Exact matched run:

- W&B run: `https://wandb.ai/carlesoctav/new_grad_accum_api_compare/runs/5p4aai6i`
- compile time: `76.07s`
- total memory: `18.4 GB`
- output: `3.7 GB`
- temp: `12.8 GB`
- argument: `1.9 GB`
- estimated TFLOPs/step: `27.39`

Notes:

- Step 0 ran successfully.
- Step 1 failed because the custom scan path promoted weights from `bf16` to `f32`.
- Root cause was the custom grad accumulator being initialized as `float32`.

## Fix applied

Changed the custom scan path to accumulate grads in param dtype instead of `float32`:

- before: `zero_grad = tree_map(lambda x: zeros_like(x, dtype=float32), train_weights)`
- after: `zero_grad = tree_map(zeros_like, train_weights)`

This change was made in `src/jaxformers/train/micro_batching_ntp.py`.

## Post-fix custom-scan matrix

All runs below used:

- script: `src/jaxformers/train/micro_batching_ntp.py`
- custom scan path: `optimizer.microbatch_impl=scan`
- `model_name=huggingface_gemma3`
- `optimizer_name=sgd`
- `global_batch_size=32`
- `optimizer.grad_accum=4`
- `packing=True`
- `remat_layer=False`
- `attn_implementation=sdpa`
- `loss_implementation=reference`
- `max_train_step=1`
- `logger_name=noop`
- `JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache`

### Results

| Parallelism | LoRA | Compile (s) | Total GB | Temp GB | Output GB | Arg GB | TFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| DP (`dp_replicate=4`, `dp_shard=1`) | No | 16.91 | 19.7 | 17.8 | 1.9 | 1.9 | 27.39 |
| FSDP (`dp_replicate=1`, `dp_shard=4`) | No | 105.00 | 16.4 | 15.9 | 0.5 | 0.5 | 27.37 |
| DP (`dp_replicate=4`, `dp_shard=1`) | Yes | 134.87 | 18.7 | 16.4 | 2.3 | 2.3 | 24.29 |
| FSDP (`dp_replicate=1`, `dp_shard=4`) | Yes | 141.37 | 16.9 | 16.1 | 0.8 | 0.8 | 24.29 |

## Takeaways

- The exact DP reference setup is still much larger with `optax.microbatch` than with our custom scan.
- Before the bf16 fix, our custom scan was not stable across steps because weights became `float32`.
- After switching the custom grad accumulator to param dtype, the one-step compile/memory runs complete for all four combinations above.
- In this matrix, FSDP is consistently lower-memory than DP.
- LoRA slightly lowers total memory in DP (`19.7 -> 18.7 GB`) and slightly increases it in FSDP (`16.4 -> 16.9 GB`) for this exact setup.

## Raw log files

- `/tmp/dp_no_lora_scan_bf16.log`
- `/tmp/fsdp_no_lora_scan_bf16.log`
- `/tmp/dp_lora_scan_bf16.log`
- `/tmp/fsdp_lora_scan_bf16.log`
- `/tmp/match_optax_dp.log`
- `/tmp/match_scan_dp_exact.log`

## Old `ntp.py` LoRA baselines

These use the old accumulation path in `src/jaxformers/train/ntp.py`.

To keep the effective batch matched to the microbatching runs (`32`), these
old-path runs used:

- `global_batch_size=8`
- `optimizer.grad_accum=4`

Other settings matched the prior experiments:

- `model_name=huggingface_gemma3`
- `optimizer_name=sgd`
- `use_lora=True`
- `packing=True`
- `remat_layer=False`
- `attn_implementation=sdpa`
- `loss_implementation=reference`
- `max_train_step=1`
- `logger_name=noop`
- `JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache`

### Results

| Script | Parallelism | LoRA | Compile (s) | Total GB | Temp GB | Output GB | Arg GB | TFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ntp.py` | DP (`dp_replicate=4`, `dp_shard=1`) | Yes | 132.02 | 17.1 | 14.4 | 2.6 | 2.6 | 24.29 |
| `ntp.py` | FSDP (`dp_replicate=1`, `dp_shard=4`) | Yes | 23.28 | 16.2 | 15.1 | 1.1 | 1.1 | 24.29 |

### Raw logs

- `/tmp/ntp_dp_lora.log`
- `/tmp/ntp_fsdp_lora.log`

## fp32-accumulator LoRA retry

I also retried the custom scan path after switching the grad accumulator back
to `float32`:

- `zero_grad = tree_map(lambda x: zeros_like(x, dtype=float32), train_weights)`

Targeted reruns:

- DP + LoRA (`dp_replicate=4`, `dp_shard=1`)
- FSDP + LoRA (`dp_replicate=1`, `dp_shard=4`)

Observed behavior:

- both runs entered compile and stayed there for a long time
- neither run emitted the first `compile time` / `Total memory size` line
- both were manually stopped after waiting significantly longer than the bf16
  accumulator runs

Interpretation:

- for the LoRA cases, switching the custom scan grad accumulator back to
  `float32` appears to introduce a serious compile-time regression
- I do not have a reliable memory number for the fp32-accumulator LoRA runs
  because compilation did not reach the first-step memory print

Raw logs:

- `/tmp/dp_lora_scan_fp32.log`
- `/tmp/fsdp_lora_scan_fp32.log`

## Custom scan + Adam + LoRA

I reran the custom scan path after restoring the old bf16 accumulator behavior:

- `zero_grad = tree_map(jnp.zeros_like, train_weights)`
- `optimizer_name=adam`
- `optimizer.microbatch_impl=scan`
- `global_batch_size=32`
- `optimizer.grad_accum=4`
- `packing=True`
- `remat_layer=False`
- `attn_implementation=sdpa`
- `loss_implementation=reference`
- `max_train_step=1`
- `logger_name=noop`
- `JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache`

I also had to strip `optimizer.microbatch_impl` before calling the optimizer
factory in `src/jaxformers/train/micro_batching_ntp.py`, otherwise
`adam.make()` failed with an unexpected keyword argument.

### Results

| Script | Parallelism | LoRA | Optimizer | Compile (s) | Total GB | Temp GB | Output GB | Arg GB | TFLOPs |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `micro_batching_ntp.py` | DP (`dp_replicate=4`, `dp_shard=1`) | Yes | Adam | 159.43 | 19.5 | 16.5 | 3.0 | 3.0 | 24.29 |
| `micro_batching_ntp.py` | FSDP (`dp_replicate=1`, `dp_shard=4`) | Yes | Adam | 161.50 | 17.6 | 16.1 | 1.4 | 1.4 | 24.29 |

### Raw logs

- `/tmp/dp_lora_scan_adam.log`
- `/tmp/fsdp_lora_scan_adam.log`

## `gemma3scan` + custom scan + Adam + LoRA + FSDP (`max_train_step=30`)

I ran the scanned-layer model from `src/jaxformers/models/huggingface/gemma3scan.py`
with the same custom microbatch scan path, LoRA enabled, Adam, and FSDP:

- script: `src/jaxformers/train/micro_batching_ntp.py`
- `model_name=huggingface_gemma3scan`
- `use_lora=True`
- `optimizer_name=adam`
- `optimizer.microbatch_impl=scan`
- `model.parallel_dims.dp_replicate=1`
- `model.parallel_dims.dp_shard=4`
- `global_batch_size=32`
- `optimizer.grad_accum=4`
- `packing=True`
- `attn_implementation=sdpa`
- `loss_implementation=reference`
- `max_train_step=30`
- `logger_name=noop`

### Results

| Remat | Compile (s) | Total GB | Temp GB | Output GB | Arg GB | TFLOPs | `program_time` | `tok/s` | Final `cum/loss` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `False` | 11.12 | 32.5 | 31.1 | 1.4 | 1.4 | 5.71 | 42.113s | 32200.00 | 0.645 |
| `True` | 16.81 | 7.5 | 6.1 | 1.4 | 1.4 | 6.03 | 39.809s | 34063.06 | 0.645 |

### Raw logs

- `/tmp/fsdp_scanlayer_lora_adam.log`
- `/tmp/fsdp_scanlayer_remat_lora_adam.log`

## `gemma3` + custom scan + Adam + LoRA + FSDP (`max_train_step=30`)

I ran the non-scanned model with the same custom microbatch scan path, LoRA,
Adam, and FSDP:

- script: `src/jaxformers/train/micro_batching_ntp.py`
- `model_name=huggingface_gemma3`
- `use_lora=True`
- `optimizer_name=adam`
- `optimizer.microbatch_impl=scan`
- `model.parallel_dims.dp_replicate=1`
- `model.parallel_dims.dp_shard=4`
- `global_batch_size=32`
- `optimizer.grad_accum=4`
- `packing=True`
- `attn_implementation=sdpa`
- `loss_implementation=reference`
- `max_train_step=30`
- `logger_name=noop`

### Results

| Remat | Compile (s) | Total GB | Temp GB | Output GB | Arg GB | TFLOPs | `program_time` | `tok/s` | Final `cum/loss` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `False` | 26.65 | 17.6 | 16.1 | 1.4 | 1.4 | 24.29 | 33.064s | 41012.03 | 0.644 |
| `True` | 198.28 | 7.7 | 6.3 | 1.4 | 1.4 | 32.65 | 41.059s | 33026.23 | 0.644 |

### Raw logs

- `/tmp/fsdp_nonscan_lora_adam.log`
- `/tmp/fsdp_nonscan_remat_lora_adam.log`
