# Refactor Scan Results

Date: 2026-03-11

## Implementation summary

- `gemma3scan.py` was removed and the logic was folded into `src/jaxformers/models/huggingface/gemma3.py`.
- Added `forward_impl = (loop, scan_layer, scan_block)`.
- Canonical model weights stay unstacked.
- LoRA is applied to the unstacked weights.
- Stacked scan views are now prepared outside the train-step JIT, after `make_lora(...)` and before optimizer init, via `jaxformers.models.prepare_weights(...)`.
- Gemma forward still has a fallback prepare path, but the common training path now hits an already-prepared tree.
- `make_scan_fwd` was refactored to use explicit `argnums`, `argnames`, and `in_axes`.
- Current local state: `scan_layer` uses `jax.lax.scan`.

## Validation

Focused tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_scan_utils.py tests/test_gemma3_forward_impl.py -q
```

Latest result:

- `4 passed, 2 warnings`

## Main TPU command

Baseline command used for most runs:

```bash
HF_HOME="/mnt/carles/.cache" PYTHONPATH=src .venv/bin/python src/jaxformers/train/ntp.py \
  --config ./config/gemma_3_1b_tunix.py \
  logger_name=wandb \
  project_name=refactor_scan \
  max_train_step=30 \
  packing=True \
  global_batch_size=32 \
  init_lora=random \
  remat_layer=True \
  forward_impl=scan_layer
```

`xla_chunked` runs added:

```bash
attn_implementation=xla_chunked loss_implementation=xla_chunked
```

## Benchmark results

### Default attention/loss

All runs below used `max_train_step=30`, `global_batch_size=32`, `init_lora=random`, `remat_layer=True`.

| Mode | Layer helper | Compile | Total mem | Temp mem | Tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| `scan_layer` | `lax.scan` | `15.84s` | `7.5 GB` | `6.1 GB` | `34176.95` |
| `scan_layer` | `fori_loop` | `16.67s` | `8.6 GB` | `7.2 GB` | `37349.61` |
| `scan_block` | `fori_loop` | `99.21s` | `13.4 GB` | `12.0 GB` | `33404.00` |

Additional `scan_block` smoke:

| Mode | Steps | Compile | Total mem | Temp mem | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scan_block` | `2` | `91.86s` | `13.4 GB` | `12.0 GB` | `29677.27` |

### Prepare outside JIT follow-up

All runs below used the current `scan_layer` + `fori_loop` path, `global_batch_size=32`, `optimizer.grad_accum=4`, `remat_layer=True`, and `max_train_step=2`.

| Setup | LoRA | Optimizer | Attn/loss | Compile | Total mem | Temp mem | Tok/s | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| before external prepare | no | `sgd` | default | `34.63s` | `8.0 GB` | `7.6 GB` | `34001.73` | old path, `prepare_weights` inside JIT |
| after external prepare | no | `sgd` | default | `9.87s` | `7.3 GB` | `6.8 GB` | `34755.74` | current path |
| after external prepare | `random` | `sgd` | default | `16.30s` | `7.9 GB` | `7.1 GB` | `38525.48` | `loraify=3.52s` |
| before external prepare | `random` | `adam` | `xla_chunked/xla_chunked` | `81.95s` | `6.5 GB` | `5.1 GB` | `30828.55` | old path |
| after external prepare | `random` | `adam` | `xla_chunked/xla_chunked` | `33.20s` | `5.4 GB` | `4.0 GB` | `33049.76` | `loraify=3.57s` |

### `xla_chunked` attention + loss

Same training config as above, plus:

```bash
attn_implementation=xla_chunked loss_implementation=xla_chunked
```

| Mode | Layer helper | Steps | Compile | Total mem | Temp mem | Program time | Tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `scan_layer` | `fori_loop` | `2` | `81.95s` | `6.5 GB` | `5.1 GB` | n/a | `30828.55` |
| `scan_layer` | `lax.scan` | `30` | `83.68s` | `5.6 GB` | `4.2 GB` | `45.903s` | `29540.91` |
| `scan_layer` | `fori_loop` | `30` | `80.89s` | `6.5 GB` | `5.1 GB` | `45.858s` | `29570.14` |
| `scan_layer` | `lax.scan` | `30` | `81.95s` | `5.6 GB` | `4.2 GB` | `45.861s` | `29568.29` |
| `scan_layer` | `lax.scan` | `2` | `33.23s` | `4.3 GB` | `2.9 GB` | `2.647s` | `33928.20` |

## Main conclusions

- `scan_block` works, but it is materially heavier than `scan_layer`.
- With default attention/loss, compile time for `scan_layer` is about `15-17s`.
- The `80s` compile time comes from `xla_chunked`, not from `scan` vs `fori_loop`.
- Under `xla_chunked`, `scan` and `fori_loop` have nearly identical throughput.
- Under `xla_chunked`, `lax.scan` uses less temp memory than `fori_loop`:
  - `lax.scan`: `4.2 GB`
  - `fori_loop`: `5.1 GB`
- With default attention/loss, `fori_loop scan_layer` gave the highest observed throughput:
  - `37349.61 tok/s`
- Moving `prepare_weights` out of JIT materially reduced compile time on the current `fori_loop` path:
  - no LoRA + `sgd`: `34.63s -> 9.87s`
  - LoRA + `adam` + `xla_chunked`: `81.95s -> 33.20s`
- Current local state: `scan_layer` uses `scan`, and the stacked weights are prepared after LoRA and before optimizer init.

## Why compile time jumped from ~16s to ~80s

The original `~80s` number was not caused by switching between `scan` and `fori_loop`; it was mostly driven by `xla_chunked`.

Apples-to-apples comparison:

- Default attention/loss:
  - `scan_layer` + `lax.scan`: `15.84s`
  - `scan_layer` + `fori_loop`: `16.67s`
- `xla_chunked` attention/loss:
  - `scan_layer` + `lax.scan`: `81.95s`
  - `scan_layer` + `fori_loop`: `80.89s`

Interpretation:

- normal attention/loss: faster compile, higher temp memory
- `xla_chunked`: slower compile, lower temp memory

Additional improvement after moving `prepare_weights` out of JIT:

- no LoRA + `sgd` + default attention/loss:
  - before: `34.63s`
  - after: `9.87s`
- LoRA + `adam` + `xla_chunked` attention/loss:
  - before: `81.95s`
  - after: `33.20s`

## Helper script

Current helper script:

```bash
.agents/refactor-scan/test_scan_layer_fori_loop.sh
```

Examples:

```bash
OPTIMIZER_NAME=sgd MAX_TRAIN_STEP=2 \
  .agents/refactor-scan/test_scan_layer_fori_loop.sh
```

```bash
OPTIMIZER_NAME=sgd INIT_LORA=random MAX_TRAIN_STEP=2 \
  .agents/refactor-scan/test_scan_layer_fori_loop.sh
```

```bash
OPTIMIZER_NAME=adam INIT_LORA=random ATTN_IMPL=xla_chunked LOSS_IMPL=xla_chunked MAX_TRAIN_STEP=2 \
  .agents/refactor-scan/test_scan_layer_fori_loop.sh
```

## Memory note

The printed memory comes from `train_step_fn.memory_analysis()` and is reported as:

```text
total = output + temp + argument - alias
```

Observed stable terms across runs:

- `output`: about `1.4 GB`
- `argument`: about `1.4 GB`
- `alias`: about `1.4 GB`

Most of the differences between modes came from `temp` memory.
