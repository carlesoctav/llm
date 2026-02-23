# Fused Cross-Entropy (Pallas TPU) notes

## Implemented

- Ported Levanter fused linear-softmax cross entropy into `src/jaxformers/kernels/pallas/fused_cross_entropy_loss/` (implementations: `reference`, `xla`, `pallas_tpu`), plus a TPU v4-safe fallback in `src/jaxformers/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py:419`.
- Added `config.loss_implementation` (`reference`, `xla`, `tpu_pallas`, plus alias `xla_chunked` → `xla`) and wired NTP to compute loss from hidden states via the fused CE kernel. The Pallas path is wrapped in `jax.shard_map` (JAX 0.8.2 requires this; Mosaic kernels can’t be auto-partitioned) in `src/jaxformers/train/ntp.py:241`.
- Refactored causal LMs (Qwen3/Gemma3) into:
  - `embed_tokens(...)`
  - `forward(...)` (returns hidden states only; no vocab projection)
  - `unembed(...)`
  and extended `Model` to carry `embed`/`unembed` callables (`src/jaxformers/modeling_utils.py:65`).
- Added dummy NTP dataset + generalized dataloader to accept Grain `MapDataset`/`IterDataset` (HF datasets are auto-wrapped). `transforms=None` is supported.
  - `src/jaxformers/data/dummy.py:1`
  - `src/jaxformers/data/training.py:157`
- Added SGD optimizer: `src/jaxformers/optimizers/sgd.py:1`.
- Added Qwen3-4B dummy test config: `config/test_config_qwen4b_ntp.py:1` (uses `attn_implementation=xla_chunked`, `optimizer_name=sgd`, `data_name=dummy`).

## OOM snapshot (TPU v4, 4 chips, Qwen3-4B, `grad_accum=1`)

### `loss_implementation=tpu_pallas`
- **B=1**: works at **T=15360**, fails at **T=16384** (runtime HBM reserve OOM).
- **T=2048**: works up to **B=12**, fails at **B=13** (runtime reserve OOM). `B=14` fails at compile-time HBM OOM.

### `loss_implementation=reference`
- **B=1**: works at **T=13312**, fails at **T=14336** (compile-time HBM OOM).
- **T=2048**: works up to **B=12**, fails at **B=13** (runtime reserve OOM).

## Notes / limitations

- NTP loss currently **reshards the output embedding (`lm_head` / tied embedding) to fully replicated** before CE (`src/jaxformers/train/ntp.py:270`). This avoids sharding/VJP issues but means we’re **not** doing true TP-parallel vocab CE yet, and it impacts max batch/seq.
- Hidden states are reshared to replicated `(B,T,H)` before flattening to avoid token-dim sharding interacting with vocab-dim work (`src/jaxformers/train/ntp.py:233`).
- TPU v4 default Pallas block sizes are forced more conservative (h/v tiles of 128) to avoid Mosaic VMEM spill / compile-time VMEM OOMs (`src/jaxformers/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py:419`).

