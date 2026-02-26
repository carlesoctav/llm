# CE Bench Notes (TPU v4)

Date: 2026-02-26

This doc captures quick memory + throughput measurements for `cross_entropy_loss` on a TPU v4 (local 4 devices; runs were single-process on `jax.default_backend() == "tpu"`).

Notes:
- `tokens = batch * pos = B`
- Memory numbers come from `compiled_executable.memory_analysis()` and are **per compiled executable per device** (XLA HBM), reported via `print_compiled_memory_stats`.
- Unless stated otherwise: `x, w` are `bf16`, `y` is `int32`, compute `dtype=float32`, `logit_soft_cap=None`, `reduction="mean"`.


## xla_chunked (streaming custom VJP)

Shape:
- `batch=64`, `pos=2048` => `B=131072`
- `H=512`, `V=128256`

Measured variants (same shape):

| b_block | v_block | fwd total/temp (GB) | bwd total/temp (GB) | fwd tok/s | bwd tok/s | status |
|---|---:|---:|---:|---:|---:|---|
| 1024 | 8192  | 0.4 / 0.1 | 0.9 / 0.4 | 837,138 | 192,189 | ok |
| 1024 | 32768 | 0.4 / 0.1 | 1.2 / 0.7 | 1,199,004 | 196,458 | ok |
| B (=131072) | 8192  | 4.4 / 4.1 | 9.2 / 8.8 | 543,715 | 114,459 | ok |
| B (=131072) | 32768 | 16.4 / 16.1 | bwd compile OOM | 776,222 | - | bwd_fail |

OOM details for `b_block=B, v_block=32768` (compile-time):
- TPU HBM exceeded (needed ~32.38G program HBM vs ~30.75G available).
- Largest temps were `f32[B, 32768]` allocations (~16GB each), i.e. effectively materializing huge `[B, v_block]` tiles.


## reference (non-streaming)

`implementation="reference"` computes full logits `[B, V]` (`einsum("bh,vh->bv")`), so it does not scale to large `B,V`.

### Large shape (same as above): does not compile

Shape: `B=131072, H=512, V=128256`

Both fwd and bwd compilation fail with `RESOURCE_EXHAUSTED` attempting to allocate the logits output:
- requested: `f32[131072,128256]` = `67243081728` bytes (~62.6 GiB)
- device memory limit in the error: `34359738368` bytes (32 GiB)

So: mem/speed are not available for reference at this scale.

### Small shape (Marin-style): compiles + runs

Shape:
- `batch=1`, `pos=512` => `B=512`
- `H=512`, `V=128256`

Results:

| impl | shape | fwd total/temp (GB) | bwd total/temp (GB) | fwd tok/s | bwd tok/s |
|---|---|---:|---:|---:|---:|
| reference | B=512,H=512,V=128256 | 0.4 / 0.2 | 0.5 / 0.2 | 699,077 | 346,599 |

Interpretation:
- Temp size matches the full logits tile: `512 * 128256 * 4 bytes ~= 262 MB` (float32).
- Throughput at `B=512` is sensitive to overhead and may not extrapolate.

