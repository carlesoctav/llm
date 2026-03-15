# Fix: `xla_chunked` Cross-Entropy Sharding Mismatch (`scan` carry type error)

## What’s happening

Your crash is the same class of bug as the earlier `tokamax` attention issue, but now in the **chunked cross-entropy** kernel:

```
TypeError: scan body function carry input and carry output must have equal types
...
input:  float32[1024,8192]
output: float32[1024@dp_shard,8192]
```

JAX treats **sharding as part of the array type** inside `scan`/`fori_loop` carries. In `src/jaxformers/ops/cross_entropy/xla_chunked.py`, the inner H-loop initializes the accumulator with an **unsharded** `jnp.zeros(...)`, but the first iteration produces a **dp_shard-sharded** result (because `x` is sharded on the batch axis). That makes the carry type change across iterations, which JAX forbids.

The failing line is:

- `fused_cross_entropy_chunked_xla(...):`
  - `logits = jax.lax.fori_loop(..., jnp.zeros((b_block, v_block), dtype=dtype))`

## Fastest workaround (no kernel changes)

You are forcing only the broken implementation:

- In your config: `config.loss_implementation = "xla_chunked"`
- In code: `implementation=config.loss_implementation or None`

So there is no fallback. To unblock training immediately, run with the reference CE:

```bash
python src/jaxformers/train/ntp.py --config ./experiments/tunix-repro/lora_gemma3_config.py \
  c.loss_implementation=reference
```

Alternative: set `config.loss_implementation = None` (or remove the field) so `cross_entropy_loss()` can try `("xla_chunked", "reference")` and fall back automatically when `xla_chunked` fails.

## Proper fix (keep `xla_chunked` and `dp_shard`)

Make the kernel allocate its loop carries with the **same sharding** as the loop body output.

### 1) Shard the H-loop accumulator (`acc`)

In `src/jaxformers/ops/cross_entropy/xla_chunked.py` inside `v_body`:

- Today: `acc0 = jnp.zeros((b_block, v_block), dtype=dtype)` (unsharded)
- Needed: `acc0` must be sharded like the eventual `logits` (sharded on batch axis with `dp_shard`)

Use one of these approaches:

- Derive from `x` sharding (most robust if you run with different meshes):
  - 2D accumulator: `out_sharding = x.sharding` (if `x` is `[B, H]` sharded on axis 0)
- Or explicitly shard on `dp_shard` (works for your current setup):
  - `out_sharding = PartitionSpec("dp_shard", None)`

Then create zeros with that `out_sharding`.

### 2) Also shard any other `fori_loop` carries that become sharded

Even if the current traceback points at the H-loop accumulator, the same principle applies to other carries:

- `lse0, loss0 = jnp.zeros((B,), ...)`
- `lse_b`, `label_logits_b` (currently created via `jnp.full(..., -inf)`)

If any of those become `@dp_shard` during the loop, their initial values must be created with matching sharding too.

Note: `jnp.full()` doesn’t take `out_sharding` in your JAX version, so if you need a sharded `-inf` initializer, build it from a sharded zero:

- `sharded_zero + (-jnp.inf)` (keeps sharding)
- or `jnp.zeros_like(..., out_sharding=...)` + constant

### 3) Verify

Re-run a single compile (the first training step). If the fix is correct:

- the `scan/fori_loop` type error disappears
- you should either start stepping or hit a different (real) numerical/shape issue

## Last-resort options

- Set `dp_shard=1` (disables batch sharding, sidesteps the carry-type mismatch).
- Stop using the chunked CE kernel: keep `dp_shard`, but use `loss_implementation=reference` (more memory).

## How tokamax avoids this (pattern to copy)

Tokamax solves the exact same class of error for its `xla_chunked` attention by **switching execution into `shard_map` when sharding is requested**.

In `tokamax/_src/ops/attention/base.py`, `DotProductAttention.__call__` does:

- If `q_sharding is None`: call the kernel directly.
- Else: build `in_specs/out_specs` from `q_sharding.mesh` + `q_sharding.spec`, then call `jax.experimental.shard_map.shard_map(fwd_closed, ...)`.

See:

- `.venv/lib/python3.11/site-packages/tokamax/_src/ops/attention/base.py:399-481`

Why this helps: inside `shard_map`, the function body runs on **local shard arrays**, so intermediate allocations and loop carries don’t acquire global sharding annotations that can change across iterations.

If you want to introduce `shard_map` for CE, mirror that structure:

- Shard-map over your `dp_shard` mesh axis.
- `x` and `labels` should be sharded on the batch axis, `w` replicated.
- Return per-token `loss` and `lse` as sharded vectors; keep the final `sum/mean` reduction outside so global reductions still happen in normal JAX code.

One extra detail for CE: if you shard the batch, the kernel sees `B_local`, so block-size inference must be compatible with `B_local` (not just global `B`).
