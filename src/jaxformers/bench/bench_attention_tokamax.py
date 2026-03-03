# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# Minimal attention microbench to compare Tokamax implementations on TPU/GPU/CPU.
# Mirrors the style of `bench_cross_entropy_chunked_xla.py` (compile + steady,
# memory analysis for fwd + bwd).
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import tokamax

from jaxformers.benchmark_utils import print_compiled_memory_stats


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    return default if v == "" else int(v)


def _make_bool_causal_mask(batch: int, seqlen: int) -> jax.Array:
    # Shape matches what our model passes into tokamax: [B, 1, T, S].
    base = jnp.tril(jnp.ones((seqlen, seqlen), dtype=jnp.bool_))
    return jnp.broadcast_to(base, (batch, 1, seqlen, seqlen))


def main() -> None:
    print("devices:", jax.devices())

    batch = _env_int("BATCH", 4)
    seqlen = _env_int("SEQLEN", 2048)
    num_q_heads = _env_int("Q_HEADS", 4)
    num_kv_heads = _env_int("KV_HEADS", 1)
    head_dim = _env_int("HEAD_DIM", 256)
    layers = _env_int("LAYERS", 1)

    # Implementation passed to tokamax.
    # - unset / "auto": let tokamax pick (mosaic -> triton -> xla on TPU).
    # - "xla", "xla_chunked", "mosaic", ...
    impl = os.environ.get("TOKAMAX_ATTENTION_IMPL", "auto").strip().lower()
    if impl in ("", "auto", "sdpa"):
        impl = None

    # Mask mode:
    # - "bool": pass a [B,1,T,S] boolean mask (matches our training path)
    # - "causal": pass mask=None + is_causal=True (lets mosaic flash-attn run)
    # - "none": no masking at all
    mask_mode = os.environ.get("ATTENTION_MASK", "bool").strip().lower()
    if mask_mode not in ("bool", "causal", "none"):
        raise ValueError(f"Unsupported ATTENTION_MASK={mask_mode!r}")

    use_q_sharding = os.environ.get("USE_Q_SHARDING", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    q = jax.random.normal(kq, (batch, seqlen, num_q_heads, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (batch, seqlen, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (batch, seqlen, num_kv_heads, head_dim), dtype=jnp.bfloat16)

    is_causal = mask_mode == "causal"
    mask = _make_bool_causal_mask(batch, seqlen) if mask_mode == "bool" else None

    q_sharding = None
    if use_q_sharding:
        # Match the common dp-shard setup: shard the leading batch axis across devices.
        # This uses `shard_map` inside tokamax attention.
        devs = np.array(jax.devices())
        mesh = jax.sharding.Mesh(devs, ("dp_shard",))
        q_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("dp_shard", None, None, None)
        )
        q = jax.device_put(q, q_sharding)
        k = jax.device_put(k, q_sharding)
        v = jax.device_put(v, q_sharding)
        if mask is not None:
            mask = jax.device_put(mask, q_sharding)

    print("batch", batch)
    print("seqlen", seqlen)
    print("q_heads", num_q_heads)
    print("kv_heads", num_kv_heads)
    print("head_dim", head_dim)
    print("tokamax_impl", impl or "auto")
    print("mask_mode", mask_mode)
    print("use_q_sharding", use_q_sharding)
    print("layers", layers)

    def attn_fn(q_in, k_in, v_in, mask_in):
        return tokamax.dot_product_attention(
            q_in,
            k_in,
            v_in,
            mask=mask_in,
            is_causal=is_causal,
            precision=jax.lax.Precision.HIGHEST,
            implementation=impl,
            q_sharding=q_sharding,
        )

    def stacked_attn_fn(q_in, k_in, v_in, mask_in):
        # Stack multiple attention calls to mimic per-layer activation saving.
        x = attn_fn(q_in, k_in, v_in, mask_in)
        if layers <= 1:
            return x

        # Residual-style stack. After the first layer, derive K/V from X to keep
        # shapes consistent for MQA/GQA settings.
        x = x + q_in
        for _ in range(layers - 1):
            k_x = x[:, :, :num_kv_heads, :]
            v_x = x[:, :, :num_kv_heads, :]
            x = x + attn_fn(x, k_x, v_x, mask_in)
        return x

    def loss_fn(q_in, k_in, v_in, mask_in):
        out = stacked_attn_fn(q_in, k_in, v_in, mask_in)
        return jnp.sum(out, dtype=jnp.float32)

    fwd_jit = jax.jit(stacked_attn_fn)
    bwd_jit = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2)))

    # Forward compile + memory.
    t0 = time.perf_counter()
    out = fwd_jit(q, k, v, mask)
    out.block_until_ready()
    fwd_compile_s = time.perf_counter() - t0
    print("fwd_compile_time_s", fwd_compile_s)
    print_compiled_memory_stats(fwd_jit.lower(q, k, v, mask).compile().memory_analysis())

    # Forward steady.
    steps = _env_int("STEPS", 5)
    t0 = time.perf_counter()
    for _ in range(steps):
        out = fwd_jit(q, k, v, mask)
        out.block_until_ready()
    fwd_steady_s = (time.perf_counter() - t0) / steps
    print("fwd_steady_time_s", fwd_steady_s)

    # Backward compile + memory.
    t0 = time.perf_counter()
    dq, dk, dv = bwd_jit(q, k, v, mask)
    dq.block_until_ready()
    dk.block_until_ready()
    dv.block_until_ready()
    bwd_compile_s = time.perf_counter() - t0
    print("bwd_compile_time_s", bwd_compile_s)
    print_compiled_memory_stats(bwd_jit.lower(q, k, v, mask).compile().memory_analysis())

    # Backward steady.
    t0 = time.perf_counter()
    for _ in range(steps):
        dq, dk, dv = bwd_jit(q, k, v, mask)
        dq.block_until_ready()
        dk.block_until_ready()
        dv.block_until_ready()
    bwd_steady_s = (time.perf_counter() - t0) / steps
    print("bwd_steady_time_s", bwd_steady_s)

    tokens = batch * seqlen
    print("tokens", tokens)
    print("fwd_tokens_per_s", tokens / fwd_steady_s)
    print("bwd_tokens_per_s", tokens / bwd_steady_s)


if __name__ == "__main__":
    main()
