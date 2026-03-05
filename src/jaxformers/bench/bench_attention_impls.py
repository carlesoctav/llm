# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# Attention microbench that can compare tokamax implementations vs our
# chunked XLA attention implementations. Mirrors the CE bench style:
# - compile time
# - steady-state time
# - compiled memory analysis (fwd + bwd)
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import tokamax

from jaxformers.benchmark_utils import print_compiled_memory_stats
from jaxformers.ops.attention import (
    chunked_manual_dot_product_attention,
    flash_attention_dot_product_attention,
    tokamax_remat_chunked_xla_dot_product_attention,
    xla_chunked_dot_product_attention,
)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    return default if v == "" else int(v)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "")
    return default if v == "" else v


def _make_bool_causal_mask(batch: int, seqlen: int) -> jax.Array:
    # Match what our training path passes around: [B, 1, T, S].
    base = jnp.tril(jnp.ones((seqlen, seqlen), dtype=jnp.bool_))
    return jnp.broadcast_to(base, (batch, 1, seqlen, seqlen))


def main() -> None:
    print("devices:", jax.devices())

    B = _env_int("BATCH", 16)
    T = _env_int("SEQLEN", 2048)
    N = _env_int("Q_HEADS", 4)
    K = _env_int("KV_HEADS", 1)
    H = _env_int("HEAD_DIM", 256)
    layers = _env_int("LAYERS", 1)
    steps = _env_int("STEPS", 5)

    impl = _env_str("ATTENTION_IMPL", "xla_chunked").strip().lower()
    if impl not in (
        "sdpa",
        "xla",
        "tokamax_xla_chunked",
        "tokamax_remat_xla_chunked",
        "flash_attention",
        "xla_chunked",
        "chunked_manual",
    ):
        raise ValueError(f"Unsupported ATTENTION_IMPL={impl!r}")

    mask_mode = _env_str("ATTENTION_MASK", "bool").strip().lower()
    if mask_mode not in ("bool", "causal", "none"):
        raise ValueError(f"Unsupported ATTENTION_MASK={mask_mode!r}")

    use_q_sharding = _env_str("USE_Q_SHARDING", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    query_chunk_size = _env_int("QUERY_CHUNK_SIZE", 128)
    key_chunk_size = _env_int("KEY_CHUNK_SIZE", 128)

    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    q = jax.random.normal(kq, (B, T, N, H), dtype=jnp.bfloat16)  # [B, T, N, H]
    k = jax.random.normal(kk, (B, T, K, H), dtype=jnp.bfloat16)  # [B, S, K, H]
    v = jax.random.normal(kv, (B, T, K, H), dtype=jnp.bfloat16)  # [B, S, K, H]

    is_causal = mask_mode == "causal"
    mask = _make_bool_causal_mask(B, T) if mask_mode == "bool" else None
    if impl == "chunked_manual" and mask_mode != "bool":
        # `chunked_manual` only accepts explicit masks. Keep semantics consistent.
        mask = _make_bool_causal_mask(B, T)
        is_causal = False

    q_sharding = None
    if use_q_sharding:
        devs = np.array(jax.devices())
        mesh_b = _env_int("MESH_B", len(devs))
        mesh_n = _env_int("MESH_N", 1)
        if mesh_b * mesh_n != len(devs):
            raise ValueError(f"MESH_B*MESH_N must equal device count; got {mesh_b}*{mesh_n} != {len(devs)}")
        mesh = jax.sharding.Mesh(devs.reshape(mesh_b, mesh_n), ("B", "N"))

        q_spec = jax.sharding.PartitionSpec("B" if mesh_b > 1 else None, None, "N" if mesh_n > 1 else None, None)
        kv_spec = jax.sharding.PartitionSpec("B" if mesh_b > 1 else None, None, ("N" if (mesh_n > 1 and K == N) else None), None)
        mask_spec = jax.sharding.PartitionSpec("B" if mesh_b > 1 else None, None, None, None)

        q_sharding = jax.sharding.NamedSharding(mesh, q_spec)
        kv_sharding = jax.sharding.NamedSharding(mesh, kv_spec)
        mask_sharding = jax.sharding.NamedSharding(mesh, mask_spec)

        q = jax.device_put(q, q_sharding)
        k = jax.device_put(k, kv_sharding)
        v = jax.device_put(v, kv_sharding)
        if mask is not None:
            mask = jax.device_put(mask, mask_sharding)

    print("B", B)
    print("T", T)
    print("N (q_heads)", N)
    print("K (kv_heads)", K)
    print("H (head_dim)", H)
    print("q_shape (B,T,N,H)", q.shape)
    print("k_shape (B,S,K,H)", k.shape)
    print("layers", layers)
    print("steps", steps)
    print("impl", impl)
    print("mask_mode", mask_mode)
    print("use_q_sharding", use_q_sharding)
    if impl in ("xla_chunked", "chunked_manual"):
        print("query_chunk_size", query_chunk_size)
        print("key_chunk_size", key_chunk_size)
    if impl == "tokamax_remat_xla_chunked":
        print("query_chunk_size", query_chunk_size)
        print("key_chunk_size", key_chunk_size)

    def attn_fn(q_in, k_in, v_in, mask_in):
        if impl == "chunked_manual":
            return chunked_manual_dot_product_attention(
                q_in,
                k_in,
                v_in,
                mask=mask_in,
                q_sharding=q_sharding,
                precision=jax.lax.Precision.HIGHEST,
                query_chunk_size=query_chunk_size,
                key_chunk_size=key_chunk_size,
            )

        if impl == "flash_attention":
            return flash_attention_dot_product_attention(
                q_in,
                k_in,
                v_in,
                mask=mask_in if mask_mode == "bool" else None,
                is_causal=is_causal,
                precision=jax.lax.Precision.HIGHEST,
                q_sharding=q_sharding,
            )

        if impl == "tokamax_remat_xla_chunked":
            return tokamax_remat_chunked_xla_dot_product_attention(
                q_in,
                k_in,
                v_in,
                mask=mask_in if mask_mode == "bool" else None,
                is_causal=is_causal,
                precision=jax.lax.Precision.HIGHEST,
                query_chunk_size=query_chunk_size,
                key_chunk_size=key_chunk_size,
                q_sharding=q_sharding,
            )

        if impl == "xla_chunked":
            return xla_chunked_dot_product_attention(
                q_in,
                k_in,
                v_in,
                mask=mask_in if mask_mode == "bool" else None,
                is_causal=is_causal,
                precision=jax.lax.Precision.HIGHEST,
                query_chunk_size=query_chunk_size,
                key_chunk_size=key_chunk_size,
                q_sharding=q_sharding,
            )

        tokamax_impl = None if impl == "sdpa" else ("xla_chunked" if impl == "tokamax_xla_chunked" else impl)
        return tokamax.dot_product_attention(
            q_in,
            k_in,
            v_in,
            mask=mask_in if mask_mode == "bool" else None,
            is_causal=is_causal,
            precision=jax.lax.Precision.HIGHEST,
            implementation=tokamax_impl,
            q_sharding=q_sharding,
        )

    def stacked_attn_fn(q_in, k_in, v_in, mask_in):
        x = attn_fn(q_in, k_in, v_in, mask_in)
        if layers <= 1:
            return x

        x = x + q_in
        for _ in range(layers - 1):
            k_x = x[:, :, :K, :]
            v_x = x[:, :, :K, :]
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

    tokens = B * T
    print("tokens", tokens)
    print("fwd_tokens_per_s", tokens / fwd_steady_s)
    print("bwd_tokens_per_s", tokens / bwd_steady_s)


if __name__ == "__main__":
    main()
