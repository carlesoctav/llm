from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxformers.inference.attention_metadata import AttentionMetadata

try:
    from jax.experimental.pallas.ops.tpu.ragged_paged_attention import (
        ragged_paged_attention as _tpu_ragged_paged_attention,
    )
except Exception:  # pragma: no cover - optional TPU-only dependency surface.
    _tpu_ragged_paged_attention = None


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)


def get_kv_cache_shape(
    total_num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int, int]:
    # Layout used in this repository: [num_pages, page_size, 2 * kv_heads, head_dim]
    # with combined KV heads interleaved: [K0, V0, K1, V1, ...].
    return (total_num_pages, page_size, 2 * num_kv_heads, head_dim)


def _repeat_kv_heads(
    key: Float[Array, "T K H"],
    value: Float[Array, "T K H"],
    num_query_heads: int,
) -> tuple[Array, Array]:
    kv_heads = key.shape[1]
    if kv_heads == num_query_heads:
        return key, value
    if kv_heads <= 0 or num_query_heads % kv_heads != 0:
        raise ValueError(
            "Number of query heads must be a positive multiple of key/value heads."
        )
    repeat_factor = num_query_heads // kv_heads
    return (
        jnp.repeat(key, repeat_factor, axis=1),
        jnp.repeat(value, repeat_factor, axis=1),
    )


def ragged_attention_kernel(
    queries: Float[Array, "TOT_Q N H"],
    keys: Float[Array, "TOT_Q K H"],
    values: Float[Array, "TOT_Q K H"],
    kv_cache: Float[Array, "PAGES PAGE_SIZE TWO_K H"],
    attn_metadata: AttentionMetadata,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> tuple[Array, Array]:
    # This is a thin wrapper:
    # 1) write new K/V into the paged cache (static shapes, token-level metadata)
    # 2) run ragged paged attention over the updated cache.
    del q_scale, k_scale, v_scale

    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    # --- KV cache update (combined heads interleaved: [K0,V0,K1,V1,...]) ---
    total_tokens = attn_metadata.query_start_loc[-1]
    token_ids = jax.lax.iota(jnp.int32, queries.shape[0])
    valid = token_ids < total_tokens

    max_num_reqs = attn_metadata.seq_lens.shape[0]
    pages_per_req = attn_metadata.block_tables.shape[0] // max_num_reqs
    page_indices = attn_metadata.block_tables.reshape((max_num_reqs, pages_per_req))

    num_pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1]
    req_indices = attn_metadata.token_req_indices
    positions = attn_metadata.input_positions

    req_indices_safe = jnp.where(valid, req_indices, 0)
    positions_safe = jnp.where(valid, positions, 0)
    page_ids = positions_safe // page_size
    offsets = positions_safe - page_ids * page_size
    page_ids_safe = jnp.where(valid, page_ids, 0)

    pages = page_indices[req_indices_safe, page_ids_safe]
    slots = pages * page_size + offsets
    invalid_slot = jnp.asarray(num_pages * page_size, dtype=slots.dtype)
    slots_safe = jnp.where(valid, slots, invalid_slot)

    _, _, two_k, head_dim = kv_cache.shape
    kv_flat = kv_cache.reshape((num_pages * page_size, two_k, head_dim))
    kv_updates = jnp.stack([keys, values], axis=2).reshape((keys.shape[0], two_k, head_dim))

    updated_kv_flat = kv_flat.at[slots_safe].set(kv_updates, mode="drop")
    updated_kv_cache = updated_kv_flat.reshape(kv_cache.shape)

    # --- Attention ---
    num_seqs = attn_metadata.request_distribution[-1:].astype(jnp.int32)
    if jax.default_backend() == "tpu" and _tpu_ragged_paged_attention is not None:
        # The default JAX ragged paged attention kernel can choose a KV prefetch block
        # size that is too large for TPU VMEM when max_model_len is big (e.g. 2048).
        # Limiting KV pages per block keeps the scratch buffers under the VMEM cap.
        num_kv_pages_per_block = min(pages_per_req, 8)
        num_queries_per_block = 64
        out = _tpu_ragged_paged_attention(
            queries,
            updated_kv_cache,
            attn_metadata.seq_lens,
            page_indices,
            attn_metadata.query_start_loc,
            num_seqs,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=mask_value,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=16 * 1024 * 1024,
        )
        return out, updated_kv_cache

    # Reference (eager) attention path for non-TPU backends.
    num_kv_heads = two_k // 2
    num_query_heads = queries.shape[1]

    outputs: list[Array] = []
    num_reqs_int = int(num_seqs[0])
    for req_idx in range(num_reqs_int):
        q_start = int(attn_metadata.query_start_loc[req_idx])
        q_end = int(attn_metadata.query_start_loc[req_idx + 1])
        q_len = q_end - q_start
        if q_len <= 0:
            continue

        kv_len = int(attn_metadata.seq_lens[req_idx])
        q = queries[q_start:q_end]

        req_pages = page_indices[req_idx]
        gathered = updated_kv_cache[req_pages].reshape(-1, two_k, head_dim)[:kv_len]
        full_k = gathered[:, 0::2, :]
        full_v = gathered[:, 1::2, :]
        full_k, full_v = _repeat_kv_heads(full_k, full_v, num_query_heads)

        attn = jnp.einsum(
            "qnh,knh->nqk",
            q,
            full_k,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        attn = attn * sm_scale

        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        causal_mask = q_span < kv_span
        if sliding_window is not None:
            causal_mask = jnp.logical_or(causal_mask, q_span - sliding_window >= kv_span)

        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        attn = attn + jnp.where(causal_mask, mask_value, 0.0)
        probs = jax.nn.softmax(attn, axis=-1).astype(full_v.dtype)
        out = jnp.einsum(
            "nqk,knh->qnh",
            probs,
            full_v,
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        ).astype(queries.dtype)
        outputs.append(out)

    out_valid = jnp.concatenate(outputs, axis=0) if outputs else queries[:0]
    out = jnp.zeros_like(queries)
    if out_valid.shape[0] > 0:
        out = out.at[: out_valid.shape[0]].set(out_valid)
    return out, updated_kv_cache


def ragged_paged_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B T K H"],
    value: Float[Array, "B T K H"],
    bias: Array | None = None,
    mask: Array | None = None,
    *,
    kv_cache: Array,
    attn_metadata: AttentionMetadata,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    **kwargs,
) -> tuple[Array, Array]:
    del bias, mask, kwargs

    if attn_metadata is None:
        raise ValueError("`attn_metadata` is required for ragged paged attention.")
    if kv_cache is None:
        raise ValueError("`kv_cache` is required for ragged paged attention.")

    # Runtime policy for this repository: keep batch_size=1 and flatten tokens.
    bsz, seq_len, num_heads, head_dim = query.shape
    if bsz != 1:
        raise ValueError(
            f"ragged paged attention expects batch_size=1, got batch_size={bsz}."
        )
    num_kv_heads = key.shape[2]
    query_flat = query.reshape(bsz * seq_len, num_heads, head_dim)
    key_flat = key.reshape(bsz * seq_len, num_kv_heads, head_dim)
    value_flat = value.reshape(bsz * seq_len, num_kv_heads, head_dim)

    out_flat, updated_kv_cache = ragged_attention_kernel(
        query_flat,
        key_flat,
        value_flat,
        kv_cache,
        attn_metadata,
        sm_scale=head_dim**-0.5,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return out_flat.reshape((bsz, seq_len, num_heads, head_dim)), updated_kv_cache
