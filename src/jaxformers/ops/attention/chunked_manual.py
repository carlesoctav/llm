import functools
import math
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def _pad_axis0(x: jax.Array, pad_len: int) -> jax.Array:
    if pad_len == 0:
        return x
    pad = jnp.zeros((pad_len,) + x.shape[1:], dtype=x.dtype)
    return jnp.concatenate([x, pad], axis=0)


def _pad_axis_last(x: jax.Array, pad_len: int) -> jax.Array:
    if pad_len == 0:
        return x
    pad = jnp.zeros(x.shape[:-1] + (pad_len,), dtype=x.dtype)
    return jnp.concatenate([x, pad], axis=-1)


def _pad_q_axis(x: jax.Array, pad_len: int) -> jax.Array:
    # q-axis is axis 0 for our internal layout: [Q, H, ...]
    return _pad_axis0(x, pad_len)


def _canonicalize_qhk(x: jax.Array, num_q: int) -> jax.Array:
    """Return x shaped [Q, H, K] with broadcastable H, keeping Q on axis 0.

    Accepts [Q, K] or [H, Q, K] or [Q, H, K].
    """
    if x.ndim == 2:
        # [Q, K] -> [Q, 1, K]
        if x.shape[0] != num_q:
            raise ValueError(f"Expected x.shape[0]==num_q, got {x.shape[0]} vs {num_q}")
        return x[:, None, :]
    if x.ndim == 3:
        if x.shape[0] == num_q:
            # [Q, H, K]
            return x
        if x.shape[1] == num_q:
            # [H, Q, K] -> [Q, H, K]
            return jnp.transpose(x, (1, 0, 2))
    raise ValueError(f"Unsupported (q,h,k) layout: shape={x.shape}")


def _query_chunk_attention(
    query: Float[Array, "Q H D"],
    key: Float[Array, "K H D"],
    value: Float[Array, "K H V"],
    *,
    precision: jax.lax.PrecisionLike,
    key_chunk_size: int,
    bias_qhk: Float[Array, "Q #H K"] | None,
    mask_qhk: Bool[Array, "Q #H K"] | None,
) -> Float[Array, "Q H V"]:
    """Attention for a fixed query chunk, chunking over the KV axis.

    This is a manual, checkpointed (remat) implementation intended for debugging
    activation memory. It is not optimized.
    """
    num_kv, num_heads, k_features = key.shape
    v_features = value.shape[-1]

    if key_chunk_size <= 0:
        raise ValueError("key_chunk_size must be > 0")
    key_chunk_size = min(key_chunk_size, num_kv)

    # Pad K/V (and bias/mask) so we can use a fixed `dynamic_slice` size.
    num_chunks = math.ceil(num_kv / key_chunk_size)
    padded_kv = num_chunks * key_chunk_size
    pad_kv = padded_kv - num_kv

    if pad_kv:
        key = cast(Float[Array, "K H D"], _pad_axis0(key, pad_kv))
        value = cast(Float[Array, "K H V"], _pad_axis0(value, pad_kv))
        if bias_qhk is not None:
            bias_qhk = cast(Float[Array, "Q #H K"], _pad_axis_last(bias_qhk, pad_kv))
        if mask_qhk is not None:
            mask_qhk = cast(Bool[Array, "Q #H K"], _pad_axis_last(mask_qhk, pad_kv))

    query = query / jnp.sqrt(jnp.array(k_features, dtype=jnp.float32)).astype(query.dtype)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(
        query: Float[Array, "Q H D"],
        key: Float[Array, "k H D"],
        value: Float[Array, "k H V"],
        bias: Float[Array, "Q #H k"] | None,
        mask: Bool[Array, "Q #H k"] | None,
    ):
        attn_weights = jnp.einsum(
            "qhd,khd->qhk",
            query,
            key,
            precision=precision,
            preferred_element_type=query.dtype,
        ).astype(jnp.float32)

        if bias is not None:
            attn_weights = attn_weights + bias.astype(jnp.float32)

        if mask is not None:
            mask_value = jnp.finfo(attn_weights.dtype).min
            attn_weights = jnp.where(mask, attn_weights, mask_value)

        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)

        exp_values = jnp.einsum(
            "khd,qhk->qhd",
            value.astype(jnp.float32),
            exp_weights,
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        return (
            exp_values,  # [Q, H, V]
            exp_weights.sum(axis=-1),  # [Q, H]
            jnp.squeeze(max_score, axis=-1),  # [Q, H]
        )

    def chunk_scanner(chunk_idx: jax.Array):
        key_chunk = jax.lax.dynamic_slice_in_dim(key, chunk_idx, key_chunk_size, axis=0)
        value_chunk = jax.lax.dynamic_slice_in_dim(
            value, chunk_idx, key_chunk_size, axis=0
        )
        bias_chunk = (
            None
            if bias_qhk is None
            else jax.lax.dynamic_slice_in_dim(
                bias_qhk, chunk_idx, key_chunk_size, axis=-1
            )
        )
        mask_chunk = (
            None
            if mask_qhk is None
            else jax.lax.dynamic_slice_in_dim(
                mask_qhk, chunk_idx, key_chunk_size, axis=-1
            )
        )
        return summarize_chunk(query, key_chunk, value_chunk, bias_chunk, mask_chunk)

    chunk_starts = jnp.arange(0, padded_kv, key_chunk_size, dtype=jnp.int32)
    chunk_values, chunk_weights, chunk_max = jax.lax.map(chunk_scanner, chunk_starts)

    global_max = jnp.max(chunk_max, axis=0)  # [Q, H]
    max_diffs = jnp.exp(chunk_max - global_max[None, :, :])  # [C, Q, H]
    chunk_values = chunk_values * max_diffs[..., None]  # [C, Q, H, V]
    chunk_weights = chunk_weights * max_diffs  # [C, Q, H]

    all_values = chunk_values.sum(axis=0)  # [Q, H, V]
    all_weights = chunk_weights.sum(axis=0)[..., None]  # [Q, H, 1]
    return (all_values / all_weights).astype(query.dtype)


def attention(
    query: Float[Array, "T H D"],
    key: Float[Array, "S H D"],
    value: Float[Array, "S H V"],
    *,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 4096,
    bias_ths: Float[Array, "T #H S"] | None = None,
    mask_ths: Bool[Array, "T #H S"] | None = None,
) -> Float[Array, "T H V"]:
    """Memory-ish efficient attention by chunking Q and KV.

    Internal layout is [T, H, D] (no batch axis). Caller is expected to `vmap`
    over batch.
    """
    num_q, num_heads, q_features = query.shape

    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be > 0")
    query_chunk_size = min(query_chunk_size, num_q)

    # Pad Q (and bias/mask) so we can use a fixed `dynamic_slice` size.
    num_q_chunks = math.ceil(num_q / query_chunk_size)
    padded_q = num_q_chunks * query_chunk_size
    pad_q = padded_q - num_q

    if pad_q:
        query = cast(Float[Array, "T H D"], _pad_axis0(query, pad_q))
        if bias_ths is not None:
            bias_ths = cast(Float[Array, "T #H S"], _pad_q_axis(bias_ths, pad_q))
        if mask_ths is not None:
            mask_ths = cast(Bool[Array, "T #H S"], _pad_q_axis(mask_ths, pad_q))

    def q_chunk_fn(chunk_idx: jax.Array):
        q_chunk = jax.lax.dynamic_slice_in_dim(
            query, chunk_idx, query_chunk_size, axis=0
        )
        bias_chunk = (
            None
            if bias_ths is None
            else jax.lax.dynamic_slice_in_dim(
                bias_ths, chunk_idx, query_chunk_size, axis=0
            )
        )
        mask_chunk = (
            None
            if mask_ths is None
            else jax.lax.dynamic_slice_in_dim(
                mask_ths, chunk_idx, query_chunk_size, axis=0
            )
        )
        return _query_chunk_attention(
            q_chunk,
            key,
            value,
            precision=precision,
            key_chunk_size=key_chunk_size,
            bias_qhk=bias_chunk,
            mask_qhk=mask_chunk,
        )

    q_chunk_starts = jnp.arange(0, padded_q, query_chunk_size, dtype=jnp.int32)
    out_chunks = jax.lax.map(q_chunk_fn, q_chunk_starts)  # [C, q, H, V]
    out = out_chunks.reshape((padded_q, num_heads, value.shape[-1]))
    return out[:num_q]


def chunked_manual_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, " B #N T S"] | None = None,
    q_sharding: jax.NamedSharding | None = None,
    **kwargs,
) -> Float[Array, "B T N H"]:
    """Manual chunked attention for debugging activation memory.

    Notes:
      - When `q_sharding` is provided, we run under `shard_map` on the batch axis to
        avoid `vmap` sharding restrictions (masks are often replicated).
      - `bias` is supported for completeness, but most model paths pass `None`.
    """
    def axis_spec_is_nontrivial(mesh, axis_spec) -> bool:
        if axis_spec is None:
            return False
        if isinstance(axis_spec, tuple):
            return any(mesh.shape.get(a, 1) > 1 for a in axis_spec)
        return mesh.shape.get(axis_spec, 1) > 1

    precision = kwargs.pop("precision", jax.lax.Precision.HIGHEST)
    query_chunk_size = int(kwargs.pop("query_chunk_size", 1024))
    key_chunk_size = int(kwargs.pop("key_chunk_size", 4096))

    query = cast(Float[Array, "B T N H"], query)
    key = cast(Float[Array, "B S K H"], key)
    value = cast(Float[Array, "B S K H"], value)

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Expected query/key/value to be rank-4 [B,T,heads,dim]")

    bsz, q_len, num_q_heads, head_dim = query.shape
    bsz_k, kv_len, num_kv_heads, head_dim_k = key.shape
    bsz_v, kv_len_v, num_kv_heads_v, head_dim_v = value.shape

    if bsz_k != bsz or bsz_v != bsz:
        raise ValueError("Batch size mismatch between query/key/value")
    if kv_len_v != kv_len:
        raise ValueError("Key/value sequence length mismatch")
    if num_kv_heads_v != num_kv_heads:
        raise ValueError("Key/value head-count mismatch")
    if head_dim_k != head_dim or head_dim_v != head_dim:
        raise ValueError("Query/key/value head-dim mismatch")

    # Expand MQA/GQA heads: [B,S,K,H] -> [B,S,N,H].
    if num_kv_heads != num_q_heads:
        if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
            raise ValueError(
                "Number of query heads must be a positive multiple of key/value heads"
            )
        repeat_factor = num_q_heads // num_kv_heads
        key = jnp.repeat(key, repeat_factor, axis=-2)
        value = jnp.repeat(value, repeat_factor, axis=-2)
        num_kv_heads = num_q_heads

    # If the caller provides an explicit `q_sharding`, run the batch dimension under
    # `shard_map` to avoid `vmap`'s requirement that all mapped inputs have identical
    # sharding on the mapped axis. (In practice, our causal mask is often replicated.)
    if q_sharding is not None:
        mesh = q_sharding.mesh
        q_spec = tuple(q_sharding.spec)
        batch_axis = q_spec[0] if len(q_spec) >= 1 else None
        seq_axis = q_spec[1] if len(q_spec) >= 2 else None
        heads_axis = q_spec[2] if len(q_spec) >= 3 else None
        dim_axis = q_spec[3] if len(q_spec) >= 4 else None

        if batch_axis is not None and axis_spec_is_nontrivial(mesh, batch_axis):
            # Only support batch sharding for now.
            if axis_spec_is_nontrivial(mesh, seq_axis) or axis_spec_is_nontrivial(
                mesh, heads_axis
            ) or axis_spec_is_nontrivial(mesh, dim_axis):
                raise NotImplementedError(
                    "chunked_manual attention only supports sharding on the batch axis."
                )

            from jax.experimental import shard_map
            from jax.sharding import PartitionSpec as P

            # Always shard on the batch axis only. Any additional axes in `q_sharding`
            # are treated as replicated.
            qkv_spec = P(batch_axis, None, None, None)
            mask_spec = P(batch_axis, None, None, None)

            if bias is None and mask is None:

                def per_shard(q_local, k_local, v_local):
                    return chunked_manual_dot_product_attention(
                        q_local,
                        k_local,
                        v_local,
                        bias=None,
                        mask=None,
                        q_sharding=None,
                        precision=precision,
                        query_chunk_size=query_chunk_size,
                        key_chunk_size=key_chunk_size,
                    )

                per_shard_mapped = shard_map.shard_map(
                    per_shard,
                    mesh,
                    in_specs=(qkv_spec, qkv_spec, qkv_spec),
                    out_specs=qkv_spec,
                    check_rep=False,
                )
                return per_shard_mapped(query, key, value)

            if bias is None and mask is not None:

                def per_shard(q_local, k_local, v_local, mask_local):
                    return chunked_manual_dot_product_attention(
                        q_local,
                        k_local,
                        v_local,
                        bias=None,
                        mask=mask_local,
                        q_sharding=None,
                        precision=precision,
                        query_chunk_size=query_chunk_size,
                        key_chunk_size=key_chunk_size,
                    )

                per_shard_mapped = shard_map.shard_map(
                    per_shard,
                    mesh,
                    in_specs=(qkv_spec, qkv_spec, qkv_spec, mask_spec),
                    out_specs=qkv_spec,
                    check_rep=False,
                )
                return per_shard_mapped(query, key, value, mask)

            if bias is not None and mask is None:

                def per_shard(q_local, k_local, v_local, bias_local):
                    return chunked_manual_dot_product_attention(
                        q_local,
                        k_local,
                        v_local,
                        bias=bias_local,
                        mask=None,
                        q_sharding=None,
                        precision=precision,
                        query_chunk_size=query_chunk_size,
                        key_chunk_size=key_chunk_size,
                    )

                per_shard_mapped = shard_map.shard_map(
                    per_shard,
                    mesh,
                    in_specs=(qkv_spec, qkv_spec, qkv_spec, mask_spec),
                    out_specs=qkv_spec,
                    check_rep=False,
                )
                return per_shard_mapped(query, key, value, bias)

            assert bias is not None and mask is not None

            def per_shard(q_local, k_local, v_local, bias_local, mask_local):
                return chunked_manual_dot_product_attention(
                    q_local,
                    k_local,
                    v_local,
                    bias=bias_local,
                    mask=mask_local,
                    q_sharding=None,
                    precision=precision,
                    query_chunk_size=query_chunk_size,
                    key_chunk_size=key_chunk_size,
                )

            per_shard_mapped = shard_map.shard_map(
                per_shard,
                mesh,
                in_specs=(qkv_spec, qkv_spec, qkv_spec, mask_spec, mask_spec),
                out_specs=qkv_spec,
                check_rep=False,
            )
            return per_shard_mapped(query, key, value, bias, mask)

    bias_arr = None
    if bias is not None:
        bias_arr = jnp.asarray(bias)
        # Expect [B, #H, T, S] or [#H, T, S] or [T, S]. Convert per-batch to [T, #H, S].
        if bias_arr.ndim == 2:
            bias_arr = bias_arr[None, None, :, :]
        elif bias_arr.ndim == 3:
            bias_arr = bias_arr[None, :, :, :]
        elif bias_arr.ndim != 4:
            raise ValueError(f"Unsupported bias shape {bias_arr.shape}")

    mask_arr = None
    if mask is not None:
        mask_arr = jnp.asarray(mask, dtype=jnp.bool_)
        # Expect [B, #H, T, S] or [#H, T, S] or [T, S]. Convert per-batch to [T, #H, S].
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[None, None, :, :]
        elif mask_arr.ndim == 3:
            mask_arr = mask_arr[None, :, :, :]
        elif mask_arr.ndim != 4:
            raise ValueError(f"Unsupported mask shape {mask_arr.shape}")

    def per_batch(
        q_b: Float[Array, "T N H"],
        k_b: Float[Array, "S N H"],
        v_b: Float[Array, "S N H"],
        bias_b: Array | None,
        mask_b: Array | None,
    ) -> Float[Array, "T N H"]:
        # Internal layout is [T, H, D], so swap heads and features accordingly.
        q_thd = jnp.asarray(q_b)
        k_shd = jnp.asarray(k_b)
        v_shv = jnp.asarray(v_b)

        bias_ths = None
        if bias_b is not None:
            bias_b = _canonicalize_qhk(jnp.asarray(bias_b), num_q=q_len)  # [T, h, S]
            bias_ths = cast(Float[Array, "T #H S"], bias_b.astype(jnp.float32))

        mask_ths = None
        if mask_b is not None:
            mask_b = _canonicalize_qhk(jnp.asarray(mask_b), num_q=q_len)  # [T, h, S]
            mask_ths = cast(Bool[Array, "T #H S"], mask_b.astype(jnp.bool_))

        return cast(
            Float[Array, "T N H"],
            attention(
                q_thd,
                k_shd,
                v_shv,
                precision=precision,
                query_chunk_size=query_chunk_size,
                key_chunk_size=key_chunk_size,
                bias_ths=bias_ths,
                mask_ths=mask_ths,
            ),
        )

    if bias_arr is None and mask_arr is None:
        return jax.vmap(
            lambda q_b, k_b, v_b: per_batch(q_b, k_b, v_b, None, None),
            in_axes=(0, 0, 0),
            out_axes=0,
        )(query, key, value)

    if bias_arr is None and mask_arr is not None:
        return jax.vmap(
            lambda q_b, k_b, v_b, m_b: per_batch(q_b, k_b, v_b, None, m_b),
            in_axes=(0, 0, 0, 0),
            out_axes=0,
        )(query, key, value, mask_arr)

    if bias_arr is not None and mask_arr is None:
        return jax.vmap(
            lambda q_b, k_b, v_b, b_b: per_batch(q_b, k_b, v_b, b_b, None),
            in_axes=(0, 0, 0, 0),
            out_axes=0,
        )(query, key, value, bias_arr)

    assert bias_arr is not None and mask_arr is not None
    return jax.vmap(
        lambda q_b, k_b, v_b, b_b, m_b: per_batch(q_b, k_b, v_b, b_b, m_b),
        in_axes=(0, 0, 0, 0, 0),
        out_axes=0,
    )(query, key, value, bias_arr, mask_arr)
