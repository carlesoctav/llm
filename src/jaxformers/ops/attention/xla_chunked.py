import functools
import math
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def _pad_axis(x: jax.Array, axis: int, pad_len: int, *, pad_value=0) -> jax.Array:
    if pad_len == 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[axis] = pad_len
    pad = jnp.full(tuple(pad_shape), pad_value, dtype=x.dtype)
    return jnp.concatenate([x, pad], axis=axis)


def _pad_qk(
    x: jax.Array,
    pad_q: int,
    pad_k: int,
    *,
    pad_value=0,
) -> jax.Array:
    # Expected layout: [..., Q, K] as the last two axes.
    if pad_q:
        x = _pad_axis(x, -2, pad_q, pad_value=pad_value)
    if pad_k:
        x = _pad_axis(x, -1, pad_k, pad_value=pad_value)
    return x


def _canonicalize_bhqs(
    x: Array | None,
    *,
    B: int,
    N: int,
    T: int,
    S: int,
    name: str,
) -> tuple[jax.Array, int, int] | None:
    """Return (x_bhqs, Bx, Hx) where x_bhqs has shape [Bx, Hx, T, S].

    Bx is either 1 or B, Hx is either 1 or N. Caller is expected to broadcast
    Bx/Hx as needed (preferably after slicing).
    """
    if x is None:
        return None

    x_arr = jnp.asarray(x)
    if x_arr.ndim == 2:  # [T, S]
        if x_arr.shape != (T, S):
            raise ValueError(f"{name} expected shape (T,S)=({T},{S}), got {x_arr.shape}")
        x_arr = x_arr[None, None, :, :]
    elif x_arr.ndim == 3:
        if x_arr.shape == (B, T, S):  # [B, T, S] -> [B, 1, T, S]
            x_arr = x_arr[:, None, :, :]
        elif x_arr.shape == (N, T, S):  # [N, T, S] -> [1, N, T, S]
            x_arr = x_arr[None, :, :, :]
        else:
            raise ValueError(
                f"{name} expected shape (B,T,S)=({B},{T},{S}) or (N,T,S)=({N},{T},{S}); got {x_arr.shape}"
            )
    elif x_arr.ndim == 4:
        if x_arr.shape[2:] != (T, S):
            raise ValueError(
                f"{name} expected trailing dims (T,S)=({T},{S}), got {x_arr.shape}"
            )
    else:
        raise ValueError(f"{name} must be rank-2/3/4, got shape {x_arr.shape}")

    Bx, Hx, _, _ = x_arr.shape
    if Bx not in (1, B):
        raise ValueError(f"{name} batch dim must be 1 or B={B}, got {Bx}")
    if Hx not in (1, N):
        raise ValueError(f"{name} head dim must be 1 or N={N}, got {Hx}")
    return cast(jax.Array, x_arr), Bx, Hx


def xla_chunked_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, " B #N T S"] | None = None,
    q_sharding: jax.sharding.NamedSharding | None = None,
    **kwargs,
) -> Float[Array, "B T N H"]:
    """Memory-efficient attention via XLA loops (no `shard_map`).

    Shapes / notation:
      - query:  [B, T, N, H]
      - key:    [B, S, K, H]
      - value:  [B, S, K, H]
      - logits: [B, N, T, S] (conceptual; never materialized fully)
      - out:    [B, T, N, H]
    """
    precision = kwargs.pop("precision", jax.lax.Precision.HIGHEST)
    query_chunk_size = int(kwargs.pop("query_chunk_size", 1024))
    key_chunk_size = int(kwargs.pop("key_chunk_size", 4096))
    is_causal = bool(kwargs.pop("is_causal", False))

    query = cast(Float[Array, "B T N H"], query)
    key = cast(Float[Array, "B S K H"], key)
    value = cast(Float[Array, "B S K H"], value)

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Expected query/key/value rank-4: [B,T,N,H], [B,S,K,H], [B,S,K,H]")

    B, T, N, H = query.shape
    Bk, S, K, Hk = key.shape
    Bv, Sv, Kv, Hv = value.shape
    if (Bk, Bv) != (B, B):
        raise ValueError("Batch size mismatch between query/key/value")
    if (Sv, Kv, Hv) != (S, K, Hk):
        raise ValueError("Key/value shape mismatch")
    if Hk != H:
        raise ValueError("Query/key/value head-dim mismatch")
    if K <= 0 or N % K != 0:
        raise ValueError("Number of query heads must be a positive multiple of key/value heads")

    if query_chunk_size <= 0 or key_chunk_size <= 0:
        raise ValueError("query_chunk_size and key_chunk_size must be > 0")

    # Pre-scale Q in its own dtype to avoid allocating a full float32 Q.
    scale = jnp.array(1.0 / math.sqrt(H), dtype=query.dtype)
    query = query * scale

    pad_t = (-T) % query_chunk_size
    pad_s = (-S) % key_chunk_size
    if pad_t:
        query = cast(Float[Array, "B T N H"], _pad_axis(query, 1, pad_t, pad_value=0))
    if pad_s:
        key = cast(Float[Array, "B S K H"], _pad_axis(key, 1, pad_s, pad_value=0))
        value = cast(Float[Array, "B S K H"], _pad_axis(value, 1, pad_s, pad_value=0))

    Tpad = T + pad_t
    Spad = S + pad_s
    num_t = Tpad // query_chunk_size
    num_s = Spad // key_chunk_size

    bias_bhqs = _canonicalize_bhqs(bias, B=B, N=N, T=T, S=S, name="bias")
    mask_bhqs = _canonicalize_bhqs(mask, B=B, N=N, T=T, S=S, name="mask")

    if bias_bhqs is not None:
        bias_arr, bias_B, bias_H = bias_bhqs
        bias_arr = _pad_qk(bias_arr, pad_t, pad_s, pad_value=0)
    else:
        bias_arr, bias_B, bias_H = None, 1, 1

    if mask_bhqs is not None:
        mask_arr, mask_B, mask_H = mask_bhqs
        mask_arr = _pad_qk(mask_arr.astype(jnp.bool_), pad_t, pad_s, pad_value=False)
    else:
        mask_arr, mask_B, mask_H = None, 1, 1

    key_valid = jnp.arange(Spad, dtype=jnp.int32) < S  # [Spad]

    def _exp_values(
        v_block: Float[Array, "B s K H"],
        exp_scores: Float[Array, "B t N s"],
    ) -> Float[Array, "B t N H"]:
        # exp_scores is float32; accumulate in float32.
        if K == 1:
            v_s = v_block[:, :, 0, :].astype(jnp.float32)  # [B, s, H]
            return jnp.einsum(
                "btns,bsh->btnh",
                exp_scores,
                v_s,
                precision=precision,
                preferred_element_type=jnp.float32,
            )
        if K == N:
            v_s = v_block.astype(jnp.float32)  # [B, s, N, H]
            return jnp.einsum(
                "btns,bsnh->btnh",
                exp_scores,
                v_s,
                precision=precision,
                preferred_element_type=jnp.float32,
            )

        group = N // K
        exp_kg = exp_scores.reshape(B, query_chunk_size, K, group, key_chunk_size)
        v_s = v_block.astype(jnp.float32)  # [B, s, K, H]
        out = jnp.einsum(
            "btkgs,bskh->btkgh",
            exp_kg,
            v_s,
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        return out.reshape(B, query_chunk_size, N, H)

    def _dot_scores(
        q_chunk: Float[Array, "B t N H"],
        k_block: Float[Array, "B s K H"],
    ) -> Float[Array, "B t N s"]:
        if K == 1:
            k_s = k_block[:, :, 0, :]  # [B, s, H]
            return (
                jnp.einsum(
                    "btnh,bsh->btns",
                    q_chunk,
                    k_s,
                    precision=precision,
                    preferred_element_type=query.dtype,
                )
                .astype(jnp.float32)
            )
        if K == N:
            return (
                jnp.einsum(
                    "btnh,bsnh->btns",
                    q_chunk,
                    k_block,
                    precision=precision,
                    preferred_element_type=query.dtype,
                )
                .astype(jnp.float32)
            )

        group = N // K
        q_kg = q_chunk.reshape(B, query_chunk_size, K, group, H)
        scores_kg = jnp.einsum(
            "btkgh,bskh->btkgs",
            q_kg,
            k_block,
            precision=precision,
            preferred_element_type=query.dtype,
        ).astype(jnp.float32)
        return scores_kg.reshape(B, query_chunk_size, N, key_chunk_size)

    key_starts = jnp.arange(0, Spad, key_chunk_size, dtype=jnp.int32)
    query_starts = jnp.arange(0, Tpad, query_chunk_size, dtype=jnp.int32)

    @functools.partial(jax.remat, prevent_cse=False)
    def q_body(_, t0):
        q_chunk = jax.lax.dynamic_slice(
            query, (0, t0, 0, 0), (B, query_chunk_size, N, H)
        )

        m = jnp.full((B, query_chunk_size, N), -jnp.inf, dtype=jnp.float32)
        l = jnp.zeros((B, query_chunk_size, N), dtype=jnp.float32)
        o = jnp.zeros((B, query_chunk_size, N, H), dtype=jnp.float32)

        @functools.partial(jax.remat, prevent_cse=False)
        def k_body(carry, s0):
            m, l, o = carry
            k_block = jax.lax.dynamic_slice(
                key, (0, s0, 0, 0), (B, key_chunk_size, K, H)
            )
            v_block = jax.lax.dynamic_slice(
                value, (0, s0, 0, 0), (B, key_chunk_size, K, H)
            )

            scores = _dot_scores(q_chunk, k_block)  # [B, t, N, s]

            if bias_arr is not None:
                bias_blk = jax.lax.dynamic_slice(
                    bias_arr,
                    (0, 0, t0, s0),
                    (bias_B, bias_H, query_chunk_size, key_chunk_size),
                ).astype(jnp.float32)
                if bias_B == 1 and B != 1:
                    bias_blk = jnp.broadcast_to(
                        bias_blk, (B, bias_H, query_chunk_size, key_chunk_size)
                    )
                if bias_H == 1:
                    bias_blk = jnp.broadcast_to(
                        bias_blk, (B, N, query_chunk_size, key_chunk_size)
                    )
                else:
                    bias_blk = bias_blk.reshape(B, N, query_chunk_size, key_chunk_size)
                scores = scores + jnp.swapaxes(bias_blk, 1, 2)  # [B,t,N,s]

            if mask_arr is not None:
                mask_blk = jax.lax.dynamic_slice(
                    mask_arr,
                    (0, 0, t0, s0),
                    (mask_B, mask_H, query_chunk_size, key_chunk_size),
                ).astype(jnp.bool_)
                if mask_B == 1 and B != 1:
                    mask_blk = jnp.broadcast_to(
                        mask_blk, (B, mask_H, query_chunk_size, key_chunk_size)
                    )
                if mask_H == 1:
                    mask_blk = jnp.broadcast_to(
                        mask_blk, (B, N, query_chunk_size, key_chunk_size)
                    )
                else:
                    mask_blk = mask_blk.reshape(B, N, query_chunk_size, key_chunk_size)
                scores = jnp.where(jnp.swapaxes(mask_blk, 1, 2), scores, -jnp.inf)

            valid_s = jax.lax.dynamic_slice_in_dim(
                key_valid, s0, key_chunk_size, axis=0
            )
            scores = jnp.where(valid_s[None, None, None, :], scores, -jnp.inf)

            if is_causal:
                q_idx = t0 + jnp.arange(query_chunk_size, dtype=jnp.int32)
                k_idx = s0 + jnp.arange(key_chunk_size, dtype=jnp.int32)
                causal = q_idx[:, None] >= k_idx[None, :]
                scores = jnp.where(causal[None, :, None, :], scores, -jnp.inf)

            block_max = jnp.max(scores, axis=-1)  # [B, t, N]
            m_new = jnp.maximum(m, block_max)

            alpha = jnp.exp(m - m_new)
            alpha = jnp.where(
                jnp.isfinite(m) & jnp.isfinite(m_new), alpha, jnp.array(0.0, jnp.float32)
            )

            exp_scores = jnp.exp(scores - m_new[..., None])
            exp_scores = jnp.where(
                jnp.isfinite(scores), exp_scores, jnp.array(0.0, jnp.float32)
            )

            l_new = l * alpha + exp_scores.sum(axis=-1)
            o_new = o * alpha[..., None] + _exp_values(v_block, exp_scores)

            return (m_new, l_new, o_new), ()

        (m, l, o), _ = jax.lax.scan(k_body, (m, l, o), key_starts)
        out_chunk = jnp.where(l[..., None] > 0, o / l[..., None], 0.0)
        return (), out_chunk.astype(query.dtype)

    _, out_chunks = jax.lax.scan(q_body, (), query_starts)
    out = jnp.swapaxes(out_chunks, 0, 1).reshape((B, Tpad, N, H))
    out = out[:, :T, :, :]

    if q_sharding is not None:
        out = jax.lax.with_sharding_constraint(out, q_sharding)

    return cast(Float[Array, "B T N H"], out)
