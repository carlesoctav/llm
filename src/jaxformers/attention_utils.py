from __future__ import annotations

from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

import tokamax

from jaxformers.ops.attention import (
    chunked_manual_dot_product_attention,
    flash_attention_dot_product_attention,
    xla_chunked_dot_product_attention,
)
from jaxformers.utils import GeneralInterface


def _normalize_mask(
    mask: Bool[Array, "..."],
    B: int,
    N: int,
    T: int,
    S: int,
) -> Bool[Array, "B N T S"]:
    """Normalize a mask to `[B, N, T, S]`.

    Accepted layouts:
      - `[B, 1, T, S]` (common causal mask)
      - `[B, N, T, S]`
    """
    m = jnp.asarray(mask, dtype=jnp.bool_)
    if m.shape == (B, N, T, S):
        return m
    if m.shape == (B, 1, T, S):
        return jnp.broadcast_to(m, (B, N, T, S))
    raise ValueError(f"Unsupported mask shape {m.shape}; expected (B,1,T,S) or (B,N,T,S)")


class AttentionImpl(Protocol):
    def __call__(
        query: Float[Array, "B T N H"],
        key: Float[Array, "B S K H"],
        value: Float[Array, "B S K H"],
        bias: Array | None = None,
        mask: Bool[Array, " B #N T S"] | None = None,
        q_sharding: jax.NamedSharding | None = None,
        **kwargs,
    ):
        ...

def eager_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, " B #N T S"] | None = None,
    *,
    dropout_rate: float = 0.0,
    dropout_rng: PRNGKeyArray | None = None,
    broadcast_dropout: bool = True,
    q_sharding: jax.NamedSharding | None = None,
    **kwargs,
) -> Float[Array, "B T N H"]:
    query = query / jnp.sqrt(query.shape[-1])

    B, T, N, H = query.shape
    Bk, S, K, Hk = key.shape
    Bv, Sv, Kv, Hv = value.shape

    if Hk != H or Hv != H:
        raise ValueError("Query, key, and value must share the same head dimension")

    if Kv != K:
        raise ValueError("Value tensor must share the key head axis for attention")

    if K != N:
        if K <= 0 or N % K != 0:
            raise ValueError(
                "Number of query heads must be a positive multiple of key/value heads"
            )
        repeat_factor = N // K
        key = jnp.repeat(key, repeat_factor, axis=-2, out_sharding = q_sharding)
        value = jnp.repeat(value, repeat_factor, axis=-2, out_sharding = q_sharding)
        K = N

    scores = jnp.einsum(
        "btnh, bsnh -> bnts",
        query,
        key,
        preferred_element_type=query.dtype,
        precision=jax.lax.Precision.HIGHEST,
    )

    if bias is not None:
        scores = scores + bias

    if mask is not None:
        neg_inf = jnp.array(jnp.finfo(scores.dtype).min, dtype=scores.dtype)
        scores = jnp.where(mask, scores, neg_inf)

    dtype = jnp.result_type(scores.dtype, jnp.float32)

    weights = jax.nn.softmax(scores.astype(dtype), axis=-1).astype(scores.dtype)

    if dropout_rate > 0.0:
        if dropout_rng is None:
            raise TypeError("dropout_rate > 0 but no dropout_rng provided")
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            keep = jax.random.bernoulli(dropout_rng, keep_prob, weights.shape)
            keep = jnp.broadcast_to(keep, weights.shape)
        multiplier = keep.astype(weights.dtype) / keep_prob
        weights = weights * multiplier

    attn = jnp.einsum("bnts, bsnh -> btnh", weights, value, preferred_element_type=query.dtype)
    return attn

class AttentionInterface(GeneralInterface[str, AttentionImpl]):
    _global_mapping = {
        "eager": eager_dot_product_attention,
        "sdpa": partial(tokamax.dot_product_attention, precision = jax.lax.Precision.HIGHEST),
        "flash_attention": flash_attention_dot_product_attention,
        "xla_chunked": xla_chunked_dot_product_attention,
        "chunked_manual": chunked_manual_dot_product_attention,
    }

ATTENTION_INTERFACE = AttentionInterface()
