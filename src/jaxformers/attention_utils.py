from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
import tokamax
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from jaxformers.ops.attention.xla_chunked import (
    TokamaxRematXlaChunkedDotProductAttention,
)
from jaxformers.utils import GeneralInterface


class AttentionImpl(Protocol):
    def __call__(
        query: Float[Array, "B T N H"],
        key: Float[Array, "B S K H"],
        value: Float[Array, "B S K H"],
        bias: Array | None = None,
        mask: Bool[Array, " B #N T S"] | None = None,
        *,
        scale: float | None = None,
        logits_soft_cap: float | None = None,
        q_sharding: jax.NamedSharding | None = None,
        **kwargs,
    ): ...

def eager_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, " B #N T S"] | None = None,
    *,
    scale: float | None = None,
    logits_soft_cap: float | None = None,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    dropout_rate: float = 0.0,
    dropout_rng: PRNGKeyArray | None = None,
    broadcast_dropout: bool = True,
    q_sharding: jax.NamedSharding | None = None,
    **kwargs,
) -> Float[Array, "B T N H"]:
    B, T, N, H = query.shape
    _, S, K, Hk = key.shape
    _, _, Kv, Hv = value.shape

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
        key = jnp.repeat(key, repeat_factor, axis=-2, out_sharding=q_sharding)
        value = jnp.repeat(value, repeat_factor, axis=-2, out_sharding=q_sharding)

    if scale is None:
        scale = 1 / (H**0.5)

    scores = jnp.einsum(
        "btnh, bsnh -> bnts",
        query,
        key,
        preferred_element_type=query.dtype,
        precision=precision,
    )

    scores = scores * scale

    if bias is not None:
        scores = scores + bias

    if logits_soft_cap is not None:
        scores = logits_soft_cap * jnp.tanh(scores / logits_soft_cap)

    if mask is not None:
        mask_value = float(jnp.finfo(scores.dtype).min)
        scores = jnp.where(mask, scores, mask_value)

    weights = jax.nn.softmax(scores, axis=-1)

    if dropout_rate > 0.0:
        if dropout_rng is None:
            raise TypeError("dropout_rate > 0 but no dropout_rng provided")
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            dropout_shape = list(weights.shape)
            dropout_shape[2] = 1
            keep = jax.random.bernoulli(dropout_rng, keep_prob, tuple(dropout_shape))
            keep = jnp.broadcast_to(keep, weights.shape)
        else:
            keep = jax.random.bernoulli(dropout_rng, keep_prob, weights.shape)
        multiplier = keep.astype(weights.dtype) / keep_prob
        weights = weights * multiplier

    return jnp.einsum(
        "bnts, bsnh -> btnh",
        weights,
        value,
        preferred_element_type=query.dtype,
        precision=precision,
    )

class AttentionInterface(GeneralInterface[str, AttentionImpl]):
    _global_mapping = {
        #eager is faster please use eager, if memory isnt the constraint
        "eager": eager_dot_product_attention,
        "sdpa": partial(
            tokamax.dot_product_attention,
            precision=jax.lax.Precision.HIGHEST,
            implementation="xla",
        ),
        "xla_chunked": partial(
            tokamax.dot_product_attention,
            implementation=TokamaxRematXlaChunkedDotProductAttention(
                chunk_size=(1024, 2048)
            ),
        ),
    }


ATTENTION_INTERFACE = AttentionInterface()
