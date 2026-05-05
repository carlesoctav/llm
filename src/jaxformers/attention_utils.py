from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
import tokamax
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from jaxformers.ops.attention import chunked_manual_dot_product_attention
from jaxformers.ops.attention.xla_chunked import (
    TokamaxRematXlaChunkedDotProductAttention,
)
from jaxformers.ops.ragged_paged_attention import ragged_paged_dot_product_attention
from jaxformers.utils import GeneralInterface


class AttentionImpl(Protocol):
    def __call__(
        query: Float[Array, "B T N H"],
        key: Float[Array, "B S K H"],
        value: Float[Array, "B S K H"],
        bias: Array | None = None,
        mask: Bool[Array, " B #N T S"] | Array | None = None,
        *,
        q_sharding: jax.NamedSharding | None = None,
        **kwargs,
    ): ...


def _normalize_mask(
    mask: Array,
    batch_size: int,
    num_heads: int,
    tgt_len: int,
    src_len: int,
) -> Array:
    mask_array = jnp.asarray(mask)
    if mask_array.ndim == 2:
        if mask_array.shape != (tgt_len, src_len):
            raise ValueError(
                f"Mask shape {mask_array.shape} must match ({tgt_len}, {src_len})"
            )
        mask_array = mask_array[None, None, :, :]
        return jnp.broadcast_to(mask_array, (batch_size, num_heads, tgt_len, src_len))
    if mask_array.ndim == 3:
        if mask_array.shape == (batch_size, tgt_len, src_len):
            mask_array = mask_array[:, None, :, :]
            return jnp.broadcast_to(mask_array, (batch_size, num_heads, tgt_len, src_len))
        if mask_array.shape[1:] != (tgt_len, src_len):
            raise ValueError(
                f"Mask shape {mask_array.shape} must match ({num_heads}, {tgt_len}, {src_len})"
            )
        mask_array = mask_array[None, :, :, :]
        return jnp.broadcast_to(mask_array, (batch_size, num_heads, tgt_len, src_len))
    if mask_array.ndim == 4:
        if mask_array.shape[0] == batch_size and mask_array.shape[2:] == (
            tgt_len,
            src_len,
        ):
            if mask_array.shape[1] not in (1, num_heads):
                raise ValueError(
                    f"Mask shape {mask_array.shape} must match ({batch_size}, {num_heads}, {tgt_len}, {src_len})"
                )
            return jnp.broadcast_to(mask_array, (batch_size, num_heads, tgt_len, src_len))
        if mask_array.shape[0] == batch_size and mask_array.shape[1] == tgt_len:
            if mask_array.shape[2] not in (1, num_heads) or mask_array.shape[3] != src_len:
                raise ValueError(
                    f"Mask shape {mask_array.shape} must match ({batch_size}, {tgt_len}, {num_heads}, {src_len})"
                )
            mask_array = jnp.transpose(mask_array, (0, 2, 1, 3))
            return jnp.broadcast_to(mask_array, (batch_size, num_heads, tgt_len, src_len))
    raise ValueError(f"Mask rank must be 2, 3, or 4 but got shape {mask_array.shape}")


def eager_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, " B #N T S"] | Array | None = None,
    *,
    dropout_rate: float = 0.0,
    dropout_rng: PRNGKeyArray | None = None,
    broadcast_dropout: bool = True,
    q_sharding: jax.NamedSharding | None = None,
    scale: float | None = None,
    **kwargs,
) -> Float[Array, "B T N H"]:
    if scale is None:
        scale = query.shape[-1] ** -0.5
    query = query * jnp.asarray(scale, dtype=query.dtype)

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
        key = jnp.repeat(key, repeat_factor, axis=-2, out_sharding=q_sharding)
        value = jnp.repeat(value, repeat_factor, axis=-2, out_sharding=q_sharding)
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
        mask_array = _normalize_mask(mask, B, N, T, S)
        if mask_array.dtype == jnp.bool_:
            neg_inf = jnp.array(jnp.finfo(scores.dtype).min, dtype=scores.dtype)
            scores = jnp.where(mask_array, scores, neg_inf)
        else:
            scores = scores + mask_array

    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(scores.dtype, jnp.float32)

    weights = jax.nn.softmax(scores.astype(dtype), axis=-1).astype(scores.dtype)

    if dropout_rate > 0.0:
        if dropout_rng is None:
            raise TypeError("dropout_rate > 0 but no dropout_rng provided")
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            dropout_shape = list(weights.shape)
            if len(dropout_shape) >= 3:
                dropout_shape[2] = 1
            keep = jax.random.bernoulli(dropout_rng, keep_prob, tuple(dropout_shape))
            keep = jnp.broadcast_to(keep, weights.shape)
        else:
            keep = jax.random.bernoulli(dropout_rng, keep_prob, weights.shape)
        multiplier = keep.astype(weights.dtype) / keep_prob
        weights = weights * multiplier

    attn = jnp.einsum(
        "bnts, bsnh -> btnh", weights, value, preferred_element_type=query.dtype
    )
    return attn


class AttentionInterface(GeneralInterface[str, AttentionImpl]):
    _global_mapping = {
        "eager": eager_dot_product_attention,
        "sdpa": partial(
            tokamax.dot_product_attention, precision=jax.lax.Precision.HIGHEST
        ),
        "ragged_paged_dot_product_attention": ragged_paged_dot_product_attention,
        # Historically, "xla_chunked" referred to Tokamax's chunked XLA attention.
        # In this codebase we instead map it to the manual chunked implementation,
        # because Tokamax's xla_chunked backward can have very large temp memory.
        "xla_chunked": partial(
            tokamax.dot_product_attention,
            implementation=TokamaxRematXlaChunkedDotProductAttention(
                chunk_size=(1024, 2048)
            ),
        ),
        "chunked_manual": chunked_manual_dot_product_attention,
    }


ATTENTION_INTERFACE = AttentionInterface()
