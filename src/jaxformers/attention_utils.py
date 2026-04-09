from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
import tokamax
from einops import rearrange
from jax import P
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
    splash_attention_mask as mask_lib,
)
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jaxformers.ops.attention import chunked_manual_dot_product_attention
from jaxformers.ops.attention.xla_chunked import (
    TokamaxRematXlaChunkedDotProductAttention,
)
from jaxformers.sharding_utils import auto_axes_out, uses_auto_mesh_axis
from jaxformers.utils import GeneralInterface


class AttentionImpl(Protocol):
    def __call__(
        query: Float[Array, "B T N H"],
        key: Float[Array, "B S K H"],
        value: Float[Array, "B S K H"],
        bias: Array | None = None,
        mask: Bool[Array, " B #N T S"] | None = None,
        *,
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
        if q_sharding is not None and uses_auto_mesh_axis():
            repeat = lambda x: jnp.repeat(x, repeat_factor, axis=-2)
            key = auto_axes_out(repeat, q_sharding, key)
            value = auto_axes_out(repeat, q_sharding, value)
        else:
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

    attn = jnp.einsum(
        "bnts, bsnh -> btnh", weights, value, preferred_element_type=query.dtype
    )
    return attn


def _normalize_partition_axis(axis):
    if axis is None or axis == ():
        return None
    if isinstance(axis, tuple) and len(axis) == 1:
        return axis[0]
    return axis


def _mesh_axis_size(mesh, axis) -> int:
    axis = _normalize_partition_axis(axis)
    if axis is None:
        return 1
    if isinstance(axis, tuple):
        size = 1
        for axis_name in axis:
            size *= mesh.shape[axis_name]
        return size
    return mesh.shape[axis]


def tpu_flash_atttention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, " B #N T S"] | None = None,
    *,
    segment_ids: Int[Array, " B T "] | None = None,
    q_sharding: jax.NamedSharding | None = None,
    is_causal: bool = True,
    is_sliding: bool = False,
    window_size: int | None = None,
    flash_attention_block_size=512,
):
    del q_sharding

    if bias is not None:
        raise ValueError("tpu_flash does not support bias")
    if mask is not None and mask.shape != (1,):
        raise ValueError(
            "tpu_flash does not support dense attention_mask; use structural "
            "sliding/causal selection and segment_ids instead"
        )

    _B, T, N, _H = query.shape
    _, S, K, _ = key.shape
    assert T == S

    is_mqa = K == 1

    query = rearrange(query, "b t n h -> b n t h")
    key = rearrange(key, "b s k h -> b k s h")
    value = rearrange(value, "b s k h -> b k s h")

    if is_mqa:
        key = jnp.squeeze(key, axis=1)
        value = jnp.squeeze(value, axis=1)

    if is_causal:
        splash_mask = mask_lib.CausalMask((T, S))
    else:
        splash_mask = mask_lib.FullMask((T, S))

    if is_sliding:
        splash_mask &= mask_lib.LocalMask(
            (T, S),
            (window_size, None),
            offset=0,
        )

    multi_head_mask = mask_lib.MultiHeadMask([splash_mask for _ in range(N)])
    block_size = flash_attention_block_size
    while T % block_size != 0 or S % block_size != 0:
        block_size //= 2

    block_sizes = splash.BlockSizes(
        block_q=block_size,
        block_kv=block_size,
        block_kv_compute=block_size,
        block_q_dkv=block_size,
        block_kv_dkv=block_size,
        block_kv_dkv_compute=block_size,
        block_q_dq=block_size,
        block_kv_dq=block_size,
    )

    query_sharding = jax.typeof(query).sharding
    mesh = query_sharding.mesh
    query_spec = query_sharding.spec
    key_spec = jax.typeof(key).sharding.spec
    value_spec = jax.typeof(value).sharding.spec
    head_axis = _normalize_partition_axis(query_spec[1])
    q_seq_axis = _normalize_partition_axis(query_spec[2])

    kernel = splash._make_splash_attention(
        mask=multi_head_mask,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        head_shards=_mesh_axis_size(mesh, head_axis),
        q_seq_shards=_mesh_axis_size(mesh, q_seq_axis),
    )
    kernel_spec = P(head_axis, q_seq_axis)
    kernel_spec = kernel.manual_sharding_spec(jax.NamedSharding(mesh, kernel_spec))
    segment_ids_spec = P(query_spec[0], q_seq_axis)

    if segment_ids is None:

        @partial(
            jax.shard_map,
            in_specs=(kernel_spec, query_spec, key_spec, value_spec),
            out_specs=query_spec,
            mesh=mesh,
            check_vma=False,
        )
        def flash_attention_fn(kernel, query, key, value):
            return jax.vmap(kernel)(query, key, value)

        attn_out = flash_attention_fn(kernel, query, key, value)
    else:

        @partial(
            jax.shard_map,
            in_specs=(kernel_spec, query_spec, key_spec, value_spec, segment_ids_spec),
            out_specs=query_spec,
            mesh=mesh,
            check_vma=False,
        )
        def flash_attention_fn(kernel, query, key, value, segment_ids):
            return jax.vmap(kernel)(
                query,
                key,
                value,
                segment_ids=splash.SegmentIds(segment_ids, segment_ids),
            )

        attn_out = flash_attention_fn(kernel, query, key, value, segment_ids)

    return rearrange(attn_out, "b n t h -> b t n h")

class AttentionInterface(GeneralInterface[str, AttentionImpl]):
    _global_mapping = {
        "eager": eager_dot_product_attention,
        "sdpa": partial(
            tokamax.dot_product_attention, precision=jax.lax.Precision.HIGHEST
        ),
        "xla_chunked": partial(
            tokamax.dot_product_attention,
            implementation=TokamaxRematXlaChunkedDotProductAttention(
                chunk_size=(1024, 2048)
            ),
        ),
        "chunked_manual": chunked_manual_dot_product_attention,
        "tpu_flash": tpu_flash_atttention,
    }


ATTENTION_INTERFACE = AttentionInterface()
