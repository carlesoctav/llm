from functools import partial
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import tokamax
from einops import rearrange
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jaxformers.masking_utils import (
    make_bidirectional_mask,
    make_causal_mask,
    make_sliding_window_causal_mask,
    slliding_window_full_mask,
)
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
        q_sharding: jax.sharding.NamedSharding | None = None,
        **kwargs,
    ): ...


class PrepareAttentionArgs(Protocol):
    def __call__(
        input_embeds: Float[Array, "B T D"],
        attention_mask: Bool[Array, "B T"] | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        *,
        is_causal: bool = True,
        is_sliding: bool = False,
        is_mqa: bool = False,
        window_size: int | None = None,
        **kwargs,
    ) -> dict[str, Any]: ...


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
        if mask_array.shape == (num_heads, tgt_len, src_len):
            mask_array = mask_array[None, :, :, :]
            return jnp.broadcast_to(mask_array, (batch_size, num_heads, tgt_len, src_len))
        raise ValueError(
            f"Mask shape {mask_array.shape} must match "
            f"({batch_size}, {tgt_len}, {src_len}) or ({num_heads}, {tgt_len}, {src_len})"
        )
    if mask_array.ndim == 4:
        if mask_array.shape == (batch_size, 1, tgt_len, src_len):
            return jnp.broadcast_to(mask_array, (batch_size, num_heads, tgt_len, src_len))
        if mask_array.shape == (batch_size, num_heads, tgt_len, src_len):
            return mask_array
        if mask_array.shape == (batch_size, tgt_len, num_heads, src_len):
            return jnp.transpose(mask_array, (0, 2, 1, 3))
        raise ValueError(
            f"Mask shape {mask_array.shape} must match "
            f"({batch_size}, 1, {tgt_len}, {src_len}), "
            f"({batch_size}, {num_heads}, {tgt_len}, {src_len}), or "
            f"({batch_size}, {tgt_len}, {num_heads}, {src_len})"
        )
    raise ValueError(f"Mask rank must be 2, 3, or 4 but got shape {mask_array.shape}")


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
    q_sharding: jax.sharding.NamedSharding | None = None,
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
        mask_array = _normalize_mask(mask, B, N, T, S)
        if mask_array.dtype == jnp.bool_:
            mask_value = float(jnp.finfo(scores.dtype).min)
            scores = jnp.where(mask_array, scores, mask_value)
        else:
            scores = scores + mask_array

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


def _infer_named_sharding(x):
    sharding = getattr(x, "sharding", None)
    if sharding is None:
        aval = getattr(x, "aval", None)
        sharding = getattr(aval, "sharding", None) if aval is not None else None
    return sharding if isinstance(sharding, jax.sharding.NamedSharding) else None


def encode_tpu_flash_attention_code(
    *, is_sliding: bool, is_causal: bool, is_mqa: bool
):
    return (
        (jnp.asarray(is_sliding, dtype=jnp.int32) << 0)
        | (jnp.asarray(is_causal, dtype=jnp.int32) << 1)
        | (jnp.asarray(is_mqa, dtype=jnp.int32) << 2)
    )


def _make_attention_mask(
    input_embeds: Float[Array, "B T D"],
    attention_mask: Bool[Array, "B T"] | None,
    segment_ids: Int[Array, "B T"] | None,
    *,
    is_causal: bool,
    is_sliding: bool,
    window_size: int | None,
):
    mask_code = (
        (jnp.asarray(is_sliding, dtype=jnp.int32) << 0)
        | (jnp.asarray(is_causal, dtype=jnp.int32) << 1)
    )

    def make_full_mask(_):
        return make_bidirectional_mask(
            "eager",
            input_embeds,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )

    def make_sliding_full_mask(_):
        if window_size is None:
            return make_full_mask(None)
        return slliding_window_full_mask(
            "eager",
            input_embeds,
            window_size,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )

    def make_causal_attention_mask(_):
        return make_causal_mask(
            "eager",
            input_embeds,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )

    def make_sliding_causal_attention_mask(_):
        if window_size is None:
            return make_causal_attention_mask(None)
        return make_sliding_window_causal_mask(
            "eager",
            input_embeds,
            window_size,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )

    return jax.lax.switch(
        mask_code,
        (
            make_full_mask,
            make_sliding_full_mask,
            make_causal_attention_mask,
            make_sliding_causal_attention_mask,
        ),
        operand=None,
    )


def _prepare_dense_attention_kwargs(
    input_embeds: Float[Array, "B T D"],
    attention_mask: Bool[Array, "B T"] | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
    *,
    is_causal: bool = True,
    is_sliding: bool = False,
    window_size: int | None = None,
    **kwargs,
) -> dict[str, Any]:
    return {
        "mask": _make_attention_mask(
            input_embeds,
            attention_mask,
            segment_ids,
            is_causal=is_causal,
            is_sliding=is_sliding,
            window_size=window_size,
        )
    }


def _prepare_tpu_flash_attention_kwargs(
    input_embeds: Float[Array, "B T D"],
    attention_mask: Bool[Array, "B T"] | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
    *,
    is_causal: bool = True,
    is_sliding: bool = False,
    is_mqa: bool = False,
    window_size: int | None = None,
    **kwargs,
) -> dict[str, Any]:
    del input_embeds

    if segment_ids is None:
        prepared_segment_ids = None
        if attention_mask is not None:
            prepared_segment_ids = jnp.asarray(attention_mask, dtype=jnp.int32)
    else:
        prepared_segment_ids = jnp.asarray(segment_ids, dtype=jnp.int32)
        if attention_mask is not None:
            prepared_segment_ids = jnp.where(
                attention_mask,
                prepared_segment_ids,
                jnp.zeros_like(prepared_segment_ids),
            )

    return {
        "mask": jnp.ones((1,), dtype=jnp.bool_),
        "segment_ids": prepared_segment_ids,
        "attention_code": encode_tpu_flash_attention_code(
            is_sliding=is_sliding,
            is_causal=is_causal,
            is_mqa=is_mqa,
        ),
        "window_size": window_size,
    }


def prepare_attention_kwargs(
    attn_impl: str,
    input_embeds: Float[Array, "B T D"],
    attention_mask: Bool[Array, "B T"] | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
    *,
    is_causal: bool = True,
    is_sliding: bool = False,
    is_mqa: bool = False,
    window_size: int | None = None,
    **kwargs,
) -> dict[str, Any]:
    return PREPARE_ATTENTION_ARGS_INTERFACE[attn_impl](
        input_embeds,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        is_causal=is_causal,
        is_sliding=is_sliding,
        is_mqa=is_mqa,
        window_size=window_size,
        **kwargs,
    )


def tpu_flash_attention_impl(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, " B #N T S"] | None = None,
    *,
    scale: float | None = None,
    logits_soft_cap: float | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
    q_sharding: jax.sharding.NamedSharding | None = None,
    is_causal: bool = True,
    is_sliding: bool = False,
    is_mqa: bool = False,
    window_size: int | None = None,
    flash_attention_block_size: int = 512,
):
    del q_sharding

    if bias is not None:
        raise ValueError("tpu_flash does not support bias")
    if logits_soft_cap is not None:
        raise ValueError("tpu_flash does not support logits_soft_cap")
    if mask is not None and jnp.asarray(mask).shape != (1,):
        raise ValueError(
            "tpu_flash does not support dense attention_mask; use structural "
            "sliding/causal selection and segment_ids instead"
        )

    _B, T, N, H = query.shape
    _, S, K, _ = key.shape
    if T != S:
        raise ValueError(f"tpu_flash requires T == S, got T={T} and S={S}")

    if scale is None:
        scale = 1 / (H**0.5)
    query = query * jnp.asarray(scale, dtype=query.dtype)

    query = rearrange(query, "b t n h -> b n t h")
    key = rearrange(key, "b s k h -> b k s h")
    value = rearrange(value, "b s k h -> b k s h")

    if is_mqa:
        if K != 1:
            raise ValueError(f"tpu_flash expected K == 1 for MQA, got K={K}")
        key = jnp.squeeze(key, axis=1)
        value = jnp.squeeze(value, axis=1)

    from jax.experimental import shard_map
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,
        splash_attention_mask as mask_lib,
    )

    if is_causal:
        splash_mask = mask_lib.CausalMask((T, S))
    else:
        splash_mask = mask_lib.FullMask((T, S))

    if is_sliding:
        if window_size is None:
            raise ValueError("tpu_flash sliding attention requires window_size")
        splash_mask &= mask_lib.LocalMask((T, S), (window_size, None), offset=0)

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

    query_sharding = _infer_named_sharding(query)
    key_sharding = _infer_named_sharding(key)
    value_sharding = _infer_named_sharding(value)
    if query_sharding is None or key_sharding is None or value_sharding is None:
        raise ValueError("tpu_flash requires NamedSharding on query, key, and value")

    mesh = query_sharding.mesh
    query_spec = query_sharding.spec
    key_spec = key_sharding.spec
    value_spec = value_sharding.spec
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
    kernel_spec = kernel.manual_sharding_spec(
        jax.sharding.NamedSharding(mesh, kernel_spec)
    )
    segment_ids_spec = P(query_spec[0], q_seq_axis)

    if segment_ids is None:

        @partial(
            shard_map.shard_map,
            in_specs=(kernel_spec, query_spec, key_spec, value_spec),
            out_specs=query_spec,
            mesh=mesh,
            check_rep=False,
        )
        def flash_attention_fn(kernel, query, key, value):
            return jax.vmap(kernel)(query, key, value)

        attn_out = flash_attention_fn(kernel, query, key, value)
    else:

        @partial(
            shard_map.shard_map,
            in_specs=(kernel_spec, query_spec, key_spec, value_spec, segment_ids_spec),
            out_specs=query_spec,
            mesh=mesh,
            check_rep=False,
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


def tpu_flash_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, " B #N T S"] | None = None,
    *,
    scale: float | None = None,
    logits_soft_cap: float | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
    q_sharding: jax.sharding.NamedSharding | None = None,
    is_causal: bool = True,
    is_sliding: bool = False,
    is_mqa: bool | None = None,
    attention_code: int | Array | None = None,
    window_size: int | None = None,
    flash_attention_block_size: int = 512,
    **kwargs,
):
    if attention_code is None:
        if is_mqa is None:
            is_mqa = key.shape[2] == 1
        attention_code = encode_tpu_flash_attention_code(
            is_sliding=is_sliding,
            is_causal=is_causal,
            is_mqa=is_mqa,
        )
    # jax.debug.print("{attention_code}", attention_code = attention_code)

    operand = (
        query,
        key,
        value,
        bias,
        mask,
        scale,
        logits_soft_cap,
        segment_ids,
    )

    def make_case(case_code: int):
        def case(operand):
            (
                query,
                key,
                value,
                bias,
                mask,
                scale,
                logits_soft_cap,
                segment_ids,
            ) = operand
            return tpu_flash_attention_impl(
                query,
                key,
                value,
                bias=bias,
                mask=mask,
                scale=scale,
                logits_soft_cap=logits_soft_cap,
                segment_ids=segment_ids,
                q_sharding=q_sharding,
                is_sliding=bool(case_code & 0b001),
                is_causal=bool(case_code & 0b010),
                is_mqa=bool(case_code & 0b100),
                window_size=window_size,
                flash_attention_block_size=flash_attention_block_size,
            )

        return case

    return jax.lax.switch(attention_code, tuple(make_case(i) for i in range(8)), operand)


class AttentionInterface(GeneralInterface[str, AttentionImpl]):
    _global_mapping = {
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
        "tpu_flash": tpu_flash_attention,
    }


class PrepareAttentionArgsInterface(GeneralInterface[str, PrepareAttentionArgs]):
    _global_mapping = {
        "eager": _prepare_dense_attention_kwargs,
        "sdpa": _prepare_dense_attention_kwargs,
        "xla_chunked": _prepare_dense_attention_kwargs,
        "tpu_flash": _prepare_tpu_flash_attention_kwargs,
    }


ATTENTION_INTERFACE = AttentionInterface()
PREPARE_ATTENTION_ARGS_INTERFACE = PrepareAttentionArgsInterface()
