"""Tokamax-derived XLA chunked attention with q-loop rematerialization.

This is a local fork of Tokamax's `xla_chunked` attention implementation with a
single change: the q-chunk scan body is wrapped in `jax.remat` to reduce reverse
mode memory. The q-chunk size is kept static by closing over it.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import tokamax
from jaxtyping import Array, Bool, Float, Int
from tokamax._src import jaxtyping
from tokamax._src import quantization
from tokamax._src import shape as shape_lib
from tokamax._src.ops import op
from tokamax._src.ops.attention import base
from typing_extensions import override

Mask = base.Mask
QArray = base.QArray
PagingInfo = base.PagingInfo


@jaxtyping.jaxtyped
def _attend_chunk(
    q: Float[Array, "*B T H D"],
    k: Float[Array, "*B t #H D"],
    v: Float[Array, "*B t #H d"],
    accum: Float[Array, "*B T H d"],
    x_max: Float[Array, "*B H T"],
    denom: Float[Array, "*B H T"],
    *,
    precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
    logits_dtype: jnp.dtype,
    logits_scale: float,
    bias: Float[Array, "*#B #H #T #t"] | None,
    logits_soft_cap: float | None,
    mask: Bool[Array, "*#B #H #T #t"] | None,
    dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
    dropout_rate: float,
) -> tuple[Float[Array, "*B T H d"], Float[Array, "*B H T"], Float[Array, "*B H T"]]:
    """Compute a single KV chunk for a fixed Q chunk."""
    q_k_dot_precision, weights_v_dot_precision = precision
    bias = jnp.array([]) if bias is None else bias

    @functools.partial(jax.remat, prevent_cse=False)
    def chunk(q, k, v, bias, mask, accum, x_max, denom):
        bias = bias if len(bias) else None

        # TODO: Can we be more efficient for multi-query attention?
        logits = jnp.einsum(
            "...qhd,...khd->...hqk",
            q,
            k,
            precision=q_k_dot_precision,
            preferred_element_type=q_k_dot_precision.accumulation_type,
        ).astype(logits_dtype)

        logits *= logits_scale

        if bias is not None:
            logits = (logits + bias).astype(logits.dtype)

        if logits_soft_cap is not None:
            logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)

        if mask is not None:
            mask_value = float(jnp.finfo(logits.dtype).min)
            logits = jnp.where(jnp.asarray(mask), logits, mask_value)

        logits = logits.astype(jnp.promote_types(logits.dtype, jnp.float32))
        loc_x_max = jnp.max(logits, axis=-1)
        new_x_max = jnp.maximum(x_max, loc_x_max)
        weights = jnp.exp(logits - new_x_max[..., None])
        alpha = jnp.exp(x_max - new_x_max)

        x_max = new_x_max
        accum *= alpha.mT[..., None]
        denom = (denom * alpha) + weights.sum(axis=-1)

        if dropout_mask is not None:
            weights *= dropout_mask.astype(weights.dtype) / (1 - dropout_rate)

        accum += jnp.einsum(
            "...hqk,...khd->...qhd",
            weights,
            v,
            precision=weights_v_dot_precision,
            preferred_element_type=weights_v_dot_precision.accumulation_type,
        ).astype(accum.dtype)

        return accum, x_max, denom

    return chunk(
        q=q,
        k=k,
        v=v,
        bias=bias,
        mask=mask,
        accum=accum,
        x_max=x_max,
        denom=denom,
    )


@jaxtyping.jaxtyped
def _attend_chunked(
    q: Float[Array, "*B T H D"],
    k: Float[Array, "*B t #H D"],
    v: Float[Array, "*B t #H d"],
    *,
    precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
    logits_dtype: jnp.dtype,
    logits_scale: float,
    bias: Float[Array, "*#B #H #T #t"] | None,
    logits_soft_cap: float | None,
    mask: Mask,
    dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
    dropout_rate: float,
    paging_info: PagingInfo | None,
    q_indices: Int[Array, "*#B #H T"] | None,
    k_indices: Int[Array, "*#B #h t"] | None,
    normalize_output: bool,
    chunk_size: tuple[int, int],
) -> tuple[Float[Array, "*B T H d"], None]:
    """Compute chunked attention (linear memory in sequence length)."""
    if paging_info is not None:
        raise NotImplementedError("Paged attention not supported.")

    *b, seq_q, h, _ = q.shape
    *_, seq_k, _, d_out = v.shape

    q_chunk_size, kv_chunk_size = chunk_size

    q_len_or_indices = seq_q if q_indices is None else q_indices
    k_len_or_indices = seq_k if k_indices is None else k_indices
    mask = mask.as_array(q_len_or_indices, k_len_or_indices)

    def get_chunk(x, idx, size, axis):
        if x is None:
            return None
        if x.shape[axis] == 1:
            return x
        return jax.lax.dynamic_slice_in_dim(x, idx, size, axis)

    # Remat the q-body while keeping q_chunk_size static by closing over it.
    def make_q_loop_fn(q_chunk_size_local: int):
        @functools.partial(jax.remat, prevent_cse=False)
        def q_loop_fn(q_chunk_idx, _):
            def get_q_chunk(x, axis):
                return get_chunk(x, q_chunk_idx, q_chunk_size_local, axis)

            q_chunk = get_q_chunk(q, -3)
            bias_chunk = get_q_chunk(bias, -2)
            mask_chunk = get_q_chunk(mask, -2)
            dropout_mask_chunk = get_q_chunk(dropout_mask, -2)

            intermediates_shape = (*b, h, q_chunk.shape[-3])
            acc = jnp.zeros(q_chunk.shape[:-1] + (d_out,), jnp.float32)
            x_max = jnp.full(intermediates_shape, float("-inf"))
            denom = jnp.zeros(intermediates_shape, jnp.float32)

            def kv_loop_fn(_, carry, *, kv_chunk_size_local):
                kv_chunk_idx, accum, x_max, denom = carry

                def get_kv_chunk(x, axis):
                    return get_chunk(x, kv_chunk_idx, kv_chunk_size_local, axis)

                out = _attend_chunk(
                    q_chunk,
                    get_kv_chunk(k, -3),
                    get_kv_chunk(v, -3),
                    accum,
                    x_max,
                    denom,
                    bias=get_kv_chunk(bias_chunk, -1),
                    mask=get_kv_chunk(mask_chunk, -1),
                    dropout_mask=get_kv_chunk(dropout_mask_chunk, -1),
                    precision=precision,
                    logits_dtype=logits_dtype,
                    logits_scale=logits_scale,
                    logits_soft_cap=logits_soft_cap,
                    dropout_rate=dropout_rate,
                )
                return kv_chunk_idx + kv_chunk_size_local, *out

            even_chunks = seq_k // kv_chunk_size
            carry = (0, acc, x_max, denom)

            # Main kv loop
            if seq_k >= kv_chunk_size:
                loop_fn = functools.partial(kv_loop_fn, kv_chunk_size_local=kv_chunk_size)
                carry = jax.lax.fori_loop(0, even_chunks, loop_fn, carry)

            # Remainder kv loop
            if (k_remainder := (seq_k % kv_chunk_size)) != 0:
                carry = kv_loop_fn(even_chunks + 1, carry, kv_chunk_size_local=k_remainder)

            # Final normalization by the denominator.
            _, acc, _, denom = carry
            out = acc / denom.mT[..., None] if normalize_output else acc
            return q_chunk_idx + q_chunk_size_local, out.astype(q.dtype)

        return q_loop_fn

    q_chunk_idx, out = 0, None

    # Main q loop
    if seq_q >= q_chunk_size:
        loop_fn = make_q_loop_fn(q_chunk_size)
        length = seq_q // q_chunk_size
        q_chunk_idx, out = jax.lax.scan(loop_fn, init=0, length=length)
        out = shape_lib.einshape("q...thd->...(qt)hd")(out)

    # Remainder q loop
    if (q_remainder := (seq_q % q_chunk_size)) != 0:
        _, rem_out = make_q_loop_fn(q_remainder)(q_chunk_idx, None)
        out = rem_out if out is None else jnp.concatenate([out, rem_out], axis=-3)

    return out, None


@jaxtyping.jaxtyped
def _attend_paged(
    q: Float[Array, "*B T H D"],
    k: Float[Array, "*b t #H D"],
    v: Float[Array, "*b t #H d"],
    *,
    precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
    logits_dtype: jnp.dtype,
    logits_scale: float,
    bias: Float[Array, "*#B #H #T #t"] | None,
    logits_soft_cap: float | None,
    mask: Mask,
    dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
    dropout_rate: float,
    paging_info: PagingInfo | None,
    q_indices: Int[Array, "*#B #H T"] | None,
    k_indices: Int[Array, "*#B #h t"] | None,
    normalize_output: bool,
    chunk_size: int,
) -> tuple[Float[Array, "*B T H d"], None]:
    """Paged attention (unchanged from Tokamax)."""
    del q_indices  # Unused.
    assert paging_info is not None

    if mask or any(x is not None for x in (bias, dropout_mask, k_indices)):
        raise NotImplementedError("Paging only supports qkv as inputs.")

    _, seq_q, h, _ = q.shape

    def bq_loop_fn(bidx, q_batch):
        num_pages = paging_info.num_active_pages[bidx]
        page_indices = paging_info.active_page_indices[bidx]

        def sq_loop_fn(_, q_chunk):
            intermediates_shape = (h, q_chunk.shape[0])
            acc = jnp.zeros(q_chunk.shape[:-1] + (v.shape[-1],), jnp.float32)
            x_max = jnp.full(intermediates_shape, float("-inf"))
            denom = jnp.zeros(intermediates_shape, jnp.float32)

            def kv_loop_fn(i, carry):
                accum, x_max, denom = carry
                return _attend_chunk(
                    q_chunk,
                    k[page_indices[i]],
                    v[page_indices[i]],
                    accum,
                    x_max,
                    denom,
                    bias=None,
                    mask=None,
                    dropout_mask=None,
                    precision=precision,
                    logits_dtype=logits_dtype,
                    logits_scale=logits_scale,
                    logits_soft_cap=logits_soft_cap,
                    dropout_rate=dropout_rate,
                )

            carry = jax.lax.fori_loop(0, num_pages, kv_loop_fn, (acc, x_max, denom))
            acc, _, denom = carry
            out = acc / denom.mT[..., None] if normalize_output else acc
            return -1, out.astype(q.dtype)

        if seq_q % chunk_size != 0:
            raise NotImplementedError(f"{seq_q=} must be a multiple of {chunk_size}.")

        q_batch = q_batch.reshape(seq_q // chunk_size, chunk_size, h, -1)
        _, out = jax.lax.scan(sq_loop_fn, init=0, xs=q_batch)
        return bidx + 1, out.reshape(seq_q, *out.shape[2:])

    return jax.lax.scan(bq_loop_fn, init=0, xs=q)[1], None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TokamaxRematXlaChunkedDotProductAttention(base.DotProductAttention[op.NullConfig, None]):
    """Tokamax XLA chunked dot product attention with remat on q-loop."""

    chunk_size: int | tuple[int, int] = 128

    @jaxtyping.jaxtyped
    @override
    def _fwd(
        self,
        q: Float[Array | QArray, "*B T H D"],
        k: Float[Array | QArray, "*b t h D"],
        v: Float[Array | QArray, "*b t h d"],
        *,
        precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset],
        logits_dtype: jnp.dtype,
        logits_scale: float,
        bias: Float[Array, "*#B #H #T #t"] | None,
        logits_soft_cap: float | None,
        mask: Mask,
        dropout_mask: Bool[Array, "*#B #H #T #t"] | None,
        dropout_rate: float,
        paging_info: PagingInfo | None,
        q_indices: Int[Array, "*#B #H T"] | None,
        k_indices: Int[Array, "*#B #h t"] | None,
        normalize_output: bool,
        return_residuals: bool,
        config: op.NullConfig,
    ) -> tuple[Float[Array, "*B T H d"], None]:
        del config
        if return_residuals:
            raise NotImplementedError("`return_residuals=True` not supported.")

        q, k, v = map(quantization.as_array, (q, k, v))
        if k.shape[-2] not in (1, q.shape[-2]):
            repeats = q.shape[-2] // k.shape[-2]
            with shape_lib.upcast_broadcast():
                k = jnp.repeat(k, repeats, axis=-2)
                v = jnp.repeat(v, repeats, axis=-2)

        is_paged = paging_info is not None
        single_chunk = isinstance(self.chunk_size, int)
        if is_paged and not single_chunk:
            raise ValueError("Paged attention does not support multiple chunk sizes.")

        if not is_paged and single_chunk:
            chunk_size = (self.chunk_size,) * 2
        else:
            chunk_size = self.chunk_size

        attn_fn = _attend_paged if is_paged else _attend_chunked
        return attn_fn(
            q,
            k,
            v,
            precision=precision,
            logits_dtype=logits_dtype,
            logits_scale=logits_scale,
            bias=bias,
            logits_soft_cap=logits_soft_cap,
            mask=mask,
            dropout_mask=dropout_mask,
            dropout_rate=dropout_rate,
            paging_info=paging_info,
            q_indices=q_indices,
            k_indices=k_indices,
            normalize_output=normalize_output,
            chunk_size=chunk_size,
        )


def tokamax_remat_chunked_xla_dot_product_attention(
    query: Float[Array, "*B T N H"],
    key: Float[Array, "*B S K H"],
    value: Float[Array, "*B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, "*#B #N #T #S"] | None = None,
    q_sharding: jax.sharding.NamedSharding | None = None,
    **kwargs,
) -> Float[Array, "*B T N H"]:
    """Public wrapper matching the rest of jaxformers attention ops.

    This calls `tokamax.dot_product_attention` with a custom implementation that
    mirrors Tokamax's `xla_chunked`, but with q-loop rematerialization.
    """
    precision = kwargs.pop("precision", jax.lax.Precision.HIGHEST)
    is_causal = bool(kwargs.pop("is_causal", False) or kwargs.pop("causal", False))

    chunk_size = kwargs.pop("chunk_size", None)
    query_chunk_size = kwargs.pop("query_chunk_size", None)
    key_chunk_size = kwargs.pop("key_chunk_size", None)
    if chunk_size is None:
        if query_chunk_size is not None or key_chunk_size is not None:
            if query_chunk_size is None or key_chunk_size is None:
                raise ValueError("Provide both query_chunk_size and key_chunk_size, or chunk_size.")
            chunk_size = (int(query_chunk_size), int(key_chunk_size))
        else:
            chunk_size = 128

    impl = TokamaxRematXlaChunkedDotProductAttention(chunk_size=chunk_size)

    # Forward tokamax-specific knobs (match tokamax.dot_product_attention).
    scale = kwargs.pop("scale", None)
    query_seq_lengths = kwargs.pop("query_seq_lengths", None)
    key_value_seq_lengths = kwargs.pop("key_value_seq_lengths", None)
    local_window_size = kwargs.pop("local_window_size", None)
    logits_soft_cap = kwargs.pop("logits_soft_cap", None)

    if kwargs:
        raise TypeError(f"Unexpected kwargs: {sorted(kwargs.keys())}")

    return tokamax.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        scale=scale,
        is_causal=is_causal,
        query_seq_lengths=query_seq_lengths,
        key_value_seq_lengths=key_value_seq_lengths,
        local_window_size=local_window_size,
        logits_soft_cap=logits_soft_cap,
        precision=precision,
        implementation=impl,
        q_sharding=q_sharding,
    )

