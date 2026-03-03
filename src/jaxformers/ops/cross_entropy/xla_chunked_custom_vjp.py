import jax
import jax.numpy as jnp
from functools import partial
from jaxtyping import Array, Float, Int

from .config import BlockSizes


def _apply_logit_soft_cap(
    logits: Float[Array, "B V"], logit_soft_cap: float | None = None
) -> Float[Array, "B V"]:
    if logit_soft_cap is None:
        return logits
    return jnp.tanh(logits / logit_soft_cap) * logit_soft_cap


def _materialize_cotangent(
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero, reference: jax.Array
) -> jax.Array:
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros_like(reference)
    return jnp.asarray(cotangent, dtype=reference.dtype)


def _apply_logit_soft_cap_with_deriv(
    logits: jax.Array, logit_soft_cap: float | None
) -> tuple[jax.Array, jax.Array]:
    if logit_soft_cap is None:
        return logits, jnp.asarray(1.0, dtype=logits.dtype)
    tanh_arg = logits / logit_soft_cap
    tanh_val = jnp.tanh(tanh_arg)
    return tanh_val * logit_soft_cap, (1.0 - tanh_val**2).astype(logits.dtype)


def _fused_cross_entropy_chunked_xla_fwd_impl(
    x: Float[Array, "B H"],
    labels: Int[Array, " B"],
    w: Float[Array, "V H"],
    *,
    block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    logit_soft_cap: float | None = None,
    precision: jax.lax.PrecisionLike = None,
) -> tuple[jax.Array, jax.Array]:
    """Forward: returns per-token loss and logsumexp."""
    B, H = x.shape
    V, _ = w.shape
    b_block = block_sizes.b
    v_block = block_sizes.v
    h_block = block_sizes.h

    if B % b_block != 0:
        raise ValueError(f"B={B} must be divisible by b_block={b_block}")
    if H % h_block != 0:
        raise ValueError(f"H={H} must be divisible by h_block={h_block}")

    Vpad = (-V) % v_block
    w_pad = jnp.pad(w, ((0, Vpad), (0, 0)))

    num_b = B // b_block
    num_h = H // h_block
    num_v = (V + Vpad) // v_block

    def b_body(bi, val):
        lse, loss = val
        b0 = b_block * bi
        yb = jax.lax.dynamic_slice(labels, (b0,), (b_block,))  # [B_chunk]

        lse_b = jnp.full((b_block,), -jnp.inf, dtype=dtype)
        label_logits_b = jnp.full((b_block,), -jnp.inf, dtype=dtype)

        @jax.checkpoint
        def v_body(vi, val):
            lse_b, label_logits_b = val
            v0 = vi * v_block

            def h_body(hi, acc):
                h0 = h_block * hi
                x_bh = jax.lax.dynamic_slice(x, (b0, h0), (b_block, h_block))
                w_hv = jax.lax.dynamic_slice(w_pad, (v0, h0), (v_block, h_block))
                return acc + jax.lax.dot_general(
                    x_bh,
                    w_hv,
                    (((1,), (1,)), ((), ())),
                    precision,
                    preferred_element_type=dtype,
                )

            logits = jax.lax.fori_loop(
                0, num_h, h_body, jnp.zeros((b_block, v_block), dtype=dtype)
            )  # [B_block, V_block]
            logits = _apply_logit_soft_cap(logits, logit_soft_cap)
            valid = v0 + jnp.arange(v_block) < V
            logits = jnp.where(valid, logits, -jnp.inf)

            in_block = (yb >= v0) & (yb < v0 + v_block)
            idx = jnp.where(in_block, yb - v0, 0)
            block_label_logits = logits[jnp.arange(b_block), idx]
            label_logits_b = jnp.where(in_block, block_label_logits, label_logits_b)

            block_lse = jax.nn.logsumexp(logits, axis=-1)  # (B_block, )
            lse_b = jnp.logaddexp(lse_b, block_lse)  # (B_block,)

            return lse_b, label_logits_b

        lse_b, label_logits_b = jax.lax.fori_loop(0, num_v, v_body, (lse_b, label_logits_b))
        loss_b = lse_b - label_logits_b
        lse = jax.lax.dynamic_update_slice(lse, lse_b, (b0,))
        loss = jax.lax.dynamic_update_slice(loss, loss_b, (b0,))
        return lse, loss

    lse0, loss0 = jnp.zeros((B,), dtype), jnp.zeros((B,), dtype)
    (lse, loss) = jax.lax.fori_loop(0, num_b, b_body, (lse0, loss0))
    return loss, lse


def _fused_cross_entropy_chunked_xla_bwd_impl(
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    lse: jax.Array,
    dout_loss: jax.Array,
    dout_lse: jax.Array,
    *,
    block_sizes: BlockSizes,
    dtype: jnp.dtype,
    logit_soft_cap: float | None,
    precision: jax.lax.PrecisionLike,
) -> tuple[jax.Array, jax.Array]:
    """Backward: returns grads for (x, w)."""
    B, H = x.shape
    V, _ = w.shape
    b_block = block_sizes.b
    v_block = block_sizes.v
    h_block = block_sizes.h

    if b_block is None or v_block is None or h_block is None:
        raise ValueError(f"xla_chunked requires non-None block sizes, got {block_sizes}")
    if B % b_block != 0:
        raise ValueError(f"B={B} must be divisible by b_block={b_block}")
    if H % h_block != 0:
        raise ValueError(f"H={H} must be divisible by h_block={h_block}")

    Vpad = (-V) % v_block
    w_pad = jnp.pad(w, ((0, Vpad), (0, 0)))

    num_b = B // b_block
    num_h = H // h_block
    num_v = (V + Vpad) // v_block

    dx = jnp.zeros((B, H), dtype=jnp.float32)
    dw = jnp.zeros((V + Vpad, H), dtype=jnp.float32)

    row_indices = jnp.arange(b_block, dtype=labels.dtype)
    v_ids = jnp.arange(v_block, dtype=labels.dtype)

    def v_body(vi, state):
        dx, dw = state
        v0 = vi * v_block
        dw_block = jnp.zeros((v_block, H), dtype=jnp.float32)

        def b_body(bi, b_state):
            dx, dw_block = b_state
            b0 = bi * b_block

            yb = jax.lax.dynamic_slice(labels, (b0,), (b_block,))
            lse_b = jax.lax.dynamic_slice(lse, (b0,), (b_block,))
            dout_loss_b = jax.lax.dynamic_slice(dout_loss, (b0,), (b_block,))
            dout_lse_b = jax.lax.dynamic_slice(dout_lse, (b0,), (b_block,))

            # logits tile: [b_block, v_block]
            def h_body(hi, acc):
                h0 = h_block * hi
                x_bh = jax.lax.dynamic_slice(x, (b0, h0), (b_block, h_block))
                w_vh = jax.lax.dynamic_slice(w_pad, (v0, h0), (v_block, h_block))
                return acc + jax.lax.dot_general(
                    x_bh,
                    w_vh,
                    (((1,), (1,)), ((), ())),
                    precision=precision,
                    preferred_element_type=dtype,
                )

            logits = jax.lax.fori_loop(
                0, num_h, h_body, jnp.zeros((b_block, v_block), dtype=dtype)
            )

            logits, cap_deriv = _apply_logit_soft_cap_with_deriv(logits, logit_soft_cap)
            valid = (v0 + v_ids) < V
            logits = jnp.where(valid, logits, -jnp.inf)

            # Match forward's blockwise logsumexp structure:
            block_lse = jax.nn.logsumexp(logits, axis=-1)
            block_weight = jnp.exp(block_lse - lse_b.astype(block_lse.dtype))
            probs = jnp.exp(logits - block_lse[:, None]) * block_weight[:, None]

            delta = (dout_loss_b[:, None].astype(logits.dtype) + dout_lse_b[:, None].astype(logits.dtype)) * probs

            in_block = (yb >= v0) & (yb < v0 + v_block)
            label_idx = yb - v0
            safe_idx = jnp.where(in_block, label_idx, 0)
            delta = delta.at[row_indices, safe_idx].add(
                jnp.where(in_block, -dout_loss_b.astype(logits.dtype), 0.0)
            )
            delta = (delta * cap_deriv).astype(logits.dtype)

            # dx update for this (b,v) tile
            dx_slice = jax.lax.dynamic_slice(dx, (b0, 0), (b_block, H))

            def h_dx_body(hi, acc):
                h0 = h_block * hi
                w_vh = jax.lax.dynamic_slice(w_pad, (v0, h0), (v_block, h_block))
                dx_h = jax.lax.dot_general(
                    delta,
                    w_vh,
                    (((1,), (0,)), ((), ())),
                    precision=precision,
                    preferred_element_type=jnp.float32,
                )
                acc_h = jax.lax.dynamic_slice(acc, (0, h0), (b_block, h_block))
                return jax.lax.dynamic_update_slice(
                    acc, acc_h + dx_h.astype(acc.dtype), (0, h0)
                )

            dx_contrib = jax.lax.fori_loop(
                0, num_h, h_dx_body, jnp.zeros((b_block, H), dtype=dx_slice.dtype)
            )

            dx = jax.lax.dynamic_update_slice(
                dx, dx_slice + dx_contrib.astype(dx_slice.dtype), (b0, 0)
            )

            # dw accumulation for this v-block (sum over b blocks)
            x_b = jax.lax.dynamic_slice(x, (b0, 0), (b_block, H))
            dw_contrib = jax.lax.dot_general(
                delta,
                x_b,
                (((0,), (0,)), ((), ())),
                precision=precision,
                preferred_element_type=jnp.float32,
            )
            dw_block = dw_block + dw_contrib.astype(dw_block.dtype)
            return dx, dw_block

        dx, dw_block = jax.lax.fori_loop(0, num_b, b_body, (dx, dw_block))
        dw = jax.lax.dynamic_update_slice(dw, dw_block, (v0, 0))
        return dx, dw

    dx, dw = jax.lax.fori_loop(0, num_v, v_body, (dx, dw))
    return dx.astype(x.dtype), dw[:V, :].astype(w.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _fused_cross_entropy_chunked_xla_custom_vjp(
    block_sizes: BlockSizes,
    dtype: jnp.dtype,
    logit_soft_cap: float | None,
    precision: jax.lax.PrecisionLike,
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _fused_cross_entropy_chunked_xla_fwd_impl(
        x,
        labels,
        w,
        block_sizes=block_sizes,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )


def _fused_cross_entropy_chunked_xla_custom_vjp_fwd(
    block_sizes: BlockSizes,
    dtype: jnp.dtype,
    logit_soft_cap: float | None,
    precision: jax.lax.PrecisionLike,
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
):
    loss, lse = _fused_cross_entropy_chunked_xla_fwd_impl(
        x,
        labels,
        w,
        block_sizes=block_sizes,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )
    return (loss, lse), (x, labels, w, lse)


def _fused_cross_entropy_chunked_xla_custom_vjp_bwd(
    block_sizes: BlockSizes,
    dtype: jnp.dtype,
    logit_soft_cap: float | None,
    precision: jax.lax.PrecisionLike,
    residuals,
    cotangents,
):
    x, labels, w, lse = residuals
    dout_loss, dout_lse = cotangents

    dout_loss_arr = _materialize_cotangent(dout_loss, lse).astype(lse.dtype)
    dout_lse_arr = _materialize_cotangent(dout_lse, lse).astype(lse.dtype)

    dx, dw = _fused_cross_entropy_chunked_xla_bwd_impl(
        x,
        labels,
        w,
        lse,
        dout_loss_arr,
        dout_lse_arr,
        block_sizes=block_sizes,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )
    return dx, None, dw


_fused_cross_entropy_chunked_xla_custom_vjp.defvjp(
    _fused_cross_entropy_chunked_xla_custom_vjp_fwd,
    _fused_cross_entropy_chunked_xla_custom_vjp_bwd,
)


@partial(
    jax.jit, static_argnames=["block_sizes", "dtype", "logit_soft_cap", "precision"]
)
def fused_cross_entropy_chunked_xla_custom_vjp(
    x: Float[Array, "B H"],
    labels: Int[Array, " B"],
    w: Float[Array, "V H"],
    *,
    block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    logit_soft_cap: float | None = None,
    precision: jax.lax.PrecisionLike = None,
) -> tuple[jax.Array, jax.Array]:
    """Chunked CE with a custom VJP to reduce backward temp memory."""
    return _fused_cross_entropy_chunked_xla_custom_vjp(
        block_sizes, dtype, logit_soft_cap, precision, x, labels, w
    )

