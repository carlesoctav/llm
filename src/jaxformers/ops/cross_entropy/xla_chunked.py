import jax
import jax.numpy as jnp
from functools import partial
from jaxtyping import Array, Float, Int

from .config import BlockSizes

def _apply_logit_soft_cap(logits: Float[Array, "B V"], logit_soft_cap: float | None = None) -> Float[Array, "B V"]:
    if logit_soft_cap is None:
        return logits
    return jnp.tanh(logits / logit_soft_cap) * logit_soft_cap


@partial(jax.jit, static_argnames = ["block_sizes", "dtype", "precision"])
def fused_cross_entropy_chunked_xla(
    x: Float[Array, "B H"],
    labels: Int[Array, " B"],
    w: Float[Array, "H V"],
    *,
    block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    logit_soft_cap: float | None = None,
    precision: jax.lax.PrecisionLike = None,
):
    B, H = x.shape
    _, V = w.shape
    b_block = block_sizes.b
    v_block = block_sizes.v
    h_block = block_sizes.h

    if B % b_block != 0:
        raise ValueError(f"B={B} must be divisible by b_block={b_block}")
    if H % h_block != 0:
        raise ValueError(f"H={H} must be divisible by h_block={h_block}")

    Vpad = (-V) % v_block
    w_pad = jnp.pad(w, ((0, 0), (0, Vpad)))

    num_b = B // b_block
    num_h = H // h_block
    num_v = (V + Vpad) // v_block

    def b_body(bi, val):
        lse, loss = val
        b0 = b_block * bi
        yb = jax.lax.dynamic_slice(labels, (b0,), (b_block,))  # [B_chunk]

        lse_b = jnp.full((b_block,), -jnp.inf, dtype=dtype)
        label_logits_b = jnp.full((b_block,), -jnp.inf, dtype=dtype)

        @jax.remat
        def v_body(vi, val):
            lse_b, label_logits_b = val
            v0 = vi * v_block

            @jax.remat
            def h_body(hi, acc):
                h0 = h_block * hi
                x_bh = jax.lax.dynamic_slice(x, (b0, h0), (b_block, h_block))
                w_hv = jax.lax.dynamic_slice(w_pad, (h0, v0), (h_block, v_block))
                return acc + jax.lax.dot_general(x_bh, w_hv, (((1,), (0,)), ((), ())), precision, preferred_element_type = dtype)

            logits = jax.lax.fori_loop(
                0, num_h, h_body, jnp.zeros((b_block, v_block), dtype = dtype)
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

        lse_b, label_logits_b = jax.lax.fori_loop(
            0, num_v, v_body, (lse_b, label_logits_b)
        )  # (B_block)
        loss_b = lse_b - label_logits_b
        lse = jax.lax.dynamic_update_slice(lse, lse_b, (b0,))
        loss = jax.lax.dynamic_update_slice(loss, loss_b, (b0,))
        return lse, loss

    lse0, loss0 = jnp.zeros((B,), dtype), jnp.zeros((B,), dtype)
    (lse, loss) = jax.lax.fori_loop(0, num_b, b_body, (lse0, loss0))

    return loss, lse
