from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jaxformers.dispatch import einsum
from jaxformers.ops.cross_entropy.config import BlockSizes


@partial(jax.jit, static_argnames=["block_sizes", "dtype", "precision"])
def cross_entropy_reference(
    x: Float[Array, "B H"],
    labels: Int[Array, " B"],
    w: Float[Array, "V H"],
    *,
    block_sizes: BlockSizes | None = None,
    dtype: jnp.dtype | None = None,
    logit_soft_cap: float | None = None,
    precision: jax.lax.PrecisionLike = None,
):
    del block_sizes  # unused for reference impl

    logits = einsum(
        "bh,vh -> bv",
        x,
        w,
        precision=precision,
        preferred_element_type=jnp.float32,
    )
    if logit_soft_cap is not None:
        logits = jnp.tanh(logits / logit_soft_cap) * logit_soft_cap

    lse = jax.nn.logsumexp(logits, axis=-1)
    label_logits = jnp.sum(
        jax.nn.one_hot(labels, logits.shape[-1], dtype=logits.dtype) * logits,
        axis=-1,
    )
    loss = lse - label_logits

    return loss, lse
