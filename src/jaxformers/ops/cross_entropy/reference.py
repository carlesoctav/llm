from functools import partial

import jax
import jax.numpy as jnp
import optax
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

    def _inner(x: jax.Array, labels: jax.Array, w: jax.Array):
        logits = einsum(
            "bh,vh -> bv",
            x,
            w,
            precision=precision,
            preferred_element_type=dtype,
        )
        if logit_soft_cap is not None:
            logits = jnp.tanh(logits / logit_soft_cap) * logit_soft_cap
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        lse = jax.nn.logsumexp(logits, axis=-1)
        return loss, lse

    return _inner(x, labels, w)
