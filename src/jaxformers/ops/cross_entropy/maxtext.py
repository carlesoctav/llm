from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jaxformers.dispatch import einsum
from jaxformers.sharding_utils import from_logical_rules


@jax.custom_vjp
def cross_entropy_with_integer_labels(
    logits: Float[Array, "B V"],
    labels: Int[Array, " B"],
    z_loss: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    logits_sum = jax.scipy.special.logsumexp(logits, axis=-1)
    label_logits = jnp.take_along_axis(logits, labels[:, None], axis=-1).squeeze(-1)
    total_z_loss = z_loss * jax.lax.square(logits_sum)
    loss = logits_sum - label_logits + total_z_loss
    return loss, total_z_loss


def _cross_entropy_with_integer_labels_fwd(
    logits: jax.Array,
    labels: jax.Array,
    z_loss: float = 0.0,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    max_logit = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logit
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
    label_logits = jnp.take_along_axis(logits, labels[:, None], axis=-1).squeeze(-1)
    total_z_loss = z_loss * jax.lax.square(log_z)
    loss = log_z - label_logits + total_z_loss
    return (loss, total_z_loss), (labels, z_loss, exp_shifted, sum_exp, log_z)


def _cross_entropy_with_integer_labels_bwd(
    res: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    g: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, None, None]:
    g = g[0]
    labels, z_loss, exp_shifted, sum_exp, log_z = res
    deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp
    deriv = deriv - jax.nn.one_hot(labels, deriv.shape[-1], dtype=deriv.dtype)
    g_logits = jnp.expand_dims(g, axis=-1) * deriv
    return jnp.asarray(g_logits, deriv.dtype), None, None


cross_entropy_with_integer_labels.defvjp(
    _cross_entropy_with_integer_labels_fwd,
    _cross_entropy_with_integer_labels_bwd,
)


def linear_cross_entropy_maxtext(
    x: Float[Array, "B H"],
    labels: Int[Array, " B"],
    w: Float[Array, "V H"],
    *,
    reduction: str | None = "sum",
    mask: Float[Array, " B"] | None = None,
    precision: jax.lax.PrecisionLike = None,
    z_loss: float = 0.0,
) -> jax.Array:
    logits = einsum(
        "bh,vh -> bv",
        x,
        w,
        precision=precision,
        preferred_element_type=jnp.float32,
        out_sharding=from_logical_rules(("batch", None)),
    )
    loss, _ = cross_entropy_with_integer_labels(logits, labels, z_loss=z_loss)

    if mask is not None:
        loss = loss * mask.astype(loss.dtype)

    if reduction is None:
        return loss
    if reduction == "sum":
        return jnp.sum(loss)
    if reduction == "mean":
        if mask is None:
            return jnp.mean(loss)
        denom = jnp.sum(mask.astype(loss.dtype))
        return jnp.where(denom != 0, jnp.sum(loss) / denom, jnp.zeros_like(denom))
    raise ValueError(f"Unsupported reduction: {reduction}")
