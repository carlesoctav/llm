import warnings
from collections.abc import Callable, Sequence
from typing import cast, Literal, Optional, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .config import BlockSizes, infer_block_sizes
from .reference import cross_entropy_reference
from .xla_chunked import fused_cross_entropy_chunked_xla


Implementation: TypeAlias = Literal["xla_chunked", "reference"]
Reduction: TypeAlias = Literal["sum", "mean"] | None


ArrayImpl = Callable[..., tuple[jax.Array, jax.Array]]


IMPLEMENTATIONS: dict[str, ArrayImpl] = {
    "xla_chunked": fused_cross_entropy_chunked_xla,
    "reference": cross_entropy_reference,
}

_DEFAULT_IMPLEMENTATION: tuple[Implementation, ...] = ("xla_chunked", "reference")
# try:
#     from .pallas_tpu import PallasUnsupportedError, linear_softmax_cross_entropy_loss_pallas

#     IMPLEMENTATIONS["pallas_tpu"] = linear_softmax_cross_entropy_loss_pallas
#     _DEFAULT_IMPLEMENTATION = ("pallas_tpu",) + _DEFAULT_IMPLEMENTATION
# except ImportError:
#     PallasUnsupportedError = NotImplementedError  # type: ignore[assignment]


def _validate_inputs(x: jax.Array, labels: jax.Array, w: jax.Array) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [B, H], got shape {x.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"labels must be rank-1 [B], got shape {labels.shape}.")
    if w.ndim != 2:
        raise ValueError(f"w must be rank-2 [H, V], got shape {w.shape}.")
    if x.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Batch mismatch: x has B={x.shape[0]}, labels has B={labels.shape[0]}."
        )
        raise ValueError(
            f"Batch mismatch: x has B={x.shape[0]}, labels has B={labels.shape[0]}."
        )
    if x.shape[1] != w.shape[1]:
        raise ValueError(
            f"Hidden mismatch: x has H={x.shape[1]}, w has H={w.shape[0]}."
        )
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError(f"labels must be integer dtype, got {labels.dtype}.")


def _apply_reduction(
    loss: jax.Array, reduction: Reduction, weight: jax.Array | None
) -> jax.Array:
    if weight is not None:
        weight = weight.astype(loss.dtype)
        loss = loss * weight

    if reduction is None:
        return loss
    if reduction == "sum":
        return jnp.sum(loss)
    if reduction == "mean":
        if weight is None:
            return jnp.mean(loss)
        denom = jnp.sum(weight)
        return jnp.where(denom != 0, jnp.sum(loss) / denom, jnp.zeros_like(denom))
    raise ValueError(f"Unsupported reduction: {reduction}")


def cross_entropy_loss(
    x: Float[Array, "B H"],
    labels: Int[Array, " B"],
    w: Float[Array, "V H"],
    block_sizes: BlockSizes | None = None,
    *,
    reduction: Reduction = "sum",
    weight: Float[Array, " B"] | None = None,
    logsumexp_weight: float | None = 0.0,
    dtype: jnp.dtype | None = jnp.float32,
    logit_soft_cap: float | None = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: Implementation | Sequence[Implementation | ArrayImpl] | None = None,
) -> jax.Array:

    _validate_inputs(x, labels, w)

    if implementation is None:
        impls: Sequence[Implementation | ArrayImpl] = _DEFAULT_IMPLEMENTATION
        # explicit = False
    elif isinstance(implementation, Sequence) and not isinstance(
        implementation, (str, bytes)
    ):
        impls = cast(Sequence[Implementation | ArrayImpl], implementation)
        # explicit = len(impls) == 1
    else:
        impls = (cast(Implementation, implementation),)
        # explicit = True

    errors: list[Exception] = []
    B, H = x.shape
    V, H = w.shape
    for impl in impls:
        fn = IMPLEMENTATIONS.get(impl)
        if fn is None:
            raise ValueError(f"Unsupported implementation: {impl}")
        if block_sizes is None:
            block_sizes_for_impl = infer_block_sizes(impl, B, H, V, dtype = dtype)
        else:
            block_sizes_for_impl = block_sizes
        try:
            loss, lse = fn(
                x,
                labels,
                w,
                logit_soft_cap=logit_soft_cap,
                block_sizes=block_sizes_for_impl,
                dtype=dtype,
                precision=precision,
            )
        except Exception as e:
            errors.append(e)
            continue

        if logsumexp_weight is not None and logsumexp_weight != 0.0:
            loss = loss + logsumexp_weight * (lse**2)
        return _apply_reduction(loss, reduction, weight)
    raise ExceptionGroup("all implementations failed", errors)
