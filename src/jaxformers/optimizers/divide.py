from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


class DivideByState(NamedTuple):
    count: jax.Array
    denom_acc: jax.Array


def divide_every(
    every_k: int = 1,
    *,
    denom_key: str = "count",
    eps: float = 1e-8,
) -> optax.GradientTransformationExtraArgs:
    """Divide updates by an accumulated scalar denominator.

    This is useful when gradients are accumulated with `optax.apply_every(k)`
    (which sums gradients) and you want a *mean* update, e.g. mean over tokens
    across the k microbatches.

    Requirements:
    - Pass `denom_key` as an extra kwarg to `tx.update(...)` every microstep.
    - Use the same `every_k` here as in `optax.apply_every(every_k)` so the
      emit/reset boundaries stay aligned.
    """

    if every_k < 1:
        raise ValueError(f"`every_k` must be >= 1, got {every_k}.")

    def init_fn(params):
        del params
        return DivideByState(
            count=jnp.zeros([], dtype=jnp.int32),
            denom_acc=jnp.zeros([], dtype=jnp.float32),
        )

    def update_fn(updates, state, params=None, **extra_args):
        del params
        if denom_key not in extra_args:
            raise ValueError(
                f"Missing extra arg {denom_key!r}. Pass it to `tx.update(..., {denom_key}=...)`."
            )

        denom = jnp.asarray(extra_args[denom_key], dtype=jnp.float32)
        if denom.shape != ():
            denom = jnp.sum(denom)

        c = state.count % every_k
        emit = c == (every_k - 1)

        denom_total = state.denom_acc + denom
        denom_safe = jnp.maximum(denom_total, jnp.asarray(eps, jnp.float32))
        inv = (1.0 / denom_safe).astype(jnp.float32)

        def _scale(u):
            inv_u = inv.astype(u.dtype) if jnp.issubdtype(u.dtype, jnp.floating) else inv
            return jnp.where(emit, u * inv_u, u)

        updates = jax.tree.map(_scale, updates)

        count_next = optax.safe_int32_increment(state.count) % every_k
        denom_acc_next = jnp.where(emit, jnp.zeros_like(denom_total), denom_total)
        return updates, DivideByState(count=count_next, denom_acc=denom_acc_next)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
