from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
import jax.tree_util as jtu


class LogGradNormState(NamedTuple):
    grad_norm: jax.Array


def log_grad_norm() -> optax.GradientTransformation:
    """Log the (pre-clip) global norm of updates into optimizer state.

    Intended usage:
    - Place this *after* any grad-accum scaling (e.g. token-mean division).
    - Place this *before* clipping so the logged value is unclipped.

    This transform updates its stored `grad_norm` only when it observes
    non-zero updates (or NaNs). When updates are all zeros (e.g. non-emit
    microsteps with `optax.apply_every(k)`), it keeps the previous value.
    """

    def init_fn(params):
        del params
        return LogGradNormState(grad_norm=jnp.zeros([], dtype=jnp.float32))

    def update_fn(updates, state, params=None):
        del params
        grad_norm = optax.global_norm(updates).astype(jnp.float32)
        should_update = jnp.isnan(grad_norm) | (grad_norm > 0)
        grad_norm_next = jnp.where(should_update, grad_norm, state.grad_norm)
        return updates, LogGradNormState(grad_norm=grad_norm_next)

    return optax.GradientTransformation(init_fn, update_fn)


def get_logged_grad_norm(opt_state):
    found: list[jax.Array] = []

    def _maybe_collect(leaf):
        if isinstance(leaf, LogGradNormState):
            found.append(leaf.grad_norm)

    jtu.tree_map(_maybe_collect, opt_state, is_leaf=lambda x: isinstance(x, LogGradNormState))
    return found[0] if found else None
