"""Gradient transformations."""

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax.tree
from jaxtyping import Array, Int
from optax import GradientTransformationExtraArgs
from optax._src import base, numerics


class GradAccumState(NamedTuple):
    """Contains a counter and a gradient accumulator."""

    count: Int[Array, ""]
    denom: Int[Array, ""]
    grad_acc: base.Updates
    inner_opt_state: Any


def gradient_accumulation(
    tx: optax.GradientTransformationExtraArgs,
    every_k: int = 1,
    *,
    denom_key: str = "count",
):
    def init(params):
        grad_acc = jax.tree.map(lambda p: jnp.zeros_like(p, dtype=p.dtype), params)
        return GradAccumState(
            count=jnp.zeros([], dtype=jnp.int32),
            denom=jnp.zeros([], dtype=jnp.float32),
            grad_acc=grad_acc,
            inner_opt_state=tx.init(params),
        )

    def update(
        updates: base.Updates,
        state: GradAccumState,
        params: base.Params | None = None,
        **kwargs,
    ):
        if denom_key not in kwargs:
            raise ValueError(
                f"Missing extra arg {denom_key!r}. Pass it to `tx.update(..., {denom_key}=...)`."
        )

        c = state.count % every_k
        acc = c != 0
        emit = c == (every_k - 1)
        grad_acc = (jax.tree.map(lambda g, ga: acc * ga + g, updates, state.grad_acc),)
        denom_acc = (
            acc * state.denom + jnp.asarray(kwargs[denom_key], dtype=jnp.float32),
        )
        inner_updates, updated_opt_state = tx.update(
            grad_acc, state.inner_opt_state, params
        )

        new_state = GradAccumState(
            count=numerics.safe_increment(state.count) % every_k,
            grad_acc=grad_acc,
            denom_acc=denom_acc,
            inner_opt_state=jax.tree.map(
                lambda new, old: jnp.where(emit, new, old),
                updated_opt_state,
                state.inner_opt_state,
            ),
        )

        def make_final_updates(iu):
            inv = (1 / denom_acc).astype(iu.dtype)
            return emit * iu * inv

        inner_updates = jax.tree_map(make_final_updates, inner_updates)
        return inner_updates, new_state

    return GradientTransformationExtraArgs(init, update)
