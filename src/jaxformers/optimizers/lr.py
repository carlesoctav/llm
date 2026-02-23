from typing import Callable, NamedTuple

import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jaxtyping import Scalar


class ScaleByLearningRateState(NamedTuple):
    count: Scalar
    learning_rate: Scalar
    step_size: Scalar


def custom_scale_by_learning_rate(
    learning_rate: float | Callable[[Scalar], Scalar] | None = None,
    *,
    flip_sign: bool = True,
    apply_every_k: int | None = None,
):
    if learning_rate is None:
        return optax.identity()

    if apply_every_k is not None and apply_every_k < 1:
        raise ValueError("apply_every_k must be >= 1")

    sign = jnp.asarray(-1.0 if flip_sign else 1.0, dtype=jnp.float32)

    if callable(learning_rate):
        lr_fn: Callable[[Scalar], Scalar] = learning_rate
    else:
        lr_const = jnp.asarray(float(learning_rate), dtype=jnp.float32)
        lr_fn = lambda count: lr_const

    def init_fn(params):
        del params
        count0 = jnp.zeros([], jnp.int32)
        lr0 = jnp.asarray(lr_fn(count0), dtype=jnp.float32)
        step0 = sign * lr0
        return ScaleByLearningRateState(count=count0, learning_rate=lr0, step_size=step0)

    def update_fn(updates, state, params=None):
        del params
        lr = jnp.asarray(lr_fn(state.count), dtype=jnp.float32)
        step_size = sign * lr
        update_norm = optax.global_norm(updates).astype(jnp.float32)

        def _scale(g):
            step = step_size.astype(g.dtype) if jnp.issubdtype(g.dtype, jnp.floating) else step_size
            return step * g

        updates = jtu.tree_map(_scale, updates)

        emit = jnp.isnan(update_norm) | (update_norm > 0)
        count_next = jnp.where(emit, optax.safe_increment(state.count), state.count)
        lr_next = jnp.where(emit, lr, state.learning_rate)
        step_next = jnp.where(emit, step_size, state.step_size)
        return updates, ScaleByLearningRateState(
            count=count_next,
            learning_rate=lr_next,
            step_size=step_next,
        )

    return optax.GradientTransformation(init_fn, update_fn)


def _find_lr_state(opt_state) -> ScaleByLearningRateState | None:
    found: list[ScaleByLearningRateState] = []

    def _maybe_collect(leaf):
        if isinstance(leaf, ScaleByLearningRateState):
            found.append(leaf)

    jtu.tree_map(_maybe_collect, opt_state, is_leaf=lambda x: isinstance(x, ScaleByLearningRateState))
    return found[0] if found else None


def get_logged_learning_rate(opt_state):
    state = _find_lr_state(opt_state)
    return None if state is None else state.learning_rate


def get_logged_step_size(opt_state):
    state = _find_lr_state(opt_state)
    return None if state is None else state.step_size
