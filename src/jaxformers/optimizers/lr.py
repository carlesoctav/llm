from typing import Callable, NamedTuple

import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jaxtyping import Scalar


optax.scale_by_learning_rate


class ScaleByLearningRateState(NamedTuple):
    count: Scalar
    learning_rate: Scalar


def custom_scale_by_learning_rate(
    learning_rate: float | Callable[[Scalar], Scalar] | None = None,
    *,
    flip_sign: bool = True,
):
    if learning_rate is None:
        return optax.identity()
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        step_size_fn = lambda count: m * learning_rate(count)

        def init_fn(params):
            del params
            return ScaleByLearningRateState(
                count=jnp.zeros([], jnp.int32), learning_rate=jnp.zeros([], jnp.float32)
            )

        def update_fn(updates, state, params=None):
            step_size = step_size_fn(state.count)
            updates = jtu.tree_map(lambda g: jnp.array(step_size) * g, updates)
            return updates, ScaleByLearningRateState(count = optax.safe_increment(state.count), learning_rate = jnp.array(step_size))

        return optax.GradientTransformation(init_fn, update_fn)
    else:
        def init_fn(params):
            del params
            return ScaleByLearningRateState(
                count=jnp.zeros([], jnp.int32), learning_rate=jnp.zeros([], jnp.float32)
            )

        def update_fn(updates, state, params=None):
            updates = jtu.tree_map(lambda g: jnp.array( m * learning_rate) * g, updates)
            return updates, ScaleByLearningRateState(count = optax.safe_increment(state.count), learning_rate = jnp.array(learning_rate))

        return optax.GradientTransformation(init_fn, update_fn)
