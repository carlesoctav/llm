import optax
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from jaxformers import tree_util
from jaxformers.optimizers.lr import ScaleByLearningRateState

from .base import Callback

class LogGradNormState(NamedTuple):
    grad_norm: jax.Array

def log_grad_norm() -> Callback:
    def init(weights, opt_state):
        del weights, opt_state
        return LogGradNormState(jnp.zeros([], dtype = jnp.float32))

    def update(callback_state, grad, updates, opt_state, weights, aux):
        del updates, opt_state, weights, aux
        grad_norm = optax.global_norm(grad).astype(jnp.float32)

        # do we need this check?
        # should_update = jnp.isnan(grad_norm) | (grad_norm > 0)
        # grad_norm_next = jnp.where(should_update, grad_norm, callback_state.grad_norm)
        return LogGradNormState(grad_norm = grad_norm)

    def process(output, callback_state, aux):
        output["grad/grad_norm"] = callback_state.grad_norm
        return output, callback_state

    return Callback(init, update, process)


def make():
    return log_grad_norm()
