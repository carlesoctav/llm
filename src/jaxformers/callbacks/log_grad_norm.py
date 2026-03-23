from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .base import Callback


class LogGradNormState(NamedTuple):
    grad_norm: jax.Array


def log_grad_norm() -> Callback:
    def init(weights, opt_state):
        del weights, opt_state
        return LogGradNormState(jnp.zeros([], dtype=jnp.float32))

    def update(model, callback_state, grad, updates, aux):
        del callback_state, updates, aux
        grad_norm = optax.global_norm(grad).astype(jnp.float32)
        return model, LogGradNormState(grad_norm=grad_norm)

    def process(output, model, callback_state, aux):
        del aux
        output["grad/grad_norm"] = callback_state.grad_norm
        return output, model, callback_state

    return Callback(init, update, process)


def make():
    return log_grad_norm()
