import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree, Int

from .base import Callback

class LogPerformanceState(NamedTuple):
    time: Float[Array, ""]
    denom_count: PyTree[int]


def log_performance(
    denom_keys: list[str]

) -> Callback:
    def init(weights, opt_state):
        del weights, opt_state
        denom_count = {}
        for denom in denom_keys:
            denom_count[denom] = jnp.zeros([], dtype = jnp.dtype32)

        return LogPerformanceState(time = time.monotonic(), denom_count = denom_count)

    def update(callback_state, grad, updates, opt_state, weights, aux):
        del grad, updates, opt_state, weights
        denom_count = {}
        for denom in denom_keys:
            if denom not in aux:
                raise ValueError
            denom_count[denom] = aux[denom]
        return LogPerformanceState(time = callback_state.time, denom_count = denom_count)

    def process(output, callback_state, aux):
        delta = float(time.monotonic() - callback_state.time)
        for k, v in callback_state.denom_count.items():
            output[f"performance/{k}_s"] = float(v / delta)

        denom_count = {}
        for denom in denom_keys:
            denom_count[denom] = jnp.zeros([], dtype = jnp.dtype32)

        return output, LogPerformanceState(time = time.monotonic(), denom_count = denom_count)
    return Callback(init, update, process)

def make(denom_keys: list[str]):
    return log_performance(denom_keys)
