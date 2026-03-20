import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

from .base import Callback


class LogPerformanceState(NamedTuple):
    denom_count: dict[str, Int[Array, ""]]


def log_performance(
    denom_keys: list[str],
    real_step_threshold: int = 0,
) -> Callback:

    host_state = {
        "last_time": None,
        "step": 0,
    }

    def init(weights, opt_state):
        del weights, opt_state
        denom_count = {}
        for denom in denom_keys:
            denom_count[denom] = jnp.zeros([], dtype=jnp.int32)

        host_state["last_time"] = time.monotonic()

        return LogPerformanceState(denom_count=denom_count)

    def update(callback_state, grad, updates, opt_state, weights, aux):
        del callback_state
        del grad, updates, opt_state, weights
        denom_count = {}
        for denom in denom_keys:
            if denom in aux:
                denom_count[denom] = aux[denom]
            else:
                try:
                    available_keys = list(aux.keys())
                except Exception:
                    available_keys = f"<non-mapping type: {type(aux).__name__}>"
                raise ValueError(
                    f"Missing required performance denominator '{denom}' in aux. "
                    f"Expected one of: {denom_keys!r}. Available keys: {available_keys}"
                )
        return LogPerformanceState(denom_count=denom_count)

    def process(output, callback_state, aux):
        dispatch_delta = time.monotonic() - host_state["last_time"]
        for k, v in callback_state.denom_count.items():
            output[f"performance/dispatch_{k}_per_s"] = float(v / dispatch_delta)

        output["performance/dispatch_time_per_step"] = float(dispatch_delta)

        if host_state["step"] < real_step_threshold:
            jax.block_until_ready(aux)
            real_delta = time.monotonic() - host_state["last_time"]
            for k, v in callback_state.denom_count.items():
                output[f"performance/real_{k}_per_s"] = float(v / real_delta)

            output["performance/real_time_per_step"] = real_delta

        host_state["last_time"] = time.monotonic()
        host_state["step"] += 1
        return output, callback_state

    return Callback(init, update, process)


def make(denom_keys: list[str], real_step_threshold: int = 0):
    return log_performance(
        denom_keys,
        real_step_threshold=real_step_threshold,
    )
