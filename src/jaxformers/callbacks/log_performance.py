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

    def init(model):
        del model
        denom_count = {}
        for denom in denom_keys:
            denom_count[denom] = jnp.zeros([], dtype=jnp.int32)

        host_state["last_time"] = time.monotonic()

        return LogPerformanceState(denom_count=denom_count)

    def update(model, callback_state, grad, updates, aux):
        del callback_state, grad, updates
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
        return model, LogPerformanceState(denom_count=denom_count)

    def process(output, model, callback_state, aux):
        del aux
        dispatch_delta = time.monotonic() - host_state["last_time"]
        for k, v in callback_state.denom_count.items():
            output[f"performance/dispatch_{k}_per_s"] = float(v / dispatch_delta)

        output["performance/dispatch_time_per_step"] = float(dispatch_delta)

        if host_state["step"] < real_step_threshold:
            jax.block_until_ready(callback_state.denom_count)
            real_delta = time.monotonic() - host_state["last_time"]
            for k, v in callback_state.denom_count.items():
                output[f"performance/real_{k}_per_s"] = float(v / real_delta)

            output["performance/real_time_per_step"] = real_delta

        host_state["last_time"] = time.monotonic()
        host_state["step"] += 1
        return output, model, callback_state

    return Callback(init, update, process)


def make(denom_keys: list[str], real_step_threshold: int = 0):
    return log_performance(
        denom_keys,
        real_step_threshold=real_step_threshold,
    )
