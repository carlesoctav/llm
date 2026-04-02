from typing import NamedTuple

import jax
import jax.numpy as jnp

from .base import Callback


class ProfilerState(NamedTuple):
    step: jax.Array


def profiler(path: str) -> Callback:
    host_state = {"active": False, "done": False}

    def init(train_state):
        del train_state
        return ProfilerState(step=jnp.zeros([], dtype=jnp.int32))

    def update(train_state, callback_state, grad, updates, aux):
        del grad, updates, aux
        return train_state, ProfilerState(step=callback_state.step + 1)

    def process(output, train_state, callback_state, aux):
        del aux
        if jax.process_index() != 0:
            return output, train_state, callback_state

        step = int(jax.device_get(callback_state.step))
        if not host_state["active"] and not host_state["done"]:
            jax.block_until_ready(train_state.model)
            jax.profiler.start_trace(path)
            host_state["active"] = True
            print(f"Started profiler trace at step {step} -> {path}")

        elif host_state["active"]:
            jax.block_until_ready(train_state.model)
            jax.profiler.stop_trace()
            host_state["active"] = False
            host_state["done"] = True
            print(f"Stopped profiler trace at step {step} -> {path}")

        return output, train_state, callback_state

    return Callback(init, update, process)


def make(path: str):
    return profiler(path)
