from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from jaxformers import tree_util
from jaxformers.optimizers.lr import ScaleByLearningRateState

from .base import Callback


def find_learning_rate(opt_state):
    is_lr_state = lambda x: isinstance(x, ScaleByLearningRateState)
    res = {}

    def f(path, leaf):
        if is_lr_state(leaf):
            log_key = tree_util.optimizerstr(path)
            log_key = f"{log_key}/lr" if log_key else "lr"
            res[f"optim/{log_key}"] = leaf.learning_rate

    jax.tree.map_with_path(f, opt_state, is_leaf=is_lr_state)
    return res


class LogLearningRateState(NamedTuple):
    learning_rate: PyTree[Float[Array, ""]]


def log_learning_rate() -> Callback:
    def init(model):
        shape = jax.eval_shape(find_learning_rate, model.opt_state)
        zeros = jax.tree.map(
            lambda shape: jnp.zeros_like(shape, dtype=jnp.float32), shape
        )
        return LogLearningRateState(zeros)

    def update(model, callback_state, grad, updates, aux):
        del callback_state, grad, updates, aux
        return model, LogLearningRateState(
            learning_rate=find_learning_rate(model.opt_state)
        )

    def process(output, model, callback_state, aux):
        del aux
        output.update(callback_state.learning_rate)
        return output, model, callback_state

    return Callback(init, update, process)


def make():
    return log_learning_rate()
