from dataclasses import replace

import equinox
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from jaxformers import tree_util
from jaxformers.dispatch.lora import LoraArray
from jaxformers.optimizers.lr import ScaleByLearningRateState


def mask_non_lora(params):
    is_lora_array = lambda x: isinstance(x, LoraArray)

    def label(leaf):
        if isinstance(leaf, LoraArray):
            return replace(leaf, _w=True, a=False, b=False)
        return True

    return jtu.tree_map(label, params, is_leaf=is_lora_array)


def find_learning_rate(opt_state):
    is_lr_state = lambda x: isinstance(x, ScaleByLearningRateState)
    res = {}

    def f(path, leaf):
        if is_lr_state(leaf):
            log_key = tree_util.optimizerstr(path)
            log_key = f"{log_key}/lr" if log_key else "lr"
            res[f"optim/{log_key}"] = leaf.learning_rate

    jtu.tree_map_with_path(f, opt_state, is_leaf=is_lr_state)
    return res

def find_grad_norm(opt_state):
    return {"grad/egrad_norm": opt_state[2].grad_norm}
