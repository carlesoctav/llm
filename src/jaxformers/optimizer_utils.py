from dataclasses import replace

import equinox
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from jaxformers import tree_util
from jaxformers.dispatch.lora import LoraArray
from jaxformers.optimizers.lr import ScaleByLearningRateState


def mask_trainable_lora(params):
    is_lora_array = lambda x: isinstance(x, LoraArray)

    def label(leaf):
        if isinstance(leaf, LoraArray):
            return replace(leaf, _w=False, a=True, b=True)
        return False

    return jtu.tree_map(label, params, is_leaf=is_lora_array)
