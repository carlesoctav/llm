import fnmatch
from typing import Callable

import jax
import jax.tree_util as jtu
import optax
from jaxtyping import PyTree

from jaxformers import tree_util
from jaxformers.print_utils import tree_pformat

from .lr import custom_scale_by_learning_rate as custom_scale_by_learning_rate


DEFAULT_WEIGHT_DECAY_PATH = [
    "*.q_proj.weight",
    "*.k_proj.weight",
    "*.v_proj.weight",
    "*.o_proj.weight",
    "*.gate_proj.weight",
    "*.up_proj.weight",
    "*.down_proj.weight",
]

DEFAULT_WEIGHT_DECAY_PATH_LORA = [
    "*.q_proj.weight.a",
    "*.k_proj.weight.a",
    "*.v_proj.weight.a",
    "*.o_proj.weight.a",
    "*.gate_proj.weight.a",
    "*.up_proj.weight.a",
    "*.down_proj.weight.a",
    "*.q_proj.weight.b",
    "*.k_proj.weight.b",
    "*.v_proj.weight.b",
    "*.o_proj.weight.b",
    "*.gate_proj.weight.b",
    "*.up_proj.weight.b",
    "*.down_proj.weight.b",
]


def make(
    learning_rate: float | Callable[[int], float],
    model: PyTree | None = None,
    max_grad_norm: float | None = 1.0,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    weights_decay: float | Callable[[int], float] = 1e-4,
    weights_decay_path: list[str] = None,
):
    if not weights_decay_path and not model.is_lora:
        weights_decay_path = DEFAULT_WEIGHT_DECAY_PATH
    elif not weights_decay_path and model.is_lora:
        weights_decay_path = DEFAULT_WEIGHT_DECAY_PATH_LORA

    decayed_weights = []

    def make_weight_decay_mask(weights, weights_decay_path):
        def _f(path, leaf):
            keystr = jtu.keystr(path, simple=True, separator=".")
            for pattern_to_match in weights_decay_path:
                if fnmatch.fnmatch(keystr, pattern_to_match):
                    decayed_weights.append(keystr)
                    return True
            return False

        return jax.tree.map_with_path(_f, weights, is_leaf=lambda x: x is None)

    train_weights, _ = tree_util.partition(model.weights, model.train_mask)
    weight_decay_mask = make_weight_decay_mask(train_weights, weights_decay_path)

    if decayed_weights:
        print("Model weights that will be decayed:", tree_pformat(decayed_weights))
    else:
        print("No model weights matched weight decay patterns.")

    components = []
    if max_grad_norm:
        components.append(optax.clip_by_global_norm(max_grad_norm))
    components.append(
        optax.scale_by_adam(
            b2=b2,
            b1=b1,
            eps=eps,
        ),
    )
    components.append(
        optax.add_decayed_weights(
            weights_decay,
            weight_decay_mask,
        )
    )

    components.append(custom_scale_by_learning_rate(learning_rate))
    return optax.chain(*components)
