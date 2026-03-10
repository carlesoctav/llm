import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, TypedDict, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax import P
from jaxtyping import Bool, Float, PyTree
from safetensors import safe_open
from transformers import PreTrainedConfig, PreTrainedTokenizerFast

from jaxformers.print_utils import tree_pformat


LayerWeights = TypeVar("LayerWeights")
ModelWeights = TypeVar("ModelWeights")


def logical_to_physical(logical, rules):
    spec = [rules[lo] for lo in logical]
    flat_leaves = jtu.tree_leaves(spec)
    if len(flat_leaves) != len(set(flat_leaves)):
        raise ValueError(
            f"Colliding physical axes from translating logical spec {logical} -> {spec}"
        )

    return P(*spec)


class AdditionalConfig(TypedDict):
    remat_layer: bool

    attn_implementation: str = "sdpa"
    sequence_parallelism: bool = True


DEFAULT_ADDITIONAL_CONFIG = {
    "remat_layer": False,
    "attn_implementation": "sdpa",
    "sequence_parallelism": True,
}


@partial(
    jtu.register_dataclass,
    data_fields=[
        "weights",
        "opt_state",
        "step",
        "callback_state",
    ],
    meta_fields=[
        "name",
        "tokenizer",
        "forward",
        "config",
        "tx",
        "is_lora",
        "train_mask",
        "embed",
        "unembed",
        "lm_head_key",
        "callbacks",
    ],
)
@dataclass
class Model:
    name: str
    config: PreTrainedConfig
    weights: PyTree[Float, "ModelWeights"]
    forward: Callable
    embed: Callable
    unembed: Callable
    tokenizer: PreTrainedTokenizerFast
    lm_head_key: str

    opt_state: PyTree["ModelWeights"] | None = None
    tx: optax.GradientTransformation | None = None
    step: int | None = None

    callback_state: PyTree | None = None
    callbacks: Any | None = None

    train_mask: PyTree[Bool] | None = None
    is_lora: bool = False

    def __repr__(self):
        return self.name + "\n" + tree_pformat(self.weights)


def load_weights(model_ckpt_dir, param_dtype, sharding_rules, get_sharding):
    weights = {}
    for file in model_ckpt_dir.glob("*.safetensors"):
        with safe_open(file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = jax.device_put(
                    f.get_tensor(key).astype(param_dtype),
                    get_sharding(key, sharding_rules),
                )
    return weights


def load_weights_vectorize(
    prefix, layer_size, model_ckpt_dir, param_dtype, sharding_rules, get_sharding
):
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)\.(.*)")
    weights = {}
    for file in model_ckpt_dir.glob("*.safetensors"):
        with safe_open(file, framework="numpy") as f:
            for key in f.keys():
                match = pattern.match(key)
                if match:
                    idx = int(match.group(1))
                    rg_key = match.group(2)
                    weights[rg_key] = weights.get(
                        rg_key, [None for _ in range(layer_size)]
                    )
                    weights[rg_key][idx] = jax.device_put(
                        f.get_tensor(key).astype(param_dtype),
                        get_sharding(key, sharding_rules),
                    )
                else:
                    weights[key] = jax.device_put(
                        f.get_tensor(key).astype(param_dtype),
                        get_sharding(key, sharding_rules),
                    )
    for k, v in weights.items():
        if isinstance(v, list):
            weights[k] = jnp.stack(v)

    return weights
