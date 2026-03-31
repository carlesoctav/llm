import re
from dataclasses import dataclass
from enum import auto, StrEnum
from functools import partial
from typing import Any, TypedDict, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax.sharding import Mesh
from jaxtyping import Bool, PyTree
from safetensors import safe_open

from jaxformers import tree_util
from jaxformers.dispatch.lora import lora_get_w
from jaxformers.distributed.parallel import from_logical_rules
from jaxformers.module_utils import Stackable, StackModule
from jaxformers.print_utils import tree_pformat


LayerWeights = TypeVar("LayerWeights")
ModelWeights = TypeVar("ModelWeights")


class StoreWeights(StrEnum):
    STACK = auto()
    FREE = auto()


class ForwardImpl(StrEnum):
    LOOP = auto()
    SCAN_LAYER = auto()


def logical_to_physical(logical, rules):
    spec = from_logical_rules(logical, rules)
    flat_leaves = jtu.tree_leaves(spec)
    if len(flat_leaves) != len(set(flat_leaves)):
        raise ValueError(
            f"Colliding physical axes from translating logical spec {logical} -> {tuple(spec)}"
        )

    return spec


class AdditionalConfig(TypedDict):
    remat_layer: bool

    attn_impl: str = "sdpa"
    sequence_parallelism: bool = True
    weights_impl: str = "stack"
    forward_impl: str = "loop"


DEFAULT_ADDITIONAL_CONFIG = {
    "remat_layer": False,
    "attn_impl": "sdpa",
    "forward_impl": "loop",
}


@partial(
    jtu.register_dataclass,
    data_fields=[
        "model",
        "opt_state",
        "step",
        "callback_state",
    ],
    meta_fields=[
        "tx",
        "is_lora",
        "train_mask",
        "callbacks",
        "mesh",
        "rule",
    ],
)
@dataclass
class TrainState:
    model: eqx.Module
    mesh: Mesh
    rule: tuple[tuple[str, str | tuple[str, ...] | None], ...] = ()

    opt_state: PyTree["ModelWeights"] | None = None
    tx: optax.GradientTransformation | None = None
    step: int | None = None

    callback_state: PyTree | None = None
    callbacks: Any | None = None

    train_mask: PyTree[Bool] | None = None
    is_lora: bool = False

    def __repr__(self):
        return tree_pformat(self.model)

    @property
    def params(
        self,
    ):
        return self.model

    @property
    def trainable_params(self) -> tuple[PyTree, PyTree]:
        return tree_util.partition(self.model, self.train_mask)

    @property
    def base_params(self):
        if self.is_lora:
            return lora_get_w(self.model)
        else:
            return self.model


def get_model_config(weights):
    if hasattr(weights, "config"):
        return weights.config
    return weights.model.config


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


class PreTrainedModel(eqx.Module):
    @classmethod
    def init(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__}.init() is not implemented.")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError(
            f"{cls.__name__}.from_pretrained() is not implemented."
        )

    def stack(self):
        _is_leaf = lambda x: isinstance(x, list)

        def f(path, leaf):
            if isinstance(leaf, list):
                l0 = leaf[0]
                if isinstance(l0, Stackable):
                    print(
                        f"{jtu.keystr(path, simple=True, separator='.')} is a stackable list, converthing to stack"
                    )
                    return StackModule(
                        type(l0),
                        leaf,
                        l0.argnums,
                        argnames=l0.argnames,
                        in_axes=l0.in_axes,
                        remat=l0.remat,
                    )
                else:
                    return leaf
            else:
                return leaf

        return jax.tree.map_with_path(f, self, is_leaf=_is_leaf)
