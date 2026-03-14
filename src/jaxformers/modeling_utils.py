import copy
import re
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, TypedDict, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax import P
from jax.sharding import AxisType
from jaxtyping import Bool, Float, PyTree
from safetensors import safe_open
from transformers import PreTrainedConfig, PreTrainedTokenizerFast

from jaxformers.distributed.parallel import mutate_sharding_rule_parallel_dims
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
    forward_impl: str = "loop"


DEFAULT_ADDITIONAL_CONFIG = {
    "remat_layer": False,
    "attn_implementation": "sdpa",
    "sequence_parallelism": True,
    "forward_impl": "loop",
}

DEFAULT_SHARDING_RULES = {
    "none": None,
    "batch": ("dp_replicate", "dp_shard"),
    "fsdp": ("dp_shard", "cp"),
    "model": ("tp",),
    "sequence": ("tp", "cp"),
    "context": ("cp",),
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
        "mesh",
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
    mesh: Any | None = None

    opt_state: PyTree["ModelWeights"] | None = None
    tx: optax.GradientTransformation | None = None
    step: int | None = None

    callback_state: PyTree | None = None
    callbacks: Any | None = None

    train_mask: PyTree[Bool] | None = None
    is_lora: bool = False

    def __repr__(self):
        return self.name + "\n" + tree_pformat(self.weights)


def make_mesh(parallel_dims, devices: list | None = None):
    axis_shapes = tuple(parallel_dims.values())
    axis_names = tuple(parallel_dims.keys())
    axis_types = tuple(AxisType.Explicit for _ in axis_names)
    return jax.make_mesh(
        axis_shapes,
        axis_names,
        axis_types=axis_types,
        devices=devices,
    )


def clone_model_with_mesh(
    model: Model,
    parallel_dims,
    *,
    devices: list | None = None,
) -> Model:
    config = copy.deepcopy(model.config)
    additional_config = {
        **DEFAULT_ADDITIONAL_CONFIG,
        **getattr(config, "additional_config", {}),
    }
    sharding_rules = mutate_sharding_rule_parallel_dims(
        dict(DEFAULT_SHARDING_RULES),
        parallel_dims,
        sequence_parallelism=additional_config["sequence_parallelism"],
    )
    config.additional_config = additional_config
    config.parallel_dims = parallel_dims
    config.sharding_rules = sharding_rules
    mesh = make_mesh(
        parallel_dims,
        devices=devices,
    )
    config.mesh = mesh

    def _rebind(fn: Callable):
        if isinstance(fn, partial):
            args = fn.args
            if args:
                return partial(
                    fn.func,
                    config,
                    *args[1:],
                    **(fn.keywords or {}),
                )
            return partial(fn.func, config, **(fn.keywords or {}))
        bound_fn = getattr(fn, "__func__", None)
        if bound_fn is not None:
            return partial(bound_fn, config)
        return fn

    return replace(
        model,
        config=config,
        forward=_rebind(model.forward),
        embed=_rebind(model.embed),
        unembed=_rebind(model.unembed),
        mesh=mesh,
    )


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
