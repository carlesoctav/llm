from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, TypedDict, TypeVar

import equinox as eqx
import jax.tree_util as jtu
import optax
from jax import P
from jaxtyping import Array, Bool, Float, PyTree
from transformers import PreTrainedConfig, PreTrainedTokenizerFast


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
    # training
    gradient_checkpointing: bool
    remat_layer: bool
    remat_loss: bool

    # training and inference
    attn_implementation: str
    sequence_parallelism: bool
    loss_parallel: bool


DEFAULT_ADDITIONAL_CONFIG = {
    "gradient_checkpointing": True,
    "attn_implementation": "sdpa",
    "sequence_parallelism": True,
    "loss_parallel": True,
    "remat_layer": False,
    "remat_loss": False,
}


@partial(
    jtu.register_dataclass,
    data_fields=["weights", "opt_state", "step"],
    meta_fields=[
        "tokenizer",
        "embed",
        "forward",
        "unembed",
        "config",
        "tx",
        "is_lora",
        "train_mask",
    ],
)
@dataclass
class Model:
    config: PreTrainedConfig
    weights: PyTree[Float, "ModelWeights"]
    embed: Callable | None
    forward: Callable
    unembed: Callable | None
    tokenizer: PreTrainedTokenizerFast

    opt_state: PyTree["ModelWeights"] | None = None
    tx: optax.GradientTransformation | None = None
    step: int | None = None

    train_mask: PyTree[Bool] | None = None
    is_lora: bool = False
