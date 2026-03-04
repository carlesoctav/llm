from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, TypedDict, TypeVar

import jax.tree_util as jtu
import optax
from jax import P
from jaxtyping import Bool, Float, PyTree
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
    # training
    gradient_checkpointing: bool = True

    # Rematerialization / checkpointing
    remat_layer: bool
    remat_attention: bool

    # training and inference
    attn_implementation: str = "sdpa"
    sequence_parallelism: bool = True
    loss_parallel: True


DEFAULT_ADDITIONAL_CONFIG = {
    "gradient_checkpointing": True,
    "remat_layer": False,
    "remat_attention": False,
    "attn_implementation": "sdpa",
    "sequence_parallelism": True,
}


@partial(
    jtu.register_dataclass,
    data_fields=["weights", "opt_state", "step"],
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

    train_mask: PyTree[Bool] | None = None
    is_lora: bool = False

    def __repr__(self):
        return self.name + "\n" + tree_pformat(self.weights)
