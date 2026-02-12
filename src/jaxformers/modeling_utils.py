from dataclasses import dataclass
from typing import Any, Callable, TypedDict, TypeVar
from functools import partial

import jax.tree_util as jtu
from jax import P
from jaxtyping import Array, PyTree
from transformers import PreTrainedTokenizerFast
import optax


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

    # training and inference
    attn_implementation: str = "sdpa"
    sequence_parallelism: bool = True


DEFAULT_ADDITIONAL_CONFIG = {
    "gradient_checkpointing": True,
    "attn_implementation": "sdpa",
    "sequence_parallelism": True,
}


@partial(
    jtu.register_dataclass,
    data_fields = ["weights", "opt_state", "step"],
    meta_fields = ["tokenizer", "forward", "config", "tx"]
)
@dataclass
class Model:
    config: dict[str, Any]
    weights: PyTree[Array, "ModelWeights"]
    forward: Callable
    tokenizer: PreTrainedTokenizerFast

    opt_state: PyTree["ModelWeights"] | None = None
    tx: optax.GradientTransformation | None = None
    step: int | None = None
