from dataclasses import dataclass
from functools import partial
from typing import Any

import jax.tree_util as jtu
import optax
from jax.sharding import Mesh
from jaxtyping import Bool, PyTree

from jaxformers import tree_util
from jaxformers.module_utils import AbstractModel
from jaxformers.print_utils import tree_pformat


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
    model: AbstractModel
    mesh: Mesh
    rule: tuple[tuple[str, str | tuple[str, ...] | None], ...] = ()

    opt_state: PyTree | None = None
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
            from jaxformers.dispatch.lora import lora_get_w

            return lora_get_w(self.model)
        else:
            return self.model
