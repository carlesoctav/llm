import dataclasses
import importlib
import time
from functools import partial

from jaxformers import tree_util
from jaxformers.benchmark_utils import print_timing


@print_timing
def make_optimizer(optimizer_name: str, train_state, scheduler, optimizer_config: dict):
    optimizer_module = importlib.import_module(
        f"jaxformers.optimizers.{optimizer_name}"
    )
    if not getattr(optimizer_module, "make"):
        raise ValueError(
            f"{optimizer_module!r} does not have a 'make' method; please ensure you're using the correct optimizer_name."
        )

    train_mask = None
    if train_state.is_lora:
        from jaxformers.optimizer_utils import mask_trainable_lora

        train_mask = mask_trainable_lora(train_state.model)
        train_state = dataclasses.replace(train_state, train_mask=train_mask)

    train_weights, _ = tree_util.partition(train_state.model, train_mask)
    tx = optimizer_module.make(scheduler, train_state, **optimizer_config)
    opt_state = tx.init(train_weights)
    return dataclasses.replace(train_state, opt_state=opt_state, tx=tx, train_mask=train_mask)
