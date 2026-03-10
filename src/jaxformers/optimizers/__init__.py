import dataclasses
import importlib

from jaxformers import tree_util
from jaxformers.optimizer_utils import mask_trainable_lora


def make_scheduler(scheduler_name: str | None, learning_rate):
    if scheduler_name in (None, "constant"):
        return learning_rate

    raise NotImplementedError(
        f"Unsupported lr scheduler {scheduler_name!r}; only constant is supported."
    )


def make_optimizer(optimizer_name: str, model, scheduler, optimizer_config: dict):
    optimizer_module = importlib.import_module(f"jaxformers.optimizers.{optimizer_name}")

    train_mask = None
    if model.is_lora:
        train_mask = mask_trainable_lora(model.weights)

    train_weights, _ = tree_util.partition(model.weights, train_mask)
    tx = optimizer_module.make(scheduler, **optimizer_config)
    opt_state = tx.init(train_weights)
    return dataclasses.replace(model, opt_state=opt_state, tx=tx, train_mask=train_mask)
