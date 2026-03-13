import dataclasses
import importlib

from jaxformers import tree_util
from jaxformers.optimizer_utils import mask_trainable_lora




def make_optimizer(optimizer_name: str, model, scheduler, optimizer_config: dict):
    optimizer_module = importlib.import_module(f"jaxformers.optimizers.{optimizer_name}")
    if not getattr(optimizer_module, "make"):
        raise ValueError(f"{optimizer_module!r} does not have a 'make' method; please ensure you're using the correct optimizer_name.")

    train_mask = None
    if model.is_lora:
        train_mask = mask_trainable_lora(model.weights)

    train_weights, _ = tree_util.partition(model.weights, train_mask)
    tx = optimizer_module.make(scheduler, **optimizer_config)
    opt_state = tx.init(train_weights)
    return dataclasses.replace(model, opt_state=opt_state, tx=tx, train_mask=train_mask)
