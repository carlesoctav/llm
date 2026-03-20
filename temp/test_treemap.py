import numpy as np
import optax
from jax.tree_util import tree_map


optax.MultiSteps
optax.apply_every
optax.global_norm


list_of = [{"a": 1, "b": 100}, {"a": 10, "b": 1000}, {"a": 100, "b": 10000, "c": 10}]


def add(*all):
    return np.mean(all)


b = tree_map(add, *list_of)
print("DEBUGPRINT {b}:", b)
