from jax.tree_util import tree_map
import numpy as np


list_of = [{"a": 1, "b": 100}, {"a": 10, "b": 1000}, {"a": 100, "b": 10000}]

def add (*all):
    return np.mean(all)

b = tree_map(add, *list_of)
print("DEBUGPRINT {b}:", b)
