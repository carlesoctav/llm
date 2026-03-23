from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Scalar


np.sum

from jaxformers import tree_util


def process_aux(accum_aux: dict[str, Any], namespace=""):
    def finalize(v: Any) -> Any:
        if isinstance(v, tuple):
            numer, denom = v
            return numer / denom if denom else 0.0
        return v

    prefix = f"{namespace}/" if namespace else ""
    return {f"{prefix}{k}": finalize(v) for k, v in accum_aux.items()}


def jittable_add_aux(*aux, reduce_method: dict | None):
    is_tuple = lambda x: isinstance(x, tuple)
    use_predefine_method = True if reduce_method else False

    def f(path, *leaf):
        if use_predefine_method:
            method = reduce_method[jtu.keystr(path, simple=True)]
            match method:
                case "max":
                    return jnp.max(jnp.asarray(leaf))
                case "sum":
                    return jnp.sum(jnp.asarray(leaf))
                case "min":
                    return jnp.max(jnp.asarray(leaf))
                case "mean":
                    t0, t1 = zip(*leaf)
                    return (jnp.sum(jnp.asarray(t0)), jnp.sum(jnp.asarray(t1)))
        else:
            if isinstance(leaf[0], tuple):
                t0, t1 = zip(*leaf)
                return (jnp.sum(jnp.asarray(t0)), jnp.sum(jnp.asarray(t1)))
            elif isinstance(leaf[0], Scalar):
                return jnp.sum(leaf)
            else:
                path_str = jtu.keystr(path, simple=True)
                raise ValueError(
                    f"Unsupported leaf type(s) at path '{path_str}'."
                    "Expected leafs to be either tuples or jaxtyping.Array instances."
                )

    return jtu.tree_map_with_path(f, *aux, is_leaf=is_tuple)

def host_add_aux(*aux, reduce_method: dict | None):
    is_tuple = lambda x: isinstance(x, tuple)
    use_predefine_method = True if reduce_method else False

    def f(path, *leaf):
        if use_predefine_method:
            method = reduce_method[jtu.keystr(path, simple=True)]
            match method:
                case "max":
                    return np.max(leaf)
                case "sum":
                    return np.sum(leaf)
                case "min":
                    return np.max(leaf)
                case "mean":
                    t0, t1 = zip(*leaf)
                    return (np.sum(t0), np.sum(t1))
        else:
            if isinstance(leaf[0], tuple):
                t0, t1 = zip(*leaf)
                return (np.sum(t0), np.sum(t1))
            elif isinstance(leaf[0], (int, float, np.ndarray)):
                return np.sum(leaf)
            else:
                path_str = jtu.keystr(path, simple=True)
                raise ValueError(
                    f"Unsupported leaf type(s) at path '{path_str}'."
                    "Expected leafs to be either tuples or jaxtyping.Array instances."
                )

    return jtu.tree_map_with_path(f, *aux, is_leaf=is_tuple)


def to_host(tree, flatten=False, unpack=False):
    tree = jax.device_get(tree)
    tree = (
        tree_util.flatten(tree, is_leaf=lambda x: isinstance(x, tuple))
        if flatten
        else tree
    )

    if unpack:
        def _item(value):
            if isinstance(value, np.ndarray) and value.shape == ():
                return value.item()
            if isinstance(value, np.generic):
                return value.item()
            return value
        return jax.tree.map(_item, tree)

    return tree
