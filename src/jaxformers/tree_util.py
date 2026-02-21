from jaxformers.print_utils import tree_pprint
import jax.tree_util as jtu
from jax.tree_util import (
    DictKey,
    FlattenedIndexKey,
    GetAttrKey,
    KeyEntry,
    KeyPath,
    SequenceKey,
)


def optimizerstr(keys: KeyPath, separator: str = "/") -> str:
    str_fn = _optimizer_entrystr
    return separator.join(map(str_fn, keys))


def partition(pytree, filter, replace=None, is_leaf=None):
    if filter is None:
        return pytree, None

    left = jtu.tree_map(lambda m, v: v if m else None, filter, pytree, is_leaf=is_leaf)
    right = jtu.tree_map(
        lambda m, v: v if not m else None, filter, pytree, is_leaf=is_leaf
    )
    return left, right


def combine(left, right, is_leaf = None):
    def _combine(*args):
        for arg in args:
            if arg is not None:
                return arg

    is_none = lambda x: x is None
    _is_leaf = is_none if is_leaf is None else lambda x: is_none(x) or is_leaf(x)
    return jtu.tree_map(_combine, left, right, is_leaf = _is_leaf)


def apply_updates(weights, updates):
    def _f(w, u):
        if u is None:
            return w
        else:
            return w + u

    is_none = lambda x: x is None
    return jtu.tree_map(_f, weights, updates, is_leaf = is_none)


def _optimizer_entrystr(key: KeyEntry) -> str:
    match key:
        case DictKey(key=key) | GetAttrKey(name=key) | FlattenedIndexKey(key=key):
            return str(key)
        case SequenceKey(idx=key):
            return ""
        case _:
            return str(key)
