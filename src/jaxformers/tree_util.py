import re
from functools import partial
from operator import itemgetter

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.tree_util import (
    DictKey,
    FlattenedIndexKey,
    GetAttrKey,
    KeyEntry,
    KeyPath,
    SequenceKey,
)
from jaxtyping import Array


def optimizerstr(keys: KeyPath, separator: str = "/") -> str:
    str_fn = _optimizer_entrystr
    return separator.join(map(str_fn, keys))


def partition(pytree, filter=None, replace=None, is_leaf=None):
    if filter is None:
        return pytree, jtu.tree_map(lambda x: None, pytree)

    left = jtu.tree_map(lambda m, v: v if m else None, filter, pytree, is_leaf=is_leaf)
    right = jtu.tree_map(
        lambda m, v: v if not m else None, filter, pytree, is_leaf=is_leaf
    )
    return left, right


def combine(*val, is_leaf=None):
    def _combine(*args):
        for arg in args:
            if arg is not None:
                return arg

    is_none = lambda x: x is None
    _is_leaf = is_none if is_leaf is None else lambda x: is_none(x) or is_leaf(x)
    return jtu.tree_map(_combine, *val, is_leaf=_is_leaf)


def apply_updates(model, updates):
    def _f(m, u):
        if u is None:
            return m
        else:
            return (m + u).astype(jnp.asarray(m).dtype)

    is_none = lambda x: x is None
    return jtu.tree_map(_f, model, updates, is_leaf=is_none)


def _optimizer_entrystr(key: KeyEntry) -> str:
    match key:
        case DictKey(key=key) | GetAttrKey(name=key) | FlattenedIndexKey(key=key):
            return str(key)
        case SequenceKey(idx=key):
            return ""
        case _:
            return str(key)


def copy(tree, stop_gradient=True):
    def _f(leaf):
        if stop_gradient:
            return jax.lax.stop_gradient(leaf.copy())
        else:
            leaf.copy()

    return jax.tree.map(_f, tree)


def stack(*trees):
    return jax.tree.map(lambda *leaf: jnp.stack(leaf), *trees)


def unstack(trees):
    trees = jax.tree.map(lambda leaf: jnp.unstack(leaf), trees)
    N = len(jax.tree.leaves(trees)[0])
    return [jax.tree.map(lambda leaf: leaf[i], trees) for i in range(N)]


def maybe_stack(trees: list | dict):
    if not isinstance(trees, list):
        return trees
    return jax.tree.map(lambda *leaf: jnp.stack(leaf), *trees)


def maybe_unstack(trees: list | dict):
    if not isinstance(trees, dict):
        return trees
    trees = jax.tree.map(lambda leaf: jnp.unstack(leaf), trees)
    N = len(jax.tree.leaves(trees, is_leaf=lambda x: isinstance(x, tuple))[0])
    return [
        jax.tree.map(
            lambda leaf: leaf[i], trees, is_leaf=lambda x: isinstance(x, tuple)
        )
        for i in range(N)
    ]


def flatten(tree, separator=".", is_leaf=None):
    res = {}

    def _f(path, leaf):
        res[jtu.keystr(path, simple=True, separator=separator)] = leaf

    jax.tree.map_with_path(_f, tree, is_leaf=is_leaf)
    return res


def unflatten(arg):
    """Unflatten nested dict/array data.

    This function takes a single argument which may either be a
    ``dict`` (or any object having a dict-like ``.items()`` or
    ``.iteritems()`` method) or a sequence of ``(key, value)`` pairs.
    The keys in the ``dict`` or sequence should must all be strings.

    Examples
    --------

    Nested ``dict``\s::

    >>> unflatten({'foo.bar': 'val'})
    {'foo': {'bar': 'val'}}

    Nested ``list``::

    >>> unflatten({'foo[0]': 'val', 'foo[1]': 'bar'})
    {'foo': ['val', 'bar']}

    Nested ``list``\s::

    >>> unflatten({'foo[0][0]': 'val'})
    {'foo': [['val']]}

    Lists of ``dict``\s::

    >>> unflatten({'foo[0].bar': 'val',
    ...            'foo[1].baz': 'x'})
    {'foo': [{'bar': 'val'}, {'baz': 'x'}]}

    """

    class Holder(dict):
        def __init__(self, flat_key):
            self.flat_key = flat_key
            self.data = {}

        def __contains__(self, key):
            return key in self.data

        def __getitem__(self, key):
            return self.data[key]

        def get(self, key):
            return self.data.get(key)

        def __setitem__(self, key, value):
            self.data[key] = value

    class DictHolder(Holder):
        node_type = dict

        def getvalue(self):
            return self.data

    class ListHolder(Holder):
        node_type = list

        def getvalue(self):
            items = sorted(self.data.items(), key=itemgetter(0))
            value = []
            for n, (key, val) in enumerate(items):
                if key != n:
                    assert key > n
                    missing_key = f"{self.flat_key}[{n}]"
                    raise ValueError(f"missing key {missing_key!r}")
                value.append(val)
            return value

    def node_type(value):
        if isinstance(value, Holder):
            return (value.node_type,)
        return "terminal"

    dot_or_indexes_re = re.compile(r"(\.|(?:\[\d+\])+(?=\.|\Z))")

    def parse_key(flat_key):
        if not isinstance(flat_key, str):
            raise TypeError("keys must be strings")

        split_key = dot_or_indexes_re.split(flat_key)
        parts = [split_key[0]]
        for i in range(1, len(split_key), 2):
            sep = split_key[i]
            if sep == ".":
                parts.append(split_key[i + 1])
            else:
                parts.extend(map(int, re.findall(r"\d+", sep)))
        return parts

    def unparse_key(parsed):
        bits = []
        for part in parsed:
            if isinstance(part, str):
                fmt = ".%s" if bits else "%s"
            else:
                fmt = "[%d]"
            bits.append(fmt % part)
        return "".join(bits)

    if hasattr(arg, "iteritems"):
        items = arg.iteritems()
    elif hasattr(arg, "items"):
        items = arg.items()
    else:
        items = arg

    data = {}
    holders = []
    for flat_key, val in items:
        parsed_key = parse_key(flat_key)
        obj = data
        for depth, (key, next_key) in enumerate(zip(parsed_key, parsed_key[1:]), 1):
            if isinstance(next_key, str):
                holder_type = DictHolder
            else:
                holder_type = ListHolder

            if key not in obj:
                obj[key] = holder_type(unparse_key(parsed_key[:depth]))
                holders.append((obj, key))
            elif not isinstance(obj[key], holder_type):
                raise ValueError(
                    f"conflicting types {node_type(obj[key])} and {holder_type.node_type} "
                    f"for key {unparse_key(parsed_key[:depth])!r}"
                )
            obj = obj[key]

        last_key = parsed_key[-1]
        if isinstance(obj.get(last_key), Holder):
            raise ValueError(
                f"conflicting types {node_type(obj[last_key])} and terminal for key {flat_key!r}"
            )
        obj[last_key] = val

    for obj, key in reversed(holders):
        obj[key] = obj[key].getvalue()

    return data


def to_abstract(tree):
    def _f(leaf):
        if isinstance(leaf, Array):
            return jax.ShapeDtypeStruct(
                shape=leaf.shape, dtype=leaf.dtype, sharding=leaf.sharding
            )
        else:
            return leaf

    return jax.tree.map(_f, tree)


def get_by_path(obj, path):
    for key in path:
        match key:
            case jtu.GetAttrKey(name):
                obj = getattr(obj, name)
            case jtu.DictKey(name):
                obj = obj[name]
            case jtu.SequenceKey(idx):
                obj = obj[idx]
            case jtu.FlattenedIndexKey(idx):
                obj = obj[idx]
            case _:
                raise TypeError(f"Unsupported key path element: {key!r}")
    return obj
