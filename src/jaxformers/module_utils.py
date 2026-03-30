import dataclasses
from enum import auto, StrEnum
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxformers.scan_utils import make_scan_fwd


M = TypeVar("M", bound=eqx.Module)


class StackImpl(StrEnum):
    STACK = auto()
    FREE = auto()


def module_replace(module, **kwargs):
    "dataclasses replace for eqx.Module"


@dataclasses.dataclass
class Stackable:
    argnums: tuple[int, ...] = eqx.field(static=True, default=0)
    argnames: tuple[str, ...] = eqx.field(static=True, default=())
    in_axes: int = eqx.field(static=True, default=0)

    weights_impl: StackImpl = eqx.field(static=True, default="free")


class StackModule(eqx.Module, Generic[M]):
    layers: M
    module: type[M] = eqx.field(static=True)

    argnums: tuple[int, ...] = eqx.field(static=True)
    argnames: tuple[str, ...] = eqx.field(static=True)
    in_axes: tuple[int, ...] = eqx.field(static=True)
    length: int = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __init__(
        self,
        module: type[M],
        layers: list[M],
        argnums: int | tuple[int, ...],
        *,
        argnames: str | tuple[str, ...] = (),
        in_axes: int | tuple[int, ...] = 0,
        remat: bool = False,
    ):
        def _stack(*leaf):
            if leaf[0] is None:
                return None
            return jnp.stack(leaf)

        self.module = module
        self.argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
        self.argnames = (argnames,) if isinstance(argnames, str) else tuple(argnames)
        if isinstance(in_axes, int):
            self.in_axes = (in_axes,) * (len(self.argnums) + len(self.argnames))
        else:
            self.in_axes = tuple(in_axes)
        self.remat = remat

        if isinstance(layers, list):
            self.length = len(layers)
            self.layers = jax.tree.map(_stack, *layers, is_leaf=lambda x: x is None)
            return

        self.layers = layers
        self.length = next(
            leaf.shape[0]
            for leaf in jax.tree.leaves(layers, is_leaf=lambda x: x is None)
            if leaf is not None
        )

    def unstack(self):
        def _unstack_leaf(leaf):
            if leaf is None:
                return None
            return jnp.unstack(leaf)

        trees = jax.tree.map(_unstack_leaf, self.layers, is_leaf=lambda x: x is None)
        length = None
        for leaf in jax.tree.leaves(trees, is_leaf=lambda x: isinstance(x, tuple)):
            if isinstance(leaf, tuple):
                length = len(leaf)
                break
        if length is None:
            raise ValueError("Cannot unstack a layer tree without array leaves.")

        return [
            jax.tree.map(
                lambda leaf: None if leaf is None else leaf[layer_idx],
                trees,
                is_leaf=lambda x: x is None or isinstance(x, tuple),
            )
            for layer_idx in range(length)
        ]

    def __call__(self, *args, **kwargs):
        module_call = (
            jax.remat(self.module.__call__) if self.remat else self.module.__call__
        )

        def stack_fwd(carry, layer, *fwd_args, **fwd_kwargs):
            return module_call(layer, carry, *fwd_args, **fwd_kwargs)

        return make_scan_fwd(
            stack_fwd,
            self.length,
            self.argnums,
            argnames=self.argnames,
            in_axes=self.in_axes,
        )(*args, self.layers, **kwargs)
