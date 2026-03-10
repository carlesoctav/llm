from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import jax

from jaxformers.module_utils import Module, ParamTree


@dataclass
class Attention(Module):
    pass


@dataclass
class MLP(Module):
    pass


@dataclass
class TransformersLayer(Module):
    pass


@dataclass
class Transformers(Module):
    layers: Sequence[TransformersLayer] = field(default_factory=list)

    def init(self, rngs) -> ParamTree:
        if not self.layers:
            return {}
        keys = jax.random.split(rngs, len(self.layers))
        return {
            "layers": [
                layer.init(key)
                for layer, key in zip(self.layers, keys, strict=True)
            ]
        }
