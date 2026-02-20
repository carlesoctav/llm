from typing import Callable

import optax
from jaxtyping import Bool, PyTree

from .divide import divide_every
from .log_grad_norm import log_grad_norm


def make(
    grad_accum: int,
    max_grad_norm: float | None = 1.0,
):
    components = []
    components.append(optax.apply_every(grad_accum))
    components.append(divide_every(grad_accum))
    components.append(log_grad_norm())

    if max_grad_norm:
        components.append(optax.clip_by_global_norm(max_grad_norm))

    return optax.chain(*components)
