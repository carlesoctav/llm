from typing import Callable

import optax

from .divide import divide_every
from .log_grad_norm import log_grad_norm

def make(
    learning_rate: float | Callable[[int], float],
    grad_accum: int,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    max_grad_norm: float | None  = 1.0,
    **kwargs,
):
    components = []
    components.append(optax.apply_every(grad_accum))
    components.append(divide_every(grad_accum))
    components.append(log_grad_norm())

    if max_grad_norm:
        components.append(optax.clip_by_global_norm(max_grad_norm))

    components.append(
        optax.adam(
            learning_rate = learning_rate,
            b2 = b2,
            b1 = b1,
            eps = eps,
        ),
    )

    return optax.chain(*components)
