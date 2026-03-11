from typing import Callable

import optax

from .lr import custom_scale_by_learning_rate as custom_scale_by_learning_rate


def make(
    learning_rate: float | Callable[[int], float],
    max_grad_norm: float | None = 1.0,
    momentum: float = 0.0,
    nesterov: bool = False,
    **kwargs,
):
    components = []
    if max_grad_norm:
        components.append(optax.clip_by_global_norm(max_grad_norm))

    if momentum:
        components.append(optax.trace(decay=momentum, nesterov=nesterov))

    components.append(custom_scale_by_learning_rate(learning_rate))
    return optax.chain(*components)
