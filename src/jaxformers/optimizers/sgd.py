from typing import Callable

import optax

from .base import make_opt_base_components
from .lr import custom_scale_by_learning_rate


def make(
    learning_rate: float | Callable[[int], float],
    grad_accum: int,
    max_grad_norm: float | None = 1.0,
    momentum: float | None = None,
    nesterov: bool = False,
    weight_decay: float | None = None,
    **kwargs,
):
    del kwargs

    components = make_opt_base_components(grad_accum)
    if max_grad_norm:
        components.append(optax.clip_by_global_norm(max_grad_norm))

    if weight_decay:
        components.append(optax.add_decayed_weights(weight_decay))

    if momentum:
        components.append(optax.trace(decay=momentum, nesterov=nesterov))

    components.append(custom_scale_by_learning_rate(learning_rate))
    return optax.chain(*components)
