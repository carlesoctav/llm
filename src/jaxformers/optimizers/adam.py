from typing import Callable

import optax
from jaxtyping import Bool, PyTree

from .base import make as make_base
from .lr import custom_scale_by_learning_rate as custom_scale_by_learning_rate


def make(
    learning_rate: float | Callable[[int], float],
    grad_accum: int,
    max_grad_norm: float | None = 1.0,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    *,
    freeze_mask: PyTree[Bool] | None = None,
    **kwargs,
):

    components = []
    components.append(
        make_base(grad_accum, max_grad_norm),
    )

    components.append(
        optax.scale_by_adam(
            b2=b2,
            b1=b1,
            eps=eps,
        ),
    )

    components.append(
        custom_scale_by_learning_rate(learning_rate)
    )

    if freeze_mask:
        components.append(optax.freeze(freeze_mask))

    return optax.chain(*components)
