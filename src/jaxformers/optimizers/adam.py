from typing import Callable

import jax.tree_util as jtu
import optax
from jaxtyping import Bool, PyTree

from .base import make_opt_base_components
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

    train_mask = None
    if freeze_mask is not None:
        # `freeze_mask=True` means "frozen". Optax masking expects `True` for the
        # leaves we want to *apply* transforms to, so we invert it.
        train_mask = jtu.tree_map(lambda m: not m, freeze_mask)

    def maybe_mask(tx: optax.GradientTransformation) -> optax.GradientTransformation:
        return optax.masked(tx, train_mask) if train_mask is not None else tx

    components = [maybe_mask(tx) for tx in make_opt_base_components(grad_accum)]

    if max_grad_norm:
        components.append(maybe_mask(optax.clip_by_global_norm(max_grad_norm)))

    components.append(
        maybe_mask(
            optax.scale_by_adam(
                b2=b2,
                b1=b1,
                eps=eps,
            )
        ),
    )

    components.append(
        maybe_mask(custom_scale_by_learning_rate(learning_rate))
    )

    if freeze_mask is not None:
        components.append(optax.freeze(freeze_mask))

    return optax.chain(*components)
