import optax
from typing import Callable

def make(
    learning_rate: float | Callable[[int], float],
    grad_accum: int,
    use_grad_accum_mean: bool = True,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    max_grad_norm: float | None  = 1.0,
    **kwargs,
):
    components = []
    if max_grad_norm:
        components.append(optax.clip_by_global_norm(max_grad_norm))

    components.append(
        optax.sgd(
            learning_rate = learning_rate,
            # b1 = b1,
            # b2 = b2,
            # eps = eps,
        ),
    )

    if grad_accum > 1:
       tx = optax.chain(*components)
       return optax.MultiSteps(tx, grad_accum, use_grad_accum_mean)
    else:
        return optax.chain(*components)
