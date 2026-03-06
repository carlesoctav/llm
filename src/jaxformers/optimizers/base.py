from typing import Callable

import optax
from jaxtyping import Bool, PyTree

from .divide import divide_every
from .log_grad_norm import log_grad_norm


def make_opt_base_components( grad_accum: int):
    components = []
    # components.append(optax.apply_every(grad_accum))
    # components.append(divide_every(grad_accum))
    components.append(log_grad_norm())
    return components
