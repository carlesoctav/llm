import dataclasses
import importlib
from enum import auto, StrEnum

from jaxtyping import PRNGKeyArray

from jaxformers.benchmark_utils import print_timing


class InitMethod(StrEnum):
    PRETRAINED = auto()
    RANDOM = auto()
    PYTREE = auto()


@print_timing
def make_model(
    model_name: str,
    init_method: str | None,
    model_config: dict,
    *,
    rngs: PRNGKeyArray | None = None,
):
    model_module = importlib.import_module(f"jaxformers.models.{model_name}")

    if init_method is None or init_method == InitMethod.PRETRAINED:
        model = model_module.load(**model_config)
    elif init_method == InitMethod.RANDOM:
        model = model_module.init(**model_config, rngs=rngs)
    elif init_method == InitMethod.PYTREE:
        model = model_module.load_pytree(**model_config)
    else:
        raise ValueError(f"Unsupported model init method: {init_method!r}")
    return model
