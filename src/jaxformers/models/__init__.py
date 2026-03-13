import dataclasses
import importlib
from enum import StrEnum, auto
from jaxtyping import PRNGKeyArray


class InitMethod(StrEnum):
    PRETRAINED = auto()
    RANDOM = auto()
    PYTREE = auto()


def make_model(
    model_name: str,
    init_method: InitMethod | str | None,
    model_config: dict,
    *,
    rngs:PRNGKeyArray | None = None
):
    model_module = importlib.import_module(
        f"jaxformers.models.{model_name}"
    )

    if init_method in (None, InitMethod.PRETRAINED, "pretrained"):
        model = model_module.load(**model_config)
    elif init_method in (InitMethod.RANDOM, "random"):
        model = model_module.init(**model_config, rngs = rngs)
    elif init_method in (InitMethod.PYTREE, "pytree"):
        model = model_module.load_pytree(**model_config)
    else:
        raise ValueError(f"Unsupported model init method: {init_method!r}")
    return model


def prepare_weights(model_name: str, model):
    model_module = importlib.import_module(f"jaxformers.models.{model_name}")
    prepare_fn = getattr(model_module, "prepare_weights", None)
    if prepare_fn is None:
        return model

    weights = prepare_fn(model.config, model.weights)
    if weights is model.weights:
        return model
    return dataclasses.replace(model, weights=weights)
