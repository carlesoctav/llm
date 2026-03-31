import importlib
import inspect
import os
from collections.abc import Callable
from typing import Any

from jaxformers.benchmark_utils import print_timing


MODEL_DIR = os.environ.get("MODEL_DIR", "jaxformers.models")


def resolve_model_target(model_name: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(model_name):
        return model_name

    parts = model_name.split(".")
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        module_name = f"{MODEL_DIR}.{module_name}"
        try:
            target = importlib.import_module(module_name)
            break
        except ModuleNotFoundError:
            continue
    else:
        raise ValueError(f"Could not resolve model target {model_name!r}")

    for attr in parts[i:]:
        target = getattr(target, attr)

    if not callable(target):
        raise TypeError(f"Model target {model_name!r} is not callable")

    return target


def model_accepts_kwarg(model_name: str | Callable[..., Any], kwarg_name: str) -> bool:
    signature = inspect.signature(resolve_model_target(model_name))

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    return kwarg_name in signature.parameters


@print_timing
def make_model(model_name: str | Callable[..., Any], *model_args, **model_kwargs):
    model_factory = resolve_model_target(model_name)
    return model_factory(*model_args, **model_kwargs)
