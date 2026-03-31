import importlib
import inspect
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

import jax

from jaxformers.benchmark_utils import print_timing
from jaxformers.distributed.parallel import with_logical_axis


def resolve_model_target(model_name: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(model_name):
        return model_name

    parts = model_name.split(".")
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
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
    mesh = model_kwargs.pop("mesh", None)
    rule = model_kwargs.pop("rule", None)
    with ExitStack() as stack:
        if mesh is not None:
            stack.enter_context(jax.set_mesh(mesh))
        if rule is not None:
            stack.enter_context(with_logical_axis(rule))
        return model_factory(*model_args, **model_kwargs)
