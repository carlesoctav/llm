import importlib

from .base import callback_chain
from .log_grad_norm import log_grad_norm
from .log_learning_rate import log_learning_rate
from .log_performance import log_performance


def make_callbacks(callback_specs):
    if not callback_specs:
        return None

    callbacks = []
    for callback in callback_specs:
        if isinstance(callback, str):
            callback_module = importlib.import_module(f"jaxformers.callbacks.{callback}")
            callbacks.append(callback_module.make())
        elif isinstance(callback, tuple):
            callback_name, callback_kwargs = callback
            callback_module = importlib.import_module(
                f"jaxformers.callbacks.{callback_name}"
            )
            callbacks.append(callback_module.make(**callback_kwargs))
        else:
            raise ValueError(
                f"Invalid callback specification: {callback!r}. "
                "Expected either a callback name or a (name, kwargs) tuple."
            )

    return callback_chain(*callbacks)
