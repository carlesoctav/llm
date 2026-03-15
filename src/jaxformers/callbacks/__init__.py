import importlib
from functools import partial

from jaxformers.benchmark_utils import print_timing

from .base import callback_chain


@print_timing
def make_callbacks(
    callback_names: str | list[str],
    callback_config: dict,
):
    callbacks = []
    if isinstance(callback_names, list):
        for single_callback_name in callback_names:
            callback_module = importlib.import_module(
                f"jaxformers.callbacks.{single_callback_name}"
            )
            callback_kwargs = callback_config.get(single_callback_name, {})
            callbacks.append(callback_module.make(**callback_kwargs))
        return callback_chain(*callbacks)
    elif isinstance(callback_names, str):
        single_callback_name = callback_names
        callback_module = importlib.import_module(
            f"jaxformers.callbacks.{single_callback_name}"
        )
        callback_kwargs = (
            callback_config.get(single_callback_name, {}) or callback_config
        )
        callback = callback_module.make(**callback_kwargs)
        return callback
