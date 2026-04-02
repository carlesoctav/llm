import importlib
from functools import partial

from jaxformers.benchmark_utils import print_timing

from .base import callback_chain


@print_timing
def make_callbacks(
    callback_configs: dict,
):
    callbacks = []
    for name, callback_kwargs in callback_configs.items():
        if callback_kwargs:
            callback_module = importlib.import_module(
                f"jaxformers.callbacks.{name}"
            )
            callbacks.append(callback_module.make(**callback_kwargs))
        else:
            print(
                f"Callback '{name}' is present in config.callback but its value is None, "
                f"so it will be ignored. To enable this callback with the default configuration, "
                f"set config.callback.{name} = {{}}."
            )
    return callback_chain(*callbacks)
