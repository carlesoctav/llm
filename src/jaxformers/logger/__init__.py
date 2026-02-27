import importlib

import sws


def load(config: sws.FinalConfig, logger_name: str):
    logger_module = importlib.import_module(f"jaxformers.logger.{logger_name}")
    make = getattr(logger_module, "make", None)
    if not callable(make):
        raise ValueError(f"logger module jaxformers.logger.{logger_name} must define make(config)")
    logger = make(config)
    finish = getattr(logger, "finish", None)
    if not callable(finish):
        raise ValueError(f"logger {logger_name!r} must implement finish()")
    return logger
