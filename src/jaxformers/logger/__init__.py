from jaxformers.benchmark_utils import print_timing
import importlib


@print_timing
def make_logger(logger_name: str, logger_config: dict):
    logger_module = importlib.import_module(f"jaxformers.logger.{logger_name}")
    return logger_module.make(**logger_config)
