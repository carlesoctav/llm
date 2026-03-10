import importlib

def make_logger(logger_name: str, logger_config: dict):
    logger_module = importlib.import_module(f"jaxformers.logger.{logger_name}")
    logger = logger_module.make(logger_config)
    logger.finish
    return logger
