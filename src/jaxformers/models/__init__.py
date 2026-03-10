from enum import StrEnum, auto
import importlib


class InitMethod(StrEnum):
    PRETRAINED = auto()
    RANDOM = auto()
    PYTREE = auto()


#util for config base model loader
def make_model(model_name, init_method, model_config):
    model_module = importlib.import_module(f"jaxformers.models.{model_name}")
    if init_method == InitMethod.PRETRAINED:
        model = model_module.load(model_config)
    elif init_method == InitMethod.RANDOM:
        model = model_module.init(model_config)
    elif init_method == InitMethod.PYTREE:
        model = model_module.load_pytree(model_config)
    else:
        raise ValueError
    return model
