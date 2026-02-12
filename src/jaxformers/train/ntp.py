import importlib
import jax
import jax.numpy as jnp
import sws
import optax
from flax.training import train_state

def get_config():
    config = sws.Config()
    config.model_name = "qwen3"
    config.resume = False
    config.random_init = False

    config.model.model_id = "Qwen/Qwen3-4B-Instruct-2507"
    config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 4}
    config.model.devices = jax.devices()
    config.model.param_dtype = lambda: jnp.float32

    return config


def load_model(name: str, model_config: sws.FinalConfig):
    model_module = importlib.import_module(f"jaxformers.models.{name}")
    model = model_module.load(**model_config.to_dict())
    return model


def load_optimizer(model, opt_name, opt_config: sws.FinalConfig):
    pass



def main(config: sws.FinalConfig):
    if not config.random_init and not config.resume:
        model = load_model(config.model_name, config.model)
        model = load_optimizer(model, config.opt_name, config.opt)
    else:
        raise NotImplementedError

    print("DEBUGPRINT {model}:", model)


if __name__ == "__main__":
    sws.run(main)
    # sws.run(main)
